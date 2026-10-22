"""Independent offline processing contract tests; never open an audio device."""

import hashlib
import json
import wave

import numpy as np
import pytest

from hum_dsp.audio import read_pcm16_wav
from hum_dsp.cli import main
from hum_dsp.config import ProcessorConfig
from hum_dsp import processing


RATE = 48_000


def _write_codes(path, codes, *, rate=RATE, channels=1, width=2):
    """Encode test input independently of the production WAV writer."""
    values = np.asarray(codes, dtype="<i2")
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(channels)
        stream.setsampwidth(width)
        stream.setframerate(rate)
        if width == 2:
            stream.writeframes(values.tobytes())
        else:
            stream.writeframes(bytes(values.size * width))


def _codes(path):
    with wave.open(str(path), "rb") as stream:
        assert stream.getnchannels() == 1
        assert stream.getsampwidth() == 2
        assert stream.getframerate() == RATE
        assert stream.getcomptype() == "NONE"
        return np.frombuffer(stream.readframes(stream.getnframes()), dtype="<i2").copy()


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tone_codes(hz, *, seconds=6.0, amplitude=0.2):
    time = np.arange(round(seconds * RATE), dtype=np.float64) / RATE
    return np.rint(amplitude * np.sin(2 * np.pi * hz * time) * 32768).astype("<i2")


def test_zero_depth_preserves_every_pcm16_code_without_mutating_source(tmp_path):
    path = tmp_path / "source.wav"
    source = np.arange(-32768, 32768, dtype=np.int32).astype("<i2")
    _write_codes(path, source)
    original = path.read_bytes()
    output = tmp_path / "disabled"
    result = processing.process_wav(path, output, ProcessorConfig(max_attenuation_db=0))
    np.testing.assert_array_equal(_codes(output / "processed.wav"), source)
    assert path.read_bytes() == original
    assert result["output_gain_normalization_applied"] is False


@pytest.mark.parametrize("hz", [50, 60])
def test_stationary_tone_is_reduced_at_least_ten_db_after_settling(tmp_path, hz):
    path = tmp_path / f"tone_{hz}.wav"
    _write_codes(path, _tone_codes(hz))
    original = path.read_bytes()
    output = tmp_path / "result"
    processing.process_wav(path, output, ProcessorConfig(nominal_hum_hz=hz))
    source = _codes(path).astype(np.float64)[2 * RATE:]
    filtered = _codes(output / "processed.wav").astype(np.float64)[2 * RATE:]
    attenuation = 20 * np.log10(np.sqrt(np.mean(source**2)) / np.sqrt(np.mean(filtered**2)))
    assert attenuation >= 10.0
    assert attenuation <= 12.1  # Finite-depth filter, not unlimited cancellation.
    assert filtered.size == 4 * RATE
    assert path.read_bytes() == original


def test_report_describes_actual_saved_files_without_unavailable_quality_claims(tmp_path):
    path = tmp_path / "입력 파일.wav"
    _write_codes(path, _tone_codes(60, seconds=0.1))
    config = ProcessorConfig(nominal_hum_hz=60, max_attenuation_db=6, block_size_frames=127)
    output = tmp_path / "report output"
    result = processing.process_wav(path, output, config)
    assert result == json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert result["engine"] == "fixed_finite_depth_notch"
    assert result["config"] == config.to_dict()
    assert result["output_gain_normalization_applied"] is False
    assert result["reference_quality_metrics"] is None
    assert result["device_io"] is False
    assert result["algorithm"]["explicit_delay_samples"] == 0
    assert {file.name for file in output.iterdir()} == {"processed.wav", "report.json"}
    for key, audio_path in (("input", path), ("output", output / "processed.wav")):
        saved = read_pcm16_wav(audio_path)
        metadata = result[key]
        assert metadata["sha256"] == _hash(audio_path)
        assert metadata["file_name"] == audio_path.name
        assert metadata["frames"] == saved.size
        assert metadata["sample_rate_hz"] == RATE
        assert metadata["peak_abs"] == pytest.approx(float(np.max(np.abs(saved))), abs=1e-12)
        assert metadata["rms"] == pytest.approx(float(np.sqrt(np.mean(saved**2))), abs=1e-12)
    json.dumps(result, allow_nan=False)


def test_kernel_output_is_quantized_by_rounding_without_gain_normalization(tmp_path, monkeypatch):
    path = tmp_path / "source.wav"
    _write_codes(path, [0] * 8)
    expected_float = np.array([-1, -0.123456, -0.5 / 32768, 0, 0.5 / 32768, 1.5 / 32768, 0.234567, 0.99999])
    monkeypatch.setattr(processing, "process_audio", lambda *args, **kwargs: expected_float.copy())
    output = tmp_path / "quantized"
    result = processing.process_wav(path, output)
    expected_codes = np.clip(np.rint(expected_float * 32768), -32768, 32767).astype("<i2")
    np.testing.assert_array_equal(_codes(output / "processed.wav"), expected_codes)
    assert result["output_gain_normalization_applied"] is False


@pytest.mark.parametrize("invalid_value", [1.0, 1.1, -1.1, float("nan"), float("inf")])
def test_invalid_floating_output_is_rejected_before_destination_creation(tmp_path, monkeypatch, invalid_value):
    path = tmp_path / "source.wav"
    _write_codes(path, [0] * 16)
    before = path.read_bytes()
    monkeypatch.setattr(processing, "process_audio", lambda *args, **kwargs: np.full(16, invalid_value))
    output = tmp_path / "must not be created"
    with pytest.raises((TypeError, ValueError)):
        processing.process_wav(path, output)
    assert not output.exists()
    assert path.read_bytes() == before


@pytest.mark.parametrize("invalid_output", [np.zeros(15), np.zeros(17), np.zeros((16, 1)), np.array([], dtype=np.float64)])
def test_invalid_output_shape_or_length_cannot_silently_change_audio(tmp_path, monkeypatch, invalid_output):
    path = tmp_path / "source.wav"
    _write_codes(path, [0] * 16)
    before = path.read_bytes()
    monkeypatch.setattr(processing, "process_audio", lambda *args, **kwargs: invalid_output.copy())
    output = tmp_path / "absent"
    with pytest.raises((TypeError, ValueError)):
        processing.process_wav(path, output)
    assert not output.exists()
    assert path.read_bytes() == before


@pytest.mark.parametrize("settings", [{"rate": 8000}, {"rate": 44100}, {"channels": 2}, {"width": 1}, {"width": 3}])
def test_unsupported_input_is_rejected_without_output(tmp_path, settings):
    path = tmp_path / "unsupported.wav"
    _write_codes(path, [0] * 32, **settings)
    before = path.read_bytes()
    output = tmp_path / "not created"
    with pytest.raises(ValueError):
        processing.process_wav(path, output)
    assert not output.exists()
    assert path.read_bytes() == before


@pytest.mark.parametrize("payload", [b"", b"not a WAV", b"RIFF\x24\x00\x00\x00WAVE"])
def test_malformed_input_is_rejected_without_output(tmp_path, payload):
    path = tmp_path / "bad.wav"
    path.write_bytes(payload)
    output = tmp_path / "absent"
    with pytest.raises(ValueError):
        processing.process_wav(path, output)
    assert not output.exists()
    assert path.read_bytes() == payload


@pytest.mark.parametrize("existing_kind", ["empty_directory", "nonempty_directory", "file"])
def test_existing_destination_is_preserved_and_never_reused(tmp_path, existing_kind):
    path = tmp_path / "source.wav"
    _write_codes(path, _tone_codes(50, seconds=0.1))
    output = tmp_path / "existing"
    if existing_kind == "file":
        output.write_bytes(b"preserve me")
    else:
        output.mkdir()
        if existing_kind == "nonempty_directory":
            (output / "report.json").write_bytes(b"old report")
    before = _hash(path)
    with pytest.raises(FileExistsError):
        processing.process_wav(path, output)
    assert _hash(path) == before
    if existing_kind == "file":
        assert output.read_bytes() == b"preserve me"
    elif existing_kind == "nonempty_directory":
        assert {entry.name for entry in output.iterdir()} == {"report.json"}
        assert (output / "report.json").read_bytes() == b"old report"
    else:
        assert list(output.iterdir()) == []


def test_separate_processing_calls_do_not_reuse_filter_history(tmp_path):
    noisy, silent = tmp_path / "noisy.wav", tmp_path / "silent.wav"
    _write_codes(noisy, _tone_codes(50, seconds=0.5))
    _write_codes(silent, np.zeros(RATE // 2, dtype=np.int16))
    processing.process_wav(noisy, tmp_path / "first")
    processing.process_wav(silent, tmp_path / "second")
    np.testing.assert_array_equal(_codes(tmp_path / "second" / "processed.wav"), np.zeros(RATE // 2, dtype=np.int16))
    processing.process_wav(noisy, tmp_path / "third")
    assert (tmp_path / "first" / "processed.wav").read_bytes() == (tmp_path / "third" / "processed.wav").read_bytes()


def test_cli_runs_offline_processing_with_explicit_manual_settings(tmp_path, capsys):
    path = tmp_path / "source.wav"
    _write_codes(path, _tone_codes(60, seconds=0.1))
    output = tmp_path / "cli result"
    assert main(["process", str(path), "--output-dir", str(output), "--hum-hz", "60", "--attenuation-db", "6", "--block-size", "127"]) == 0
    printed = json.loads(capsys.readouterr().out)
    saved = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert saved["config"]["nominal_hum_hz"] == 60.0
    assert saved["config"]["max_attenuation_db"] == 6.0
    assert saved["config"]["block_size_frames"] == 127
    assert printed  # CLI may print report directly or a compact output summary.
    assert (output / "processed.wav").is_file()


@pytest.mark.parametrize("hum_setting", [[], ["--hum-hz", "55"]])
def test_cli_requires_supported_explicit_hum_frequency(tmp_path, capsys, hum_setting):
    path = tmp_path / "source.wav"
    _write_codes(path, [0] * 32)
    output = tmp_path / "must not exist"
    with pytest.raises(SystemExit) as failure:
        main(["process", str(path), "--output-dir", str(output), *hum_setting])
    assert failure.value.code == 2
    assert not output.exists()
    capsys.readouterr()
