"""Independent file/CLI contracts for tracked hum processing; no audio devices."""

import hashlib
import json
import math
import wave

import numpy as np
import pytest

from hum_dsp.audio import read_pcm16_wav
from hum_dsp.cli import main
from hum_dsp.config import ProcessorConfig
from hum_dsp import tracked_processing


RATE = 48_000


def _write_codes(path, codes, *, rate=RATE, channels=1, width=2):
    """Do not rely on the production writer to prepare independent input."""
    values = np.asarray(codes, dtype="<i2")
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(channels)
        stream.setsampwidth(width)
        stream.setframerate(rate)
        stream.writeframes(values.tobytes() if width == 2 else bytes(values.size * width))


def _codes(path):
    with wave.open(str(path), "rb") as stream:
        assert stream.getnchannels() == 1
        assert stream.getsampwidth() == 2
        assert stream.getframerate() == RATE
        assert stream.getcomptype() == "NONE"
        return np.frombuffer(stream.readframes(stream.getnframes()), dtype="<i2").copy()


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tone_codes(hz, *, seconds=3.0, amplitude=0.15):
    time = np.arange(round(seconds * RATE), dtype=np.float64) / RATE
    return np.rint(amplitude * np.sin(2 * np.pi * hz * time) * 32768).astype("<i2")


def _read_trace(output):
    return json.loads((output / "tracking.json").read_text(encoding="utf-8"))


def _assert_trace_bounds(trace, frames, config):
    assert trace["schema_version"] == 1
    assert trace["sample_rate_hz"] == RATE
    events = trace["events"]
    positions = [event["sample_index"] for event in events]
    assert positions == sorted(positions)
    for event in events:
        position = event["sample_index"]
        assert type(position) is int
        assert 0 <= position <= frames
        estimated = event["estimated_frequency_hz"]
        if estimated is not None:
            assert math.isfinite(estimated)
            assert config.nominal_hum_hz - config.tracking_half_width_hz <= estimated
            assert estimated <= config.nominal_hum_hz + config.tracking_half_width_hz
        actual = event["filter_frequency_hz"]
        assert math.isfinite(actual)
        assert config.nominal_hum_hz - config.tracking_half_width_hz <= actual
        assert actual <= config.nominal_hum_hz + config.tracking_half_width_hz
        assert isinstance(event["status"], str) and event["status"]
        assert math.isfinite(event["confidence"])
        assert 0 <= event["confidence"] <= 1
    # A proper JSON writer must not emit NaN/Infinity, including on silence.
    json.dumps(trace, allow_nan=False)


@pytest.mark.parametrize("hz", [50, 60])
def test_tracked_wav_preserves_source_and_frames_and_reports_saved_data(tmp_path, hz):
    source = tmp_path / "원본 입력.wav"
    _write_codes(source, _tone_codes(hz + 0.5, seconds=3.1))
    before = source.read_bytes()
    output = tmp_path / "tracked result"
    config = ProcessorConfig(nominal_hum_hz=hz, block_size_frames=127)
    result = tracked_processing.process_tracked_wav(source, output, config)
    assert result == json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert {entry.name for entry in output.iterdir()} == {"processed.wav", "report.json", "tracking.json"}
    assert source.read_bytes() == before
    assert _codes(source).shape == _codes(output / "processed.wav").shape
    assert result["engine"] == "tracked_finite_depth_notch"
    assert result["config"] == config.to_dict()
    assert result["output_gain_normalization_applied"] is False
    assert result["reference_quality_metrics"] is None
    assert result["device_io"] is False
    assert result["posthoc_alignment_applied"] is False
    assert result["algorithm"]["tracking_enabled"] is True
    assert result["algorithm"]["explicit_delay_samples"] == 0
    assert result["algorithm"]["samples_processed"] == _codes(source).size
    for key, path in (("input", source), ("output", output / "processed.wav")):
        decoded = read_pcm16_wav(path)
        metadata = result[key]
        assert metadata["sha256"] == _hash(path)
        assert metadata["file_name"] == path.name
        assert metadata["frames"] == decoded.size
        assert metadata["sample_rate_hz"] == RATE
        assert metadata["peak_abs"] == pytest.approx(float(np.max(np.abs(decoded))), abs=1e-12)
        assert metadata["rms"] == pytest.approx(float(np.sqrt(np.mean(decoded**2))), abs=1e-12)
    trace = _read_trace(output)
    _assert_trace_bounds(trace, _codes(source).size, config)
    assert result["tracking"]["file_name"] == "tracking.json"
    assert result["tracking"]["event_count"] == len(trace["events"])
    acquisition = result["tracking"]["first_acquired_sample"]
    assert type(acquisition) is int
    assert 2 * RATE <= acquisition <= _codes(source).size
    assert trace["events"]
    json.dumps(result, allow_nan=False)


def test_short_input_cannot_report_an_acquisition_before_required_history(tmp_path):
    source = tmp_path / "short.wav"
    _write_codes(source, _tone_codes(50.5, seconds=0.1))
    output = tmp_path / "short result"
    config = ProcessorConfig(block_size_frames=127)
    result = tracked_processing.process_tracked_wav(source, output, config)
    assert result["tracking"]["first_acquired_sample"] is None
    _assert_trace_bounds(_read_trace(output), _codes(source).size, config)


def test_zero_depth_preserves_all_pcm_codes_and_does_not_acquire(tmp_path):
    source = tmp_path / "all codes.wav"
    codes = np.tile(np.arange(-32768, 32768, dtype=np.int32).astype("<i2"), 2)
    _write_codes(source, codes)
    before = source.read_bytes()
    output = tmp_path / "disabled"
    config = ProcessorConfig(max_attenuation_db=0)
    result = tracked_processing.process_tracked_wav(source, output, config)
    np.testing.assert_array_equal(_codes(output / "processed.wav"), codes)
    assert source.read_bytes() == before
    assert result["processor_applied"] is False
    assert result["tracking"]["first_acquired_sample"] is None
    assert result["output_gain_normalization_applied"] is False
    _assert_trace_bounds(_read_trace(output), codes.size, config)


def test_silence_remains_silent_without_nonfinite_json_or_acquisition(tmp_path):
    source = tmp_path / "silence.wav"
    codes = np.zeros(3 * RATE, dtype="<i2")
    _write_codes(source, codes)
    output = tmp_path / "silent result"
    result = tracked_processing.process_tracked_wav(source, output)
    np.testing.assert_array_equal(_codes(output / "processed.wav"), codes)
    assert result["tracking"]["first_acquired_sample"] is None
    _assert_trace_bounds(_read_trace(output), codes.size, ProcessorConfig())
    json.dumps(result, allow_nan=False)


def test_processing_calls_do_not_reuse_tracking_or_filter_state(tmp_path):
    source, silent = tmp_path / "tone.wav", tmp_path / "silent.wav"
    _write_codes(source, _tone_codes(50.5, seconds=2.1))
    _write_codes(silent, np.zeros(RATE // 10, dtype="<i2"))
    tracked_processing.process_tracked_wav(source, tmp_path / "first")
    tracked_processing.process_tracked_wav(silent, tmp_path / "middle")
    tracked_processing.process_tracked_wav(source, tmp_path / "last")
    assert (tmp_path / "first" / "processed.wav").read_bytes() == (tmp_path / "last" / "processed.wav").read_bytes()
    assert _read_trace(tmp_path / "first") == _read_trace(tmp_path / "last")
    np.testing.assert_array_equal(_codes(tmp_path / "middle" / "processed.wav"), np.zeros(RATE // 10, dtype="<i2"))


@pytest.mark.parametrize("invalid_value", [1.0, 1.1, -1.1, float("nan"), float("inf")])
def test_invalid_floating_output_is_rejected_before_destination_creation(tmp_path, monkeypatch, invalid_value):
    source = tmp_path / "source.wav"
    _write_codes(source, np.zeros(16, dtype="<i2"))
    before = source.read_bytes()
    monkeypatch.setattr(
        tracked_processing.TrackedHumSuppressor,
        "process_block",
        lambda self, block: np.full(np.asarray(block).shape, invalid_value),
    )
    output = tmp_path / "absent"
    with pytest.raises((TypeError, ValueError)):
        tracked_processing.process_tracked_wav(source, output)
    assert not output.exists()
    assert source.read_bytes() == before


@pytest.mark.parametrize("invalid_output", [np.zeros(15), np.zeros(17), np.zeros((16, 1)), np.array([])])
def test_invalid_output_shape_or_length_cannot_change_audio(tmp_path, monkeypatch, invalid_output):
    source = tmp_path / "source.wav"
    _write_codes(source, np.zeros(16, dtype="<i2"))
    before = source.read_bytes()
    monkeypatch.setattr(
        tracked_processing.TrackedHumSuppressor,
        "process_block",
        lambda self, block: invalid_output.copy(),
    )
    output = tmp_path / "absent"
    with pytest.raises((TypeError, ValueError)):
        tracked_processing.process_tracked_wav(source, output)
    assert not output.exists()
    assert source.read_bytes() == before


@pytest.mark.parametrize("settings", [{"rate": 8000}, {"rate": 44100}, {"channels": 2}, {"width": 1}, {"width": 3}])
def test_unsupported_input_leaves_source_and_output_location_untouched(tmp_path, settings):
    source = tmp_path / "unsupported.wav"
    _write_codes(source, [0] * 32, **settings)
    before = source.read_bytes()
    output = tmp_path / "absent"
    with pytest.raises(ValueError):
        tracked_processing.process_tracked_wav(source, output)
    assert not output.exists()
    assert source.read_bytes() == before


@pytest.mark.parametrize("payload", [b"", b"not a WAV", b"RIFF\x24\x00\x00\x00WAVE"])
def test_malformed_input_is_rejected_without_output(tmp_path, payload):
    source = tmp_path / "malformed.wav"
    source.write_bytes(payload)
    output = tmp_path / "absent"
    with pytest.raises(ValueError):
        tracked_processing.process_tracked_wav(source, output)
    assert not output.exists()
    assert source.read_bytes() == payload


@pytest.mark.parametrize("existing_kind", ["empty_directory", "nonempty_directory", "file"])
def test_existing_destination_is_preserved(tmp_path, existing_kind):
    source = tmp_path / "source.wav"
    _write_codes(source, [0] * 32)
    before = source.read_bytes()
    output = tmp_path / "existing"
    if existing_kind == "file":
        output.write_bytes(b"preserve")
    else:
        output.mkdir()
        if existing_kind == "nonempty_directory":
            (output / "tracking.json").write_bytes(b"old tracking report")
    with pytest.raises(FileExistsError):
        tracked_processing.process_tracked_wav(source, output)
    assert source.read_bytes() == before
    if existing_kind == "file":
        assert output.read_bytes() == b"preserve"
    elif existing_kind == "nonempty_directory":
        assert {entry.name for entry in output.iterdir()} == {"tracking.json"}
        assert (output / "tracking.json").read_bytes() == b"old tracking report"
    else:
        assert list(output.iterdir()) == []


def test_cli_creates_offline_output_and_causal_trace_with_explicit_settings(tmp_path, capsys):
    source = tmp_path / "source.wav"
    _write_codes(source, _tone_codes(60, seconds=0.1))
    output = tmp_path / "cli result"
    assert main([
        "process-tracked", str(source), "--output-dir", str(output), "--hum-hz", "60",
        "--attenuation-db", "6", "--block-size", "127",
    ]) == 0
    printed = json.loads(capsys.readouterr().out)
    saved = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert printed
    assert saved["config"]["nominal_hum_hz"] == 60.0
    assert saved["config"]["max_attenuation_db"] == 6.0
    assert saved["config"]["block_size_frames"] == 127
    assert saved["device_io"] is False
    assert (output / "processed.wav").is_file()
    _assert_trace_bounds(_read_trace(output), _codes(source).size, ProcessorConfig(nominal_hum_hz=60))


@pytest.mark.parametrize("hum_setting", [[], ["--hum-hz", "55"]])
def test_cli_requires_supported_explicit_target_frequency(tmp_path, capsys, hum_setting):
    source = tmp_path / "source.wav"
    _write_codes(source, [0] * 32)
    output = tmp_path / "absent"
    with pytest.raises(SystemExit) as failure:
        main(["process-tracked", str(source), "--output-dir", str(output), *hum_setting])
    assert failure.value.code == 2
    assert not output.exists()
    capsys.readouterr()

