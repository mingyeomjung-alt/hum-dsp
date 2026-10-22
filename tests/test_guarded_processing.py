"""Independent M4 render, saved-file and command contracts; no audio devices."""

import hashlib
import json
import wave

import numpy as np
import pytest

from hum_dsp import guarded_processing
from hum_dsp.audio import read_pcm16_wav
from hum_dsp.cli import main
from hum_dsp.config import ProcessorConfig


RATE = 48_000


def _write_codes(path, codes, *, rate=RATE, channels=1, width=2):
    values = np.asarray(codes, dtype="<i2")
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(channels)
        stream.setsampwidth(width)
        stream.setframerate(rate)
        stream.writeframes(values.tobytes() if width == 2 else bytes(values.size * width))


def _codes(path):
    with wave.open(str(path), "rb") as stream:
        assert (stream.getnchannels(), stream.getsampwidth(), stream.getframerate(), stream.getcomptype()) == (1, 2, RATE, "NONE")
        return np.frombuffer(stream.readframes(stream.getnframes()), dtype="<i2").copy()


def _tone(hz, seconds=0.1, amplitude=0.15):
    time = np.arange(round(seconds * RATE), dtype=np.float64) / RATE
    return amplitude * np.sin(2 * np.pi * hz * time)


def _tone_codes(hz, seconds=0.1):
    return np.rint(_tone(hz, seconds) * 32768).astype("<i2")


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _trace(path, frames):
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["schema_version"] == 1
    assert record["sample_rate_hz"] == RATE
    assert isinstance(record["events"], list)
    positions = [event["sample_index"] for event in record["events"]]
    assert positions == sorted(positions)
    assert all(type(index) is int and 0 <= index <= frames for index in positions)
    json.dumps(record, allow_nan=False)
    return record


@pytest.mark.parametrize("hz", [50, 60])
def test_saved_guarded_report_describes_actual_files_and_causal_state(tmp_path, hz):
    source = tmp_path / "원본 입력.wav"
    _write_codes(source, _tone_codes(hz + 0.4, seconds=0.13))
    before = source.read_bytes()
    target = tmp_path / "guarded result"
    config = ProcessorConfig(nominal_hum_hz=hz, block_size_frames=127)
    report = guarded_processing.process_guarded_wav(source, target, config)
    assert report == json.loads((target / "report.json").read_text(encoding="utf-8"))
    assert {entry.name for entry in target.iterdir()} == {"processed.wav", "report.json", "guard.json", "tracking.json"}
    assert source.read_bytes() == before
    assert report["engine"] == "guarded_tracked_finite_depth_notch"
    assert report["config"] == config.to_dict()
    assert report["device_io"] is False
    assert report["output_gain_normalization_applied"] is False
    assert report["reference_quality_metrics"] is None
    assert report["algorithm"]["samples_processed"] == _codes(source).size
    for key, path in (("input", source), ("output", target / "processed.wav")):
        decoded = _codes(path).astype(np.float64) / 32768.0
        metadata = report[key]
        assert metadata["file_name"] == path.name
        assert metadata["sha256"] == _hash(path)
        assert metadata["frames"] == decoded.size
        assert metadata["sample_rate_hz"] == RATE
        assert metadata["peak_abs"] == pytest.approx(np.max(np.abs(decoded)), abs=1e-12)
        assert metadata["rms"] == pytest.approx(np.sqrt(np.mean(decoded**2)), abs=1e-12)
    assert report["input"]["frames"] == report["output"]["frames"]
    for key in ("guard", "tracking"):
        trace = _trace(target / f"{key}.json", report["input"]["frames"])
        assert report[key]["file_name"] == f"{key}.json"
        assert report[key]["event_count"] == len(trace["events"])
    floating, description, guard, tracking = guarded_processing.render_guarded(read_pcm16_wav(source), config)
    saved = read_pcm16_wav(target / "processed.wav")
    assert report["algorithm"] == description
    assert report["floating_output"]["peak_abs"] == pytest.approx(np.max(np.abs(floating)), abs=1e-12)
    assert report["floating_output"]["rms"] == pytest.approx(np.sqrt(np.mean(floating**2)), abs=1e-12)
    assert report["pcm16_quantization_max_abs_error"] == pytest.approx(np.max(np.abs(saved - floating)), abs=1e-12)
    assert guard == _trace(target / "guard.json", saved.size)["events"]
    assert tracking == _trace(target / "tracking.json", saved.size)["events"]
    json.dumps(report, allow_nan=False)


def test_zero_depth_preserves_every_pcm16_code_exactly(tmp_path):
    values = np.tile(np.arange(-32768, 32768, dtype=np.int32).astype("<i2"), 2)
    source = tmp_path / "all codes.wav"
    _write_codes(source, values)
    before = source.read_bytes()
    target = tmp_path / "zero depth"
    report = guarded_processing.process_guarded_wav(source, target, ProcessorConfig(max_attenuation_db=0))
    np.testing.assert_array_equal(_codes(target / "processed.wav"), values)
    assert report["processor_applied"] is False
    assert report["pcm16_quantization_max_abs_error"] == 0
    assert source.read_bytes() == before


@pytest.mark.parametrize("signal_kind", ["silence", "out_of_band_multitone"])
def test_no_confirmed_selected_band_signal_remains_exactly_dry(signal_kind):
    samples = np.zeros(4 * RATE)
    if signal_kind == "out_of_band_multitone":
        samples = _tone(200, seconds=4, amplitude=0.1) + _tone(317, seconds=4, amplitude=0.08)
    before = samples.copy()
    output, description, guard, tracking = guarded_processing.render_guarded(samples, ProcessorConfig(block_size_frames=511))
    np.testing.assert_array_equal(samples, before)
    np.testing.assert_array_equal(output, samples)
    assert description["samples_processed"] == samples.size
    json.dumps([description, guard, tracking], allow_nan=False)


@pytest.mark.parametrize("hz", [50, 60])
def test_eight_second_tone_is_not_permanently_bypassed(hz):
    source = _tone(hz, seconds=8)
    before = source.copy()
    output, description, guard, tracking = guarded_processing.render_guarded(source, ProcessorConfig(nominal_hum_hz=hz))
    assert output.shape == source.shape
    assert np.isfinite(output).all()
    assert np.all(output >= -1) and np.all(output < 1)
    np.testing.assert_array_equal(source, before)
    # This checks processing activation only, not perceptual quality or semantics.
    assert not np.array_equal(output[6 * RATE:], source[6 * RATE:])
    assert np.sqrt(np.mean(output[6 * RATE:] ** 2)) < np.sqrt(np.mean(source[6 * RATE:] ** 2))
    assert description["samples_processed"] == source.size
    assert guard and tracking


def test_render_controls_release_to_exact_dry_at_absolute_nonblock_boundaries():
    source = _tone(50, seconds=8)
    disable1, enable, disable2 = 5 * RATE + 11, 6 * RATE + 13, 7 * RATE + 17
    output, _, _, _ = guarded_processing.render_guarded(
        source,
        ProcessorConfig(block_size_frames=511),
        controls=((disable1, False), (enable, True), (disable2, False)),
    )
    release = RATE // 4
    np.testing.assert_array_equal(output[disable1 + release:enable], source[disable1 + release:enable])
    np.testing.assert_array_equal(output[disable2 + release:], source[disable2 + release:])
    assert not np.array_equal(output[4 * RATE:disable1], source[4 * RATE:disable1])


def test_disable_at_first_sample_keeps_entire_render_dry():
    source = _tone(50, seconds=3)
    output, _, _, _ = guarded_processing.render_guarded(source, ProcessorConfig(), controls=((0, False),))
    np.testing.assert_array_equal(output, source)


@pytest.mark.parametrize("controls", [
    ((-1, False),), ((32, False),), ((1.5, False),), ((True, False),),
    ((0, 0),), ((0, 1),), ((0, None),),
    ((2, False), (2, True)), ((3, False), (2, True)),
    ((0, False), (31, True), (32, False)),
    ((0,),), ((0, False, 1),), (None,),
])
def test_invalid_control_schedule_is_rejected_before_any_processing(monkeypatch, controls):
    called = []

    def forbidden_processing(self, block):
        called.append(True)
        raise AssertionError("input must not be processed before full control validation")

    monkeypatch.setattr(guarded_processing.GuardedHumSuppressor, "process_block", forbidden_processing)
    with pytest.raises((TypeError, ValueError)):
        guarded_processing.render_guarded(np.zeros(32), ProcessorConfig(), controls=controls)
    assert not called


@pytest.mark.parametrize("invalid_audio", [
    np.array([]), np.zeros((4, 1)), np.array([np.nan]), np.array([np.inf]),
    np.array([1.0]), np.array([-1.01]), np.array([True]), np.array([1j]),
])
def test_render_rejects_invalid_audio_before_running_engine(monkeypatch, invalid_audio):
    called = []

    def forbidden_processing(self, block):
        called.append(True)
        raise AssertionError("invalid audio must not reach the processing loop")

    monkeypatch.setattr(guarded_processing.GuardedHumSuppressor, "process_block", forbidden_processing)
    with pytest.raises((TypeError, ValueError)):
        guarded_processing.render_guarded(invalid_audio, ProcessorConfig())
    assert not called


@pytest.mark.parametrize("invalid_value", [1.0, 1.1, -1.1, float("nan"), float("inf"), -float("inf")])
def test_nonfinite_or_out_of_range_processor_output_rejected_before_directory(tmp_path, monkeypatch, invalid_value):
    source = tmp_path / "source.wav"
    _write_codes(source, np.zeros(32, dtype="<i2"))
    before = source.read_bytes()
    monkeypatch.setattr(guarded_processing.GuardedHumSuppressor, "process_block", lambda self, block: np.full(np.asarray(block).shape, invalid_value))
    target = tmp_path / "absent"
    with pytest.raises((TypeError, ValueError)):
        guarded_processing.process_guarded_wav(source, target)
    assert not target.exists()
    assert source.read_bytes() == before


@pytest.mark.parametrize("invalid_output", [np.zeros(31), np.zeros(33), np.zeros((32, 1)), np.array([]), np.full(32, True), np.full(32, 1j)])
def test_invalid_shape_or_dtype_is_rejected_without_written_artifacts(tmp_path, monkeypatch, invalid_output):
    source = tmp_path / "source.wav"
    _write_codes(source, np.zeros(32, dtype="<i2"))
    before = source.read_bytes()
    monkeypatch.setattr(guarded_processing.GuardedHumSuppressor, "process_block", lambda self, block: invalid_output.copy())
    target = tmp_path / "absent"
    with pytest.raises((TypeError, ValueError)):
        guarded_processing.process_guarded_wav(source, target)
    assert not target.exists()
    assert source.read_bytes() == before


@pytest.mark.parametrize("settings", [{"rate": 8000}, {"rate": 44100}, {"channels": 2}, {"width": 1}, {"width": 3}])
def test_unsupported_input_is_rejected_without_mutation_or_output(tmp_path, settings):
    source = tmp_path / "unsupported.wav"
    _write_codes(source, [0] * 32, **settings)
    before = source.read_bytes()
    target = tmp_path / "absent"
    with pytest.raises(ValueError):
        guarded_processing.process_guarded_wav(source, target)
    assert source.read_bytes() == before
    assert not target.exists()


@pytest.mark.parametrize("payload", [b"", b"not a WAV", b"RIFF\x24\x00\x00\x00WAVE"])
def test_malformed_container_cannot_create_output(tmp_path, payload):
    source = tmp_path / "malformed.wav"
    source.write_bytes(payload)
    target = tmp_path / "absent"
    with pytest.raises(ValueError):
        guarded_processing.process_guarded_wav(source, target)
    assert source.read_bytes() == payload
    assert not target.exists()


@pytest.mark.parametrize("existing_kind", ["empty_directory", "nonempty_directory", "file"])
def test_existing_destination_is_never_replaced(tmp_path, existing_kind):
    source = tmp_path / "source.wav"
    _write_codes(source, [0] * 32)
    target = tmp_path / "existing"
    if existing_kind == "file":
        target.write_bytes(b"preserve existing file")
    else:
        target.mkdir()
        if existing_kind == "nonempty_directory":
            (target / "guard.json").write_bytes(b"preserve existing report")
    before = source.read_bytes()
    with pytest.raises(FileExistsError):
        guarded_processing.process_guarded_wav(source, target)
    assert source.read_bytes() == before
    if existing_kind == "file":
        assert target.read_bytes() == b"preserve existing file"
    elif existing_kind == "nonempty_directory":
        assert {entry.name for entry in target.iterdir()} == {"guard.json"}
        assert (target / "guard.json").read_bytes() == b"preserve existing report"
    else:
        assert list(target.iterdir()) == []


def test_report_contains_actual_engine_description_not_a_fresh_metadata_instance(tmp_path, monkeypatch):
    source = tmp_path / "source.wav"
    _write_codes(source, _tone_codes(50, seconds=0.12))
    describe = guarded_processing.GuardedHumSuppressor.describe

    def actual_state(self):
        description = describe(self)
        description["independent_test_actual_state"] = description["samples_processed"]
        return description

    monkeypatch.setattr(guarded_processing.GuardedHumSuppressor, "describe", actual_state)
    report = guarded_processing.process_guarded_wav(source, tmp_path / "result")
    assert report["algorithm"]["independent_test_actual_state"] == _codes(source).size


def test_cli_creates_offline_guarded_output_with_explicit_settings(tmp_path, capsys):
    source = tmp_path / "source.wav"
    _write_codes(source, _tone_codes(60))
    target = tmp_path / "cli result"
    assert main(["process-guarded", str(source), "--output-dir", str(target), "--hum-hz", "60", "--attenuation-db", "6", "--block-size", "127"]) == 0
    printed = json.loads(capsys.readouterr().out)
    report = json.loads((target / "report.json").read_text(encoding="utf-8"))
    assert printed
    assert report["config"]["nominal_hum_hz"] == 60
    assert report["config"]["max_attenuation_db"] == 6
    assert report["config"]["block_size_frames"] == 127
    assert report["device_io"] is False
    assert (target / "processed.wav").is_file()
    _trace(target / "guard.json", _codes(source).size)
    _trace(target / "tracking.json", _codes(source).size)


@pytest.mark.parametrize("hum_setting", [[], ["--hum-hz", "55"]])
def test_cli_requires_supported_explicit_target(tmp_path, capsys, hum_setting):
    source = tmp_path / "source.wav"
    _write_codes(source, [0] * 32)
    target = tmp_path / "absent"
    with pytest.raises(SystemExit) as failure:
        main(["process-guarded", str(source), "--output-dir", str(target), *hum_setting])
    assert failure.value.code == 2
    assert not target.exists()
    capsys.readouterr()
