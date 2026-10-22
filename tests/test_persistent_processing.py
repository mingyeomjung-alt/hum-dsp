"""New opt-in file integration; synthetic owned samples, no devices/network."""

import hashlib
import importlib.util
import json
from pathlib import Path
import struct

import numpy as np
import pytest

from hum_dsp import persistent_processing as processing
from hum_dsp.precision_wav import (
    FLOAT32, PCM16, PCM24, PrecisionWavReader, PrecisionWavWriter,
    encode_precision_block, inspect_precision_wav,
)
from hum_dsp.sdk import HumDSPError, ProcessResult


def _write(path, values, *, rate=48000, sample_format=FLOAT32):
    values = np.asarray(values)
    if values.ndim == 1:
        values = values[:, None]
    metadata = {"sample_rate_hz": rate, "sample_format": sample_format,
                "channels": values.shape[1], "frames": len(values)}
    with PrecisionWavWriter(path, metadata) as writer:
        for offset in range(0, len(values), 4096):
            payload, _ = encode_precision_block(values[offset:offset + 4096], sample_format)
            writer.write_payload(payload)
    return path


def _values(path):
    with PrecisionWavReader(path) as reader:
        values = [values for _raw, values in reader]
    return np.concatenate(values)


def _payload(path):
    metadata = inspect_precision_wav(path)
    with path.open("rb") as handle:
        handle.seek(metadata["data_offset"])
        return handle.read(metadata["data_bytes"])


def _tone(rate=48000, *, seconds=5, channels=1, hum=60):
    time = np.arange(int(seconds * rate)) / rate
    # Exactly four persistent family lines, plus an off-family wanted tone.
    values = sum(.025 * np.sin(2 * np.pi * hum * order * time + order * .2)
                 for order in (1, 2, 3, 4))
    values += .04 * np.sin(2 * np.pi * 997 * time)
    return values[:, None] if channels == 1 else np.column_stack((values, -.6 * values))


@pytest.mark.parametrize("sample_format", [PCM16, PCM24, FLOAT32])
@pytest.mark.parametrize("rate,channels,hum", [(44100, 1, 50), (48000, 2, 60)])
def test_actual_active_family_preserves_native_format_and_reduces_lines(
    tmp_path, sample_format, rate, channels, hum
):
    source = _write(tmp_path / "owned.wav", _tone(rate, channels=channels, hum=hum),
                    rate=rate, sample_format=sample_format)
    original = source.read_bytes()
    result = processing.process_persistent_file(source, output_dir=tmp_path / "new/results", hum_hz=hum)
    assert isinstance(result, ProcessResult)
    assert result.trace_path.name == "persistent_trace.json"
    assert result.decoded_input_wav is None
    assert source.read_bytes() == original
    report = result.report
    assert report == json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["processor_applied"] and not report["processing_bypassed"]
    # PCM quantization can itself contribute weak persistent harmonics. Do not
    # misrepresent accepted lines as proof that a component is unwanted hum.
    assert {1, 2, 3, 4} <= set(report["algorithm"]["accepted_orders"])
    assert report["input"]["sha256"] == hashlib.sha256(original).hexdigest()
    assert report["output"]["sha256"] == hashlib.sha256(result.output_wav.read_bytes()).hexdigest()
    trace_bytes = result.trace_path.read_bytes()
    assert report["mask_trace"]["sha256"] == hashlib.sha256(trace_bytes).hexdigest()
    trace = json.loads(trace_bytes)
    assert trace["reason"] == "persistent_family_mask"
    assert trace["config"]["sample_rate_hz"] == rate
    assert trace["frames"] == report["output"]["frames"] == rate * 5
    assert report["output"]["sample_format"] == sample_format
    assert report["output"]["sample_rate_hz"] == rate
    assert report["output"]["channels"] == channels
    assert report["precision_io"]["native_sample_format_preserved"]
    assert report["resource_limits"]["whole_audio_retained_in_memory"]
    assert report["resource_limits"]["whole_file_fft_workspace_in_memory"]
    assert not report["resource_limits"]["bounded_stream_engine"]
    assert report["resource_limits"]["observed_max_input_read_bytes"] <= 32768
    assert report["quality_approval"] is False
    assert report["reference_quality_metrics"] is None
    assert not any(report[key] for key in (
        "output_gain_normalization_applied", "sample_rate_conversion_applied",
        "clipping_applied", "device_io", "network_use",
    ))
    before, after = _values(source), _values(result.output_wav)
    # Measure stable interior coherent components, not whole-level change or
    # a general perceptual-quality claim.
    start, stop = rate, rate * 4
    times = np.arange(start, stop) / rate
    amplitudes = []
    for frequency in (hum, 2 * hum, 3 * hum, 4 * hum, 997):
        basis = np.exp(-2j * np.pi * frequency * times)
        old = abs(np.dot(before[start:stop, 0], basis))
        new = abs(np.dot(after[start:stop, 0], basis))
        amplitudes.append(20 * np.log10(new / old))
    assert all(-13 < attenuation < -10 for attenuation in amplitudes[:4])
    assert abs(amplitudes[4]) < .01
    json.dumps(result.to_dict(), allow_nan=False)


@pytest.mark.parametrize("depth,signal", [(0, "family"), (12, "off_family"), (12, "short")])
@pytest.mark.parametrize("sample_format", [PCM16, PCM24, FLOAT32])
def test_bypass_copies_exact_raw_payload_after_validation(tmp_path, depth, signal, sample_format):
    time = np.arange(48000 if signal == "short" else 240000) / 48000
    values = (_tone()[:, 0] if signal == "family"
              else .04 * np.sin(2 * np.pi * 997 * time))
    source = _write(tmp_path / "owned.wav", values, sample_format=sample_format)
    result = processing.process_persistent_file(source, output_dir=tmp_path / "out", hum_hz=60,
                                                attenuation_db=depth)
    assert result.report["processing_bypassed"] is True
    assert result.report["processor_applied"] is False
    assert _payload(result.output_wav) == _payload(source)
    assert result.report["precision_io"]["raw_payload_identity"] is True
    assert result.report["precision_io"]["validated_raw_payload_copy"] is True


@pytest.mark.parametrize("depth,frames", [(0, 5000), (12, 240000)])
def test_negative_float_zero_raw_bits_preserved_on_bypass(tmp_path, depth, frames):
    values = np.zeros((frames, 2), dtype="<f4")
    values[::2, 0] = np.float32(-0.0)
    values[1::2, 1] = np.float32(-0.0)
    source = _write(tmp_path / "negative.wav", values)
    assert b"\0\0\0\x80" in _payload(source)
    result = processing.process_persistent_file(source, output_dir=tmp_path / "out", hum_hz=50,
                                                attenuation_db=depth)
    assert _payload(result.output_wav) == _payload(source)


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -1.01, 1.0])
def test_zero_depth_validates_late_float_samples_before_any_output(tmp_path, invalid):
    source = _write(tmp_path / "invalid.wav", np.zeros(9000))
    metadata = inspect_precision_wav(source)
    with source.open("r+b") as handle:
        handle.seek(metadata["data_offset"] + 8500 * 4)
        handle.write(struct.pack("<f", invalid))
    with pytest.raises(HumDSPError) as caught:
        processing.process_persistent_file(source, output_dir=tmp_path / "out", hum_hz=60, attenuation_db=0)
    assert caught.value.code == "processing_failed"
    assert not (tmp_path / "out").exists()
    assert not list(tmp_path.glob(".hum_persistent_*"))


@pytest.mark.parametrize("options", [
    {"hum_hz": 51}, {"hum_hz": True}, {"hum_hz": float("nan")},
    {"hum_hz": 50, "attenuation_db": -1}, {"hum_hz": 50, "attenuation_db": 19},
    {"hum_hz": 50, "attenuation_db": True}, {"hum_hz": 50, "attenuation_db": float("inf")},
])
def test_invalid_options_before_path_access(tmp_path, monkeypatch, options):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("options must be validated before path access")
    monkeypatch.setattr(processing, "_path_value", forbidden)
    with pytest.raises(HumDSPError) as caught:
        processing.process_persistent_file("missing.wav", output_dir=tmp_path / "out", **options)
    assert caught.value.code == "invalid_options"
    assert str(tmp_path) not in str(caught.value)


@pytest.mark.parametrize("remote", ["https://example.test/file.wav", "//host/share/file.wav", "\\\\host\\share\\file.wav"])
@pytest.mark.parametrize("argument", ["input", "output"])
def test_remote_paths_rejected_lexically_before_filesystem(tmp_path, monkeypatch, remote, argument):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("must reject remote paths before filesystem access")
    monkeypatch.setattr(Path, "lstat", forbidden)
    with pytest.raises(HumDSPError) as caught:
        processing.process_persistent_file(remote if argument == "input" else "owned.wav",
                                            output_dir=remote if argument == "output" else tmp_path / "out",
                                            hum_hz=50)
    assert caught.value.code == "invalid_options"


@pytest.mark.parametrize("kind", ["file", "directory"])
def test_existing_target_preserved_even_with_missing_source(tmp_path, kind):
    target = tmp_path / "out"
    if kind == "directory":
        target.mkdir()
        marker = target / "keep"
    else:
        marker = target
    marker.write_bytes(b"user evidence")
    with pytest.raises(HumDSPError) as caught:
        processing.process_persistent_file(tmp_path / "missing.wav", output_dir=target, hum_hz=60)
    assert caught.value.code == "output_exists"
    assert marker.read_bytes() == b"user evidence"


@pytest.mark.parametrize("kind", ["missing", "directory", "extension", "container", "duration"])
def test_bad_source_leaves_no_output(tmp_path, kind):
    source = tmp_path / "source.wav"
    if kind == "directory":
        source.mkdir()
    elif kind == "extension":
        source = tmp_path / "source.mp3"
        source.write_bytes(b"not a WAV")
    elif kind == "container":
        source.write_bytes(b"bad RIFF")
    elif kind == "duration":
        _write(source, np.zeros(48000 * 60 + 1), sample_format=PCM16)
    with pytest.raises(HumDSPError) as caught:
        processing.process_persistent_file(source, output_dir=tmp_path / "new/out", hum_hz=60)
    assert caught.value.code == ("input_not_found" if kind == "missing" else "unsupported_input")
    assert not (tmp_path / "new").exists()


def test_source_changed_during_render_is_not_committed(tmp_path, monkeypatch):
    source = _write(tmp_path / "owned.wav", _tone())
    original = processing.render_persistent_comb
    def mutate(samples, config):
        result = original(samples, config)
        metadata = inspect_precision_wav(source)
        with source.open("r+b") as handle:
            handle.seek(metadata["data_offset"])
            handle.write(struct.pack("<f", .1))
        return result
    monkeypatch.setattr(processing, "render_persistent_comb", mutate)
    with pytest.raises(HumDSPError) as caught:
        processing.process_persistent_file(source, output_dir=tmp_path / "out", hum_hz=60)
    assert caught.value.code == "processing_failed"
    assert not (tmp_path / "out").exists()
    assert not list(tmp_path.glob(".hum_persistent_*"))


@pytest.mark.parametrize("offset,code", [(24, 32000), (22, 3), (34, 8)])
def test_unsupported_wav_header_refused_before_output(tmp_path, offset, code):
    source = _write(tmp_path / "owned.wav", np.zeros(5000), sample_format=PCM16)
    with source.open("r+b") as handle:
        handle.seek(offset)
        handle.write(struct.pack("<I" if offset == 24 else "<H", code))
    with pytest.raises(HumDSPError) as caught:
        processing.process_persistent_file(source, output_dir=tmp_path / "out", hum_hz=60)
    assert caught.value.code == "unsupported_input"
    assert not (tmp_path / "out").exists()


def test_input_not_regular_stat_failure_is_safe_error(tmp_path, monkeypatch):
    source = _write(tmp_path / "owned.wav", np.zeros(5000))
    original = Path.stat
    def denied(path, *args, **kwargs):
        if path == source:
            raise PermissionError("test private OS path")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "stat", denied)
    with pytest.raises(HumDSPError) as caught:
        processing.process_persistent_file(source, output_dir=tmp_path / "out", hum_hz=60)
    # _local_path uses lstat, which delegates through Path.stat on this host.
    assert caught.value.code in ("invalid_options", "unsupported_input")
    assert "private OS" not in str(caught.value)
    assert str(source) not in str(caught.value)
    assert not (tmp_path / "out").exists()


def test_source_changed_between_hash_and_read_is_refused(tmp_path, monkeypatch):
    source = _write(tmp_path / "owned.wav", np.zeros(5000))
    original = processing._sha256_bounded
    def mutate(path):
        digest = original(path)
        if path == source:
            with source.open("r+b") as handle:
                handle.seek(inspect_precision_wav(source)["data_offset"])
                handle.write(struct.pack("<f", .2))
        return digest
    monkeypatch.setattr(processing, "_sha256_bounded", mutate)
    with pytest.raises(HumDSPError) as caught:
        processing.process_persistent_file(source, output_dir=tmp_path / "out", hum_hz=60)
    assert caught.value.code == "processing_failed"
    assert not (tmp_path / "out").exists()


def test_target_appearing_at_commit_is_preserved_and_stage_cleaned(tmp_path, monkeypatch):
    source = _write(tmp_path / "owned.wav", np.zeros(5000))
    original = processing._commit
    target = tmp_path / "out"
    def race(stage, destination):
        destination.mkdir()
        (destination / "keep").write_bytes(b"concurrent evidence")
        original(stage, destination)
    monkeypatch.setattr(processing, "_commit", race)
    with pytest.raises(HumDSPError) as caught:
        processing.process_persistent_file(source, output_dir=target, hum_hz=60)
    assert caught.value.code == "output_exists"
    assert (target / "keep").read_bytes() == b"concurrent evidence"
    assert list(target.iterdir()) == [target / "keep"]
    assert not list(tmp_path.glob(".hum_persistent_*"))


@pytest.mark.parametrize("failure", ["encode", "range", "shape", "memory"])
def test_failed_render_or_write_cleans_owned_stage(tmp_path, monkeypatch, failure):
    source = _write(tmp_path / "owned.wav", _tone())
    original_bytes = source.read_bytes()
    if failure == "encode":
        def bad_encode(*_args):
            raise ValueError("test write failure")
        monkeypatch.setattr(processing, "encode_precision_block", bad_encode)
    else:
        original = processing.render_persistent_comb
        def bad_render(values, config):
            if failure == "memory":
                raise MemoryError("test resource refusal")
            rendered, descriptor, trace = original(values, config)
            if failure == "range":
                rendered[0, 0] = 1.0
            else:
                rendered = rendered[:-1]
            return rendered, descriptor, trace
        monkeypatch.setattr(processing, "render_persistent_comb", bad_render)
    with pytest.raises(HumDSPError) as caught:
        processing.process_persistent_file(source, output_dir=tmp_path / "out", hum_hz=60)
    assert caught.value.code == "processing_failed"
    assert "Traceback" not in str(caught.value)
    assert source.read_bytes() == original_bytes
    assert not (tmp_path / "out").exists()
    assert not list(tmp_path.glob(".hum_persistent_*"))


def _cli():
    path = Path(__file__).resolve().parents[1] / "tools/process_persistent_hum.py"
    spec = importlib.util.spec_from_file_location("persistent_cli_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_prints_completed_json_without_auto_audio(tmp_path, capsys):
    source = _write(tmp_path / "owned.wav", np.zeros(5000))
    assert _cli().main([str(source), "--output-dir", str(tmp_path / "out"), "--hum-hz", "60"]) == 0
    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert captured.err == ""
    assert result["status"] == "completed"
    assert result["report"]["quality_approval"] is False
    assert result["trace_path"].endswith("persistent_trace.json")


def test_cli_expected_errors_are_finite_json_stderr_no_raw_trace(tmp_path, capsys):
    assert _cli().main(["private-missing.wav", "--output-dir", str(tmp_path / "out"), "--hum-hz", "60"]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    result = json.loads(captured.err)
    assert result["error"]["code"] == "input_not_found"
    assert "private-missing" not in captured.err
    assert "Traceback" not in captured.err


def test_cli_nan_depth_rejected_without_creating_output(tmp_path, capsys):
    assert _cli().main(["missing.wav", "--output-dir", str(tmp_path / "out"),
                        "--hum-hz", "60", "--attenuation-db", "nan"]) == 2
    assert json.loads(capsys.readouterr().err)["error"]["code"] == "invalid_options"
    assert not (tmp_path / "out").exists()
