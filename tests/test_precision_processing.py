"""Native precision, continuous processing, staging and frozen PCM16 route."""

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import struct

import numpy as np
import pytest

from hum_dsp import precision_processing as precision
from hum_dsp.band_control import BandControlConfig
from hum_dsp.precision_wav import PrecisionWavReader, encode_precision_block


PCM16 = "signed_PCM16_little_endian"
PCM24 = "signed_PCM24_little_endian"
FLOAT32 = "IEEE_FLOAT32_little_endian"
OFF = BandControlConfig(band_depths_db=(0, 0, 0), block_size_frames=4096)


def _write(path, values, rate=48000, sample_format=PCM24, payload=None):
    array = np.asarray(values)
    if array.ndim == 1:
        array = array[:, None]
    channels, frames = array.shape[1], len(array)
    width = {PCM16: 2, PCM24: 3, FLOAT32: 4}[sample_format]
    if payload is None:
        if sample_format == PCM24:
            codes = np.rint(array * (1 << 23)).astype(np.int32).reshape(-1)
            unsigned = codes.view(np.uint32)
            raw = np.empty((len(codes), 3), dtype=np.uint8)
            for byte in range(3):
                raw[:, byte] = (unsigned >> (8 * byte)) & 255
            payload = raw.tobytes()
        else:
            payload = (array.astype("<f4") if sample_format == FLOAT32
                       else np.rint(array * 32768).astype("<i2")).tobytes()
    tag = 3 if sample_format == FLOAT32 else 1
    fmt = struct.pack("<HHIIHH", tag, channels, rate, rate * channels * width, channels * width, width * 8)
    if tag == 3:
        fmt += b"\x00\x00"
    body = b"WAVEfmt " + struct.pack("<I", len(fmt)) + fmt
    if tag == 3:
        body += b"fact\x04\x00\x00\x00" + struct.pack("<I", frames)
    body += b"data" + struct.pack("<I", len(payload)) + payload + bytes(len(payload) % 2)
    path.write_bytes(b"RIFF" + struct.pack("<I", len(body)) + body)
    return path


def _read(path):
    with PrecisionWavReader(path) as source:
        values, payloads = [], []
        for payload, decoded in source:
            values.append(decoded.copy())
            payloads.append(payload)
        return np.concatenate(values), b"".join(payloads), source.metadata


def _signal(rate=48000, duration=2.2, channels=2):
    time = np.arange(int(duration * rate), dtype=np.float64) / rate
    left = .06 * np.sin(2 * np.pi * 50 * time) + .008 * np.sin(2 * np.pi * 731 * time)
    if channels == 1:
        return left[:, None]
    return np.column_stack((left, -.8 * left))


@pytest.mark.parametrize("rate", [44100, 48000])
@pytest.mark.parametrize("channels", [1, 2])
@pytest.mark.parametrize("sample_format", [PCM24, FLOAT32])
def test_zero_depth_native_payload_identity_no_FIR_engine_or_FFT(tmp_path, monkeypatch, rate, channels, sample_format):
    values = _signal(rate=rate, duration=.21, channels=channels)
    # Negative zero must survive bypass in float32 as a raw representation.
    if sample_format == FLOAT32:
        values[::7] = -0.0
    source = _write(tmp_path / "private.wav", values, rate, sample_format)
    original = source.read_bytes()
    def forbidden(*args, **kwargs):
        raise AssertionError("zero depths cannot initialize engine or FIR")
    import hum_dsp.stream_bands as bands
    import hum_dsp.stream_resampling as resampling
    monkeypatch.setattr(bands, "StreamBandControl", forbidden)
    monkeypatch.setattr(resampling, "StreamingResampler", forbidden)
    monkeypatch.setattr(np.fft, "rfft", forbidden)
    monkeypatch.setattr(precision, "encode_precision_block", forbidden)
    report = precision.process_precision_wav(source, tmp_path / "out", OFF)
    old_values, old_payload, _ = _read(source)
    out_values, out_payload, metadata = _read(tmp_path / "out/processed.wav")
    assert old_payload == out_payload
    np.testing.assert_array_equal(old_values, out_values)
    assert source.read_bytes() == original
    assert metadata["sample_format"] == sample_format
    assert metadata["sample_rate_hz"] == rate
    assert metadata["channels"] == channels
    assert report["precision_io"]["raw_payload_identity"] is True
    assert report["precision_io"]["zero_depth_raw_payload_copy"] is True
    assert report["precision_io"]["PCM16_intermediate_used"] is False
    assert report["precision_io"]["dither_applied"] is False
    assert report["native_quantization_max_abs_error"] == 0
    assert report["band_trace"]["event_count"] == 0
    assert report["algorithm"]["shared_original_input_fft_count"] == 0
    assert report["clocks"]["engine_frames"] == 0
    assert report["conversion"]["to_engine"]["conversion_applied"] is False
    assert report["conversion"]["to_native"]["conversion_applied"] is False
    assert report["processor_applied"] is False
    assert report["waveforms"]["input"] == report["waveforms"]["output"]
    assert (tmp_path / "out/long_trace.jsonl").read_bytes() == b""
    assert str(tmp_path) not in json.dumps(report)
    if sample_format == FLOAT32:
        assert np.signbit(out_values[::7]).all()


@pytest.mark.parametrize("rate", [44100, 48000])
def test_pcm16_dynamically_delegates_frozen_adapter_without_wrapping_result(tmp_path, monkeypatch, rate):
    source = _write(tmp_path / "pcm16.wav", np.zeros(5), rate, PCM16)
    import hum_dsp.long_compatible_processing as compatible
    expected, calls = {"preserved": "old_report"}, []
    def delegate(input_path, output_directory, config):
        calls.append((input_path, output_directory, config))
        return expected
    monkeypatch.setattr(compatible, "process_long_compatible_wav", delegate)
    actual = precision.process_precision_wav(source, tmp_path / "out", OFF)
    assert actual is expected
    assert calls == [(source, tmp_path / "out", OFF)]


def test_pcm24_sub_PCM16_LSB_is_retained_in_active_output(tmp_path):
    rate = 48000
    time = np.arange(105600) / rate
    values = (8 / (1 << 23) * np.sin(2 * np.pi * 50 * time))[:, None]
    source = _write(tmp_path / "quiet.wav", values)
    report = precision.process_precision_wav(source, tmp_path / "out")
    output, payload, metadata = _read(tmp_path / "out/processed.wav")
    assert np.any(output != 0)
    assert np.max(np.abs(output)) < 1 / 32768
    assert metadata["bits_per_sample"] == 24
    assert report["precision_io"]["PCM16_intermediate_used"] is False
    assert report["native_quantization_max_abs_error"] <= .5 / (1 << 23)
    assert len(payload) == len(values) * 3


@pytest.mark.parametrize("rate", [44100, 48000])
@pytest.mark.parametrize("sample_format", [PCM24, FLOAT32])
def test_active_native_format_exact_frames_and_float_reference_continuity(tmp_path, rate, sample_format):
    from hum_dsp.stream_bands import StreamBandControl
    from hum_dsp.stream_resampling import StreamingResampler
    source = _write(tmp_path / "source.wav", _signal(rate, duration=2.12), rate, sample_format)
    values, original_payload, _ = _read(source)
    config = BandControlConfig(block_size_frames=997)
    report = precision.process_precision_wav(source, tmp_path / "out", config)
    output, payload, metadata = _read(tmp_path / "out/processed.wav")
    analysis = values
    if rate == 44100:
        converter = StreamingResampler(rate, 48000, len(values), 2)
        analysis_parts = [converter.process_block(values[position:position + 4096])
                          for position in range(0, len(values), 4096)]
        analysis_parts.append(converter.finish())
        analysis = np.concatenate(analysis_parts)
    rms = np.sqrt(np.mean(analysis * analysis, axis=0))
    engine = StreamBandControl(config, channels=2, reference_channel_index=int(np.argmax(rms)))
    processed = np.concatenate([engine.process_block(analysis[position:position + 733])
                                for position in range(0, len(analysis), 733)])
    if rate == 44100:
        backward = StreamingResampler(48000, rate, len(processed), 2, target_frames=len(values))
        output_parts = [backward.process_block(processed[position:position + 4096])
                        for position in range(0, len(processed), 4096)]
        output_parts.append(backward.finish())
        processed = np.concatenate(output_parts)
    expected_parts = [encode_precision_block(processed[position:position + 4096], sample_format)[0]
                      for position in range(0, len(processed), 4096)]
    assert payload == b"".join(expected_parts)
    assert metadata["sample_rate_hz"] == rate
    assert metadata["frames"] == len(values)
    assert metadata["sample_format"] == sample_format
    assert report["processor_applied"] is (payload != original_payload)
    assert report["precision_io"]["raw_payload_identity"] is (payload == original_payload)
    assert report["precision_io"]["native_sample_format_preserved"] is True
    assert report["long_file"]["sample_rate_conversion_applied"] is (rate == 44100)
    assert report["resource_limits"]["max_engine_block_frames_observed"] <= 997
    assert report["resource_limits"]["whole_audio_retained_in_memory"] is False
    assert report["resource_limits"]["whole_trace_retained_in_memory"] is False
    assert report["resource_limits"]["temporary_float_audio_written"] is False
    assert report["clocks"]["engine_frames"] == len(analysis)
    assert report["band_trace"]["event_count"] > 0
    events = [json.loads(line) for line in (tmp_path / "out/long_trace.jsonl").read_text().splitlines()]
    assert len(events) == report["band_trace"]["event_count"]
    assert report["band_trace"]["sample_rate_hz"] == 48000
    assert report["output"]["sha256"] == hashlib.sha256((tmp_path / "out/processed.wav").read_bytes()).hexdigest()
    assert np.max(np.abs(output[:, 0] * .8 + output[:, 1])) <= (2 / (1 << 23) if sample_format == PCM24 else 1e-8)


@pytest.mark.parametrize("sample_format", [PCM24, FLOAT32])
def test_nonzero_filter_partition_invariance(tmp_path, sample_format):
    source = _write(tmp_path / "source.wav", _signal(duration=2.15), 44100, sample_format)
    first = precision.process_precision_wav(source, tmp_path / "one", BandControlConfig(block_size_frames=480))
    second = precision.process_precision_wav(source, tmp_path / "two", BandControlConfig(block_size_frames=4096))
    assert _read(tmp_path / "one/processed.wav")[1] == _read(tmp_path / "two/processed.wav")[1]
    assert first["algorithm"]["shared_original_input_fft_count"] == second["algorithm"]["shared_original_input_fft_count"]


def test_actual_61_second_PCM24_bypass_bounded_reads_and_waveform_bins(tmp_path):
    frames, rate = 61 * 48000, 48000
    # Fixture writes incrementally as well; no entire long audio in memory.
    header = struct.pack("<4sI4s4sIHHIIHH4sI", b"RIFF", 36 + frames * 3, b"WAVE", b"fmt ", 16,
                         1, 1, rate, rate * 3, 3, 24, b"data", frames * 3)
    source = tmp_path / "61s.wav"
    with source.open("wb") as handle:
        handle.write(header)
        for offset in range(0, frames, 4096):
            count = min(4096, frames - offset)
            codes = (np.arange(offset, offset + count) % 100 - 50).astype(np.int32)
            raw = np.empty((count, 3), dtype=np.uint8)
            for byte in range(3):
                raw[:, byte] = (codes.view(np.uint32) >> (8 * byte)) & 255
            handle.write(raw.tobytes())
    report = precision.process_precision_wav(source, tmp_path / "out", OFF)
    assert report["output"]["frames"] == frames
    assert report["timing"]["duration_seconds"] == 61
    assert report["precision_io"]["raw_payload_identity"] is True
    assert len(report["waveforms"]["input"]) == 480
    assert report["resource_limits"]["observed_max_read_bytes"] <= 32768
    assert report["input"]["sample_width_bytes"] == report["output"]["sample_width_bytes"] == 3


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), 1., 1.2, -1.01])
@pytest.mark.parametrize("off", [False, True])
def test_float_headroom_nonfinite_and_endpoint_rejected_without_result(tmp_path, value, off):
    source = _write(tmp_path / "invalid.wav", np.array([0., value]), sample_format=FLOAT32)
    with pytest.raises(ValueError):
        precision.process_precision_wav(source, tmp_path / "out", OFF if off else None)
    assert not (tmp_path / "out").exists()
    assert not list(tmp_path.glob(".hum_precision_*"))


@pytest.mark.parametrize("failure", ["raise", "overshoot", "nonfinite", "shape"])
def test_engine_fault_removes_all_staging_and_no_clipped_fallback(tmp_path, monkeypatch, failure):
    source = _write(tmp_path / "source.wav", _signal(duration=.02))
    import hum_dsp.stream_bands as bands
    original = bands.StreamBandControl.process_block
    def fail(self, block):
        if failure == "raise":
            raise RuntimeError("engine failed")
        result = original(self, block)
        if failure == "overshoot":
            result[0, 0] = 1.1
        elif failure == "nonfinite":
            result[0, 0] = np.nan
        else:
            result = result[:, :1]
        return result
    monkeypatch.setattr(bands.StreamBandControl, "process_block", fail)
    with pytest.raises((RuntimeError, ValueError)):
        precision.process_precision_wav(source, tmp_path / "out")
    assert not (tmp_path / "out").exists()
    assert not list(tmp_path.glob(".hum_precision_*"))


def test_writer_fault_removes_partial_output(tmp_path, monkeypatch):
    source = _write(tmp_path / "source.wav", np.zeros(9000))
    original = precision.PrecisionWavWriter.write_payload
    calls = 0
    def fail(self, payload):
        nonlocal calls
        original(self, payload)
        calls += 1
        if calls == 2:
            raise OSError("disk full")
    monkeypatch.setattr(precision.PrecisionWavWriter, "write_payload", fail)
    with pytest.raises(OSError, match="disk full"):
        precision.process_precision_wav(source, tmp_path / "out", OFF)
    assert not (tmp_path / "out").exists()
    assert not list(tmp_path.glob(".hum_precision_*"))


def test_source_change_between_passes_is_refused(tmp_path, monkeypatch):
    source = _write(tmp_path / "source.wav", np.zeros(80))
    original = precision.PrecisionWavReader.__enter__
    calls = 0
    def change(self):
        nonlocal calls
        calls += 1
        if calls == 2:
            with source.open("r+b") as handle:
                handle.seek(44)
                handle.write(b"\x01")
        return original(self)
    monkeypatch.setattr(precision.PrecisionWavReader, "__enter__", change)
    with pytest.raises(ValueError, match="changed between"):
        precision.process_precision_wav(source, tmp_path / "out", OFF)
    assert not (tmp_path / "out").exists()
    assert not list(tmp_path.glob(".hum_precision_*"))


def test_source_final_hash_change_cannot_commit(tmp_path, monkeypatch):
    source = _write(tmp_path / "source.wav", np.zeros(80))
    original = precision._sha256_bounded
    def modified(path):
        return "0" * 64 if Path(path) == source else original(path)
    monkeypatch.setattr(precision, "_sha256_bounded", modified)
    with pytest.raises(ValueError, match="input changed while processing"):
        precision.process_precision_wav(source, tmp_path / "out", OFF)
    assert not (tmp_path / "out").exists()
    assert not list(tmp_path.glob(".hum_precision_*"))


def test_existing_target_untouched_and_raced_target_not_replaced(tmp_path, monkeypatch):
    source = _write(tmp_path / "source.wav", np.zeros(80))
    target = tmp_path / "existing"
    target.mkdir()
    (target / "keep").write_text("user")
    with pytest.raises(FileExistsError):
        precision.process_precision_wav(source, target, OFF)
    assert (target / "keep").read_text() == "user"
    late = tmp_path / "late"
    original = precision._sha256_bounded
    def create(path):
        result = original(path)
        if Path(path) == source:
            late.mkdir()
            (late / "keep").write_text("concurrent user")
        return result
    monkeypatch.setattr(precision, "_sha256_bounded", create)
    with pytest.raises(FileExistsError, match="appeared"):
        precision.process_precision_wav(source, late, OFF)
    assert (late / "keep").read_text() == "concurrent user"
    assert not list(tmp_path.glob(".hum_precision_*"))


@pytest.mark.parametrize("bad_config", [{}, False, 3, "off"])
def test_invalid_config_refused_before_read_or_output(tmp_path, bad_config):
    with pytest.raises(ValueError, match="validated BandControlConfig"):
        precision.process_precision_wav(tmp_path / "not_existing.wav", tmp_path / "out", bad_config)
    assert not (tmp_path / "out").exists()


def test_tampered_frozen_config_revalidated(tmp_path):
    config = replace(OFF)
    object.__setattr__(config, "block_size_frames", 4097)
    with pytest.raises(ValueError, match="block_size_frames"):
        precision.process_precision_wav(tmp_path / "no.wav", tmp_path / "out", config)


def test_inspection_failure_never_makes_output_or_staging(tmp_path):
    source = tmp_path / "bad.wav"
    source.write_bytes(b"RIFFbad")
    with pytest.raises(ValueError):
        precision.process_precision_wav(source, tmp_path / "out", OFF)
    assert not (tmp_path / "out").exists()
    assert not list(tmp_path.glob(".hum_precision_*"))


def test_container_byte_cap_enforced_before_stage(tmp_path, monkeypatch):
    source = _write(tmp_path / "source.wav", np.zeros(80))
    import hum_dsp.precision_wav as io
    monkeypatch.setattr(io, "MAX_FILE_BYTES", source.stat().st_size - 1)
    with pytest.raises(ValueError, match="128 MiB"):
        precision.process_precision_wav(source, tmp_path / "out", OFF)
    assert not (tmp_path / "out").exists()
    assert not list(tmp_path.glob(".hum_precision_*"))


def test_native_duration_cap_before_stage(tmp_path, monkeypatch):
    source = _write(tmp_path / "source.wav", np.zeros(48001))
    import hum_dsp.precision_wav as io
    monkeypatch.setattr(io, "MAX_SECONDS", 1)
    with pytest.raises(ValueError, match="at most 600 seconds"):
        precision.process_precision_wav(source, tmp_path / "out", OFF)
    assert not (tmp_path / "out").exists()


def test_first_analysis_FIR_overshoot_rejected_before_output(tmp_path, monkeypatch):
    source = _write(tmp_path / "source.wav", _signal(44100, duration=.1), 44100, FLOAT32)
    import hum_dsp.stream_resampling as resampling
    original = resampling.StreamingResampler.process_block
    def overshoot(self, values):
        result = original(self, values)
        if len(result):
            result[0, 0] = 1.001
        return result
    monkeypatch.setattr(resampling.StreamingResampler, "process_block", overshoot)
    with pytest.raises(ValueError, match="invalid normalized audio"):
        precision.process_precision_wav(source, tmp_path / "out")
    assert not (tmp_path / "out").exists()
    assert not list(tmp_path.glob(".hum_precision_*"))


@pytest.mark.parametrize("sample_format,values", [
    (PCM24, np.array([-1., -8388607 / 8388608, -1 / 8388608, 0., 1 / 8388608, 8388607 / 8388608])),
    (FLOAT32, np.array([-1., np.nextafter(np.float32(-1), np.float32(0)), -0., 0.,
                        np.nextafter(np.float32(0), np.float32(1)), np.nextafter(np.float32(1), np.float32(0))])),
])
def test_native_endpoints_and_subnormal_bits_preserved_on_bypass(tmp_path, sample_format, values):
    source = _write(tmp_path / "edges.wav", values, sample_format=sample_format)
    report = precision.process_precision_wav(source, tmp_path / "out", OFF)
    assert _read(source)[1] == _read(tmp_path / "out/processed.wav")[1]
    assert report["precision_io"]["raw_payload_identity"] is True
