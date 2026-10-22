"""Long native-rate file contract, bounded conversion, integrity and staging."""

import hashlib
import json
from pathlib import Path
import struct
import wave

import numpy as np
import pytest

from hum_dsp import long_compatible_processing as long
from hum_dsp.band_control import BandControlConfig


OFF = BandControlConfig(band_depths_db=(0, 0, 0), block_size_frames=4096)


def _write(path, codes, rate=44100, width=2):
    values = np.asarray(codes)
    if values.ndim == 1:
        values = values[:, None]
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(values.shape[1])
        handle.setsampwidth(width)
        handle.setframerate(rate)
        handle.writeframes(values.astype("<i2").tobytes() if width == 2 else bytes(values.size * width))
    return path


def _read(path):
    with wave.open(str(path), "rb") as handle:
        return np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2").reshape(-1, handle.getnchannels())


def _chunks(path, fmt=None, data=None, before=b"", after=b""):
    fmt = struct.pack("<HHIIHH", 1, 1, 44100, 88200, 2, 16) if fmt is None else fmt
    data = np.arange(40, dtype="<i2").tobytes() if data is None else data
    body = b"WAVE" + before + b"fmt " + struct.pack("<I", len(fmt)) + fmt + bytes(len(fmt) % 2)
    body += b"data" + struct.pack("<I", len(data)) + data + bytes(len(data) % 2) + after
    path.write_bytes(b"RIFF" + struct.pack("<I", len(body)) + body)
    return path


def _extensible(channels, mask, valid_bits=16, guid=None, extension_size=22):
    guid = long._PCM_GUID if guid is None else guid
    return struct.pack("<HHIIHHHHI16s", 0xFFFE, channels, 44100, 44100 * channels * 2,
                       channels * 2, 16, extension_size, valid_bits, mask, guid)


@pytest.mark.parametrize("channels", [1, 2])
def test_zero_depth_exact_all_PCM_codes_without_resampler_engine_or_FFT(tmp_path, monkeypatch, channels):
    codes = np.arange(-32768, 32768, dtype=np.int32).astype(np.int16)
    values = codes[:, None] if channels == 1 else np.column_stack((codes, codes[::-1]))
    source = _write(tmp_path / "private.wav", values)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    def forbidden(*args, **kwargs):
        raise AssertionError("zero depths must skip conversion and engine")
    import hum_dsp.stream_bands as engine
    import hum_dsp.stream_resampling as resampling
    monkeypatch.setattr(engine, "StreamBandControl", forbidden)
    monkeypatch.setattr(resampling, "StreamingResampler", forbidden)
    report = long.process_long_compatible_wav(source, tmp_path / "out", OFF)
    np.testing.assert_array_equal(_read(tmp_path / "out/processed.wav"), values)
    assert report["input"]["sha256"] == digest
    assert hashlib.sha256(source.read_bytes()).hexdigest() == digest
    assert report["processing_bypassed"] is True
    assert report["processor_applied"] is False
    assert report["algorithm"]["shared_original_input_fft_count"] == 0
    assert report["conversion"]["to_engine"]["conversion_applied"] is False
    assert report["conversion"]["to_native"]["conversion_applied"] is False
    assert report["band_trace"]["event_count"] == 0
    assert (tmp_path / "out/long_trace.jsonl").read_bytes() == b""
    assert report["long_file"]["sample_rate_conversion_applied"] is False
    assert report["clocks"]["engine_frames"] == 0
    assert report["resource_limits"]["max_resampler_output_frames_observed"] == 0
    assert str(tmp_path) not in json.dumps(report)


def test_65_second_44100_stereo_identity_and_native_waveforms(tmp_path):
    frames = 65 * 44100
    values = np.empty((frames, 2), dtype=np.int16)
    values[:, 0] = np.arange(frames, dtype=np.int32) % 20000 - 10000
    values[:, 1] = -values[:, 0]
    source = _write(tmp_path / "65s.wav", values)
    report = long.process_long_compatible_wav(source, tmp_path / "out", OFF)
    np.testing.assert_array_equal(_read(tmp_path / "out/processed.wav"), values)
    assert report["input"]["sample_rate_hz"] == report["output"]["sample_rate_hz"] == 44100
    assert report["output"]["frames"] == frames
    assert report["timing"]["duration_seconds"] == 65
    edges = np.arange(481, dtype=np.int64) * frames // 480
    expected = [[float(values[a:b].min()) / 32768, float(values[a:b].max()) / 32768]
                for a, b in zip(edges[:-1], edges[1:])]
    assert report["waveforms"]["input"] == report["waveforms"]["output"] == expected
    assert report["resource_limits"]["observed_max_read_bytes"] <= 16384


@pytest.mark.parametrize("rate,channels", [(44100, 1), (44100, 2), (48000, 1), (48000, 2)])
def test_inspector_accepts_only_declared_rates_and_native_metadata(tmp_path, rate, channels):
    source = _write(tmp_path / "source.wav", np.zeros((4097, channels)), rate=rate)
    metadata = long.inspect_long_compatible_wav(source)
    assert metadata["sample_rate_hz"] == rate
    assert metadata["channels"] == channels
    assert metadata["frames"] == 4097
    assert metadata["duration_seconds"] == 4097 / rate


@pytest.mark.parametrize("channels,mask", [(1, 0), (1, 4), (2, 0), (2, 3)])
def test_canonical_44100_extensible_metadata_is_dropped_not_audio(tmp_path, channels, mask):
    payload = np.arange(80 * channels, dtype="<i2").tobytes()
    junk = b"JUNK" + struct.pack("<I", 3) + b"abc\x00"
    source = _chunks(tmp_path / "ext.wav", _extensible(channels, mask), payload, before=junk, after=junk)
    report = long.process_long_compatible_wav(source, tmp_path / "out", OFF)
    assert report["input"]["channel_mask"] == mask
    assert report["output"]["channel_mask"] is None
    assert report["output"]["wav_format_tag"] == 1
    assert b"JUNK" not in (tmp_path / "out/processed.wav").read_bytes()
    np.testing.assert_array_equal(_read(tmp_path / "out/processed.wav").reshape(-1), np.frombuffer(payload, dtype="<i2"))


@pytest.mark.parametrize("channels,mask", [(1, 1), (1, 2), (1, 8), (2, 4), (2, 12), (2, 48)])
def test_noncanonical_masks_rejected(tmp_path, channels, mask):
    source = _chunks(tmp_path / "bad.wav", _extensible(channels, mask), bytes(20 * channels * 2))
    with pytest.raises(ValueError, match="channel masks"):
        long.process_long_compatible_wav(source, tmp_path / "out", OFF)
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("rate", [8000, 16000, 32000, 96000])
def test_other_rates_rejected_before_output(tmp_path, rate):
    source = _write(tmp_path / "bad.wav", np.zeros(12), rate=rate)
    with pytest.raises(ValueError, match="44100 or 48000"):
        long.process_long_compatible_wav(source, tmp_path / "out", OFF)
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("width", [1, 3, 4])
def test_other_widths_rejected_before_output(tmp_path, width):
    source = _write(tmp_path / "bad.wav", np.zeros(12), width=width)
    with pytest.raises(ValueError, match="PCM16"):
        long.process_long_compatible_wav(source, tmp_path / "out", OFF)
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("change", ["truncated", "trailing", "bad_magic", "partial_frame", "zero_frames", "data_first", "duplicate_data", "duplicate_fmt", "wrong_alignment", "wrong_byte_rate", "float", "wrong_valid_bits", "wrong_guid", "extension_overclaim", "short_extensible", "incomplete_chunk_header", "incomplete_chunk_payload"])
def test_malformed_container_rejected_before_result(tmp_path, change):
    source = _chunks(tmp_path / "bad.wav")
    raw = bytearray(source.read_bytes())
    if change == "truncated":
        raw = raw[:-1]
    elif change == "trailing":
        raw += b"x"
    elif change == "bad_magic":
        raw[:4] = b"RIFX"
    elif change == "partial_frame":
        raw = bytearray(_chunks(source, data=b"x").read_bytes())
    elif change == "zero_frames":
        raw = bytearray(_chunks(source, data=b"").read_bytes())
    elif change == "data_first":
        raw = raw[:12] + raw[36:] + raw[12:36]
    elif change == "duplicate_data":
        raw += raw[36:]
        struct.pack_into("<I", raw, 4, len(raw) - 8)
    elif change == "duplicate_fmt":
        raw += raw[12:36]
        struct.pack_into("<I", raw, 4, len(raw) - 8)
    elif change == "wrong_alignment":
        struct.pack_into("<H", raw, 32, 4)
    elif change == "wrong_byte_rate":
        struct.pack_into("<I", raw, 28, 123)
    elif change == "float":
        struct.pack_into("<H", raw, 20, 3)
    elif change == "wrong_valid_bits":
        raw = bytearray(_chunks(source, _extensible(1, 4, valid_bits=15)).read_bytes())
    elif change == "wrong_guid":
        raw = bytearray(_chunks(source, _extensible(1, 4, guid=bytes(16))).read_bytes())
    elif change == "extension_overclaim":
        raw = bytearray(_chunks(source, _extensible(1, 4, extension_size=23)).read_bytes())
    elif change == "short_extensible":
        raw = bytearray(_chunks(source, _extensible(1, 4)[:30]).read_bytes())
    elif change == "incomplete_chunk_header":
        raw += b"abcd"
        struct.pack_into("<I", raw, 4, len(raw) - 8)
    elif change == "incomplete_chunk_payload":
        raw += b"JUNK" + struct.pack("<I", 99) + b"xx"
        struct.pack_into("<I", raw, 4, len(raw) - 8)
    source.write_bytes(raw)
    with pytest.raises(ValueError):
        long.process_long_compatible_wav(source, tmp_path / "out", OFF)
    assert not (tmp_path / "out").exists()
    assert not list(tmp_path.glob(".hum_long_compatible_*"))


def test_duration_uses_native_rate_and_container_cap(tmp_path, monkeypatch):
    source = _write(tmp_path / "source.wav", np.zeros(44101))
    monkeypatch.setattr(long, "MAX_SECONDS", 1)
    with pytest.raises(ValueError, match="at most 600 seconds"):
        long.inspect_long_compatible_wav(source)
    monkeypatch.setattr(long, "MAX_SECONDS", 600)
    monkeypatch.setattr(long, "MAX_FILE_BYTES", source.stat().st_size - 1)
    with pytest.raises(ValueError, match="128 MiB"):
        long.inspect_long_compatible_wav(source)


def test_too_many_chunks_rejected(tmp_path):
    source = _chunks(tmp_path / "source.wav", before=(b"JUNK" + struct.pack("<I", 0)) * 4096)
    with pytest.raises(ValueError, match="4096 chunk limit"):
        long.inspect_long_compatible_wav(source)


@pytest.mark.parametrize("invalid", [{}, 42, "bad", True])
def test_invalid_config_rejected(tmp_path, invalid):
    source = _write(tmp_path / "source.wav", np.zeros(40))
    with pytest.raises(ValueError, match="config"):
        long.process_long_compatible_wav(source, tmp_path / "out", invalid)


def test_tampered_config_revalidated(tmp_path):
    source = _write(tmp_path / "source.wav", np.zeros(40))
    selected = BandControlConfig()
    object.__setattr__(selected, "band_depths_db", (99, 0, 0))
    with pytest.raises(ValueError):
        long.process_long_compatible_wav(source, tmp_path / "out", selected)


def test_48k_route_delegates_dynamically_to_unchanged_H3b(tmp_path, monkeypatch):
    source = _write(tmp_path / "source.wav", np.zeros(40), rate=48000)
    sentinel, observed = {"old_contract": True}, []
    def old_processor(*args):
        observed.append(args)
        return sentinel
    monkeypatch.setattr(long._native, "process_long_wav", old_processor)
    assert long.process_long_compatible_wav(source, tmp_path / "out", OFF) is sentinel
    assert observed[0] == (source, tmp_path / "out", OFF)


def test_existing_result_never_replaced(tmp_path):
    source = _write(tmp_path / "source.wav", np.zeros(40))
    target = tmp_path / "out"
    target.mkdir()
    (target / "keep.txt").write_text("keep")
    for destination in (source, target):
        with pytest.raises(FileExistsError):
            long.process_long_compatible_wav(source, destination, OFF)
    assert (target / "keep.txt").read_text() == "keep"


def test_inspection_skips_large_metadata_and_all_processing_reads_bounded(tmp_path, monkeypatch):
    junk = b"JUNK" + struct.pack("<I", 200_000) + bytes(200_000)
    source = _chunks(tmp_path / "source.wav", before=junk)
    expected = hashlib.sha256(source.read_bytes()).hexdigest()
    actual_open, requests = Path.open, []
    class Wrapped:
        def __init__(self, handle):
            self.handle = handle
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return self.handle.__exit__(*args)
        def read(self, size=-1):
            requests.append(size)
            assert 0 <= size <= long.MAX_READ_BYTES
            return self.handle.read(size)
        def __getattr__(self, name):
            return getattr(self.handle, name)
    def opener(path, *args, **kwargs):
        handle = actual_open(path, *args, **kwargs)
        return Wrapped(handle) if path == source else handle
    monkeypatch.setattr(Path, "open", opener)
    info = long.inspect_long_compatible_wav(source)
    assert info["frames"] == 40
    assert requests and max(requests) <= 40
    requests.clear()
    report = long.process_long_compatible_wav(source, tmp_path / "out", OFF)
    assert report["input"]["sha256"] == expected
    assert max(requests) <= 16384


@pytest.mark.parametrize("channels,policy", [(1, "mono"), (2, "same"), (2, "inverse"), (2, "right_louder")])
def test_short_active_44100_PCM_matches_frozen_H3a(tmp_path, channels, policy):
    from hum_dsp.compatible_processing import process_compatible_wav
    frames = round(3.2 * 44100) + 1
    time = np.arange(frames) / 44100
    tone = np.rint((.06 * np.sin(2 * np.pi * 50 * time + .37) + .02 * np.sin(2 * np.pi * 430 * time)) * 32768).astype(np.int16)
    if channels == 1:
        values = tone[:, None]
    elif policy == "inverse":
        values = np.column_stack((tone, -tone))
    elif policy == "right_louder":
        values = np.column_stack((tone // 2, tone))
    else:
        values = np.column_stack((tone, tone))
    source = _write(tmp_path / "source.wav", values)
    config = BandControlConfig(block_size_frames=4096)
    old = process_compatible_wav(source, tmp_path / "old", config)
    new = long.process_long_compatible_wav(source, tmp_path / "new", config)
    np.testing.assert_array_equal(_read(tmp_path / "new/processed.wav"), _read(tmp_path / "old/processed.wav"))
    assert new["output"]["frames"] == frames
    assert new["output"]["sample_rate_hz"] == 44100
    assert new["stereo_policy"]["reference_channel_index"] == old["stereo_policy"]["reference_channel_index"]
    assert new["clocks"]["engine_frames"] == (frames * 160 + 146) // 147
    assert new["band_trace"]["sample_rate_hz"] == 48000
    assert new["long_file"]["temporary_float_audio_written"] is False
    assert new["resource_limits"]["whole_audio_retained_in_memory"] is False
    assert new["conversion"]["to_engine"]["finished"] is True
    assert new["conversion"]["to_native"]["finished"] is True
    assert new["stereo_policy"]["engine_channel_rms"] == pytest.approx(old["stereo_policy"]["engine_channel_rms"], abs=1e-15)


@pytest.mark.parametrize("partition", [1, 257, 480, 4096])
def test_engine_partitioning_does_not_change_short_native_PCM(tmp_path, partition):
    frames = 6001
    t = np.arange(frames) / 44100
    source = _write(tmp_path / "source.wav", np.rint(.05 * np.sin(2 * np.pi * 50 * t) * 32768))
    report = long.process_long_compatible_wav(source, tmp_path / "out", BandControlConfig(block_size_frames=partition))
    baseline = long.process_long_compatible_wav(source, tmp_path / "base", BandControlConfig(block_size_frames=4096))
    np.testing.assert_array_equal(_read(tmp_path / "out/processed.wav"), _read(tmp_path / "base/processed.wav"))
    assert report["resource_limits"]["max_engine_block_frames_observed"] <= partition
    assert report["algorithm"]["samples_processed"] == baseline["algorithm"]["samples_processed"]


def test_output_write_fault_removes_entire_stage(tmp_path, monkeypatch):
    source = _write(tmp_path / "source.wav", np.zeros(6000))
    actual = wave.Wave_write.writeframesraw
    def fail(handle, payload):
        actual(handle, payload[:200])
        raise OSError("injected partial write")
    monkeypatch.setattr(wave.Wave_write, "writeframesraw", fail)
    with pytest.raises(OSError, match="partial write"):
        long.process_long_compatible_wav(source, tmp_path / "out", OFF)
    assert not (tmp_path / "out").exists()
    assert not list(tmp_path.glob(".hum_long_compatible_*"))


@pytest.mark.parametrize("mutation", ["same_size", "grow", "shrink"])
def test_late_source_mutation_removes_stage(tmp_path, monkeypatch, mutation):
    source = _write(tmp_path / "source.wav", np.zeros(6000))
    actual = long._sha256_bounded
    def late(path):
        if Path(path) == source:
            data = bytearray(source.read_bytes())
            if mutation == "grow":
                data += b"x"
            elif mutation == "shrink":
                data = data[:-2]
            else:
                data[-1] = 1
            source.write_bytes(data)
        return actual(path)
    monkeypatch.setattr(long, "_sha256_bounded", late)
    with pytest.raises(ValueError, match="changed"):
        long.process_long_compatible_wav(source, tmp_path / "out", OFF)
    assert not (tmp_path / "out").exists()
    assert not list(tmp_path.glob(".hum_long_compatible_*"))


def test_target_created_during_processing_is_preserved(tmp_path, monkeypatch):
    source = _write(tmp_path / "source.wav", np.zeros(6000))
    target = tmp_path / "out"
    actual = long._sha256_bounded
    def late(path):
        if Path(path) == source:
            target.mkdir()
            (target / "keep.txt").write_text("keep")
        return actual(path)
    monkeypatch.setattr(long, "_sha256_bounded", late)
    with pytest.raises(FileExistsError, match="appeared"):
        long.process_long_compatible_wav(source, target, OFF)
    assert (target / "keep.txt").read_text() == "keep"
    assert not (target / "processed.wav").exists()
    assert not list(tmp_path.glob(".hum_long_compatible_*"))


@pytest.mark.parametrize("frames", [1, 2, 146, 147, 148, 4095, 4096, 4097])
def test_native_frame_roundtrip_edges_match_H3a_without_padding_or_cropping(tmp_path, frames):
    from hum_dsp.compatible_processing import process_compatible_wav
    values = np.arange(frames, dtype=np.int32) % 2000 - 1000
    source = _write(tmp_path / "source.wav", values)
    process_compatible_wav(source, tmp_path / "old")
    report = long.process_long_compatible_wav(source, tmp_path / "new")
    assert report["output"]["frames"] == frames
    np.testing.assert_array_equal(_read(tmp_path / "new/processed.wav"), _read(tmp_path / "old/processed.wav"))
    assert report["conversion"]["to_native"]["discarded_final_frames"] in (0, 1)


def _fake_engine(monkeypatch, callback, summary_fault=None):
    import hum_dsp.stream_bands as module
    class Fake:
        def __init__(self, config, channels, reference_channel_index, trace_sink):
            self.reference = reference_channel_index
            self.sink = trace_sink
            self.frames = 0
            trace_sink({"sample_index": 0})
        def process_block(self, values):
            self.frames += len(values)
            return callback(self, values)
        def describe(self):
            return {"reference_channel_index": self.reference,
                    "samples_processed": 0 if summary_fault == "clock" else self.frames,
                    "trace_event_count": 0 if summary_fault == "trace_count" else 1}
    monkeypatch.setattr(module, "StreamBandControl", Fake)


@pytest.mark.parametrize("problem", ["raise", "nan", "inf", "overshoot", "shape", "json_nan"])
def test_engine_or_trace_fault_cleans_all_native_rate_stage_files(tmp_path, monkeypatch, problem):
    source = _write(tmp_path / "source.wav", np.zeros((6000, 2)))
    def process(engine, values):
        if problem == "raise":
            raise ValueError("injected engine failure")
        if problem == "shape":
            return values[:, :1]
        if problem == "json_nan":
            engine.sink({"bad": float("nan")})
        result = values.copy()
        if problem == "nan":
            result[0, 0] = np.nan
        elif problem == "inf":
            result[0, 0] = np.inf
        elif problem == "overshoot":
            result[0, 0] = 1.01
        return result
    _fake_engine(monkeypatch, process)
    with pytest.raises(ValueError):
        long.process_long_compatible_wav(source, tmp_path / "out")
    assert not (tmp_path / "out").exists()
    assert not list(tmp_path.glob(".hum_long_compatible_*"))


@pytest.mark.parametrize("problem", ["clock", "trace_count"])
def test_bad_engine_summary_fails_closed_before_publish(tmp_path, monkeypatch, problem):
    source = _write(tmp_path / "source.wav", np.zeros(6000))
    _fake_engine(monkeypatch, lambda engine, values: values, summary_fault=problem)
    with pytest.raises(ValueError, match="clock|trace summary"):
        long.process_long_compatible_wav(source, tmp_path / "out")
    assert not (tmp_path / "out").exists()
    assert not list(tmp_path.glob(".hum_long_compatible_*"))


@pytest.mark.parametrize("mutation", ["same_size", "grow", "shrink"])
def test_source_mutation_during_engine_processing_fails_closed(tmp_path, monkeypatch, mutation):
    source = _write(tmp_path / "source.wav", np.zeros((10000, 2)))
    mutated = False
    def process(engine, values):
        nonlocal mutated
        if not mutated:
            data = bytearray(source.read_bytes())
            if mutation == "grow":
                data += b"x"
            elif mutation == "shrink":
                data = data[:-2]
            else:
                data[-1] = 1
            source.write_bytes(data)
            mutated = True
        return values
    _fake_engine(monkeypatch, process)
    with pytest.raises(ValueError, match="changed|truncated"):
        long.process_long_compatible_wav(source, tmp_path / "out")
    assert not (tmp_path / "out").exists()
    assert not list(tmp_path.glob(".hum_long_compatible_*"))


def test_forward_FIR_overshoot_is_refused_not_clipped_or_normalized(tmp_path):
    source = _write(tmp_path / "source.wav", np.full(1000, 32767))
    with pytest.raises(ValueError, match="conversion|normalized"):
        long.process_long_compatible_wav(source, tmp_path / "out")
    assert not (tmp_path / "out").exists()
    assert not list(tmp_path.glob(".hum_long_compatible_*"))
