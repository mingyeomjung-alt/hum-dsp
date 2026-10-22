"""Bounded file, exact bypass, linked stereo, integrity and staging contracts."""

import hashlib
import json
from pathlib import Path
import struct
import wave

import numpy as np
import pytest

from hum_dsp import long_processing as long
from hum_dsp.band_control import BandControlConfig


OFF = BandControlConfig(band_depths_db=(0, 0, 0))


def _write(path, codes, rate=48000, width=2):
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
    fmt = struct.pack("<HHIIHH", 1, 1, 48000, 96000, 2, 16) if fmt is None else fmt
    data = np.arange(40, dtype="<i2").tobytes() if data is None else data
    body = b"WAVE" + before + b"fmt " + struct.pack("<I", len(fmt)) + fmt + bytes(len(fmt) % 2)
    body += b"data" + struct.pack("<I", len(data)) + data + bytes(len(data) % 2) + after
    path.write_bytes(b"RIFF" + struct.pack("<I", len(body)) + body)
    return path


def _extensible(channels, mask, valid_bits=16, guid=None, extension_size=22):
    guid = long._PCM_GUID if guid is None else guid
    return struct.pack("<HHIIHHHHI16s", 0xFFFE, channels, 48000, 48000 * channels * 2,
                       channels * 2, 16, extension_size, valid_bits, mask, guid)


@pytest.mark.parametrize("channels", [1, 2])
def test_inspection_and_pcm_code_identity_all_codes(tmp_path, channels, monkeypatch):
    codes = np.arange(-32768, 32768, dtype=np.int32).astype(np.int16)
    values = codes[:, None] if channels == 1 else np.column_stack((codes, codes[::-1]))
    source = _write(tmp_path / "private_name.wav", values)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    def forbidden(*args, **kwargs):
        raise AssertionError("zero-depth processing must not create an engine or FFT")
    import hum_dsp.stream_bands as engine
    monkeypatch.setattr(engine, "StreamBandControl", forbidden)
    report = long.process_long_wav(source, tmp_path / "out", OFF)
    np.testing.assert_array_equal(_read(tmp_path / "out/processed.wav"), values)
    assert report["input"]["sha256"] == digest
    assert hashlib.sha256(source.read_bytes()).hexdigest() == digest
    assert report["input"]["file_name"] == "private_name.wav"
    assert str(tmp_path) not in json.dumps(report)
    assert report["processing_bypassed"] is True
    assert report["processor_applied"] is False
    assert report["algorithm"]["shared_original_input_fft_count"] == 0
    assert report["band_trace"]["event_count"] == 0
    assert (tmp_path / "out/long_trace.jsonl").read_bytes() == b""
    assert report["stereo_policy"]["reference_channel_index"] is None
    assert report["resource_limits"]["observed_max_read_bytes"] <= 16384
    assert report["resource_limits"]["whole_audio_retained_in_memory"] is False
    assert report["resource_limits"]["whole_trace_retained_in_memory"] is False
    assert not list(tmp_path.glob(".hum_long_*"))


def test_65_second_stereo_zero_bypass_and_waveform_no_signed_downmix(tmp_path):
    frames = 65 * 48000
    values = np.zeros((frames, 2), dtype=np.int16)
    values[:, 0] = np.arange(frames, dtype=np.int32) % 20000 - 10000
    values[:, 1] = -values[:, 0]
    source = _write(tmp_path / "65s.wav", values)
    info = long.inspect_long_wav(source)
    assert info["duration_seconds"] == 65
    report = long.process_long_wav(source, tmp_path / "out", OFF)
    np.testing.assert_array_equal(_read(tmp_path / "out/processed.wav"), values)
    assert len(report["waveforms"]["input"]) == 480
    expected_edges = np.arange(481, dtype=np.int64) * frames // 480
    expected = [[float(values[start:stop].min()) / 32768,
                 float(values[start:stop].max()) / 32768]
                for start, stop in zip(expected_edges[:-1], expected_edges[1:])]
    assert report["waveforms"]["input"] == expected
    assert report["waveforms"]["output"] == expected
    assert min(value[0] for value in expected) < -0.25
    assert max(value[1] for value in expected) > 0.25


@pytest.mark.parametrize("frames", [1, 7, 479, 480, 481, 4095, 4096, 4097, 9601])
def test_waveform_bins_include_every_frame_exactly(tmp_path, frames):
    values = np.arange(frames * 2, dtype=np.int32).reshape(frames, 2) % 30001 - 15000
    source = _write(tmp_path / "source.wav", values)
    report = long.process_long_wav(source, tmp_path / "out", OFF)
    count = min(480, frames)
    edges = np.linspace(0, frames, count + 1, dtype=np.int64)
    expected = [[float(values[a:b].min()) / 32768, float(values[a:b].max()) / 32768]
                for a, b in zip(edges[:-1], edges[1:])]
    assert report["waveforms"]["input"] == expected


@pytest.mark.parametrize("channels,mask", [(1, 0), (1, 4), (2, 0), (2, 3)])
def test_canonical_extensible_and_metadata_drop(tmp_path, channels, mask):
    payload = np.arange(80 * channels, dtype="<i2").tobytes()
    junk = b"JUNK" + struct.pack("<I", 3) + b"abc\x00"
    source = _chunks(tmp_path / "ext.wav", _extensible(channels, mask), payload, before=junk, after=junk)
    report = long.process_long_wav(source, tmp_path / "out", OFF)
    assert report["input"]["channel_mask"] == mask
    assert report["output"]["channel_mask"] is None
    assert report["output"]["wav_format_tag"] == 1
    assert report["long_file"]["metadata_chunks_copied"] is False
    assert b"JUNK" not in (tmp_path / "out/processed.wav").read_bytes()
    np.testing.assert_array_equal(_read(tmp_path / "out/processed.wav").reshape(-1), np.frombuffer(payload, dtype="<i2"))


@pytest.mark.parametrize("channels,mask", [(1, 1), (1, 2), (1, 8), (2, 4), (2, 12), (2, 48)])
def test_noncanonical_masks_rejected(tmp_path, channels, mask):
    source = _chunks(tmp_path / "bad.wav", _extensible(channels, mask), bytes(20 * channels * 2))
    with pytest.raises(ValueError, match="channel masks"):
        long.process_long_wav(source, tmp_path / "out", OFF)
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("width", [1, 3, 4])
def test_non_pcm16_width_rejected(tmp_path, width):
    source = _write(tmp_path / "bad.wav", np.zeros(12), width=width)
    with pytest.raises(ValueError, match="PCM16"):
        long.process_long_wav(source, tmp_path / "out", OFF)
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("rate", [8000, 44100, 96000])
def test_non_48k_rejected_with_explicit_short_mode_advice(tmp_path, rate):
    source = _write(tmp_path / "bad.wav", np.zeros(12), rate=rate)
    with pytest.raises(ValueError, match="48000 Hz only; use general WAV mode"):
        long.process_long_wav(source, tmp_path / "out", OFF)


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
        source = _chunks(source, data=b"x")
        raw = bytearray(source.read_bytes())
    elif change == "zero_frames":
        source = _chunks(source, data=b"")
        raw = bytearray(source.read_bytes())
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
        long.process_long_wav(source, tmp_path / "out", OFF)
    assert not (tmp_path / "out").exists()
    assert not list(tmp_path.glob(".hum_long_*"))


def test_duration_and_container_upper_bounds_before_audio_reads(tmp_path, monkeypatch):
    assert long.MAX_SECONDS == 600
    assert long.MAX_FRAMES == 28_800_000
    assert long.MAX_FILE_BYTES == 128 * 1024 * 1024
    source = _write(tmp_path / "source.wav", np.zeros(120))
    monkeypatch.setattr(long, "MAX_FRAMES", 119)
    with pytest.raises(ValueError, match="at most 600 seconds"):
        long.inspect_long_wav(source)
    monkeypatch.setattr(long, "MAX_FILE_BYTES", source.stat().st_size - 1)
    with pytest.raises(ValueError, match="128 MiB"):
        long.inspect_long_wav(source)


def test_inspection_skips_large_metadata_without_unbounded_read(tmp_path, monkeypatch):
    junk = b"JUNK" + struct.pack("<I", 200_000) + bytes(200_000)
    source = _chunks(tmp_path / "source.wav", before=junk)
    actual_open = Path.open
    requests = []
    class Wrapped:
        def __init__(self, handle):
            self.handle = handle
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return self.handle.__exit__(*args)
        def read(self, size=-1):
            requests.append(size)
            assert 0 <= size <= 40
            return self.handle.read(size)
        def __getattr__(self, name):
            return getattr(self.handle, name)
    def opener(path, *args, **kwargs):
        handle = actual_open(path, *args, **kwargs)
        return Wrapped(handle) if path == source else handle
    monkeypatch.setattr(Path, "open", opener)
    metadata = long.inspect_long_wav(source)
    assert metadata["frames"] == 40
    assert sum(requests) < 100


def test_all_processing_reads_are_bounded_and_container_hash_includes_metadata(tmp_path, monkeypatch):
    junk = b"JUNK" + struct.pack("<I", 200_000) + bytes(200_000)
    source = _chunks(tmp_path / "source.wav", before=junk)
    expected = hashlib.sha256(source.read_bytes()).hexdigest()
    actual_open = Path.open
    requests = []
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
    report = long.process_long_wav(source, tmp_path / "out", OFF)
    assert report["input"]["sha256"] == expected
    assert requests and max(requests) <= 16384


@pytest.mark.parametrize("invalid", [None, {}, 42, "bad"])
def test_invalid_config_rejected(tmp_path, invalid):
    source = _write(tmp_path / "source.wav", np.zeros(40))
    if invalid is None:
        selected = BandControlConfig()
        object.__setattr__(selected, "band_depths_db", (99, 0, 0))
    else:
        selected = invalid
    with pytest.raises(ValueError):
        long.process_long_wav(source, tmp_path / "out", selected)
    assert not (tmp_path / "out").exists()


def test_existing_output_and_input_never_replaced(tmp_path):
    source = _write(tmp_path / "source.wav", np.zeros(40))
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    target = tmp_path / "out"
    target.mkdir()
    (target / "keep.txt").write_text("keep")
    for destination in (source, target):
        with pytest.raises(FileExistsError):
            long.process_long_wav(source, destination, OFF)
    assert (target / "keep.txt").read_text() == "keep"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == digest


def _fake_engine(monkeypatch, callback):
    import hum_dsp.stream_bands as module
    class Fake:
        def __init__(self, config, channels, reference_channel_index, trace_sink):
            self.reference = reference_channel_index
            self.sink = trace_sink
            self.frames = 0
            trace_sink({"sample_index": 0})
        def process_block(self, values):
            self.frames += values.shape[0]
            return callback(self, values)
        def describe(self):
            return {"reference_channel_index": self.reference, "samples_processed": self.frames,
                    "trace_event_count": 1}
    monkeypatch.setattr(module, "StreamBandControl", Fake)


@pytest.mark.parametrize("problem", ["raise", "nan", "inf", "overshoot", "positive_rounding_endpoint", "shape", "json_nan"])
def test_processing_or_trace_fault_removes_entire_stage(tmp_path, monkeypatch, problem):
    source = _write(tmp_path / "source.wav", np.zeros((6000, 2)))
    def process(engine, values):
        if problem == "raise":
            raise ValueError("injected processing fault")
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
        elif problem == "positive_rounding_endpoint":
            result[0, 0] = .999999
        return result
    _fake_engine(monkeypatch, process)
    with pytest.raises(ValueError):
        long.process_long_wav(source, tmp_path / "out")
    assert not (tmp_path / "out").exists()
    assert not list(tmp_path.glob(".hum_long_*"))


@pytest.mark.parametrize("mutation", ["same_size", "grow", "shrink", "final_hash"])
def test_input_mutation_fails_closed_and_cleans_stage(tmp_path, monkeypatch, mutation):
    source = _write(tmp_path / "source.wav", np.zeros((6000, 2)))
    mutated = False
    def mutate():
        nonlocal mutated
        if mutated:
            return
        raw = bytearray(source.read_bytes())
        if mutation == "grow":
            raw += b"x"
        elif mutation == "shrink":
            raw = raw[:-2]
        else:
            raw[-4] = 1
        source.write_bytes(raw)
        mutated = True
    def process(engine, values):
        if mutation != "final_hash":
            mutate()
        return values
    _fake_engine(monkeypatch, process)
    if mutation == "final_hash":
        actual = long._sha256_bounded
        def late(path):
            if Path(path) == source:
                mutate()
            return actual(path)
        monkeypatch.setattr(long, "_sha256_bounded", late)
    with pytest.raises(ValueError, match="changed|truncated"):
        long.process_long_wav(source, tmp_path / "out")
    assert not (tmp_path / "out").exists()
    assert not list(tmp_path.glob(".hum_long_*"))


def test_target_appearing_during_processing_never_replaced(tmp_path, monkeypatch):
    source = _write(tmp_path / "source.wav", np.zeros((6000, 2)))
    target = tmp_path / "out"
    def process(engine, values):
        target.mkdir(exist_ok=True)
        (target / "keep.txt").write_text("keep")
        return values
    _fake_engine(monkeypatch, process)
    with pytest.raises(FileExistsError):
        long.process_long_wav(source, target)
    assert (target / "keep.txt").read_text() == "keep"
    assert not (target / "processed.wav").exists()
    assert not list(tmp_path.glob(".hum_long_*"))


def test_partial_output_write_fault_does_not_publish_result(tmp_path, monkeypatch):
    source = _write(tmp_path / "source.wav", np.zeros((6000, 2)))
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    original = wave.Wave_write.writeframesraw
    def fail_after_partial_write(handle, payload):
        original(handle, payload[:400])
        raise OSError("injected disk write fault after partial PCM payload")
    monkeypatch.setattr(wave.Wave_write, "writeframesraw", fail_after_partial_write)
    with pytest.raises(OSError, match="disk write fault"):
        long.process_long_wav(source, tmp_path / "out", OFF)
    assert not (tmp_path / "out").exists()
    assert not list(tmp_path.glob(".hum_long_*"))
    assert hashlib.sha256(source.read_bytes()).hexdigest() == digest


def test_too_many_chunks_rejected_with_bounded_header_work(tmp_path):
    source = _chunks(tmp_path / "source.wav", before=(b"JUNK" + struct.pack("<I", 0)) * 4096)
    with pytest.raises(ValueError, match="4096 chunk limit"):
        long.inspect_long_wav(source)


@pytest.mark.parametrize("problem", ["clock", "trace_count"])
def test_inconsistent_engine_summary_fails_before_commit(tmp_path, monkeypatch, problem):
    source = _write(tmp_path / "source.wav", np.zeros((6000, 2)))
    _fake_engine(monkeypatch, lambda engine, values: values)
    import hum_dsp.stream_bands as module
    original = module.StreamBandControl.describe
    def wrong(engine):
        result = original(engine)
        result["samples_processed" if problem == "clock" else "trace_event_count"] = 0
        return result
    monkeypatch.setattr(module.StreamBandControl, "describe", wrong)
    with pytest.raises(ValueError, match="clock|trace summary"):
        long.process_long_wav(source, tmp_path / "out")
    assert not (tmp_path / "out").exists()
    assert not list(tmp_path.glob(".hum_long_*"))


@pytest.mark.parametrize("block_frames", [1, 257, 480, 4096])
def test_configured_engine_chunks_are_actually_fed_without_reset(tmp_path, monkeypatch, block_frames):
    source = _write(tmp_path / "source.wav", np.zeros((8201, 2)))
    observed, identities = [], set()
    def process(engine, values):
        identities.add(id(engine))
        observed.append(values.shape[0])
        return values
    _fake_engine(monkeypatch, process)
    config = BandControlConfig(block_size_frames=block_frames)
    report = long.process_long_wav(source, tmp_path / "out", config)
    assert len(identities) == 1
    assert sum(observed) == 8201
    assert max(observed) == block_frames
    assert all(1 <= value <= block_frames for value in observed)
    assert report["resource_limits"]["configured_engine_block_frames"] == block_frames
    assert report["resource_limits"]["max_engine_block_frames_observed"] == block_frames


@pytest.mark.parametrize("channels,policy", [(1, "mono"), (2, "same"), (2, "inverse"), (2, "right_louder")])
def test_real_engine_linked_channels_reference_and_jsonl(tmp_path, channels, policy):
    frames = round(3.4 * 48000)
    time = np.arange(frames) / 48000
    signal = np.rint((.06 * np.sin(2 * np.pi * 50 * time + .37) + .02 * np.sin(2 * np.pi * 430 * time)) * 32768).astype(np.int16)
    if channels == 1:
        values = signal[:, None]
    elif policy == "inverse":
        values = np.column_stack((signal, -signal))
    elif policy == "right_louder":
        values = np.column_stack((signal // 2, signal))
    else:
        values = np.column_stack((signal, signal))
    source = _write(tmp_path / "source.wav", values)
    report = long.process_long_wav(source, tmp_path / "out")
    output = _read(tmp_path / "out/processed.wav")
    assert output.shape == values.shape
    assert report["processor_applied"] is True
    assert report["stereo_policy"]["reference_channel_index"] == (1 if policy == "right_louder" else 0)
    events = [json.loads(line) for line in (tmp_path / "out/long_trace.jsonl").read_text().splitlines()]
    assert events and events[0]["sample_index"] == 0
    assert len(events) == report["band_trace"]["event_count"]
    assert all(event["sample_index"] <= frames for event in events)
    if policy == "same":
        np.testing.assert_array_equal(output[:, 0], output[:, 1])
    elif policy == "inverse":
        np.testing.assert_array_equal(output[:, 0], -output[:, 1])
    assert report["reference_quality_metrics"] is None
    assert report["timing"]["real_time_device_performance_verified"] is False
