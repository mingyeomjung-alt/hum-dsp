"""Independent native-code WAV fixtures, bounded IO and fail-closed encoding."""

import hashlib
from pathlib import Path
import struct

import numpy as np
import pytest
from scipy.io import wavfile

import hum_dsp.precision_wav as precision


FORMATS = [precision.PCM16, precision.PCM24, precision.FLOAT32]


def _chunk(tag, payload):
    return tag + struct.pack("<I", len(payload)) + payload + (b"\x00" if len(payload) % 2 else b"")


def _fmt(sample_format=precision.PCM24, channels=1, rate=48000, extension=False, mask=None):
    tag, bits, width = {precision.PCM16: (1, 16, 2), precision.PCM24: (1, 24, 3),
                        precision.FLOAT32: (3, 32, 4)}[sample_format]
    prefix = struct.pack("<HHIIHH", 0xFFFE if extension else tag, channels, rate,
                         rate * channels * width, channels * width, bits)
    if extension:
        mask = (4 if channels == 1 else 3) if mask is None else mask
        return prefix + struct.pack("<HHII", 22, bits, mask, tag) + bytes.fromhex("00001000800000aa00389b71")
    return prefix


def _wav(path, payload=None, sample_format=precision.PCM24, channels=1, rate=48000,
         fmt=None, before=b"", after=b"", fact=None):
    if payload is None:
        payload = bytes({precision.PCM16: 2, precision.PCM24: 3, precision.FLOAT32: 4}[sample_format] * channels * 3)
    fmt = _fmt(sample_format, channels, rate) if fmt is None else fmt
    fact_chunk = _chunk(b"fact", struct.pack("<I", fact)) if fact is not None else b""
    chunks = before + _chunk(b"fmt ", fmt) + fact_chunk + _chunk(b"data", payload) + after
    path.write_bytes(b"RIFF" + struct.pack("<I", 4 + len(chunks)) + b"WAVE" + chunks)
    return path


def _native_payload(codes, sample_format):
    if sample_format == precision.PCM16:
        return b"".join(struct.pack("<h", int(code)) for code in codes)
    if sample_format == precision.PCM24:
        return b"".join(int(code).to_bytes(3, "little", signed=True) for code in codes)
    return b"".join(struct.pack("<f", code) for code in codes)


def _read(path):
    with precision.PrecisionWavReader(path) as reader:
        blocks = list(reader)
        return reader, b"".join(raw for raw, _ in blocks), np.concatenate([values for _, values in blocks])


@pytest.mark.parametrize("sample_format", FORMATS)
@pytest.mark.parametrize("rate", [44100, 48000])
@pytest.mark.parametrize("channels", [1, 2])
def test_native_format_metadata_and_no_intermediate_precision(tmp_path, sample_format, rate, channels):
    if sample_format == precision.FLOAT32:
        codes = [-1., -.5, -0., 0., .125, float(np.nextafter(np.float32(1), np.float32(0)))]
        expected = np.array(codes, dtype=np.float32).astype(np.float64)
    else:
        scale = 32768 if sample_format == precision.PCM16 else 8388608
        codes = [-scale, -scale + 1, -257, -1, 0, 1, 257, scale - 1]
        expected = np.array(codes, dtype=np.float64) / scale
    payload = _native_payload(codes * channels, sample_format)
    source = _wav(tmp_path / "source.wav", payload, sample_format, channels, rate)
    reader, raw, decoded = _read(source)
    metadata = reader.metadata
    assert metadata["sample_format"] == sample_format
    assert metadata["sample_rate_hz"] == rate
    assert metadata["channels"] == channels
    assert metadata["frames"] == len(codes)
    assert metadata["sample_width_bytes"] == len(payload) // (len(codes) * channels)
    assert metadata["duration_seconds"] == len(codes) / rate
    assert metadata["data_bytes"] == len(payload)
    assert metadata["container_bytes"] == source.stat().st_size
    assert metadata["wav_format_tag"] == (3 if sample_format == precision.FLOAT32 else 1)
    assert raw == payload
    np.testing.assert_array_equal(decoded.reshape(-1), np.tile(expected, channels))
    assert decoded.dtype == np.float64
    assert reader.sha256 == hashlib.sha256(source.read_bytes()).hexdigest()
    assert reader.payload_sha256 == hashlib.sha256(payload).hexdigest()


def test_packed_pcm24_literal_sign_extension_and_low_eight_bits():
    payload = bytes.fromhex("000080010080fffffeffffff000000010000ff0000000100feff7fffff7f")
    codes = [-8388608, -8388607, -65537, -1, 0, 1, 255, 256, 8388606, 8388607]
    decoded = precision.decode_precision_payload(payload, {"sample_format": precision.PCM24, "channels": 1})
    np.testing.assert_array_equal(decoded[:, 0] * 8388608, codes)
    assert decoded[5, 0] != 0
    assert decoded[6, 0] != decoded[7, 0]


@pytest.mark.parametrize("sample_format", FORMATS)
@pytest.mark.parametrize("channels", [1, 2])
@pytest.mark.parametrize("rate", [44100, 48000])
def test_writer_roundtrip_and_scipy_independent_interoperability(tmp_path, sample_format, channels, rate):
    codes = np.arange(-14, 16).reshape(-1, 1) / 32
    values = np.repeat(codes, channels, axis=1)
    payload, quantized = precision.encode_precision_block(values, sample_format)
    path = tmp_path / "written.wav"
    metadata = {"sample_format": sample_format, "channels": channels, "sample_rate_hz": rate,
                "frames": len(values), "sample_width_bytes": len(payload) // values.size}
    with precision.PrecisionWavWriter(path, metadata) as writer:
        writer.write_payload(payload)
        assert writer.payload_sha256 is None
    assert writer.frames_written == len(values)
    assert writer.payload_sha256 == hashlib.sha256(payload).hexdigest()
    info = precision.inspect_precision_wav(path)
    assert info["fact_frames"] == (len(values) if sample_format == precision.FLOAT32 else None)
    _, raw, decoded = _read(path)
    assert raw == payload
    np.testing.assert_array_equal(decoded, quantized)
    scipy_rate, scipy_data = wavfile.read(path)
    assert scipy_rate == rate
    scipy_data = scipy_data.reshape(-1, channels)
    if sample_format == precision.PCM16:
        scipy_decoded = scipy_data.astype(np.float64) / 32768
    elif sample_format == precision.PCM24:
        assert scipy_data.dtype == np.int32
        # scipy uses left-justified PCM24 in int32; this fixture validates native packing.
        scipy_decoded = scipy_data.astype(np.float64) / 2147483648
    else:
        assert scipy_data.dtype == np.float32
        scipy_decoded = scipy_data.astype(np.float64)
    np.testing.assert_array_equal(scipy_decoded, decoded)


def test_pcm24_odd_data_has_zero_pad_and_exact_riff_size(tmp_path):
    path = tmp_path / "odd.wav"
    payload = b"\x01\x00\x00"
    metadata = {"sample_format": precision.PCM24, "channels": 1, "sample_rate_hz": 44100,
                "frames": 1, "sample_width_bytes": 3}
    with precision.PrecisionWavWriter(path, metadata) as writer:
        writer.write_payload(payload)
    raw = path.read_bytes()
    assert len(raw) == 48
    assert raw[-4:] == payload + b"\x00"
    assert struct.unpack_from("<I", raw, 4)[0] == len(raw) - 8
    info = precision.inspect_precision_wav(path)
    assert info["data_bytes"] == 3


def test_float_output_header_has_fmt18_empty_extension_and_fact(tmp_path):
    path = tmp_path / "float.wav"
    metadata = {"sample_format": precision.FLOAT32, "channels": 1, "sample_rate_hz": 48000,
                "frames": 1, "sample_width_bytes": 4}
    with precision.PrecisionWavWriter(path, metadata) as writer:
        writer.write_payload(struct.pack("<f", -.5))
    raw = path.read_bytes()
    assert raw[12:20] == b"fmt " + struct.pack("<I", 18)
    assert raw[36:38] == b"\x00\x00"
    assert raw[38:50] == b"fact" + struct.pack("<II", 4, 1)
    assert raw[50:58] == b"data" + struct.pack("<I", 4)
    assert len(raw) == 62


def test_float_signed_zero_raw_payload_is_preserved(tmp_path):
    payload = bytes.fromhex("0000008000000000000080bf0100000001000080ffff7f3f")
    source = _wav(tmp_path / "zero.wav", payload, precision.FLOAT32)
    reader, raw, values = _read(source)
    assert np.signbit(values[0, 0]) and not np.signbit(values[1, 0])
    with precision.PrecisionWavWriter(tmp_path / "copy.wav", reader.metadata) as writer:
        writer.write_payload(raw)
    copied, copy_raw, _ = _read(tmp_path / "copy.wav")
    assert copy_raw == payload
    assert copied.payload_sha256 == reader.payload_sha256


@pytest.mark.parametrize("sample_format", FORMATS)
def test_encode_native_quantization_bound_and_ties_even(sample_format):
    rng = np.random.default_rng(852)
    values = rng.uniform(-.99, .99, (4096, 2))
    payload, quantized = precision.encode_precision_block(values, sample_format)
    assert len(payload) == values.size * precision._FORMATS[sample_format][2]
    bound = {precision.PCM16: .5 / 32768, precision.PCM24: .5 / 8388608,
             precision.FLOAT32: .5 * 2**-24}[sample_format]
    assert np.max(np.abs(quantized - values)) <= bound
    if sample_format != precision.FLOAT32:
        scale = 32768 if sample_format == precision.PCM16 else 8388608
        _, ties = precision.encode_precision_block(np.array([[-2.5, -1.5], [-.5, .5], [1.5, 2.5]]) / scale, sample_format)
        np.testing.assert_array_equal(ties * scale, [[-2, -2], [0, 0], [2, 2]])


@pytest.mark.parametrize("sample_format", FORMATS)
@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf, 1., -1.00000001])
def test_nonfinite_or_unnormalized_output_is_refused(sample_format, invalid):
    with pytest.raises(ValueError, match="finite.*no clipping"):
        precision.encode_precision_block(np.array([[invalid]]), sample_format)


@pytest.mark.parametrize("sample_format", FORMATS)
def test_positive_rounding_endpoint_is_refused_no_clip(sample_format):
    scale = {precision.PCM16: 32768, precision.PCM24: 8388608, precision.FLOAT32: 2**25}[sample_format]
    with pytest.raises(ValueError, match="without clipping"):
        precision.encode_precision_block(np.array([[1 - .25 / scale]]), sample_format)
    payload, quantized = precision.encode_precision_block(np.array([[-1.]]), sample_format)
    assert quantized[0, 0] == -1


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, 1., 1.000001, -1.000001])
def test_invalid_float_samples_are_refused_even_native_bypass(tmp_path, value):
    raw = struct.pack("<f", value)
    source = _wav(tmp_path / "bad.wav", raw, precision.FLOAT32)
    precision.inspect_precision_wav(source)  # Header inspection is deliberately sample-independent.
    with precision.PrecisionWavReader(source) as reader:
        with pytest.raises(ValueError, match="float32 WAV samples"):
            list(reader)
        assert reader.sha256 is None and reader.payload_sha256 is None
    metadata = {"sample_format": precision.FLOAT32, "channels": 1, "sample_rate_hz": 48000,
                "frames": 1, "sample_width_bytes": 4}
    with pytest.raises(ValueError, match="float32 WAV samples"):
        with precision.PrecisionWavWriter(tmp_path / "badcopy.wav", metadata) as writer:
            writer.write_payload(raw)
    assert writer.payload_sha256 is None


@pytest.mark.parametrize("sample_format", FORMATS)
@pytest.mark.parametrize("channels,mask", [(1, 0), (1, 4), (2, 0), (2, 3)])
def test_canonical_extensible_formats(sample_format, channels, mask, tmp_path):
    source = _wav(tmp_path / "ext.wav", sample_format=sample_format, channels=channels,
                  fmt=_fmt(sample_format, channels, extension=True, mask=mask))
    reader, _, _ = _read(source)
    assert reader.metadata["wav_format_tag"] == 0xFFFE
    assert reader.metadata["channel_mask"] == mask
    assert reader.metadata["valid_bits_per_sample"] == reader.metadata["bits_per_sample"]


@pytest.mark.parametrize("sample_format", FORMATS)
def test_classic_fmt18_empty_extension_accepted(sample_format, tmp_path):
    source = _wav(tmp_path / "emptyext.wav", sample_format=sample_format,
                  fmt=_fmt(sample_format) + b"\x00\x00")
    assert precision.inspect_precision_wav(source)["sample_format"] == sample_format


@pytest.mark.parametrize("sample_format", FORMATS)
def test_optional_fact_is_unique_and_matches_frames(sample_format, tmp_path):
    source = _wav(tmp_path / "fact.wav", sample_format=sample_format, fact=3)
    assert precision.inspect_precision_wav(source)["fact_frames"] == 3
    bad = _wav(tmp_path / "badfact.wav", sample_format=sample_format, fact=2)
    with pytest.raises(ValueError, match="fact frame count"):
        precision.inspect_precision_wav(bad)
    duplicate = _wav(tmp_path / "duplicate.wav", sample_format=sample_format, fact=3,
                     after=_chunk(b"fact", struct.pack("<I", 3)))
    with pytest.raises(ValueError, match="fact chunk"):
        precision.inspect_precision_wav(duplicate)


@pytest.mark.parametrize("kind", ["wrong_tag", "PCM32", "float64", "unknown_bits", "bad_rate", "three_channels",
                                  "byte_rate", "align", "short_fmt", "fmt17", "fmt18_nonzero", "fmt20",
                                  "short_extensible", "valid_bits", "guid", "guid_subtype", "extension_size",
                                  "extension_overclaim", "mono_mask", "stereo_mask"])
def test_unsupported_or_inconsistent_formats_rejected(tmp_path, kind):
    fmt = bytearray(_fmt())
    if kind == "wrong_tag":
        struct.pack_into("<H", fmt, 0, 7)
    elif kind == "PCM32":
        fmt = bytearray(struct.pack("<HHIIHH", 1, 1, 48000, 192000, 4, 32))
    elif kind == "float64":
        fmt = bytearray(struct.pack("<HHIIHH", 3, 1, 48000, 384000, 8, 64))
    elif kind == "unknown_bits":
        struct.pack_into("<H", fmt, 14, 20)
    elif kind == "bad_rate":
        struct.pack_into("<I", fmt, 4, 32000)
    elif kind == "three_channels":
        struct.pack_into("<H", fmt, 2, 3)
    elif kind == "byte_rate":
        struct.pack_into("<I", fmt, 8, 123)
    elif kind == "align":
        struct.pack_into("<H", fmt, 12, 2)
    elif kind == "short_fmt":
        fmt = fmt[:15]
    elif kind == "fmt17":
        fmt += b"\x00"
    elif kind == "fmt18_nonzero":
        fmt += b"\x01\x00"
    elif kind == "fmt20":
        fmt += b"\x02\x00xx"
    else:
        fmt = bytearray(_fmt(extension=True))
        if kind == "short_extensible":
            fmt = fmt[:30]
        elif kind == "valid_bits":
            struct.pack_into("<H", fmt, 18, 20)
        elif kind == "guid":
            fmt[39] = 0
        elif kind == "guid_subtype":
            struct.pack_into("<I", fmt, 24, 7)
        elif kind == "extension_size":
            struct.pack_into("<H", fmt, 16, 21)
        elif kind == "extension_overclaim":
            struct.pack_into("<H", fmt, 16, 23)
        elif kind == "mono_mask":
            struct.pack_into("<I", fmt, 20, 1)
        elif kind == "stereo_mask":
            fmt = bytearray(_fmt(channels=2, extension=True, mask=4))
    source = _wav(tmp_path / "bad.wav", fmt=bytes(fmt))
    with pytest.raises(ValueError):
        precision.inspect_precision_wav(source)


@pytest.mark.parametrize("kind", ["magic", "RF64", "RIFX", "truncated", "trailing", "zero_data", "partial_frame",
                                  "data_first", "duplicate_fmt", "duplicate_data", "missing_fmt", "missing_data",
                                  "incomplete_header", "incomplete_chunk", "missing_pad", "short_fact", "long_fact"])
def test_strict_container_failures(tmp_path, kind):
    source = _wav(tmp_path / "bad.wav")
    raw = bytearray(source.read_bytes())
    fmt_chunk = bytes(raw[12:36])
    data_chunk = bytes(raw[36:])
    chunks = None
    if kind in ("magic", "RF64", "RIFX"):
        raw[:4] = {"magic": b"xxxx", "RF64": b"RF64", "RIFX": b"RIFX"}[kind]
    elif kind == "truncated":
        raw = raw[:-1]
    elif kind == "trailing":
        raw += b"x"
    elif kind == "zero_data":
        chunks = fmt_chunk + _chunk(b"data", b"")
    elif kind == "partial_frame":
        chunks = fmt_chunk + _chunk(b"data", b"\x00\x00")
    elif kind == "data_first":
        chunks = data_chunk + fmt_chunk
    elif kind == "duplicate_fmt":
        chunks = fmt_chunk + fmt_chunk + data_chunk
    elif kind == "duplicate_data":
        chunks = fmt_chunk + data_chunk + data_chunk
    elif kind == "missing_fmt":
        chunks = data_chunk
    elif kind == "missing_data":
        chunks = fmt_chunk
    elif kind == "incomplete_header":
        chunks = fmt_chunk + data_chunk + b"xxxx"
    elif kind == "incomplete_chunk":
        chunks = fmt_chunk + data_chunk + b"JUNK" + struct.pack("<I", 20) + b"x"
    elif kind == "missing_pad":
        chunks = fmt_chunk + b"data" + struct.pack("<I", 3) + b"xxx"
    elif kind in ("short_fact", "long_fact"):
        chunks = fmt_chunk + _chunk(b"fact", bytes(3 if kind == "short_fact" else 8)) + data_chunk
    if chunks is not None:
        raw = bytearray(b"RIFF" + struct.pack("<I", 4 + len(chunks)) + b"WAVE" + chunks)
    source.write_bytes(raw)
    with pytest.raises(ValueError):
        precision.inspect_precision_wav(source)


def test_inspection_skips_unknown_large_metadata_and_bounded_header_reads(tmp_path, monkeypatch):
    source = _wav(tmp_path / "metadata.wav", before=_chunk(b"JUNK", bytes(200001)),
                  after=_chunk(b"LIST", b"abc"))
    original = Path.open
    requests = []
    class Watched:
        def __init__(self, handle):
            self.handle = handle
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return self.handle.__exit__(*args)
        def read(self, amount=-1):
            requests.append(amount)
            assert 0 <= amount <= 40
            return self.handle.read(amount)
        def __getattr__(self, name):
            return getattr(self.handle, name)
    monkeypatch.setattr(Path, "open", lambda path, *args, **kwargs:
                        Watched(original(path, *args, **kwargs)) if path == source else original(path, *args, **kwargs))
    assert precision.inspect_precision_wav(source)["frames"] == 3
    assert sum(requests) < 100


@pytest.mark.parametrize("sample_format", FORMATS)
def test_reads_and_decode_blocks_are_bounded_full_hash_includes_metadata_pad(tmp_path, monkeypatch, sample_format):
    width = precision._FORMATS[sample_format][2]
    frames = 9001
    payload = bytes(frames * 2 * width)
    source = _wav(tmp_path / "bounded.wav", payload, sample_format, channels=2,
                  before=_chunk(b"JUNK", bytes(100003)), after=_chunk(b"LIST", bytes(100001)))
    expected_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    original = Path.open
    requests = []
    class Watched:
        def __init__(self, handle):
            self.handle = handle
        def read(self, amount=-1):
            requests.append(amount)
            assert 0 <= amount <= 32768
            return self.handle.read(amount)
        def __getattr__(self, name):
            return getattr(self.handle, name)
    monkeypatch.setattr(Path, "open", lambda path, *args, **kwargs:
                        Watched(original(path, *args, **kwargs)) if path == source else original(path, *args, **kwargs))
    with precision.PrecisionWavReader(source) as reader:
        shapes = [values.shape for _, values in reader]
    assert shapes == [(4096, 2), (4096, 2), (809, 2)]
    assert reader.max_read_bytes <= 32768
    assert reader.sha256 == expected_sha
    assert reader.payload_sha256 == hashlib.sha256(payload).hexdigest()
    assert requests and max(requests) <= 32768


def test_chunk_count_limit_before_unbounded_header_work(tmp_path):
    source = _wav(tmp_path / "chunks.wav", before=_chunk(b"JUNK", b"") * 4096)
    with pytest.raises(ValueError, match="4096 chunk limit"):
        precision.inspect_precision_wav(source)


@pytest.mark.parametrize("rate", [44100, 48000])
def test_duration_bound_checked_from_header_before_payload_read(tmp_path, rate):
    source = tmp_path / "too_long.wav"
    frames = rate * 600 + 1
    fmt = _fmt(precision.PCM16, rate=rate)
    data_bytes = frames * 2
    header = b"RIFF" + struct.pack("<I", 36 + data_bytes) + b"WAVE" + _chunk(b"fmt ", fmt) + b"data" + struct.pack("<I", data_bytes)
    with source.open("xb") as handle:
        handle.write(header)
        handle.seek(len(header) + data_bytes - 1)
        handle.write(b"\x00")
    with pytest.raises(ValueError, match="600 seconds"):
        precision.inspect_precision_wav(source)


def test_128_mib_container_and_writer_output_limit(tmp_path):
    source = tmp_path / "large.wav"
    with source.open("xb") as handle:
        handle.seek(precision.MAX_FILE_BYTES)
        handle.write(b"\x00")
    with pytest.raises(ValueError, match="128 MiB"):
        precision.inspect_precision_wav(source)
    metadata = {"sample_format": precision.FLOAT32, "channels": 2, "sample_rate_hz": 48000,
                "frames": 48000 * 600, "sample_width_bytes": 4}
    with pytest.raises(ValueError, match="128 MiB"):
        precision.PrecisionWavWriter(tmp_path / "no.wav", metadata)
    assert not (tmp_path / "no.wav").exists()


def test_exact_128_mib_pcm24_header_and_output_capacity(tmp_path):
    path = tmp_path / "exact_limit.wav"
    frames = (precision.MAX_FILE_BYTES - 44) // 6
    data_bytes = frames * 6
    assert 44 + data_bytes == precision.MAX_FILE_BYTES
    header = (b"RIFF" + struct.pack("<I", 36 + data_bytes) + b"WAVE"
              + _chunk(b"fmt ", _fmt(precision.PCM24, 2))
              + b"data" + struct.pack("<I", data_bytes))
    with path.open("xb") as handle:
        handle.write(header)
        handle.seek(precision.MAX_FILE_BYTES - 1)
        handle.write(b"\x00")
    metadata = precision.inspect_precision_wav(path)
    assert metadata["frames"] == frames
    writer = precision.PrecisionWavWriter(tmp_path / "no_output_created.wav", metadata)
    assert len(writer.header) + data_bytes == precision.MAX_FILE_BYTES
    assert not writer.path.exists()


def test_exact_container_limit_metadata_hash_is_bounded(tmp_path):
    path = tmp_path / "metadata_limit.wav"
    # One stereo PCM24 frame plus a large unknown chunk: the parser seeks over
    # metadata; the reader later hashes all bytes in fixed-size blocks.
    junk_size = precision.MAX_FILE_BYTES - 58
    fmt_chunk = _chunk(b"fmt ", _fmt(precision.PCM24, 2))
    with path.open("xb") as handle:
        handle.write(b"RIFF" + struct.pack("<I", precision.MAX_FILE_BYTES - 8) + b"WAVE")
        handle.write(b"JUNK" + struct.pack("<I", junk_size))
        handle.seek(20 + junk_size)
        handle.write(fmt_chunk + _chunk(b"data", bytes(6)))
    with precision.PrecisionWavReader(path) as reader:
        blocks = list(reader)
    assert reader.metadata["container_bytes"] == precision.MAX_FILE_BYTES
    assert reader.max_read_bytes == precision.MAX_READ_BYTES
    assert blocks[0][0] == bytes(6)
    independent = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(10007):
            independent.update(block)
    assert reader.sha256 == independent.hexdigest()


@pytest.mark.parametrize("sample_format,channels", [(precision.PCM24, 2), (precision.FLOAT32, 1)])
def test_600_second_legal_header_and_128_mib_bound_are_independent(tmp_path, sample_format, channels):
    rate, frames = 48000, 48000 * 600
    width = precision._FORMATS[sample_format][2]
    data_bytes = frames * channels * width
    path = tmp_path / "duration.wav"
    header = b"RIFF" + struct.pack("<I", 36 + data_bytes) + b"WAVE" + _chunk(b"fmt ", _fmt(sample_format, channels)) + b"data" + struct.pack("<I", data_bytes)
    with path.open("xb") as handle:
        handle.write(header)
        handle.seek(len(header) + data_bytes - 1)
        handle.write(b"\x00")
    if data_bytes + len(header) > precision.MAX_FILE_BYTES:
        with pytest.raises(ValueError, match="128 MiB"):
            precision.inspect_precision_wav(path)
    else:
        assert precision.inspect_precision_wav(path)["duration_seconds"] == 600


def test_reader_detects_growth_truncation_and_expected_metadata_mismatch(tmp_path):
    source = _wav(tmp_path / "source.wav", bytes(5000 * 3))
    with precision.PrecisionWavReader(source) as reader:
        iterator = iter(reader)
        next(iterator)
        with source.open("ab") as handle:
            handle.write(b"x")
        with pytest.raises(ValueError, match="changed"):
            list(iterator)
        assert reader.sha256 is None
    source = _wav(source, bytes(5000 * 3))
    with precision.PrecisionWavReader(source) as reader:
        iterator = iter(reader)
        next(iterator)
        with source.open("r+b") as handle:
            handle.truncate(reader.metadata["data_offset"] + 4096 * 3)
        with pytest.raises(ValueError, match="truncated|changed"):
            list(iterator)
    source = _wav(source)
    expected = precision.inspect_precision_wav(source)
    expected["frames"] += 1
    with pytest.raises(ValueError, match="changed between"):
        with precision.PrecisionWavReader(source, expected):
            pass


def test_reader_single_pass_and_digest_only_after_eof(tmp_path):
    source = _wav(tmp_path / "source.wav", bytes(5000 * 3))
    reader = precision.PrecisionWavReader(source)
    with pytest.raises(ValueError, match="must be open"):
        next(iter(reader))
    with reader:
        iterator = iter(reader)
        next(iterator)
        assert reader.sha256 is None and reader.payload_sha256 is None
        list(iterator)
        assert reader.sha256 and reader.payload_sha256
        with pytest.raises(ValueError, match="once"):
            list(reader)


@pytest.mark.parametrize("kind", ["short", "long", "partial", "after_fault", "same_path"])
def test_writer_fail_closed_and_exclusive_create(tmp_path, kind):
    path = tmp_path / "out.wav"
    metadata = {"sample_format": precision.PCM24, "channels": 1, "sample_rate_hz": 48000,
                "frames": 2, "sample_width_bytes": 3}
    if kind == "same_path":
        path.write_bytes(b"keep")
        with pytest.raises(FileExistsError):
            with precision.PrecisionWavWriter(path, metadata):
                pass
        assert path.read_bytes() == b"keep"
        return
    with pytest.raises(ValueError):
        with precision.PrecisionWavWriter(path, metadata) as writer:
            if kind == "short":
                writer.write_payload(bytes(3))
            elif kind == "long":
                writer.write_payload(bytes(9))
            elif kind == "partial":
                writer.write_payload(bytes(2))
            else:
                with pytest.raises(ValueError):
                    writer.write_payload(bytes(2))
                writer.write_payload(bytes(6))
    assert writer.payload_sha256 is None


@pytest.mark.parametrize("operation", ["header", "payload", "padding"])
def test_short_disk_write_is_not_advertised_as_success(tmp_path, monkeypatch, operation):
    path = tmp_path / "shortwrite.wav"
    metadata = {"sample_format": precision.PCM24, "channels": 1, "sample_rate_hz": 48000,
                "frames": 1}
    original = Path.open
    class ShortWrite:
        def __init__(self, handle):
            self.handle = handle
            self.calls = 0
        def write(self, payload):
            self.calls += 1
            fail_call = {"header": 1, "payload": 2, "padding": 3}[operation]
            if self.calls == fail_call:
                return self.handle.write(payload[:-1])
            return self.handle.write(payload)
        def __getattr__(self, name):
            return getattr(self.handle, name)
    monkeypatch.setattr(Path, "open", lambda candidate, *args, **kwargs:
                        ShortWrite(original(candidate, *args, **kwargs)) if candidate == path else original(candidate, *args, **kwargs))
    writer = precision.PrecisionWavWriter(path, metadata)
    with pytest.raises(OSError, match="incomplete WAV"):
        with writer:
            writer.write_payload(bytes(3))
    assert writer.payload_sha256 is None
    assert writer.handle is None


@pytest.mark.parametrize("payload", [b"", b"x", bytes(4097 * 3), "not bytes", [0]],
                         ids=["empty", "partial", "oversized", "text", "list"])
def test_decode_block_shape_and_type_bounds(payload):
    with pytest.raises(ValueError):
        precision.decode_precision_payload(payload, {"sample_format": precision.PCM24, "channels": 1})


@pytest.mark.parametrize("values", [[], [0.], np.zeros((0, 1)), np.zeros((4097, 1)), np.zeros((2, 3)),
                                    np.zeros((2, 1), dtype=complex), np.array([[True]]), np.array([["x"]])])
def test_encode_invalid_shape_or_nonreal_dtype(values):
    with pytest.raises(ValueError):
        precision.encode_precision_block(values, precision.PCM24)
