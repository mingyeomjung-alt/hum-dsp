"""Bounded, strict WAV containers with native PCM16, packed PCM24 or float32.

No decoder, encoder, hash or metadata operation retains a complete recording.
Normalized floating samples must be finite and in [-1, 1), including bypasses.
This is container/sample IO only; it does not resample, dither, clip or normalize.
"""

import hashlib
import os
from pathlib import Path
import stat
import struct

import numpy as np


MAX_SECONDS = 600
MAX_FILE_BYTES = 128 * 1024 * 1024
MAX_READ_FRAMES = 4096
MAX_READ_BYTES = MAX_READ_FRAMES * 2 * 4
MAX_CONTAINER_CHUNKS = 4096
SAMPLE_RATES = (44100, 48000)
PCM16 = "signed_PCM16_little_endian"
PCM24 = "signed_PCM24_little_endian"
FLOAT32 = "IEEE_FLOAT32_little_endian"
_FORMATS = {PCM16: (1, 16, 2), PCM24: (1, 24, 3), FLOAT32: (3, 32, 4)}
_GUID_SUFFIX = b"\x00\x00\x10\x00\x80\x00\x00\xaa\x00\x38\x9b\x71"


def _exact(handle, size):
    if not isinstance(size, int) or not 0 <= size <= MAX_READ_BYTES:
        raise ValueError("internal bounded read limit exceeded")
    result = handle.read(size)
    if len(result) != size:
        raise ValueError("truncated WAV container")
    return result


def _regular_bounded(handle):
    info = os.fstat(handle.fileno())
    if not stat.S_ISREG(info.st_mode):
        raise ValueError("a regular WAV file is required")
    if info.st_size > MAX_FILE_BYTES:
        raise ValueError("WAV container exceeds the 128 MiB limit")
    return info.st_size


def _format_metadata(prefix, chunk_size):
    if chunk_size < 16:
        raise ValueError("incomplete WAV format chunk")
    tag, channels, rate, byte_rate, alignment, bits = struct.unpack_from("<HHIIHH", prefix)
    if rate not in SAMPLE_RATES:
        raise ValueError("precision WAV supports 44100 or 48000 Hz only")
    if channels not in (1, 2):
        raise ValueError("only mono/stereo WAV is supported")
    mask, subtype = None, tag
    if tag == 0xFFFE:
        if chunk_size < 40:
            raise ValueError("incomplete extensible WAV format")
        extension_size, valid_bits, mask = struct.unpack_from("<HHI", prefix, 16)
        if extension_size < 22 or extension_size + 18 != chunk_size:
            raise ValueError("inconsistent extensible WAV extension size")
        subtype = struct.unpack_from("<I", prefix, 24)[0]
        if prefix[28:40] != _GUID_SUFFIX or subtype not in (1, 3):
            raise ValueError("unsupported extensible WAV subformat GUID")
        if valid_bits != bits:
            raise ValueError("valid bits must equal container bits; padded PCM is unsupported")
        if mask not in ((0, 4) if channels == 1 else (0, 3)):
            raise ValueError("only canonical mono/stereo channel masks are supported")
    elif chunk_size != 16 and not (chunk_size == 18 and struct.unpack_from("<H", prefix, 16)[0] == 0):
        raise ValueError("classic WAV format must have 16 bytes or an empty 18-byte extension")
    sample_format = next((name for name, (kind, depth, _) in _FORMATS.items()
                          if (subtype, bits) == (kind, depth)), None)
    if sample_format is None:
        raise ValueError("only PCM16, packed PCM24 or IEEE float32 WAV is supported")
    width = _FORMATS[sample_format][2]
    if alignment != channels * width or byte_rate != rate * alignment:
        raise ValueError("inconsistent WAV frame alignment or byte rate")
    return {"sample_rate_hz": rate, "channels": channels,
            "sample_format": sample_format, "sample_width_bytes": width,
            "bits_per_sample": bits, "valid_bits_per_sample": bits,
            "block_align_bytes": alignment, "byte_rate": byte_rate,
            "wav_format_tag": tag, "wav_subformat_tag": subtype,
            "channel_mask": mask,
            "channel_order": ["front_center"] if channels == 1 else ["front_left", "front_right"],
            "canonical_layout_assumption": "unspecified_masks_and_classic_WAV_use_canonical_mono_or_left_right_order"}


def _inspect_handle(handle):
    size = _regular_bounded(handle)
    handle.seek(0)
    header = _exact(handle, 12)
    if header[:4] != b"RIFF" or header[8:] != b"WAVE":
        raise ValueError("a little-endian RIFF WAVE container is required")
    riff_end = struct.unpack_from("<I", header, 4)[0] + 8
    if riff_end != size:
        raise ValueError("truncated or trailing RIFF WAV container bytes")
    position, fmt, data, fact, chunks = 12, None, None, None, 0
    while position < riff_end:
        chunks += 1
        if chunks > MAX_CONTAINER_CHUNKS:
            raise ValueError("WAV container exceeds the 4096 chunk limit")
        if position + 8 > riff_end:
            raise ValueError("truncated WAV chunk header")
        handle.seek(position)
        tag, chunk_size = struct.unpack("<4sI", _exact(handle, 8))
        start = position + 8
        end = start + chunk_size
        padded_end = end + chunk_size % 2
        if padded_end > riff_end:
            raise ValueError("truncated WAV chunk payload or pad")
        if tag == b"fmt ":
            if fmt is not None:
                raise ValueError("duplicate WAV format chunk")
            fmt = _format_metadata(_exact(handle, min(chunk_size, 40)), chunk_size)
        elif tag == b"data":
            if fmt is None or data is not None:
                raise ValueError("data must follow one WAV format chunk and occur once")
            alignment = fmt["block_align_bytes"]
            if chunk_size % alignment:
                raise ValueError("WAV payload contains an incomplete frame")
            frames = chunk_size // alignment
            if not 1 <= frames <= MAX_SECONDS * fmt["sample_rate_hz"]:
                raise ValueError("WAV duration must be greater than 0 and at most 600 seconds")
            data = {"frames": frames, "duration_seconds": frames / fmt["sample_rate_hz"],
                    "data_offset": start, "data_bytes": chunk_size}
        elif tag == b"fact":
            if fact is not None or chunk_size != 4:
                raise ValueError("one exact four-byte WAV fact chunk is permitted")
            fact = struct.unpack("<I", _exact(handle, 4))[0]
        position = padded_end
    if fmt is None or data is None:
        raise ValueError("WAV format and data chunks are required")
    if fact is not None and fact != data["frames"]:
        raise ValueError("WAV fact frame count does not match the data payload")
    if _regular_bounded(handle) != size:
        raise ValueError("input changed while inspecting WAV")
    return {**fmt, **data, "container_bytes": size,
            "fact_frames": fact, "container_chunk_count": chunks}


def inspect_precision_wav(path):
    """Inspect bounded RIFF headers; sample validity is checked when read."""
    with Path(path).open("rb") as handle:
        return _inspect_handle(handle)


def _sample_layout(metadata):
    if not isinstance(metadata, dict):
        raise ValueError("WAV metadata must be a dictionary")
    sample_format = metadata.get("sample_format")
    if not isinstance(sample_format, str) or sample_format not in _FORMATS:
        raise ValueError("unsupported precision sample format")
    channels = metadata.get("channels")
    if type(channels) is not int or channels not in (1, 2):
        raise ValueError("only mono/stereo sample blocks are supported")
    width = _FORMATS[sample_format][2]
    if metadata.get("sample_width_bytes", width) != width:
        raise ValueError("sample width and format disagree")
    return sample_format, channels, width


def decode_precision_payload(payload, metadata):
    """Decode one native block without an intermediate lower-bit-depth format."""
    sample_format, channels, width = _sample_layout(metadata)
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise ValueError("sample payload must be bytes")
    try:
        payload = memoryview(payload).cast("B")
    except (TypeError, ValueError) as error:
        raise ValueError("sample payload must be contiguous bytes") from error
    frame_bytes = channels * width
    if len(payload) % frame_bytes or not 1 <= len(payload) // frame_bytes <= MAX_READ_FRAMES:
        raise ValueError("sample payload must contain 1..4096 complete frames")
    if sample_format == PCM16:
        result = np.frombuffer(payload, dtype="<i2").astype(np.float64) / 32768.0
    elif sample_format == PCM24:
        packed = np.frombuffer(payload, dtype=np.uint8).reshape(-1, 3)
        codes = (packed[:, 0].astype(np.int32)
                 | (packed[:, 1].astype(np.int32) << 8)
                 | (packed[:, 2].astype(np.int32) << 16))
        codes = (codes ^ 0x800000) - 0x800000
        result = codes.astype(np.float64) / 8388608.0
    else:
        result = np.frombuffer(payload, dtype="<f4").astype(np.float64)
        if not np.isfinite(result).all() or np.any(result < -1) or np.any(result >= 1):
            raise ValueError("float32 WAV samples must be finite and in [-1, 1); no normalization")
    return result.reshape(-1, channels)


def encode_precision_block(values, sample_format):
    """Round-to-nearest/ties-even once, refusing saturation or lost headroom."""
    if not isinstance(sample_format, str) or sample_format not in _FORMATS:
        raise ValueError("unsupported precision sample format")
    source = np.asarray(values)
    if (source.dtype.kind not in "fiu" or source.ndim != 2
            or source.shape[1] not in (1, 2) or not 1 <= source.shape[0] <= MAX_READ_FRAMES):
        raise ValueError("output block must be 1..4096 mono/stereo real frames")
    normalized = np.asarray(source, dtype=np.float64)
    if (not np.isfinite(normalized).all() or np.any(normalized < -1)
            or np.any(normalized >= 1)):
        raise ValueError("output must be finite and in [-1, 1); no clipping or normalization")
    if sample_format == FLOAT32:
        quantized = normalized.astype("<f4")
        if (not np.isfinite(quantized).all() or np.any(quantized < -1)
                or np.any(quantized >= 1)):
            raise ValueError("output cannot be represented as normalized float32 without clipping")
        payload = quantized.tobytes()
    else:
        scale = 32768 if sample_format == PCM16 else 8388608
        rounded = np.rint(normalized * scale)
        if np.any(rounded < -scale) or np.any(rounded > scale - 1):
            raise ValueError("output cannot be represented in native PCM without clipping")
        if sample_format == PCM16:
            payload = rounded.astype("<i2").tobytes()
        else:
            codes = rounded.astype(np.int32).reshape(-1)
            packed = np.empty((codes.size, 3), dtype=np.uint8)
            packed[:, 0] = codes & 0xFF
            packed[:, 1] = (codes >> 8) & 0xFF
            packed[:, 2] = (codes >> 16) & 0xFF
            payload = packed.tobytes()
    return payload, decode_precision_payload(payload, {"sample_format": sample_format,
                                                     "channels": source.shape[1]})


class PrecisionWavReader:
    """Yield (raw payload, float64 frames); digests become final only at EOF.

    A reader is single-pass. The full-container digest includes metadata and pad
    bytes; payload_sha256 covers only the exact interleaved native sample bytes.
    """

    def __init__(self, path, expected=None):
        self.path, self.expected = Path(path), expected
        self.handle = None
        self.sha256 = self.payload_sha256 = None
        self.max_read_bytes = 0
        self._started = False

    def __enter__(self):
        if self.handle is not None or self._started:
            raise ValueError("reader can only be opened once")
        self.handle = self.path.open("rb")
        try:
            self.metadata = _inspect_handle(self.handle)
            if self.expected is not None and self.metadata != self.expected:
                raise ValueError("input changed between processing passes")
            self.handle.seek(0)
            return self
        except BaseException:
            self.handle.close()
            self.handle = None
            raise

    def __exit__(self, *args):
        if self.handle is not None:
            self.handle.close()
            self.handle = None

    def __iter__(self):
        if self.handle is None or self._started:
            raise ValueError("reader must be open and can only be iterated once")
        self._started = True
        full_digest, payload_digest = hashlib.sha256(), hashlib.sha256()
        total = 0

        def read(size):
            nonlocal total
            result = _exact(self.handle, size)
            total += len(result)
            if total > MAX_FILE_BYTES:
                raise ValueError("WAV container exceeds the 128 MiB limit")
            full_digest.update(result)
            self.max_read_bytes = max(self.max_read_bytes, len(result))
            return result

        prefix = self.metadata["data_offset"]
        while prefix:
            amount = min(prefix, MAX_READ_BYTES)
            read(amount)
            prefix -= amount
        remaining = self.metadata["frames"]
        width = self.metadata["sample_width_bytes"] * self.metadata["channels"]
        while remaining:
            count = min(remaining, MAX_READ_FRAMES)
            payload = read(count * width)
            decoded = decode_precision_payload(payload, self.metadata)
            payload_digest.update(payload)
            yield payload, decoded
            remaining -= count
        suffix = self.metadata["container_bytes"] - total
        while suffix:
            amount = min(suffix, MAX_READ_BYTES)
            read(amount)
            suffix -= amount
        if self.handle.read(1) or _regular_bounded(self.handle) != total:
            raise ValueError("input changed while reading; trailing or growing WAV file")
        self.sha256, self.payload_sha256 = full_digest.hexdigest(), payload_digest.hexdigest()


def _output_header(metadata):
    sample_format, channels, width = _sample_layout(metadata)
    rate, frames = metadata.get("sample_rate_hz"), metadata.get("frames")
    if type(rate) is not int or rate not in SAMPLE_RATES:
        raise ValueError("precision WAV supports 44100 or 48000 Hz only")
    if type(frames) is not int or not 1 <= frames <= rate * MAX_SECONDS:
        raise ValueError("WAV duration must be greater than 0 and at most 600 seconds")
    tag, bits, _ = _FORMATS[sample_format]
    if metadata.get("bits_per_sample", bits) != bits:
        raise ValueError("sample bits and format disagree")
    data_bytes = frames * channels * width
    fmt = struct.pack("<HHIIHH", tag, channels, rate, rate * channels * width, channels * width, bits)
    if sample_format == FLOAT32:
        fmt += b"\x00\x00"
    chunks = b"fmt " + struct.pack("<I", len(fmt)) + fmt
    if sample_format == FLOAT32:
        chunks += b"fact" + struct.pack("<II", 4, frames)
    chunks += b"data" + struct.pack("<I", data_bytes)
    container_bytes = 12 + len(chunks) + data_bytes + data_bytes % 2
    if container_bytes > MAX_FILE_BYTES:
        raise ValueError("output WAV container exceeds the 128 MiB limit")
    return b"RIFF" + struct.pack("<I", container_bytes - 8) + b"WAVE" + chunks


class PrecisionWavWriter:
    """Exclusive-create canonical WAV with known frames and native payloads.

    Partial files remain for the owning staging transaction to remove on failure;
    an incomplete writer never advertises a successful payload digest.
    """

    def __init__(self, path, metadata):
        self.path, self.metadata = Path(path), dict(metadata)
        self.header = _output_header(self.metadata)
        self.metadata["sample_width_bytes"] = _FORMATS[self.metadata["sample_format"]][2]
        self.handle = None
        self.frames_written = 0
        self.payload_sha256 = None
        self._digest = hashlib.sha256()
        self._failed = False

    def __enter__(self):
        if self.handle is not None or self.frames_written or self._failed:
            raise ValueError("writer can only be opened once")
        self.handle = self.path.open("xb")
        try:
            if self.handle.write(self.header) != len(self.header):
                raise OSError("incomplete WAV header write")
            return self
        except BaseException:
            self.handle.close()
            self.handle = None
            self._failed = True
            raise

    def write_payload(self, payload):
        if self.handle is None or self._failed:
            raise ValueError("writer must be open and healthy")
        try:
            values = decode_precision_payload(payload, self.metadata)
            count = values.shape[0]
            if self.frames_written + count > self.metadata["frames"]:
                raise ValueError("output payload exceeds declared frame count")
            raw = bytes(payload)
            if self.handle.write(raw) != len(raw):
                raise OSError("incomplete WAV payload write")
            self.frames_written += count
            self._digest.update(raw)
        except BaseException:
            self._failed = True
            raise

    def __exit__(self, exception_type, exception, traceback):
        try:
            if exception_type is None and not self._failed:
                if self.frames_written != self.metadata["frames"]:
                    raise ValueError("output payload is incomplete")
                total_payload = self.frames_written * self.metadata["channels"] * self.metadata["sample_width_bytes"]
                if total_payload % 2 and self.handle.write(b"\x00") != 1:
                    raise OSError("incomplete WAV pad write")
                self.handle.flush()
                if os.fstat(self.handle.fileno()).st_size != len(self.header) + total_payload + total_payload % 2:
                    raise ValueError("output WAV size does not match complete payload")
                self.payload_sha256 = self._digest.hexdigest()
            elif exception_type is None:
                raise ValueError("writer failed before completion")
        finally:
            self._failed = True
            if self.handle is not None:
                self.handle.close()
                self.handle = None
