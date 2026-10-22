"""Fixed worker entry point for optional, file-object-only PyAV decoding.

Never run arbitrary decoder commands or accept decoder options from a caller.
Stdout and stderr are discarded by the parent; one bounded JSON file is returned.
Native libraries run in a child process, not in an OS security sandbox.
"""

from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import stat
import struct
import sys


MAX_SOURCE_BYTES = 32 * 1024 * 1024
MAX_WAV_BYTES = 128 * 1024 * 1024
MAX_SECONDS = 600
BLOCK_FRAMES = 4096
FLOAT32 = "IEEE_FLOAT32_little_endian"


class DecodeError(Exception):
    pass


def _read_exact(handle, size):
    if not 0 <= size <= 65536:
        raise DecodeError("invalid_container")
    value = handle.read(size)
    if len(value) != size:
        raise DecodeError("invalid_container")
    return value


def _hash_handle(handle, limit):
    digest, total = hashlib.sha256(), 0
    handle.seek(0)
    while True:
        block = handle.read(65536)
        if not block:
            break
        total += len(block)
        if total > limit:
            raise DecodeError("decode_limit_exceeded")
        digest.update(block)
    handle.seek(0)
    return digest.hexdigest(), total


def _validate_m4a_references(handle, size):
    """Bounded ISO-BMFF walk: accept only self-contained data references.

    MOV options also disable dereferencing. This rejects external URL/URN data
    references before handing a file object to FFmpeg, without following them.
    """
    count, data_refs, moov, mdat = 0, 0, 0, 0
    containers = {b"moov", b"trak", b"mdia", b"minf", b"dinf"}

    def boxes(start, end, depth):
        nonlocal count, data_refs, moov, mdat
        if depth > 8:
            raise DecodeError("invalid_container")
        position = start
        while position < end:
            count += 1
            if count > 4096 or end - position < 8:
                raise DecodeError("invalid_container")
            handle.seek(position)
            length, kind = struct.unpack(">I4s", _read_exact(handle, 8))
            header = 8
            if length == 1:
                length, = struct.unpack(">Q", _read_exact(handle, 8))
                header = 16
            elif length == 0:
                length = end - position
            if length < header or position + length > end:
                raise DecodeError("invalid_container")
            payload, stop = position + header, position + length
            if kind == b"moov":
                moov += 1
            if kind == b"mdat":
                mdat += 1
            if kind in containers:
                boxes(payload, stop, depth + 1)
            elif kind == b"dref":
                data_refs += 1
                if stop - payload < 8:
                    raise DecodeError("invalid_container")
                handle.seek(payload)
                version_flags, entries = struct.unpack(">II", _read_exact(handle, 8))
                if version_flags != 0 or not 1 <= entries <= 16:
                    raise DecodeError("invalid_container")
                cursor = payload + 8
                for _ in range(entries):
                    if cursor + 12 > stop:
                        raise DecodeError("invalid_container")
                    handle.seek(cursor)
                    entry_size, entry_kind, flags = struct.unpack(">I4sI", _read_exact(handle, 12))
                    if entry_size != 12 or entry_kind != b"url " or flags != 1:
                        raise DecodeError("invalid_container")
                    cursor += entry_size
                if cursor != stop:
                    raise DecodeError("invalid_container")
            position = stop
        if position != end:
            raise DecodeError("invalid_container")

    boxes(0, size, 0)
    handle.seek(0)
    if moov != 1 or not mdat or not data_refs:
        raise DecodeError("invalid_container")


def _layout(layout):
    try:
        names = tuple(channel.name for channel in layout.channels)
    except (AttributeError, TypeError):
        raise DecodeError("unsupported_layout") from None
    if names == ("FC",) and layout.name == "mono":
        return 1
    if names == ("FL", "FR") and layout.name == "stereo":
        return 2
    raise DecodeError("unsupported_layout")


class _FloatWriter:
    """Unknown-length streaming Float32 WAV, finalized once after bounded EOF."""

    HEADER_BYTES = 58

    def __init__(self, path, rate, channels):
        self.path, self.rate, self.channels, self.frames = path, rate, channels, 0
        self.handle = path.open("x+b")
        self.handle.write(bytes(self.HEADER_BYTES))

    def write(self, values, np):
        if (values.ndim != 2 or values.shape[1] != self.channels
                or not 1 <= values.shape[0] <= BLOCK_FRAMES):
            raise DecodeError("decoder_failed")
        if (not np.isfinite(values).all() or np.any(values < -1) or np.any(values >= 1)):
            raise DecodeError("invalid_decoded_samples")
        frames = self.frames + values.shape[0]
        if (frames > self.rate * MAX_SECONDS
                or self.HEADER_BYTES + frames * self.channels * 4 > MAX_WAV_BYTES):
            raise DecodeError("decode_limit_exceeded")
        quantized = np.asarray(values, dtype="<f4", order="C")
        if (not np.isfinite(quantized).all() or np.any(quantized < -1) or np.any(quantized >= 1)):
            raise DecodeError("invalid_decoded_samples")
        payload = quantized.tobytes()
        self.handle.write(payload)
        self.frames = frames

    def finish(self):
        if not self.frames:
            raise DecodeError("empty_decoded_audio")
        data_bytes = self.frames * self.channels * 4
        fmt = struct.pack("<HHIIHHH", 3, self.channels, self.rate,
                          self.rate * self.channels * 4, self.channels * 4, 32, 0)
        header = (b"RIFF" + struct.pack("<I", self.HEADER_BYTES + data_bytes - 8) + b"WAVE"
                  + b"fmt " + struct.pack("<I", 18) + fmt
                  + b"fact" + struct.pack("<II", 4, self.frames)
                  + b"data" + struct.pack("<I", data_bytes))
        self.handle.seek(0)
        self.handle.write(header)
        self.handle.flush()
        self.handle.close()
        self.handle = None

    def close(self):
        if self.handle is not None:
            self.handle.close()
            self.handle = None


def _decode(source, output, source_format):
    try:
        import av
        import numpy as np
    except ImportError:
        raise DecodeError("dependency_unavailable") from None
    if not av.__version__.split(".", 1)[0] == "19":
        raise DecodeError("unsupported_dependency")
    av.logging.set_level(av.logging.PANIC)
    writer = None
    try:
        with source.open("rb") as handle:
            info = os.fstat(handle.fileno())
            if (not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= MAX_SOURCE_BYTES
                    or source.is_symlink()):
                raise DecodeError("invalid_container")
            source_digest, source_bytes = _hash_handle(handle, MAX_SOURCE_BYTES)
            if source_format == "m4a":
                _validate_m4a_references(handle, source_bytes)
            container_format = "mp3" if source_format == "mp3" else "mov"
            codec = "mp3" if source_format == "mp3" else "aac"
            options = {"enable_drefs": "0", "use_absolute_path": "0",
                       "protocol_whitelist": "pipe", "err_detect": "explode",
                       "probesize": "1048576", "analyzeduration": "5000000"}
            with av.open(handle, mode="r", format=container_format, options=options,
                         buffer_size=32768) as container:
                streams = list(container.streams)
                if len(streams) != 1 or streams[0].type != "audio":
                    raise DecodeError("unsupported_stream")
                stream = streams[0]
                context = stream.codec_context
                if context.codec.canonical_name != codec:
                    raise DecodeError("unsupported_stream")
                rate, channels = context.sample_rate, _layout(context.layout)
                if rate not in (44100, 48000):
                    raise DecodeError("unsupported_rate")
                context.thread_count = 1
                context.options = {**context.options, "err_detect": "explode"}
                convert = av.AudioResampler(format="fltp", layout=context.layout.name, rate=rate)
                writer = _FloatWriter(output, rate, channels)
                timeline_origin, timestamp_frames, missing_timestamps = None, 0, 0

                def consume(frame):
                    nonlocal timeline_origin, timestamp_frames, missing_timestamps
                    if frame.is_corrupt:
                        raise DecodeError("corrupt_audio")
                    if frame.sample_rate != rate:
                        raise DecodeError("unsupported_rate")
                    if _layout(frame.layout) != channels or not 1 <= frame.samples <= 8192:
                        raise DecodeError("unsupported_layout")
                    if frame.pts is not None and frame.time_base is not None:
                        position = Fraction(frame.pts) * Fraction(frame.time_base)
                        if timeline_origin is None:
                            timeline_origin = position - Fraction(timestamp_frames, rate)
                        expected = timeline_origin + Fraction(timestamp_frames, rate)
                        tolerance = max(Fraction(1, rate), Fraction(frame.time_base))
                        if abs(position - expected) > tolerance:
                            raise DecodeError("discontinuous_timestamps")
                    else:
                        missing_timestamps += 1
                    timestamp_frames += frame.samples
                    converted = convert.resample(frame)
                    for prepared in converted:
                        if prepared.sample_rate != rate or _layout(prepared.layout) != channels:
                            raise DecodeError("decoder_failed")
                        planar = prepared.to_ndarray()
                        if planar.shape != (channels, prepared.samples):
                            raise DecodeError("decoder_failed")
                        for start in range(0, prepared.samples, BLOCK_FRAMES):
                            writer.write(planar[:, start:start + BLOCK_FRAMES].T, np)

                for packet in container.demux(stream):
                    if packet.is_corrupt:
                        raise DecodeError("corrupt_audio")
                    for frame in packet.decode():
                        consume(frame)
                for prepared in convert.resample(None):
                    planar = prepared.to_ndarray()
                    for start in range(0, prepared.samples, BLOCK_FRAMES):
                        writer.write(planar[:, start:start + BLOCK_FRAMES].T, np)
                if writer.frames != timestamp_frames:
                    raise DecodeError("decoder_failed")
                writer.finish()
                frames = writer.frames
            if _hash_handle(handle, MAX_SOURCE_BYTES) != (source_digest, source_bytes):
                raise DecodeError("source_changed")
        with output.open("rb") as wav:
            output_digest, output_bytes = _hash_handle(wav, MAX_WAV_BYTES)
        return {"ok": True, "source_format": source_format, "source_container": container_format,
                "source_codec": codec, "source_bytes": source_bytes, "source_sha256": source_digest,
                "decoder": {"name": "PyAV", "version": av.__version__,
                            "ffmpeg_libraries": {key: list(value) for key, value in av.library_versions.items()}},
                "decoded": {"frames": frames, "sample_rate_hz": rate, "channels": channels,
                            "sample_format": FLOAT32, "container_bytes": output_bytes,
                            "sha256": output_digest},
                "duration_seconds": frames / rate, "lossy_decode": True,
                "decoder_delay_padding_handled_by_backend": True,
                "original_sample_identity_guaranteed": False,
                "timestamp_continuity": "checked_when_timestamps_available",
                "frames_without_timestamps": missing_timestamps,
                "native_decoder_os_sandbox": False,
                "network_protocols_enabled": False,
                "external_data_references_enabled": False}
    except DecodeError:
        raise
    except Exception:
        raise DecodeError("corrupt_audio") from None
    finally:
        if writer is not None:
            writer.close()


def main():
    """Invoked only by the fixed parent bootstrap; return bounded error codes."""
    if len(sys.argv) != 5:
        raise SystemExit(2)
    source, output, source_format, status = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], Path(sys.argv[4])
    if source_format not in ("mp3", "m4a"):
        raise SystemExit(2)
    existed = output.exists()
    try:
        if existed:
            raise DecodeError("decoder_failed")
        result = _decode(source, output, source_format)
        code = 0
    except DecodeError as error:
        result, code = {"ok": False, "error": str(error)}, 1
    except Exception:
        result, code = {"ok": False, "error": "decoder_failed"}, 1
    if code and not existed:
        try:
            output.unlink(missing_ok=True)
        except OSError:
            pass
    encoded = json.dumps(result, ensure_ascii=True, allow_nan=False, separators=(",", ":")).encode("ascii")
    if len(encoded) > 16384:
        encoded, code = b'{"ok":false,"error":"decoder_failed"}', 1
    try:
        with status.open("xb") as handle:
            handle.write(encoded)
    except OSError:
        raise SystemExit(2) from None
    raise SystemExit(code)


if __name__ == "__main__":
    main()
