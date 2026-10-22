"""Optional, bounded local MP3/AAC-M4A decoding in an isolated Python worker.

The parent never imports native PyAV/FFmpeg. This is process isolation and
protocol restriction, not an operating-system sandbox. Only owned temporary
files are removed; the source and any existing destination are never modified.
"""

import hashlib
from importlib import metadata as import_metadata
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import tempfile

from .precision_wav import FLOAT32, PrecisionWavReader, inspect_precision_wav


MAX_SOURCE_BYTES = 32 * 1024 * 1024
MAX_STATUS_BYTES = 16 * 1024
MAX_TIMEOUT_SECONDS = 120.0
HASH_BLOCK_BYTES = 65536
_BOOTSTRAP = (
    "import sys; sys.path.insert(0, sys.argv.pop(1)); "
    "from hum_dsp._compressed_worker import main; main()"
)
_ERRORS = {
    "dependency_unavailable": "the optional compressed-audio decoder is unavailable",
    "unsupported_dependency": "the compressed-audio decoder requires PyAV 19.x",
    "invalid_container": "the compressed container is invalid or uses external references",
    "unsupported_stream": "exactly one MP3 or AAC audio stream and no other streams are required",
    "unsupported_rate": "only 44100 or 48000 Hz compressed audio is supported",
    "unsupported_layout": "only canonical mono or stereo compressed audio is supported",
    "corrupt_audio": "corrupt or incomplete compressed audio is unsupported",
    "discontinuous_timestamps": "discontinuous compressed-audio timestamps are unsupported",
    "decode_limit_exceeded": "decoded audio exceeds the 600-second or 128 MiB limit",
    "invalid_decoded_samples": "decoded samples must be finite and in [-1, 1); no normalization",
    "empty_decoded_audio": "compressed audio contains no decoded samples",
    "source_changed": "compressed input changed during decoding",
    "decoder_failed": "compressed audio could not be decoded safely",
}


def decoder_status():
    """Read package metadata without loading PyAV or any native decoder library."""
    try:
        version = import_metadata.version("av")
    except import_metadata.PackageNotFoundError:
        version = None
    compatible = isinstance(version, str) and bool(re.fullmatch(r"19\.\d+\.\d+", version))
    return {"available": compatible, "dependency": "av", "required_version": ">=19,<20",
            "installed_version": version,
            "reason": None if compatible else (
                "optional_dependency_missing" if version is None else "unsupported_dependency_version")}


def _reparse(info):
    return stat.S_ISLNK(info.st_mode) or bool(
        getattr(info, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 1024))


def _safe_path(value, *, output=False):
    try:
        raw = os.fspath(value)
    except TypeError as error:
        raise ValueError("a local filesystem path is required") from error
    if not isinstance(raw, str) or not raw or "\x00" in raw or "://" in raw or raw.startswith(("//", "\\\\")):
        raise ValueError("a local filesystem path is required; URLs and network shares are unsupported")
    path = Path(os.path.abspath(raw))
    for item in reversed([path, *path.parents]):
        try:
            info = item.lstat()
        except FileNotFoundError:
            if output and item == path:
                continue
            raise ValueError("local input and output parent directories must exist") from None
        if _reparse(info):
            raise ValueError("symbolic links and reparse paths are unsupported")
        if item == path:
            if output:
                raise ValueError("output WAV must not already exist")
            if not stat.S_ISREG(info.st_mode):
                raise ValueError("a regular compressed audio file is required")
        elif not stat.S_ISDIR(info.st_mode):
            raise ValueError("local input and output parent directories must be directories")
    return path


def _source_digest(path, copy_to=None):
    digest = hashlib.sha256()
    total = 0
    with path.open("rb") as source:
        before = os.fstat(source.fileno())
        if not stat.S_ISREG(before.st_mode) or _reparse(before) or not 0 < before.st_size <= MAX_SOURCE_BYTES:
            raise ValueError("compressed input must contain 1 byte to 32 MiB")
        while True:
            block = source.read(HASH_BLOCK_BYTES)
            if not block:
                break
            total += len(block)
            if total > MAX_SOURCE_BYTES:
                raise ValueError("compressed input exceeds the 32 MiB limit")
            digest.update(block)
            if copy_to is not None:
                copy_to.write(block)
        after = os.fstat(source.fileno())
    if (total != before.st_size or total != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns):
        raise ValueError("compressed input changed while reading")
    return digest.hexdigest(), total


def _worker_decode(source, output, source_format, status, timeout_seconds):
    trusted_package_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    command = [sys.executable, "-I", "-c", _BOOTSTRAP, trusted_package_root,
               str(source), str(output), source_format, str(status)]
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    try:
        child = subprocess.Popen(command, shell=False, stdin=subprocess.DEVNULL,
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                 cwd=str(status.parent), creationflags=creationflags)
        try:
            returncode = child.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait(timeout=10)
            raise RuntimeError("compressed audio decoder exceeded its time limit") from None
    except OSError:
        raise RuntimeError("compressed audio decoder could not be started") from None
    try:
        size = status.stat().st_size
        if not 1 <= size <= MAX_STATUS_BYTES:
            raise ValueError
        with status.open("rb") as handle:
            raw = handle.read(MAX_STATUS_BYTES + 1)
        result = json.loads(raw)
        if not isinstance(result, dict):
            raise ValueError
    except (OSError, ValueError, UnicodeError):
        raise ValueError("compressed audio decoder returned an invalid result") from None
    if returncode != 0 or result.get("ok") is not True:
        code = result.get("error")
        message = _ERRORS.get(code, _ERRORS["decoder_failed"]) if isinstance(code, str) else _ERRORS["decoder_failed"]
        if code in ("dependency_unavailable", "unsupported_dependency"):
            raise RuntimeError(message)
        raise ValueError(message)
    return result


def _validate_result(result, output, source_format, digest, source_bytes):
    expected_container, expected_codec = ("mp3", "mp3") if source_format == "mp3" else ("mov", "aac")
    decoded = result.get("decoded")
    decoder = result.get("decoder")
    if (result.get("source_format") != source_format
            or result.get("source_container") != expected_container
            or result.get("source_codec") != expected_codec
            or type(result.get("source_bytes")) is not int or result["source_bytes"] != source_bytes
            or result.get("source_sha256") != digest
            or result.get("lossy_decode") is not True
            or result.get("decoder_delay_padding_handled_by_backend") is not True
            or result.get("original_sample_identity_guaranteed") is not False
            or not isinstance(decoded, dict) or not isinstance(decoder, dict)
            or decoder.get("name") != "PyAV" or not isinstance(decoder.get("version"), str)
            or not re.fullmatch(r"19\.\d+\.\d+", decoder["version"])):
        raise ValueError("compressed audio decoder returned inconsistent metadata")
    metadata = inspect_precision_wav(output)
    if (metadata["sample_format"] != FLOAT32 or metadata["fact_frames"] != metadata["frames"]
            or any(type(decoded.get(key)) is not int or decoded[key] != metadata[key]
                   for key in ("frames", "sample_rate_hz", "channels", "container_bytes"))
            or decoded.get("sample_format") != FLOAT32):
        raise ValueError("decoded WAV does not match the decoder result")
    with PrecisionWavReader(output, expected=metadata) as reader:
        for _raw, _values in reader:
            pass
        wav_digest = reader.sha256
    if decoded.get("sha256") != wav_digest:
        raise ValueError("decoded WAV does not match the decoder digest")
    if (type(result.get("duration_seconds")) not in (int, float)
            or result["duration_seconds"] != metadata["frames"] / metadata["sample_rate_hz"]):
        raise ValueError("decoded duration does not match the decoder result")


def decode_compressed(source_path, new_wav_path, *, source_format=None, timeout_seconds=120.0):
    """Decode local MP3 or AAC-M4A to a new normalized Float32 WAV, fail closed.

    Decoding is optional, never installed automatically, lossy, and does not
    promise original pre-encoding sample identity or exact source duration.
    """
    if (isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds) or not 0 < timeout_seconds <= MAX_TIMEOUT_SECONDS):
        raise ValueError("decoder timeout must be greater than 0 and at most 120 seconds")
    source = _safe_path(source_path)
    destination = _safe_path(new_wav_path, output=True)
    source_format = source.suffix.lower().lstrip(".") if source_format is None else source_format
    if not isinstance(source_format, str) or source_format not in ("mp3", "m4a"):
        raise ValueError("only local MP3 and AAC-M4A input is supported")
    if destination.suffix.lower() != ".wav":
        raise ValueError("decoded output must have a .wav extension")
    availability = decoder_status()
    if not availability["available"]:
        raise RuntimeError(_ERRORS["dependency_unavailable"] if availability["installed_version"] is None
                           else _ERRORS["unsupported_dependency"])
    with tempfile.TemporaryDirectory(prefix="hum_decode_", dir=destination.parent) as work_name:
        work = Path(work_name)
        staged_source = work / ("input." + source_format)
        staged_output = work / "decoded.wav"
        status = work / "status.json"
        with staged_source.open("xb") as copy:
            digest, source_bytes = _source_digest(source, copy)
        result = _worker_decode(staged_source, staged_output, source_format, status, float(timeout_seconds))
        _validate_result(result, staged_output, source_format, digest, source_bytes)
        _safe_path(source)
        if _source_digest(source) != (digest, source_bytes):
            raise ValueError("compressed input changed during decoding")
        _safe_path(destination, output=True)
        try:
            # Atomic no-replace publication on both Windows and POSIX. There is
            # no fallback to an overwrite-prone rename or truncating open.
            if os.name == "nt":
                os.rename(staged_output, destination)
            else:
                os.link(staged_output, destination)
        except FileExistsError:
            raise ValueError("output WAV must not already exist") from None
        except OSError:
            raise ValueError("decoded WAV could not be published without overwriting") from None
    return {key: value for key, value in result.items() if key != "ok"}
