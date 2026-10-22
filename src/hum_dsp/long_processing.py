"""Bounded, staged 48 kHz PCM16 file processing without whole-audio buffers.

This is an offline file wrapper, not device audio or a real-time guarantee.
One original-input channel selected by full-file RMS drives linked decisions;
channels are never summed and no entire decision trace is retained in RAM.
"""

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import stat
import struct
import tempfile
from time import perf_counter
import wave

import numpy as np

from .band_control import BandControlConfig


ALGORITHM = "bounded_long_file_linked_band_control_v1"
SAMPLE_RATE_HZ = 48000
MAX_SECONDS = 600
MAX_FRAMES = SAMPLE_RATE_HZ * MAX_SECONDS
MAX_FILE_BYTES = 128 * 1024 * 1024
MAX_READ_FRAMES = 4096
MAX_READ_BYTES = MAX_READ_FRAMES * 2 * 2
MAX_WAVEFORM_BINS = 480
MAX_CONTAINER_CHUNKS = 4096
_PCM_GUID = b"\x01\x00\x00\x00\x00\x00\x10\x00\x80\x00\x00\xaa\x00\x38\x9b\x71"


def _regular_bounded(path):
    info = Path(path).stat()
    if not stat.S_ISREG(info.st_mode):
        raise ValueError("a regular WAV file is required")
    if info.st_size > MAX_FILE_BYTES:
        raise ValueError("WAV file exceeds the 128 MiB container limit")
    return info


def _exact(handle, size):
    if not 0 <= size <= MAX_READ_BYTES:
        raise ValueError("internal bounded read limit exceeded")
    block = handle.read(size)
    if len(block) != size:
        raise ValueError("truncated WAV container")
    return block


def _inspect_handle(handle):
    """Seek over chunks; never read arbitrary metadata or the data payload."""
    initial = os.fstat(handle.fileno())
    if not stat.S_ISREG(initial.st_mode):
        raise ValueError("a regular WAV file is required")
    size = initial.st_size
    if size > MAX_FILE_BYTES:
        raise ValueError("WAV file exceeds the 128 MiB container limit")
    handle.seek(0)
    header = _exact(handle, 12)
    if header[:4] != b"RIFF" or header[8:] != b"WAVE":
        raise ValueError("invalid RIFF WAV container")
    riff_end = struct.unpack_from("<I", header, 4)[0] + 8
    if riff_end != size:
        raise ValueError("truncated or trailing RIFF WAV container bytes")
    position, fmt, data, chunk_count = 12, None, None, 0
    while position < riff_end:
        chunk_count += 1
        if chunk_count > MAX_CONTAINER_CHUNKS:
            raise ValueError("WAV container exceeds the 4096 chunk limit")
        if position + 8 > riff_end:
            raise ValueError("truncated WAV chunk header")
        handle.seek(position)
        tag, chunk_size = struct.unpack("<4sI", _exact(handle, 8))
        start = position + 8
        end = start + chunk_size
        padded_end = end + chunk_size % 2
        if padded_end > riff_end:
            raise ValueError("truncated WAV chunk payload")
        if tag == b"fmt ":
            if fmt is not None or chunk_size < 16:
                raise ValueError("ambiguous or incomplete WAV format chunk")
            prefix = _exact(handle, min(chunk_size, 40))
            format_tag, channels, rate, byte_rate, alignment, bits = struct.unpack_from("<HHIIHH", prefix)
            if rate != SAMPLE_RATE_HZ:
                raise ValueError("long WAV mode supports 48000 Hz only; use general WAV mode for short 44100 Hz files")
            if channels not in (1, 2) or bits != 16:
                raise ValueError("only mono/stereo uncompressed PCM16 WAV is supported")
            if alignment != channels * 2 or byte_rate != rate * alignment:
                raise ValueError("inconsistent PCM16 WAV format alignment or byte rate")
            channel_mask = None
            if format_tag == 0xFFFE:
                if (chunk_size < 40 or struct.unpack_from("<H", prefix, 16)[0] < 22
                        or 18 + struct.unpack_from("<H", prefix, 16)[0] > chunk_size
                        or struct.unpack_from("<H", prefix, 18)[0] != 16
                        or prefix[24:40] != _PCM_GUID):
                    raise ValueError("unsupported extensible WAV format")
                channel_mask = struct.unpack_from("<I", prefix, 20)[0]
                if channel_mask not in ((0, 4) if channels == 1 else (0, 3)):
                    raise ValueError("only canonical mono/stereo extensible WAV channel masks are supported")
            elif format_tag != 1:
                raise ValueError("uncompressed PCM16 WAV format required")
            fmt = {"sample_rate_hz": rate, "channels": channels,
                   "sample_format": "signed_PCM16_little_endian",
                   "wav_format_tag": format_tag, "channel_mask": channel_mask,
                   "channel_order": ["front_center"] if channels == 1 else ["front_left", "front_right"],
                   "canonical_layout_assumption": "unspecified_masks_and_classic_PCM_use_canonical_mono_or_left_right_order"}
        elif tag == b"data":
            if fmt is None or data is not None:
                raise ValueError("ambiguous WAV data chunk")
            alignment = fmt["channels"] * 2
            if chunk_size % alignment:
                raise ValueError("WAV payload contains an incomplete PCM16 frame")
            frames = chunk_size // alignment
            if not 1 <= frames <= MAX_FRAMES:
                raise ValueError("WAV duration must be greater than 0 and at most 600 seconds")
            data = {"frames": frames, "duration_seconds": frames / SAMPLE_RATE_HZ,
                    "data_offset": start, "data_bytes": chunk_size}
        position = padded_end
    if fmt is None or data is None:
        raise ValueError("WAV format and data chunks are required")
    if os.fstat(handle.fileno()).st_size != size:
        raise ValueError("input changed while inspecting WAV")
    return {**fmt, **data, "container_bytes": size}


def inspect_long_wav(path):
    """Inspect strict classic/canonical-extensible PCM16 without loading audio."""
    _regular_bounded(path)
    with Path(path).open("rb") as handle:
        return _inspect_handle(handle)


class _PCMBlocks:
    """Hash exact container bytes while yielding at most 4096 PCM frames."""

    def __init__(self, path, expected=None):
        self.path, self.expected = Path(path), expected
        self.handle = None
        self.sha256 = None
        self.max_read_bytes = 0

    def __enter__(self):
        _regular_bounded(self.path)
        self.handle = self.path.open("rb")
        try:
            self.metadata = _inspect_handle(self.handle)
            if self.expected is not None and self.metadata != self.expected:
                raise ValueError("input changed between processing passes")
            self.handle.seek(0)
            return self
        except BaseException:
            self.handle.close()
            raise

    def __exit__(self, *args):
        self.handle.close()

    def __iter__(self):
        digest = hashlib.sha256()
        total = 0

        def read(size):
            nonlocal total
            if size > MAX_READ_BYTES:
                raise ValueError("internal bounded read limit exceeded")
            result = _exact(self.handle, size)
            total += len(result)
            if total > MAX_FILE_BYTES:
                raise ValueError("WAV file exceeds the 128 MiB container limit")
            digest.update(result)
            self.max_read_bytes = max(self.max_read_bytes, len(result))
            return result

        prefix = self.metadata["data_offset"]
        while prefix:
            size = min(prefix, MAX_READ_BYTES)
            read(size)
            prefix -= size
        remaining = self.metadata["frames"]
        channels = self.metadata["channels"]
        while remaining:
            frames = min(remaining, MAX_READ_FRAMES)
            payload = read(frames * channels * 2)
            yield np.frombuffer(payload, dtype="<i2").reshape(frames, channels)
            remaining -= frames
        suffix = self.metadata["container_bytes"] - total
        while suffix:
            size = min(suffix, MAX_READ_BYTES)
            read(size)
            suffix -= size
        # Read one extra byte as well as checking fstat: growth is never ignored.
        if self.handle.read(1) or os.fstat(self.handle.fileno()).st_size != total:
            raise ValueError("input changed while reading; trailing or growing WAV file")
        self.sha256 = digest.hexdigest()


def _sha256_bounded(path):
    _regular_bounded(path)
    digest = hashlib.sha256()
    total = 0
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(min(MAX_READ_BYTES, MAX_FILE_BYTES + 1 - total))
            if not block:
                break
            total += len(block)
            if total > MAX_FILE_BYTES:
                raise ValueError("WAV file exceeds the 128 MiB container limit")
            digest.update(block)
        if os.fstat(handle.fileno()).st_size != total:
            raise ValueError("input changed while hashing")
    return digest.hexdigest()


class _Statistics:
    def __init__(self, frames, channels):
        self.frames, self.channels, self.position = frames, channels, 0
        self.squares = np.zeros(channels, dtype=np.float64)
        self.peak = np.zeros(channels, dtype=np.float64)
        count = min(MAX_WAVEFORM_BINS, frames)
        self.edges = np.arange(count + 1, dtype=np.int64) * frames // count
        self.minimum = np.full(count, np.inf)
        self.maximum = np.full(count, -np.inf)

    def add(self, block):
        values = np.asarray(block, dtype=np.float64)
        if (values.ndim != 2 or values.shape[1] != self.channels
                or not 1 <= values.shape[0] <= MAX_READ_FRAMES
                or self.position + values.shape[0] > self.frames
                or not np.isfinite(values).all()):
            raise ValueError("invalid bounded statistics block")
        self.squares += np.sum(values * values, axis=0)
        self.peak = np.maximum(self.peak, np.max(np.abs(values), axis=0))
        stop = self.position + values.shape[0]
        bin_index = int(np.searchsorted(self.edges, self.position, side="right") - 1)
        while self.edges[bin_index] < stop:
            start_index = max(self.position, int(self.edges[bin_index])) - self.position
            stop_index = min(stop, int(self.edges[bin_index + 1])) - self.position
            window = values[start_index:stop_index]
            self.minimum[bin_index] = min(self.minimum[bin_index], float(np.min(window)))
            self.maximum[bin_index] = max(self.maximum[bin_index], float(np.max(window)))
            bin_index += 1
            if bin_index == len(self.minimum):
                break
        self.position = stop

    def result(self):
        if self.position != self.frames or not np.isfinite(self.minimum).all() or not np.isfinite(self.maximum).all():
            raise ValueError("incomplete streaming file statistics")
        channel_rms = np.sqrt(self.squares / self.frames)
        return {"peak_abs": float(np.max(self.peak)),
                "rms": float(np.sqrt(np.sum(self.squares) / (self.frames * self.channels))),
                "channel_levels": [{"channel_index": index, "peak_abs": float(self.peak[index]),
                                    "rms": float(channel_rms[index])} for index in range(self.channels)]}

    def waveform(self):
        self.result()
        return np.column_stack((self.minimum, self.maximum)).tolist()


def _encode_block(block, shape):
    array = np.asarray(block)
    if array.dtype.kind not in "fiu" or array.shape != shape:
        raise ValueError("processor changed block shape or returned non-real samples")
    values = np.array(array, dtype=np.float64, copy=True)
    if not np.isfinite(values).all() or np.any(values < -1) or np.any(values >= 1):
        raise ValueError("processor output is nonfinite or outside [-1, 1); no clipping or normalization")
    codes = np.rint(values * 32768)
    if np.any(codes < -32768) or np.any(codes > 32767):
        raise ValueError("output cannot be represented as PCM16 without clipping")
    return codes.astype("<i2"), values


def process_long_wav(input_path, output_directory, config=None):
    """Stage and commit a new result, with no partial target on processing faults.

    Full-file input analysis fixes a reference channel before continuous block
    rendering begins. This is intentionally offline, not low-latency playback.
    All-zero depths skip the engine and preserve every signed PCM16 code.
    """
    selected = BandControlConfig() if config is None else config
    if type(selected) is not BandControlConfig:
        raise ValueError("config must be a validated BandControlConfig")
    selected = BandControlConfig(**asdict(selected))
    source, target = Path(input_path), Path(output_directory)
    if target.exists() or target.is_symlink():
        raise FileExistsError("output directory must be new; previous files are never replaced")
    started = perf_counter()
    with _PCMBlocks(source) as first:
        metadata = first.metadata
        input_stats = _Statistics(metadata["frames"], metadata["channels"])
        for codes in first:
            input_stats.add(codes.astype(np.float64) / 32768)
        original_sha = first.sha256
        maximum_read_bytes = first.max_read_bytes
    input_levels = input_stats.result()
    channel_rms = [entry["rms"] for entry in input_levels["channel_levels"]]
    reference = int(np.argmax(channel_rms)) if selected.orders else None
    frames, channels = metadata["frames"], metadata["channels"]
    floating_stats, output_stats = _Statistics(frames, channels), _Statistics(frames, channels)
    target.parent.mkdir(parents=True, exist_ok=True)
    # TemporaryDirectory removes every partial audio, trace and report on faults.
    with tempfile.TemporaryDirectory(prefix=".hum_long_", dir=target.parent) as temporary:
        stage = Path(temporary) / "result"
        stage.mkdir()
        events = 0
        with (stage / "long_trace.jsonl").open("x", encoding="utf-8", newline="\n") as trace_file:
            def sink(event):
                nonlocal events
                if type(event) is not dict:
                    raise ValueError("trace event must be a dictionary")
                trace_file.write(json.dumps(event, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n")
                events += 1

            engine = None
            if selected.orders:
                from .stream_bands import StreamBandControl
                engine = StreamBandControl(selected, channels=channels,
                                           reference_channel_index=reference, trace_sink=sink)
            quantization_error, processor_applied = 0.0, False
            maximum_engine_block_frames = 0
            with _PCMBlocks(source, metadata) as second:
                with wave.open(str(stage / "processed.wav"), "wb") as destination:
                    destination.setnchannels(channels)
                    destination.setsampwidth(2)
                    destination.setframerate(SAMPLE_RATE_HZ)
                    destination.setnframes(frames)
                    for original_codes in second:
                        values = original_codes.astype(np.float64) / 32768
                        if engine is None:
                            result_codes, floating = original_codes, values
                        else:
                            # Disk blocks remain independently bounded at 4096;
                            # user-selected engine partitioning is honored, with
                            # no state reset at either read or engine boundaries.
                            result_codes = np.empty_like(original_codes)
                            floating = np.empty_like(values)
                            for start in range(0, values.shape[0], selected.block_size_frames):
                                stop = min(start + selected.block_size_frames, values.shape[0])
                                part = values[start:stop]
                                part_codes, part_floating = _encode_block(engine.process_block(part), part.shape)
                                result_codes[start:stop] = part_codes
                                floating[start:stop] = part_floating
                                maximum_engine_block_frames = max(maximum_engine_block_frames, stop - start)
                        quantized = result_codes.astype(np.float64) / 32768
                        floating_stats.add(floating)
                        output_stats.add(quantized)
                        quantization_error = max(quantization_error, float(np.max(np.abs(quantized - floating))))
                        processor_applied = processor_applied or bool(np.any(result_codes != original_codes))
                        destination.writeframesraw(result_codes.tobytes())
                if second.sha256 != original_sha:
                    raise ValueError("input changed between processing passes")
                maximum_read_bytes = max(maximum_read_bytes, second.max_read_bytes)
        output_metadata = inspect_long_wav(stage / "processed.wav")
        if (output_metadata["frames"], output_metadata["channels"], output_metadata["sample_rate_hz"]) != (frames, channels, SAMPLE_RATE_HZ):
            raise ValueError("processor changed frame count, channels or sample rate")
        description = engine.describe() if engine is not None else {
            "algorithm": "identity_PCM16_zero_depth_bypass", "samples_processed": frames,
            "processing_bypassed": True, "shared_original_input_fft_count": 0,
            "analysis_trace_event_count": 0, "native_PCM16_code_identity": True}
        if description.get("samples_processed") != frames:
            raise ValueError("processor clock does not match complete output frame count")
        if engine is not None and description.get("trace_event_count") != events:
            raise ValueError("processor trace summary does not match written trace events")
        stereo_policy = {"mode": "linked_same_decisions_and_coefficients_no_downmix_no_crossfeed",
                         "reference_channel_index": reference,
                         "reference_selection": "first_maximum_full_file_native_raw_RMS",
                         "reference_selection_uses_entire_file": bool(selected.orders),
                         "native_channel_rms": channel_rms,
                         "same_trace_and_coefficients_all_channels": True,
                         "independent_channel_frequency_tracking": False,
                         "perceptual_stereo_preservation_verified": False,
                         "limitation": "A quieter other channel's weak hum may be missed; opposite-polarity channels are never summed."}
        report = {
            "schema_version": 1, "stage": "H3b_bounded_long_WAV_streaming",
            "engine": ALGORITHM, "config": selected.to_dict(), "algorithm": description,
            "processor_applied": processor_applied, "processing_bypassed": not selected.orders,
            "band_trace": {"file_name": "long_trace.jsonl", "format": "JSON_lines_actual_events",
                           "event_count": events, "full_trace_retained_in_memory": False},
            "device_io": False, "output_gain_normalization_applied": False,
            "posthoc_alignment_applied": False, "reference_quality_metrics": None,
            "quality_notice": "Long-file compatibility is not proof of improved sound quality. Wanted same-frequency content can still be attenuated.",
            "input": {"file_name": source.name, "sha256": original_sha, **metadata, **input_levels},
            "output": {"file_name": "processed.wav", "sha256": _sha256_bounded(stage / "processed.wav"),
                       **output_metadata, **output_stats.result()},
            "floating_output": floating_stats.result(), "stereo_policy": stereo_policy,
            "pcm16_quantization_max_abs_error": quantization_error,
            "pcm16_rounding": "nearest_ties_to_even_reject_unrepresentable_endpoint_no_saturation",
            "long_file": {"maximum_duration_seconds": MAX_SECONDS, "offline_file_processing": True,
                          "first_pass": "container_hash_channel_RMS_and_waveforms",
                          "second_pass": "container_hash_continuous_engine_and_output",
                          "final_source_identity_hash_checked": True,
                          "native_frame_count_preserved": True, "sample_rate_conversion_applied": False,
                          "metadata_chunks_copied": False, "staged_result_commit": True,
                          "atomic_no_replace_commit_verified_platform": "Windows",
                          "commit_platform_notice": "Windows rename refuses existing targets atomically. Other platforms are not release-validated; a concurrent empty-target creation race is not guaranteed safe."},
            "resource_limits": {"max_container_bytes": MAX_FILE_BYTES, "max_duration_seconds": MAX_SECONDS,
                                "max_read_frames": MAX_READ_FRAMES, "max_read_bytes": MAX_READ_BYTES,
                                "max_container_chunks": MAX_CONTAINER_CHUNKS,
                                "configured_engine_block_frames": selected.block_size_frames,
                                "max_engine_block_frames_observed": maximum_engine_block_frames,
                                "observed_max_read_bytes": maximum_read_bytes,
                                "waveform_bin_limit": MAX_WAVEFORM_BINS,
                                "whole_audio_retained_in_memory": False,
                                "whole_trace_retained_in_memory": False},
            "waveforms": {"input": input_stats.waveform(), "output": output_stats.waveform()},
            "timing": {"file_processing_seconds": perf_counter() - started,
                       "duration_seconds": frames / SAMPLE_RATE_HZ,
                       "includes_device_driver_or_playback": False,
                       "real_time_device_performance_verified": False},
            "limitations": [
                "Explicit long-file mode only: 48000 Hz mono/stereo uncompressed PCM16 WAV, at most 600 seconds and 128 MiB.",
                "44100 Hz long input, MP3/AAC, floating-point and 24-bit WAV are not accepted; no automatic conversion.",
                "Full-file RMS selection requires a preliminary disk pass; this is offline, not device real-time streaming.",
                "Only canonical layouts are accepted. Non-audio metadata chunks are dropped; output is classic PCM.",
                "At most 4096 RIFF chunks are accepted; ambiguous or excessively fragmented containers are rejected.",
                "Atomic no-replace result commit is verified on Windows only; other platforms are not release-validated.",
                "All channels use one original analysis channel; quieter-channel-only hum may be missed.",
                "Wanted content at the selected frequencies remains indistinguishable from hum; no automatic quality approval.",
                "No normalization, clipping, limiter, playback, microphone capture, spatial cancellation, cloud or network use.",
            ],
        }
        serialized = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
        (stage / "report.json").write_text(serialized + "\n", encoding="utf-8")
        if _sha256_bounded(source) != original_sha:
            raise ValueError("input changed while processing")
        if target.exists() or target.is_symlink():
            raise FileExistsError("output directory appeared during processing; never replacing it")
        # Windows rename is an atomic no-replace directory commit. The final
        # existence check also refuses already-created targets on other hosts.
        stage.rename(target)
    return report
