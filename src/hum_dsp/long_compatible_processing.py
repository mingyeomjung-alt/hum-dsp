"""Bounded 44.1/48 kHz long WAV compatibility without whole-audio storage.

The unchanged H3b wrapper owns the 48 kHz route.  At 44.1 kHz this explicit
offline adapter makes one original-input analysis pass, then a continuous
resample -> linked-filter -> resample pass.  FIR history and filter clocks do
not reset at disk-block boundaries.  No float audio is staged on disk.
"""

from dataclasses import asdict
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
from . import long_processing as _native
from .long_processing import _Statistics, _encode_block, _sha256_bounded


ALGORITHM = "bounded_long_native_rate_linked_band_control_v1"
SUPPORTED_SAMPLE_RATES_HZ = (44100, 48000)
MAX_SECONDS = 600
MAX_FILE_BYTES = 128 * 1024 * 1024
MAX_READ_FRAMES = 4096
MAX_READ_BYTES = MAX_READ_FRAMES * 2 * 2
MAX_WAVEFORM_BINS = 480
MAX_CONTAINER_CHUNKS = 4096
_PCM_GUID = _native._PCM_GUID


def _inspect_handle(handle):
    """Read only fixed headers and seek across even very large metadata chunks."""
    initial = os.fstat(handle.fileno())
    if not stat.S_ISREG(initial.st_mode):
        raise ValueError("a regular WAV file is required")
    size = initial.st_size
    if size > MAX_FILE_BYTES:
        raise ValueError("WAV file exceeds the 128 MiB container limit")
    handle.seek(0)
    header = _native._exact(handle, 12)
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
        tag, chunk_size = struct.unpack("<4sI", _native._exact(handle, 8))
        start = position + 8
        end = start + chunk_size
        padded_end = end + chunk_size % 2
        if padded_end > riff_end:
            raise ValueError("truncated WAV chunk payload")
        if tag == b"fmt ":
            if fmt is not None or chunk_size < 16:
                raise ValueError("ambiguous or incomplete WAV format chunk")
            prefix = _native._exact(handle, min(chunk_size, 40))
            tag_number, channels, rate, byte_rate, alignment, bits = struct.unpack_from("<HHIIHH", prefix)
            if rate not in SUPPORTED_SAMPLE_RATES_HZ:
                raise ValueError("long compatible WAV supports 44100 or 48000 Hz only")
            if channels not in (1, 2) or bits != 16:
                raise ValueError("only mono/stereo uncompressed PCM16 WAV is supported")
            if alignment != channels * 2 or byte_rate != rate * alignment:
                raise ValueError("inconsistent PCM16 WAV format alignment or byte rate")
            mask = None
            if tag_number == 0xFFFE:
                if (chunk_size < 40 or struct.unpack_from("<H", prefix, 16)[0] < 22
                        or 18 + struct.unpack_from("<H", prefix, 16)[0] > chunk_size
                        or struct.unpack_from("<H", prefix, 18)[0] != 16
                        or prefix[24:40] != _PCM_GUID):
                    raise ValueError("unsupported extensible WAV format")
                mask = struct.unpack_from("<I", prefix, 20)[0]
                if mask not in ((0, 4) if channels == 1 else (0, 3)):
                    raise ValueError("only canonical mono/stereo extensible WAV channel masks are supported")
            elif tag_number != 1:
                raise ValueError("uncompressed PCM16 WAV format required")
            fmt = {"sample_rate_hz": rate, "channels": channels,
                   "sample_format": "signed_PCM16_little_endian",
                   "wav_format_tag": tag_number, "channel_mask": mask,
                   "channel_order": ["front_center"] if channels == 1 else ["front_left", "front_right"],
                   "canonical_layout_assumption": "unspecified_masks_and_classic_PCM_use_canonical_mono_or_left_right_order"}
        elif tag == b"data":
            if fmt is None or data is not None:
                raise ValueError("ambiguous WAV data chunk")
            if chunk_size % (fmt["channels"] * 2):
                raise ValueError("WAV payload contains an incomplete PCM16 frame")
            frames = chunk_size // (fmt["channels"] * 2)
            if not 1 <= frames <= MAX_SECONDS * fmt["sample_rate_hz"]:
                raise ValueError("WAV duration must be greater than 0 and at most 600 seconds")
            data = {"frames": frames, "duration_seconds": frames / fmt["sample_rate_hz"],
                    "data_offset": start, "data_bytes": chunk_size}
        position = padded_end
    if fmt is None or data is None:
        raise ValueError("WAV format and data chunks are required")
    if os.fstat(handle.fileno()).st_size != size:
        raise ValueError("input changed while inspecting WAV")
    return {**fmt, **data, "container_bytes": size}


def inspect_long_compatible_wav(path):
    """Strict native-rate inspection, with no sample decode or full-file read."""
    _native._regular_bounded(path)
    with Path(path).open("rb") as handle:
        return _inspect_handle(handle)


class _PCMBlocks(_native._PCMBlocks):
    """Reuse H3b bounded/hash iterator, replacing only its explicit rate contract."""

    def __enter__(self):
        _native._regular_bounded(self.path)
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


def _parts(values, size=MAX_READ_FRAMES):
    for position in range(0, len(values), size):
        yield values[position:position + size]


def _normalized_block(values, channels):
    array = np.asarray(values)
    if (array.dtype.kind not in "fiu" or array.ndim != 2
            or array.shape[1] != channels or len(array) > MAX_READ_FRAMES
            or not np.isfinite(array).all() or np.any(array < -1) or np.any(array >= 1)):
        raise ValueError("conversion or engine produced invalid normalized audio; no clipping or normalization")
    return array


def _bypass_conversion(rate, frames):
    return {"conversion_applied": False, "reason": "all_band_depths_zero_native_PCM16_identity",
            "source_rate_hz": rate, "target_rate_hz": rate,
            "source_frames": frames, "target_frames": frames}


def process_long_compatible_wav(input_path, output_directory, config=None):
    """Process at a preserved native rate, staging every output before commit.

    All-zero controls skip the FIR and engine, preserving every PCM16 code.
    Active 44.1 kHz processing uses the same centered FIR as H3a, with finite
    lookahead and zero-extension only at the real file boundaries.  Thus this
    bounded file algorithm does not imply live or zero-latency operation.
    """
    selected = BandControlConfig(block_size_frames=MAX_READ_FRAMES) if config is None else config
    if type(selected) is not BandControlConfig:
        raise ValueError("config must be a validated BandControlConfig")
    selected = BandControlConfig(**asdict(selected))
    source, target = Path(input_path), Path(output_directory)
    if target.exists() or target.is_symlink():
        raise FileExistsError("output directory must be new; previous files are never replaced")
    metadata = inspect_long_compatible_wav(source)
    if metadata["sample_rate_hz"] == 48000:
        return _native.process_long_wav(source, target, selected)
    started = perf_counter()
    frames, channels, rate = metadata["frames"], metadata["channels"], metadata["sample_rate_hz"]
    engine_frames = (frames * 160 + 146) // 147
    native_stats = _Statistics(frames, channels)
    analysis_stats = _Statistics(engine_frames, channels) if selected.orders else None
    forward_analysis = None
    if selected.orders:
        from .stream_resampling import StreamingResampler
        forward_analysis = StreamingResampler(rate, 48000, frames, channels)

    def add_analysis(values):
        for part in _parts(values):
            analysis_stats.add(_normalized_block(part, channels))

    with _PCMBlocks(source, metadata) as first:
        for codes in first:
            values = codes.astype(np.float64) / 32768
            native_stats.add(values)
            if forward_analysis is not None:
                add_analysis(forward_analysis.process_block(values))
        if forward_analysis is not None:
            add_analysis(forward_analysis.finish())
        original_sha, maximum_read_bytes = first.sha256, first.max_read_bytes
    native_levels = native_stats.result()
    analysis_levels = [] if analysis_stats is None else [entry["rms"] for entry in analysis_stats.result()["channel_levels"]]
    reference = int(np.argmax(analysis_levels)) if analysis_levels else None
    floating_stats, output_stats = _Statistics(frames, channels), _Statistics(frames, channels)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".hum_long_compatible_", dir=target.parent) as temporary:
        stage = Path(temporary) / "result"
        stage.mkdir()
        events, quantization_error, processor_applied = 0, 0.0, False
        max_engine_block, max_resampler_output = 0, 0
        with (stage / "long_trace.jsonl").open("x", encoding="utf-8", newline="\n") as trace_file:
            def sink(event):
                nonlocal events
                if type(event) is not dict:
                    raise ValueError("trace event must be a dictionary")
                trace_file.write(json.dumps(event, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n")
                events += 1

            engine, forward, backward = None, None, None
            if selected.orders:
                from .stream_bands import StreamBandControl
                from .stream_resampling import StreamingResampler
                engine = StreamBandControl(selected, channels=channels, reference_channel_index=reference, trace_sink=sink)
                forward = StreamingResampler(rate, 48000, frames, channels)
                backward = StreamingResampler(48000, rate, engine_frames, channels, target_frames=frames)
            with _PCMBlocks(source, metadata) as second, source.open("rb") as comparison:
                comparison.seek(metadata["data_offset"])
                with wave.open(str(stage / "processed.wav"), "wb") as destination:
                    destination.setnchannels(channels)
                    destination.setsampwidth(2)
                    destination.setframerate(rate)
                    destination.setnframes(frames)

                    def write_output(values, original_codes=None):
                        nonlocal quantization_error, processor_applied, max_resampler_output
                        if engine is not None:
                            max_resampler_output = max(max_resampler_output, len(values))
                        for part in _parts(values):
                            codes, floating = _encode_block(part, part.shape)
                            if original_codes is None:
                                expected = _native._exact(comparison, len(part) * channels * 2)
                                old_codes = np.frombuffer(expected, dtype="<i2").reshape(part.shape)
                            else:
                                # Bypass writes exactly one <=4096-frame read.
                                old_codes = original_codes
                            quantized = codes.astype(np.float64) / 32768
                            floating_stats.add(floating)
                            output_stats.add(quantized)
                            quantization_error = max(quantization_error, float(np.max(np.abs(quantized - floating))))
                            processor_applied = processor_applied or bool(np.any(codes != old_codes))
                            destination.writeframesraw(codes.tobytes())

                    def render(values):
                        nonlocal max_engine_block, max_resampler_output
                        max_resampler_output = max(max_resampler_output, len(values))
                        for part in _parts(values, selected.block_size_frames):
                            _normalized_block(part, channels)
                            max_engine_block = max(max_engine_block, len(part))
                            processed = _normalized_block(engine.process_block(part), channels)
                            if processed.shape != part.shape:
                                raise ValueError("processor changed block shape")
                            write_output(backward.process_block(processed))

                    for original_codes in second:
                        values = original_codes.astype(np.float64) / 32768
                        if engine is None:
                            write_output(values, original_codes)
                        else:
                            render(forward.process_block(values))
                    if engine is not None:
                        render(forward.finish())
                        write_output(backward.finish())
                if second.sha256 != original_sha:
                    raise ValueError("input changed between processing passes")
                maximum_read_bytes = max(maximum_read_bytes, second.max_read_bytes)
        output_metadata = inspect_long_compatible_wav(stage / "processed.wav")
        if (output_metadata["frames"], output_metadata["channels"], output_metadata["sample_rate_hz"]) != (frames, channels, rate):
            raise ValueError("processor changed native frame count, channels or sample rate")
        description = engine.describe() if engine is not None else {
            "algorithm": "identity_PCM16_zero_depth_bypass", "samples_processed": frames,
            "processing_bypassed": True, "shared_original_input_fft_count": 0,
            "analysis_trace_event_count": 0, "native_PCM16_code_identity": True}
        if description.get("samples_processed") != (engine_frames if engine is not None else frames):
            raise ValueError("processor clock does not match complete engine frame count")
        if engine is not None and description.get("trace_event_count") != events:
            raise ValueError("processor trace summary does not match written trace events")
        conversions = {"to_engine": _bypass_conversion(rate, frames) if forward is None else forward.describe(),
                       "to_native": _bypass_conversion(rate, frames) if backward is None else backward.describe(),
                       "engine_sample_rate_hz": 48000, "native_sample_rate_hz": rate,
                       "native_frame_count_preserved": True}
        stereo_policy = {"mode": "linked_same_decisions_and_coefficients_no_downmix_no_crossfeed",
                         "reference_channel_index": reference,
                         "reference_selection": "first_maximum_full_file_engine_domain_raw_RMS",
                         "reference_selection_uses_entire_file": bool(selected.orders),
                         "engine_channel_rms": analysis_levels,
                         "same_trace_and_coefficients_all_channels": True,
                         "independent_channel_frequency_tracking": False,
                         "perceptual_stereo_preservation_verified": False,
                         "limitation": "A quieter other channel's weak hum may be missed; opposite-polarity channels are never summed."}
        report = {
            "schema_version": 1, "stage": "H3c_bounded_long_native_rate_WAV",
            "engine": ALGORITHM, "config": selected.to_dict(), "algorithm": description,
            "processor_applied": processor_applied, "processing_bypassed": not selected.orders,
            "band_trace": {"file_name": "long_trace.jsonl", "format": "JSON_lines_actual_events",
                           "event_count": events, "sample_rate_hz": 48000,
                           "sample_index_domain": "48000_Hz_engine_domain_not_native_44100_frames",
                           "full_trace_retained_in_memory": False},
            "conversion": conversions, "device_io": False, "output_gain_normalization_applied": False,
            "clocks": {"native_sample_rate_hz": rate, "native_frames": frames,
                       "engine_sample_rate_hz": 48000,
                       "engine_frames": engine_frames if engine is not None else 0,
                       "trace_sample_index_domain": "48000_Hz_engine_frames",
                       "native_output_duration_seconds": frames / rate,
                       "engine_bypassed": engine is None},
            "posthoc_alignment_applied": False, "reference_quality_metrics": None,
            "quality_notice": "Long-file compatibility is not proof of improved sound quality. FIR boundary/high-frequency differences and attenuation of wanted same-frequency content remain.",
            "input": {"file_name": source.name, "sha256": original_sha, **metadata, **native_levels},
            "output": {"file_name": "processed.wav", "sha256": _sha256_bounded(stage / "processed.wav"),
                       **output_metadata, **output_stats.result()},
            "floating_output": floating_stats.result(), "stereo_policy": stereo_policy,
            "pcm16_quantization_max_abs_error": quantization_error,
            "pcm16_rounding": "nearest_ties_to_even_reject_unrepresentable_endpoint_no_saturation",
            "long_file": {"maximum_duration_seconds": MAX_SECONDS, "offline_file_processing": True,
                          "first_pass": "native_hash_stats_and_bounded_conversion_engine_channel_RMS",
                          "second_pass": "continuous_forward_FIR_engine_reverse_FIR_and_native_PCM_output",
                          "final_source_identity_hash_checked": True, "native_frame_count_preserved": True,
                          "sample_rate_conversion_applied": bool(selected.orders), "metadata_chunks_copied": False,
                          "staged_result_commit": True, "temporary_float_audio_written": False,
                          "atomic_no_replace_commit_verified_platform": "Windows",
                          "commit_platform_notice": "Windows rename refuses existing targets atomically. Other platforms are not release-validated; a concurrent empty-target creation race is not guaranteed safe."},
            "resource_limits": {"max_container_bytes": MAX_FILE_BYTES, "max_duration_seconds": MAX_SECONDS,
                                "max_read_frames": MAX_READ_FRAMES, "max_read_bytes": MAX_READ_BYTES,
                                "max_container_chunks": MAX_CONTAINER_CHUNKS,
                                "configured_engine_block_frames": selected.block_size_frames,
                                "max_engine_block_frames_observed": max_engine_block,
                                "max_resampler_output_frames_observed": max_resampler_output,
                                "observed_max_read_bytes": maximum_read_bytes, "waveform_bin_limit": MAX_WAVEFORM_BINS,
                                "whole_audio_retained_in_memory": False, "whole_trace_retained_in_memory": False,
                                "memory_bound_is_not_process_RSS_claim": True},
            "waveforms": {"input": native_stats.waveform(), "output": output_stats.waveform()},
            "timing": {"file_processing_seconds": perf_counter() - started, "duration_seconds": frames / rate,
                       "includes_device_driver_or_playback": False, "real_time_device_performance_verified": False},
            "limitations": [
                "Explicit long-file mode only: 44100/48000 Hz mono/stereo uncompressed PCM16 WAV, at most 600 seconds and 128 MiB.",
                "MP3/AAC, floating-point and 24-bit WAV are not accepted; no automatic decode.",
                "Full-file engine-domain RMS reference selection requires a preliminary disk pass; centered FIR needs lookahead. This is offline, not device real time.",
                "Nonzero 44100 Hz processing includes fixed FIR conversion to 48000 Hz and back; boundary and high-frequency changes remain.",
                "Only canonical layouts are accepted; non-audio metadata is dropped. Atomic no-replace result commit is verified on Windows only.",
                "All channels share one original analysis channel; quieter-channel-only hum can be missed. Wanted audio at hum frequencies can be attenuated.",
                "All-zero controls bypass the FIR and engine, preserving native PCM16 codes, not the original non-audio container metadata.",
                "No normalization, clipping, limiter, playback, microphone capture, spatial cancellation, cloud or network use.",
            ],
        }
        serialized = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
        (stage / "report.json").write_text(serialized + "\n", encoding="utf-8")
        if _sha256_bounded(source) != original_sha:
            raise ValueError("input changed while processing")
        if target.exists() or target.is_symlink():
            raise FileExistsError("output directory appeared during processing; never replacing it")
        stage.rename(target)
    return report
