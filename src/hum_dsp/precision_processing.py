"""Bounded native-precision WAV processing, without a PCM16 intermediate.

PCM16 keeps the frozen H3c adapter. Packed PCM24 and normalized IEEE float32
use the same continuous H3b filter and H3c FIR, retaining their original sample
format, channel count, native rate and frame count. This is offline file work,
not an audio device, quality guarantee or wide-headroom float mastering tool.
"""

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
from time import perf_counter

import numpy as np

from .band_control import BandControlConfig
from .long_processing import _Statistics
from .precision_wav import (
    PrecisionWavReader, PrecisionWavWriter, encode_precision_block,
    inspect_precision_wav,
)


ALGORITHM = "bounded_native_precision_linked_band_control_v1"
MAX_SECONDS = 600
MAX_FILE_BYTES = 128 * 1024 * 1024
MAX_READ_FRAMES = 4096
MAX_READ_BYTES = MAX_READ_FRAMES * 2 * 4
MAX_WAVEFORM_BINS = 480


def _sha256_bounded(path):
    """Hash a bounded regular container, detecting growth and truncation."""
    source = Path(path)
    info = source.stat()
    if not stat.S_ISREG(info.st_mode):
        raise ValueError("a regular WAV file is required")
    size = info.st_size
    if size > MAX_FILE_BYTES:
        raise ValueError("WAV file exceeds the 128 MiB container limit")
    digest, total = hashlib.sha256(), 0
    with source.open("rb") as handle:
        while True:
            block = handle.read(min(MAX_READ_BYTES, MAX_FILE_BYTES + 1 - total))
            if not block:
                break
            total += len(block)
            if total > MAX_FILE_BYTES:
                raise ValueError("WAV file exceeds the 128 MiB container limit")
            digest.update(block)
        if total != size or os.fstat(handle.fileno()).st_size != size:
            raise ValueError("input changed while hashing")
    return digest.hexdigest()


def _parts(values, size=MAX_READ_FRAMES):
    for position in range(0, len(values), size):
        yield values[position:position + size]


def _normalized_block(values, channels):
    array = np.asarray(values)
    if (array.dtype.kind not in "fiu" or array.ndim != 2
            or array.shape[1] != channels or not 1 <= len(array) <= MAX_READ_FRAMES
            or not np.isfinite(array).all() or np.any(array < -1) or np.any(array >= 1)):
        raise ValueError("conversion or engine produced invalid normalized audio; no clipping or normalization")
    return array


def _no_conversion(rate, frames, bypass):
    return {"conversion_applied": False,
            "reason": "all_band_depths_zero_native_payload_identity" if bypass else "native_48000_Hz_engine_rate",
            "source_rate_hz": rate, "target_rate_hz": rate,
            "source_frames": frames, "target_frames": frames}


def process_precision_wav(input_path, output_directory, config=None):
    """Preserve PCM16/PCM24/float32 native precision with all-or-nothing output.

    Zero-depth mode validates normalized samples but copies the raw audio
    payload, including negative floating zero, exactly. Nonzero float input is
    rejected if any input, intermediate FIR/filter output, or rounded float32
    output lies outside [-1, 1); no limiter, normalization, clipping or dither.
    """
    selected = BandControlConfig(block_size_frames=MAX_READ_FRAMES) if config is None else config
    if type(selected) is not BandControlConfig:
        raise ValueError("config must be a validated BandControlConfig")
    selected = BandControlConfig(**asdict(selected))
    source, target = Path(input_path), Path(output_directory)
    if target.exists() or target.is_symlink():
        raise FileExistsError("output directory must be new; previous files are never replaced")
    metadata = inspect_precision_wav(source)
    if metadata["sample_format"] == "signed_PCM16_little_endian":
        # Delegate dynamically so compatibility/regression tests can establish
        # that no earlier PCM16 processing policy is silently replaced.
        from .long_compatible_processing import process_long_compatible_wav
        return process_long_compatible_wav(source, target, selected)
    started = perf_counter()
    frames, channels, rate = metadata["frames"], metadata["channels"], metadata["sample_rate_hz"]
    sample_format = metadata["sample_format"]
    active, convert = bool(selected.orders), bool(selected.orders) and rate != 48000
    engine_frames = (frames * 160 + 146) // 147 if rate == 44100 else frames
    native_stats = _Statistics(frames, channels)
    analysis_stats = _Statistics(engine_frames, channels) if convert else None
    forward_analysis = None
    if convert:
        from .stream_resampling import StreamingResampler
        forward_analysis = StreamingResampler(rate, 48000, frames, channels)

    def analyze(values):
        for part in _parts(values):
            analysis_stats.add(_normalized_block(part, channels))

    with PrecisionWavReader(source, metadata) as first:
        for _payload, values in first:
            native_stats.add(_normalized_block(values, channels))
            if forward_analysis is not None:
                analyze(forward_analysis.process_block(values))
        if forward_analysis is not None:
            analyze(forward_analysis.finish())
        original_sha, original_payload_sha = first.sha256, first.payload_sha256
        maximum_read_bytes = first.max_read_bytes
    native_levels = native_stats.result()
    engine_levels = (analysis_stats.result() if convert else native_levels) if active else None
    channel_rms = [] if engine_levels is None else [item["rms"] for item in engine_levels["channel_levels"]]
    reference = int(np.argmax(channel_rms)) if active else None
    floating_stats, output_stats = _Statistics(frames, channels), _Statistics(frames, channels)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".hum_precision_", dir=target.parent) as temporary:
        stage = Path(temporary) / "result"
        stage.mkdir()
        events, quantization_error = 0, 0.0
        max_engine_block, max_resampler_output = 0, 0
        with (stage / "long_trace.jsonl").open("x", encoding="utf-8", newline="\n") as trace_file:
            def sink(event):
                nonlocal events
                if type(event) is not dict:
                    raise ValueError("trace event must be a dictionary")
                trace_file.write(json.dumps(event, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n")
                events += 1

            engine, forward, backward = None, None, None
            if active:
                from .stream_bands import StreamBandControl
                engine = StreamBandControl(selected, channels=channels, reference_channel_index=reference, trace_sink=sink)
            if convert:
                from .stream_resampling import StreamingResampler
                forward = StreamingResampler(rate, 48000, frames, channels)
                backward = StreamingResampler(48000, rate, engine_frames, channels, target_frames=frames)
            with PrecisionWavReader(source, metadata) as second:
                with PrecisionWavWriter(stage / "processed.wav", metadata) as destination:
                    def write_output(values, raw_payload=None):
                        nonlocal quantization_error, max_resampler_output
                        if raw_payload is not None:
                            # Bypass never decodes/re-encodes audio bytes.
                            checked = _normalized_block(values, channels)
                            floating_stats.add(checked)
                            output_stats.add(checked)
                            destination.write_payload(raw_payload)
                            return
                        if convert:
                            max_resampler_output = max(max_resampler_output, len(values))
                        for part in _parts(values):
                            floating = _normalized_block(part, channels)
                            payload, quantized = encode_precision_block(floating, sample_format)
                            floating_stats.add(floating)
                            output_stats.add(quantized)
                            quantization_error = max(quantization_error, float(np.max(np.abs(quantized - floating))))
                            destination.write_payload(payload)

                    def render(values):
                        nonlocal max_engine_block, max_resampler_output
                        if convert:
                            max_resampler_output = max(max_resampler_output, len(values))
                        for part in _parts(values, selected.block_size_frames):
                            checked = _normalized_block(part, channels)
                            max_engine_block = max(max_engine_block, len(part))
                            processed = _normalized_block(engine.process_block(checked), channels)
                            if processed.shape != checked.shape:
                                raise ValueError("processor changed block shape")
                            write_output(backward.process_block(processed) if convert else processed)

                    for original_payload, values in second:
                        if not active:
                            write_output(values, original_payload)
                        elif convert:
                            render(forward.process_block(values))
                        else:
                            render(values)
                    if convert:
                        render(forward.finish())
                        write_output(backward.finish())
                if second.sha256 != original_sha or second.payload_sha256 != original_payload_sha:
                    raise ValueError("input changed between processing passes")
                maximum_read_bytes = max(maximum_read_bytes, second.max_read_bytes)
            if destination.frames_written != frames:
                raise ValueError("writer changed native frame count")
            output_payload_sha = destination.payload_sha256
        output_metadata = inspect_precision_wav(stage / "processed.wav")
        for name in ("frames", "channels", "sample_rate_hz", "sample_format", "bits_per_sample", "sample_width_bytes"):
            if output_metadata[name] != metadata[name]:
                raise ValueError("processor changed native frame count, rate, channels or sample format")
        description = engine.describe() if active else {
            "algorithm": "identity_native_precision_zero_depth_bypass", "samples_processed": frames,
            "processing_bypassed": True, "shared_original_input_fft_count": 0,
            "trace_event_count": 0, "native_payload_identity": True}
        if description.get("samples_processed") != (engine_frames if active else frames):
            raise ValueError("processor clock does not match complete engine frame count")
        if description.get("trace_event_count") != events:
            raise ValueError("processor trace summary does not match written trace events")
        payload_identical = original_payload_sha == output_payload_sha
        if not active and not payload_identical:
            raise ValueError("zero-depth processing changed the native audio payload")
        conversions = {"to_engine": _no_conversion(rate, frames, not active) if forward is None else forward.describe(),
                       "to_native": _no_conversion(rate, frames, not active) if backward is None else backward.describe(),
                       "engine_sample_rate_hz": 48000, "native_sample_rate_hz": rate,
                       "native_frame_count_preserved": True}
        stereo_policy = {"mode": "linked_same_decisions_and_coefficients_no_downmix_no_crossfeed",
                         "reference_channel_index": reference,
                         "reference_selection": "first_maximum_full_file_engine_domain_raw_RMS",
                         "reference_selection_uses_entire_file": active,
                         "engine_channel_rms": channel_rms,
                         "same_trace_and_coefficients_all_channels": True,
                         "independent_channel_frequency_tracking": False,
                         "perceptual_stereo_preservation_verified": False,
                         "limitation": "A quieter other channel's weak hum may be missed; opposite-polarity channels are never summed."}
        report = {
            "schema_version": 1, "stage": "H3d_bounded_native_precision_WAV",
            "engine": ALGORITHM, "config": selected.to_dict(), "algorithm": description,
            "processor_applied": not payload_identical, "processing_bypassed": not active,
            "precision_io": {"input_sample_format": sample_format, "output_sample_format": sample_format,
                             "native_sample_format_preserved": True, "PCM16_intermediate_used": False,
                             "internal_numeric_format": "float64", "dither_applied": False,
                             "normalized_range_policy": "finite_samples_in_[-1,1)_including_float_input_and_intermediates",
                             "original_payload_sha256": original_payload_sha, "output_payload_sha256": output_payload_sha,
                             "raw_payload_identity": payload_identical,
                             "zero_depth_raw_payload_copy": not active, "negative_float_zero_preserved_on_bypass": not active,
                             "rounding": "nearest_ties_to_even_then_reject_unrepresentable_endpoint_no_saturation",
                             "quantization_max_abs_error": quantization_error},
            "band_trace": {"file_name": "long_trace.jsonl", "format": "JSON_lines_actual_events",
                           "event_count": events, "sample_rate_hz": 48000,
                           "sample_index_domain": "48000_Hz_engine_frames_not_native_44100_frames",
                           "full_trace_retained_in_memory": False},
            "conversion": conversions, "device_io": False, "output_gain_normalization_applied": False,
            "clocks": {"native_sample_rate_hz": rate, "native_frames": frames,
                       "engine_sample_rate_hz": 48000, "engine_frames": engine_frames if active else 0,
                       "trace_sample_index_domain": "48000_Hz_engine_frames",
                       "native_output_duration_seconds": frames / rate, "engine_bypassed": not active},
            "posthoc_alignment_applied": False, "reference_quality_metrics": None,
            "quality_notice": "Native format preservation is not proof of improved sound quality. FIR boundary/high-frequency differences and attenuation of wanted same-frequency content remain.",
            "input": {"file_name": source.name, "sha256": original_sha, **metadata, **native_levels},
            "output": {"file_name": "processed.wav", "sha256": _sha256_bounded(stage / "processed.wav"),
                       **output_metadata, **output_stats.result()},
            "floating_output": floating_stats.result(), "stereo_policy": stereo_policy,
            "native_quantization_max_abs_error": quantization_error,
            "long_file": {"maximum_duration_seconds": MAX_SECONDS, "offline_file_processing": True,
                          "first_pass": "native_hash_stats_and_optional_bounded_conversion_engine_channel_RMS",
                          "second_pass": "continuous_optional_FIR_engine_and_same_precision_native_output",
                          "final_source_identity_hash_checked": True, "native_frame_count_preserved": True,
                          "sample_rate_conversion_applied": convert, "metadata_chunks_copied": False,
                          "staged_result_commit": True, "temporary_float_audio_written": False,
                          "atomic_no_replace_commit_verified_platform": "Windows",
                          "commit_platform_notice": "Windows rename refuses existing targets atomically. Other platforms are not release-validated; a concurrent empty-target creation race is not guaranteed safe."},
            "resource_limits": {"max_container_bytes": MAX_FILE_BYTES, "max_duration_seconds": MAX_SECONDS,
                                "max_read_frames": MAX_READ_FRAMES, "max_read_bytes": MAX_READ_BYTES,
                                "configured_engine_block_frames": selected.block_size_frames,
                                "max_engine_block_frames_observed": max_engine_block,
                                "max_resampler_output_frames_observed": max_resampler_output,
                                "observed_max_read_bytes": maximum_read_bytes, "waveform_bin_limit": MAX_WAVEFORM_BINS,
                                "whole_audio_retained_in_memory": False, "whole_trace_retained_in_memory": False,
                                "temporary_float_audio_written": False, "memory_bound_is_not_process_RSS_claim": True},
            "waveforms": {"input": native_stats.waveform(), "output": output_stats.waveform()},
            "timing": {"file_processing_seconds": perf_counter() - started, "duration_seconds": frames / rate,
                       "includes_device_driver_or_playback": False, "real_time_device_performance_verified": False},
            "limitations": [
                "Explicit precision mode only: 44100/48000 Hz canonical mono/stereo PCM16, packed PCM24 or IEEE float32 WAV, at most 600 seconds and 128 MiB; the byte cap may bind first.",
                "Float samples, FIR intermediates and output must be finite in [-1,1); float headroom and +1.0 are not accepted, even with zero depths.",
                "MP3/AAC, PCM32, float64 and arbitrary layouts are not decoded or automatically converted.",
                "Full-file RMS reference selection and centered FIR need preliminary passes/lookahead: this is offline, not device real time.",
                "Nonzero 44100 Hz processing converts to 48000 Hz and back; file-boundary and high-frequency changes remain.",
                "All channels share one original analysis channel; quieter-channel hum can be missed and wanted tones can be attenuated.",
                "Zero depths preserve raw audio payload, not non-audio container metadata. No dither, normalization, saturation or limiter is applied.",
                "Atomic no-replace result commit is verified on Windows only; no automatic quality approval, playback, microphone capture or network use.",
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
