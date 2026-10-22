"""Opt-in, offline persistent-family suppression for short native WAV files.

This is deliberately separate from :func:`hum_dsp.sdk.process_file`: the
existing processor and its defaults are unchanged. Container IO is blockwise,
but analysis/rendering retain the complete (at most 60 second) recording and FFT
workspace in memory. This is not a bounded-stream or device-audio engine.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import stat
import tempfile
from time import perf_counter

import numpy as np

from .long_processing import _Statistics
from .persistent_comb import CombConfig, render_persistent_comb
from .precision_processing import _sha256_bounded
from .precision_wav import (
    MAX_FILE_BYTES, MAX_READ_BYTES, MAX_READ_FRAMES, PrecisionWavReader,
    PrecisionWavWriter, encode_precision_block, inspect_precision_wav,
)
from .sdk import HumDSPError, ProcessResult, _local_path, _path_value


__all__ = ["process_persistent_file"]
MAX_SECONDS = 60
_PRESERVED_FIELDS = (
    "frames", "channels", "sample_rate_hz", "sample_format",
    "bits_per_sample", "sample_width_bytes", "channel_order",
)


def _commit(stage: Path, target: Path) -> None:
    # Windows rename is atomic and refuses even an empty existing directory.
    # Other platforms have not been release-validated for a concurrent target
    # creation race; do not advertise this as a portable no-replace primitive.
    if target.exists() or target.is_symlink():
        raise FileExistsError("result directory appeared during processing")
    stage.rename(target)


def _run(source: Path, target: Path, metadata: dict, config: CombConfig) -> dict:
    started = perf_counter()
    original_sha = _sha256_bounded(source)
    frames, channels = metadata["frames"], metadata["channels"]
    samples = np.empty((frames, channels), dtype=np.float64)
    input_stats = _Statistics(frames, channels)
    position = 0
    with PrecisionWavReader(source, metadata) as reader:
        for _raw, values in reader:
            end = position + len(values)
            samples[position:end] = values
            input_stats.add(values)
            position = end
        if position != frames or reader.sha256 != original_sha:
            raise ValueError("input changed while loading samples")
        original_payload_sha = reader.payload_sha256
        maximum_read_bytes = reader.max_read_bytes

    # Full validated input has been read before any bypass decision or output.
    rendered, descriptor, trace = render_persistent_comb(samples, config)
    if (not isinstance(rendered, np.ndarray) or rendered.shape != samples.shape
            or rendered.dtype.kind not in "fiu" or not np.isfinite(rendered).all()
            or np.any(rendered < -1) or np.any(rendered >= 1)):
        raise ValueError("persistent renderer changed shape or normalized range")
    bypass = config.max_attenuation_db == 0 or not descriptor["accepted_orders"]
    if bypass and not np.array_equal(rendered, samples):
        raise ValueError("persistent bypass changed validated samples")

    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".hum_persistent_", dir=target.parent) as temporary:
        stage = Path(temporary) / "result"
        stage.mkdir()
        output_stats = _Statistics(frames, channels)
        max_quantization_error = 0.0
        with PrecisionWavWriter(stage / "processed.wav", metadata) as writer:
            if bypass:
                # Do not re-encode negative Float32 zero or native PCM samples.
                with PrecisionWavReader(source, metadata) as second:
                    for raw, values in second:
                        writer.write_payload(raw)
                        output_stats.add(values)
                    if (second.sha256 != original_sha
                            or second.payload_sha256 != original_payload_sha):
                        raise ValueError("input changed during raw-payload bypass")
                    maximum_read_bytes = max(maximum_read_bytes, second.max_read_bytes)
            else:
                for offset in range(0, frames, MAX_READ_FRAMES):
                    values = rendered[offset:offset + MAX_READ_FRAMES]
                    payload, quantized = encode_precision_block(values, metadata["sample_format"])
                    writer.write_payload(payload)
                    output_stats.add(quantized)
                    max_quantization_error = max(
                        max_quantization_error, float(np.max(np.abs(values - quantized)))
                    )
        output_metadata = inspect_precision_wav(stage / "processed.wav")
        if (writer.frames_written != frames
                or any(output_metadata[key] != metadata[key] for key in _PRESERVED_FIELDS)):
            raise ValueError("persistent processor changed native audio metadata")
        payload_identical = writer.payload_sha256 == original_payload_sha
        if bypass and not payload_identical:
            raise ValueError("persistent bypass changed raw payload")
        trace_path = stage / "persistent_trace.json"
        trace_path.write_text(json.dumps(trace, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        report = {
            "schema_version": 1,
            "stage": "opt_in_offline_persistent_family_candidate_v1",
            "config": config.to_dict(), "algorithm": descriptor,
            "processor_applied": not payload_identical,
            "processing_bypassed": bypass,
            "input": {"file_name": source.name, "sha256": original_sha,
                      **metadata, **input_stats.result()},
            "output": {"file_name": "processed.wav",
                       "sha256": _sha256_bounded(stage / "processed.wav"),
                       **output_metadata, **output_stats.result()},
            "precision_io": {
                "native_sample_format_preserved": True,
                "native_rate_channels_and_frame_count_preserved": True,
                "input_payload_sha256": original_payload_sha,
                "output_payload_sha256": writer.payload_sha256,
                "raw_payload_identity": payload_identical,
                "validated_raw_payload_copy": bypass,
                "negative_float_zero_preserved_on_bypass": bypass,
                "internal_numeric_format": "float64",
                "quantization_max_abs_error": max_quantization_error,
                "rounding": "nearest_ties_to_even_reject_unrepresentable_endpoint",
                "dither_applied": False, "non_audio_metadata_chunks_copied": False,
            },
            "mask_trace": {
                "file_name": "persistent_trace.json",
                "sha256": _sha256_bounded(trace_path),
                "meaning": "frozen_original_mixture_evidence_and_shared_spectral_mask",
                "source_container_binding": "report_input_sha256_not_an_authenticated_certificate",
                "replay_reestimates_audio": False,
            },
            "resource_limits": {
                "maximum_seconds": MAX_SECONDS, "maximum_container_bytes": MAX_FILE_BYTES,
                "max_io_block_frames": MAX_READ_FRAMES,
                "max_io_block_bytes": MAX_READ_BYTES,
                "observed_max_input_read_bytes": maximum_read_bytes,
                "whole_audio_retained_in_memory": True,
                "whole_trace_retained_in_memory": True,
                "whole_file_fft_workspace_in_memory": True,
                "bounded_stream_engine": False,
                "process_RSS_bound_claimed": False,
            },
            "output_gain_normalization_applied": False,
            "sample_rate_conversion_applied": False, "clipping_applied": False,
            "quality_approval": False, "reference_quality_metrics": None,
            "device_io": False, "network_use": False,
            "real_time_device_performance_verified": False,
            "source_identity_hash_checked_before_read_and_before_commit": True,
            "staged_result_commit": True,
            "atomic_no_replace_commit_verified_platform": "Windows",
            "timing": {"file_processing_seconds": perf_counter() - started},
            "limitations": [
                "Explicit opt-in candidate only; existing SDK, CLI and web defaults are unchanged.",
                "Only local canonical mono/stereo PCM16, packed PCM24 or normalized Float32 WAV at 44100/48000 Hz, at most 60 seconds and 128 MiB.",
                "Full-file lookahead and FFT workspace use memory; this is offline, not a bounded stream or real-time audio-device engine.",
                "Persistent harmonic-family evidence does not identify unwanted noise: wanted same-family tones can also be attenuated.",
                "Shared real spectral masks retain channel separation, not a guarantee of perceptual stereo or wanted-content preservation.",
                "FFT masking can introduce boundary changes and pre/post ringing; no improved-quality approval is inferred from completion.",
                "No resampling, gain normalization, saturation, limiter or dither; invalid or unrepresentable samples are refused.",
                "Bypass copies validated raw audio payload, not ancillary container metadata.",
                "Atomic no-replace commit is verified on Windows only; concurrent path mutation is not a filesystem sandbox guarantee.",
            ],
        }
        (stage / "report.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        # A late mutation must not publish a result for a different source.
        _local_path(source, "input_path")
        _local_path(target, "output_dir")
        if _sha256_bounded(source) != original_sha:
            raise ValueError("input changed before result commit")
        _commit(stage, target)
    return report


def process_persistent_file(
    input_path: str | os.PathLike[str], *, output_dir: str | os.PathLike[str],
    hum_hz: float, attenuation_db: float = 12.0,
) -> ProcessResult:
    """Run the new candidate explicitly into a new directory, never overwrite.

    Select a 50/60 Hz family and [0,18] dB maximum attenuation. A family needs
    at least four consistent persistent orders; unaccepted or zero-depth inputs
    use exact validated audio-payload bypass. This is not a quality certificate.
    """
    try:
        # Rate is replaced after validated header inspection; all user options
        # are rejected before paths are touched.
        provisional = CombConfig(nominal_hum_hz=hum_hz, max_attenuation_db=attenuation_db)
    except (TypeError, ValueError) as error:
        raise HumDSPError("invalid_options", "Select hum_hz 50 or 60 and a finite attenuation_db in [0, 18].") from error

    # Reject both lexical remote paths before examining either local path.
    _path_value(input_path, "input_path")
    _path_value(output_dir, "output_dir")
    target = _local_path(output_dir, "output_dir")
    source = _local_path(input_path, "input_path")
    if target.exists():
        raise HumDSPError("output_exists", "Choose a new output directory; existing results are never replaced.")
    try:
        info = source.stat()
    except FileNotFoundError as error:
        raise HumDSPError("input_not_found", "Input file was not found; provide an existing local WAV file.") from error
    except OSError as error:
        raise HumDSPError("unsupported_input", "Input file could not be read; check local permissions.") from error
    if not stat.S_ISREG(info.st_mode) or source.suffix.lower() != ".wav":
        raise HumDSPError("unsupported_input", "This opt-in mode accepts regular local WAV files only.")
    try:
        metadata = inspect_precision_wav(source)
        if metadata["frames"] > metadata["sample_rate_hz"] * MAX_SECONDS:
            raise ValueError("persistent mode has a 60-second limit")
        config = CombConfig(sample_rate_hz=metadata["sample_rate_hz"],
                            nominal_hum_hz=provisional.nominal_hum_hz,
                            max_attenuation_db=provisional.max_attenuation_db)
    except ValueError as error:
        raise HumDSPError(
            "unsupported_input",
            "Use canonical mono/stereo PCM16, PCM24 or normalized Float32 WAV at 44.1/48 kHz, at most 60 seconds and 128 MiB.",
        ) from error
    except OSError as error:
        raise HumDSPError("processing_failed", "Input inspection failed; check access and concurrent changes.") from error
    try:
        report = _run(source, target, metadata, config)
        return ProcessResult(target / "processed.wav", target / "report.json", report,
                             target / "persistent_trace.json")
    except HumDSPError:
        raise
    except FileExistsError as error:
        raise HumDSPError("output_exists", "Choose a new output directory; existing results are never replaced.") from error
    except ImportError as error:
        raise HumDSPError("dependency_unavailable", "A processing dependency is unavailable; check this installation.") from error
    except (OSError, RuntimeError, TypeError, ValueError, KeyError, MemoryError) as error:
        raise HumDSPError(
            "processing_failed",
            "Processing was refused or could not complete. Check normalized sample ranges, local resources and concurrent file changes.",
        ) from error
