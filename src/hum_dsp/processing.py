"""Offline fixed-notch WAV processing. Never opens an audio device."""

import hashlib
from pathlib import Path
from time import perf_counter

import numpy as np

from .audio import MAX_FILE_BYTES, SAMPLE_RATE_HZ, read_pcm16_wav, validate_audio, write_pcm16_wav
from .config import ProcessorConfig
from .filter import FixedHumSuppressor, process_audio
from .fixtures import _dump_json


def _levels(samples: np.ndarray) -> dict:
    return {"peak_abs": float(np.max(np.abs(samples))), "rms": float(np.sqrt(np.mean(samples**2)))}


def _input_sha256(path: Path) -> str:
    """Hash with a hard read bound, including if a file changes after stat()."""
    if path.stat().st_size > MAX_FILE_BYTES:
        raise ValueError("WAV file exceeds the 32 MiB container limit")
    digest = hashlib.sha256()
    total = 0
    with path.open("rb") as source:
        while chunk := source.read(65536):
            total += len(chunk)
            if total > MAX_FILE_BYTES:
                raise ValueError("WAV file exceeds the 32 MiB container limit")
            digest.update(chunk)
    return digest.hexdigest()


def process_wav(input_path: str | Path, output_directory: str | Path, config: ProcessorConfig | None = None) -> dict:
    """Write a new processed.wav/report.json pair, without replacing input.

    The target directory must not exist. Unsupported inputs and out-of-range
    floating outputs are rejected before the directory is created. No overall
    gain change, clipping, automatic frequency selection or quality approval.
    """
    selected = ProcessorConfig() if config is None else config
    if not isinstance(selected, ProcessorConfig):
        raise ValueError("config must be a ProcessorConfig")
    selected = ProcessorConfig(**selected.to_dict())
    source = Path(input_path)
    target = Path(output_directory)
    if target.exists():
        raise FileExistsError("output directory must be new; existing files are never replaced")
    original_sha = _input_sha256(source)
    # The bounded reader checks container size and signal contract independently.
    samples = read_pcm16_wav(source)
    if _input_sha256(source) != original_sha:
        raise ValueError("input changed while it was being read")
    started = perf_counter()
    floating = process_audio(samples, selected)
    elapsed_seconds = perf_counter() - started
    output = validate_audio(floating)  # Reject overshoot; do not normalize it away.
    if output.shape != samples.shape:
        raise ValueError("processor changed the number of samples")
    if _input_sha256(source) != original_sha:
        raise ValueError("input changed while processing")
    target.mkdir(parents=True, exist_ok=False)
    destination = target / "processed.wav"
    write_pcm16_wav(destination, output)
    decoded = read_pcm16_wav(destination)
    output_bytes = destination.read_bytes()
    description = FixedHumSuppressor(selected).describe()
    # This separate metadata instance does not represent processing state.
    description["samples_processed"] = int(samples.size)
    record = {
        "schema_version": 1,
        "stage": "M2_fixed_filter",
        "engine": "fixed_finite_depth_notch",
        "config": selected.to_dict(),
        "algorithm": description,
        "processor_applied": selected.max_attenuation_db > 0,
        "device_io": False,
        "output_gain_normalization_applied": False,
        "posthoc_alignment_applied": False,
        "reference_quality_metrics": None,
        "quality_notice": "No clean reference is supplied. File statistics are not proof of improved perceptual quality or desired-content preservation.",
        "input": {
            "file_name": source.name,
            "sha256": original_sha,
            "sample_rate_hz": SAMPLE_RATE_HZ,
            "frames": int(samples.size),
            **_levels(samples),
        },
        "output": {
            "file_name": "processed.wav",
            "sha256": hashlib.sha256(output_bytes).hexdigest(),
            "sample_rate_hz": SAMPLE_RATE_HZ,
            "frames": int(decoded.size),
            **_levels(decoded),
        },
        "floating_output": _levels(output),
        "pcm16_quantization_max_abs_error": float(np.max(np.abs(decoded - output))),
        "pcm16_rounding": "nearest_ties_to_even_with_positive_endpoint_saturation",
        "timing": {
            "file_processing_seconds": elapsed_seconds,
            "duration_seconds": samples.size / SAMPLE_RATE_HZ,
            "includes_device_driver_or_playback": False,
            "real_time_device_performance_verified": False,
        },
        "limitations": [
            "Manual 50/60 Hz target; no tracking or harmonic suppression yet.",
            "Desired content at the selected frequency is attenuated too.",
            "Zero added sample buffer does not mean zero phase or group delay.",
            "No playback, microphone capture, spatial cancellation, cloud or network use.",
        ],
    }
    _dump_json(target / "report.json", record)
    return record

