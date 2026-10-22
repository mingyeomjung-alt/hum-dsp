"""Explicit opt-in offline tracked WAV processing. No audio device access."""

import hashlib
from pathlib import Path
from time import perf_counter

import numpy as np

from .audio import SAMPLE_RATE_HZ, read_pcm16_wav, validate_audio, write_pcm16_wav
from .config import ProcessorConfig
from .fixtures import _dump_json
from .processing import _input_sha256, _levels
from .tracking import TrackedHumSuppressor


def render_tracked(samples, config: ProcessorConfig) -> tuple[np.ndarray, dict, list]:
    """Render causal blocks and return actual final state metadata and trace."""
    source = validate_audio(samples)
    selected = ProcessorConfig(**config.to_dict())
    engine = TrackedHumSuppressor(selected)
    output = np.empty_like(source)
    for start in range(0, source.size, selected.block_size_frames):
        stop = min(start + selected.block_size_frames, source.size)
        block = np.asarray(engine.process_block(source[start:stop]))
        if block.ndim != 1 or block.shape != source[start:stop].shape:
            raise ValueError("tracked processor changed a block's shape or sample count")
        # Validate each block before assignment, including dtype/NaN/overshoot.
        output[start:stop] = validate_audio(block)
    return output, engine.describe(), engine.get_trace()


def process_tracked_wav(input_path: str | Path, output_directory: str | Path, config: ProcessorConfig | None = None) -> dict:
    selected = ProcessorConfig() if config is None else config
    if type(selected) is not ProcessorConfig:
        raise ValueError("config must be a ProcessorConfig")
    selected = ProcessorConfig(**selected.to_dict())
    source = Path(input_path)
    target = Path(output_directory)
    if target.exists():
        raise FileExistsError("output directory must be new; existing files are never replaced")
    input_sha = _input_sha256(source)
    samples = read_pcm16_wav(source)
    if _input_sha256(source) != input_sha:
        raise ValueError("input changed while it was being read")
    started = perf_counter()
    floating, algorithm, trace = render_tracked(samples, selected)
    elapsed = perf_counter() - started
    if _input_sha256(source) != input_sha:
        raise ValueError("input changed while processing")
    target.mkdir(parents=True, exist_ok=False)
    destination = target / "processed.wav"
    write_pcm16_wav(destination, floating)
    saved = read_pcm16_wav(destination)
    tracking_record = {"schema_version": 1, "sample_rate_hz": SAMPLE_RATE_HZ, "events": trace}
    _dump_json(target / "tracking.json", tracking_record)
    first = next((int(event["sample_index"]) for event in trace if event["estimated_frequency_hz"] is not None), None)
    report = {
        "schema_version": 1,
        "stage": "M3_frequency_tracking",
        "engine": "tracked_finite_depth_notch",
        "config": selected.to_dict(),
        "algorithm": algorithm,
        "tracking": {"file_name": "tracking.json", "event_count": len(trace), "first_acquired_sample": first},
        "processor_applied": selected.max_attenuation_db > 0,
        "device_io": False,
        "output_gain_normalization_applied": False,
        "posthoc_alignment_applied": False,
        "reference_quality_metrics": None,
        "quality_notice": "No clean reference supplied. Tracking confidence is not a probability or proof that the selected component is unwanted noise.",
        "input": {"file_name": source.name, "sha256": input_sha, "frames": int(samples.size), "sample_rate_hz": SAMPLE_RATE_HZ, **_levels(samples)},
        "output": {"file_name": "processed.wav", "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(), "frames": int(saved.size), "sample_rate_hz": SAMPLE_RATE_HZ, **_levels(saved)},
        "floating_output": _levels(floating),
        "pcm16_quantization_max_abs_error": float(np.max(np.abs(saved - floating))),
        "pcm16_rounding": "nearest_ties_to_even_with_positive_endpoint_saturation",
        "timing": {"file_processing_seconds": elapsed, "duration_seconds": samples.size / SAMPLE_RATE_HZ, "real_time_device_performance_verified": False},
        "limitations": [
            "Explicit selected 50/60 Hz band, +/-1 Hz maximum; not arbitrary broadband suppression.",
            "First estimate needs 2 seconds; this is estimator acquisition, not a delayed audio buffer.",
            "Same-frequency desired tones can be acquired and attenuated too.",
            "Only center tracking is implemented; confidence failure holds center, not an M4 safety gate.",
            "No microphone, playback, spatial cancellation, network or cloud use.",
        ],
    }
    _dump_json(target / "report.json", report)
    return report

