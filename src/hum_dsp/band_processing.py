"""Offline per-band controls. No playback, normalization or blind quality claim."""

import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .audio import SAMPLE_RATE_HZ, read_pcm16_wav, validate_audio, write_pcm16_wav
from .band_control import BandControlConfig, render_band_control
from .fixtures import _dump_json
from .processing import _input_sha256, _levels


def process_band_wav(input_path, output_directory, config=None):
    settings = BandControlConfig() if config is None else config
    if type(settings) is not BandControlConfig:
        raise ValueError("config must be a BandControlConfig")
    settings = BandControlConfig(
        sample_rate_hz=settings.sample_rate_hz,
        nominal_hum_hz=settings.nominal_hum_hz,
        band_depths_db=settings.band_depths_db,
        block_size_frames=settings.block_size_frames,
    )
    source, target = Path(input_path), Path(output_directory)
    if target.exists():
        raise FileExistsError("output directory must be new; previous files are never replaced")
    before = _input_sha256(source)
    samples = read_pcm16_wav(source)
    if _input_sha256(source) != before:
        raise ValueError("input changed while being read")
    started = perf_counter()
    floating, algorithm, trace = render_band_control(samples, settings)
    elapsed = perf_counter() - started
    output = validate_audio(floating)
    if output.shape != samples.shape:
        raise ValueError("per-band processing changed the sample count")
    if _input_sha256(source) != before:
        raise ValueError("input changed while processing")
    # Reject malformed engine metadata before creating any successful output.
    json.dumps((algorithm, trace), allow_nan=False)
    target.mkdir(parents=True, exist_ok=False)
    destination = target / "processed.wav"
    write_pcm16_wav(destination, output)
    decoded = read_pcm16_wav(destination)
    _dump_json(target / "bands.json", trace)
    report = {
        "schema_version": 1,
        "stage": "H2_manual_per_band_control",
        "engine": "offline_original_input_decisions_per_band_attenuation_replay",
        "config": settings.to_dict(),
        "algorithm": algorithm,
        "processor_applied": bool(np.any(output != samples)),
        "band_trace": {"file_name": "bands.json"},
        "device_io": False,
        "output_gain_normalization_applied": False,
        "posthoc_alignment_applied": False,
        "reference_quality_metrics": None,
        "quality_notice": "Manual controls only. Smaller attenuation can reduce wanted-signal change but leaves more hum. No reference-free sound quality or noise-source classification is provided.",
        "input": {"file_name": source.name, "sha256": before, "sample_rate_hz": SAMPLE_RATE_HZ, "frames": int(samples.size), **_levels(samples)},
        "output": {"file_name": "processed.wav", "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(), "sample_rate_hz": SAMPLE_RATE_HZ, "frames": int(decoded.size), **_levels(decoded)},
        "floating_output": _levels(output),
        "pcm16_quantization_max_abs_error": float(np.max(np.abs(decoded - output))),
        "timing": {"file_processing_seconds": elapsed, "duration_seconds": samples.size / SAMPLE_RATE_HZ, "real_time_device_performance_verified": False},
        "limitations": [
            "Offline two-pass analysis and replay; no streaming input/output implementation.",
            "Same-frequency wanted content and hum are not distinguishable by this method.",
            "Zero on a band removes that filter stage, not effects from other enabled nearby stages.",
            "Start at zero, add only needed attenuation, keep the source, and compare manually at low playback volume.",
            "A held-out real desired-audio clip with injected mathematical hum is not a natural-noise or perceptual validation.",
            "48 kHz mono PCM16 WAV only, at most 60 seconds; no automatic conversion, clipping or normalization.",
        ],
    }
    _dump_json(target / "report.json", report)
    return report
