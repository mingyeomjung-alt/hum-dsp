"""Explicit harmonic-bank WAV processing; never captures or plays audio."""

import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .audio import SAMPLE_RATE_HZ, read_pcm16_wav, validate_audio, write_pcm16_wav
from .fixtures import _dump_json
from .harmonics import HarmonicConfig, render_harmonic
from .processing import _input_sha256, _levels


def process_harmonic_wav(input_path, output_directory, config=None):
    settings = HarmonicConfig() if config is None else config
    if type(settings) is not HarmonicConfig:
        raise ValueError("config must be a HarmonicConfig")
    settings = HarmonicConfig(**settings.to_dict())
    source, target = Path(input_path), Path(output_directory)
    if target.exists():
        raise FileExistsError("output directory must be new; input and previous results are never replaced")
    before = _input_sha256(source)
    samples = read_pcm16_wav(source)
    if _input_sha256(source) != before:
        raise ValueError("input changed while being read")
    start = perf_counter()
    floating, algorithm, trace = render_harmonic(samples, settings)
    elapsed = perf_counter() - start
    output = validate_audio(floating)
    if output.shape != samples.shape:
        raise ValueError("harmonic processor changed the sample count")
    if _input_sha256(source) != before:
        raise ValueError("input changed while processing")
    json.dumps((algorithm, trace), allow_nan=False)
    target.mkdir(parents=True, exist_ok=False)
    destination = target / "processed.wav"
    write_pcm16_wav(destination, output)
    decoded = read_pcm16_wav(destination)
    _dump_json(target / "harmonics.json", {"schema_version": 1, "sample_rate_hz": SAMPLE_RATE_HZ, "events": trace})
    report = {
        "schema_version": 1,
        "stage": "H1_selective_harmonic_bank",
        "engine": "selective_evidence_gated_finite_depth_harmonic_bank",
        "config": settings.to_dict(),
        "algorithm": algorithm,
        "processor_applied": bool(np.any(output != samples)),
        "harmonic_trace": {"file_name": "harmonics.json", "event_count": len(trace)},
        "device_io": False,
        "output_gain_normalization_applied": False,
        "posthoc_alignment_applied": False,
        "reference_quality_metrics": None,
        "quality_notice": "No clean reference is supplied. Per-band spectral evidence is not semantic noise classification or proof of improved sound quality.",
        "input": {"file_name": source.name, "sha256": before, "sample_rate_hz": SAMPLE_RATE_HZ, "frames": int(samples.size), **_levels(samples)},
        "output": {"file_name": "processed.wav", "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(), "sample_rate_hz": SAMPLE_RATE_HZ, "frames": int(decoded.size), **_levels(decoded)},
        "floating_output": _levels(output),
        "pcm16_quantization_max_abs_error": float(np.max(np.abs(decoded - output))),
        "timing": {"file_processing_seconds": elapsed, "duration_seconds": samples.size / SAMPLE_RATE_HZ, "real_time_device_performance_verified": False},
        "limitations": [
            "Manually selected integer-multiple bands are independently observed; neither true harmonic coherence nor an electrical noise source is identified.",
            "Wanted periodic content inside selected bands is confusable and can be attenuated. Choose only bands you intend to process.",
            "Weak components below the concentration threshold may remain unprocessed.",
            "At higher orders, observation lag causes larger frequency error while pole-design bandwidth stays 1 Hz.",
            "Initial two-second observation, qualification and attack delay processing; recovery includes observation history and release.",
            "No broadband denoising, microphone path, spatial ANC, clipping or gain normalization.",
        ],
    }
    _dump_json(target / "report.json", report)
    return report
