"""Explicitly opted-in file wrapper for the unchanged experimental profile guard.

The same-file noise-only assertion belongs to the user. It is not inferred from
silence, verified by this wrapper, or treated as evidence of better sound.
"""

from decimal import Decimal, InvalidOperation, localcontext
import hashlib
import io
import json
import math
from pathlib import Path
import re
from time import perf_counter
import wave

import numpy as np

from .audio import SAMPLE_RATE_HZ, read_pcm16_wav, validate_audio, write_pcm16_wav
from .fixtures import _dump_json
from .processing import _input_sha256, _levels
from .profile_guard import ProfileGuardConfig, render_profile_guard


PROFILE_QUERY_KEYS = frozenset(("mode", "hum_hz", "attenuation_db", "orders",
                                "profile_start_seconds", "profile_end_seconds", "confirm_noise_only"))


def _time_to_sample(text):
    """Bound decimal parsing before arithmetic; never round an interval edge."""
    if (type(text) is not str or len(text) > 32
            or re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", text) is None):
        raise ValueError("profile time must be a plain decimal number of seconds")
    try:
        value = Decimal(text)
        with localcontext() as context:
            context.prec = len(text) + 8
            frames = value * SAMPLE_RATE_HZ
        if not 0 <= value <= 60 or frames != frames.to_integral_value():
            raise ValueError("profile time must be 0..60 seconds at an exact sample boundary")
        return int(frames)
    except InvalidOperation as exc:
        raise ValueError("invalid profile time") from exc


def config_from_profile_query(fields):
    """Parse the strict single-valued local UI contract; no implicit profile."""
    if (set(fields) != PROFILE_QUERY_KEYS or any(len(v) != 1 for v in fields.values())
            or fields["mode"] != ["profile"] or fields["hum_hz"][0] not in ("50", "60")
            or fields["confirm_noise_only"] != ["true"]):
        raise ValueError("all profile controls and the explicit noise-only assertion are required")
    order_text = fields["orders"][0]
    tokens = order_text.split(",")
    if order_text == "all":
        orders = tuple(range(1, 33))
    elif (len(tokens) <= 32 and all(re.fullmatch(r"(?:[1-9]|[12][0-9]|3[0-2])", v) for v in tokens)):
        orders = tuple(map(int, tokens))
    else:
        raise ValueError("select all or sorted unique orders 1..32")
    return ProfileGuardConfig(
        nominal_hum_hz=float(fields["hum_hz"][0]), orders=orders,
        max_attenuation_db=float(fields["attenuation_db"][0]),
        profile_start_sample=_time_to_sample(fields["profile_start_seconds"][0]),
        profile_end_sample=_time_to_sample(fields["profile_end_seconds"][0]),
    )


def profile_example_wav(config):
    """Owned eight-second fixture; controls never quietly alter the input noise.

Only this mathematical example has a known noise-only first two seconds. An
uploaded file receives no default interval or machine-verified assertion.
"""
    if type(config) is not ProfileGuardConfig:
        raise ValueError("config must be ProfileGuardConfig")
    time = np.arange(SAMPLE_RATE_HZ * 8, dtype=np.float64) / SAMPLE_RATE_HZ
    samples = sum(math.sqrt(2) * .012 * np.sin(2 * math.pi * order * config.nominal_hum_hz * time + .37 * order)
                  for order in (1, 6, 21, 24))
    program = time >= 2
    samples[program] += (.05 * np.sin(2 * math.pi * 233 * time[program])
                         + .05 * np.sin(2 * math.pi * 419 * time[program] + .2))
    codes = np.rint(samples * 32768).astype("<i2")
    result = io.BytesIO()
    with wave.open(result, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE_HZ)
        wav.writeframes(codes.tobytes())
    return result.getvalue()


def process_profile_wav(input_path, output_directory, config):
    """Write new WAV/report/trace only after strict same-input validation."""
    if type(config) is not ProfileGuardConfig:
        raise ValueError("config must be ProfileGuardConfig")
    settings = ProfileGuardConfig(**config.to_dict())
    if settings.profile_start_sample is None:
        raise ValueError("local profile processing requires an explicitly declared interval")
    source, target = Path(input_path), Path(output_directory)
    if target.exists():
        raise FileExistsError("output directory must be new; existing files are never replaced")
    before = _input_sha256(source)
    samples = read_pcm16_wav(source)
    if _input_sha256(source) != before:
        raise ValueError("input changed while being read")
    started = perf_counter()
    floating, description, trace = render_profile_guard(samples, settings)
    elapsed = perf_counter() - started
    output = validate_audio(floating)
    if output.shape != samples.shape:
        raise ValueError("profile processing changed the sample count")
    if _input_sha256(source) != before:
        raise ValueError("input changed while processing")
    json.dumps((description, trace), allow_nan=False)
    approved = list(trace["approved_orders"])
    bands = trace["bands"]
    summary = {
        "experimental": True,
        "noise_only_interval_user_asserted": True,
        "noise_only_interval_machine_verified": False,
        "profile_start_sample": settings.profile_start_sample,
        "profile_end_sample": settings.profile_end_sample,
        "profile_start_seconds": settings.profile_start_sample / SAMPLE_RATE_HZ,
        "profile_end_seconds": settings.profile_end_sample / SAMPLE_RATE_HZ,
        "same_input_interval": True,
        "input_wav_sha256": before,
        "offline_whole_file_future_lookahead": True,
        "real_time": False,
        "approved_orders": approved,
        "rejected_orders": [band["order"] for band in bands if band["amplitude_guard_rejected"]],
        "unsupported_orders": [band["order"] for band in bands if not band["profile_supported"]],
        "bands": [{key: band[key] for key in ("order", "frequency_hz", "profile_supported", "approved",
                   "amplitude_guard_rejected", "maximum_amplitude_ratio", "reason")} for band in bands],
        "processing_bypassed": not approved,
        "bypass_reason": trace["bypass_reason"],
        "true_noise_reduction_measured": False,
        "wanted_content_preservation_verified": False,
        "default_product_engine_changed": False,
        "commercial_release_approved": False,
        "trace_file_name": "profile_trace.json",
    }
    target.mkdir(parents=True, exist_ok=False)
    destination = target / "processed.wav"
    write_pcm16_wav(destination, output)
    decoded = read_pcm16_wav(destination)
    _dump_json(target / "profile_trace.json", trace)
    report = {
        "schema_version": 1, "stage": "H2d_opt_in_profile_guard",
        "engine": "experimental_offline_same_file_profile_guard",
        "config": settings.to_dict(), "algorithm": description,
        "processor_applied": bool(np.any(output != samples)), "profile_guard": summary,
        "device_io": False, "output_gain_normalization_applied": False,
        "posthoc_alignment_applied": False, "reference_quality_metrics": None,
        "quality_notice": "Experimental offline protection, not source separation. A wrong noise-only interval or equal-amplitude wanted overlap can still remove wanted sound. Withholding all bands leaves the input unchanged; that is not noise-reduction success.",
        "input": {"file_name": source.name, "sha256": before, "sample_rate_hz": SAMPLE_RATE_HZ,
                  "frames": int(samples.size), **_levels(samples)},
        "output": {"file_name": "processed.wav", "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
                   "sample_rate_hz": SAMPLE_RATE_HZ, "frames": int(decoded.size), **_levels(decoded)},
        "floating_output": _levels(output),
        "pcm16_quantization_max_abs_error": float(np.max(np.abs(decoded - output))),
        "timing": {"file_processing_seconds": elapsed, "duration_seconds": samples.size / SAMPLE_RATE_HZ,
                   "real_time_device_performance_verified": False},
        "limitations": [
            "48 kHz mono PCM16 WAV only, maximum 60 seconds; no automatic conversion or normalization.",
            "The user must identify at least two seconds of noise only within this same input; the assertion is not verified.",
            "Whole-file amplitude inspection includes future samples; this is offline and noncausal, not real-time protection.",
            "Growing noise is also withheld and left in the output; profile transients can weaken protection.",
            "Equal-amplitude overlap and contaminated profiles can still suppress wanted audio.",
            "Approved means supported by this profile rule, not proven safe or perceptually improved.",
            "No microphone capture, automatic playback, spatial cancellation or commercial-quality approval.",
        ],
    }
    _dump_json(target / "report.json", report)
    return report
