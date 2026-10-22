"""Raw full-RMS measurements for known-reference, frozen-decision replay.

Replaying a fixed time-varying linear filter on known synthetic components can
separate desired-signal damage from residual noise without fitting gains or
discarding phase error. It is an oracle development measurement, NOT source
separation available from an unknown real-world mixture. Detection decisions
must come from the mixture once and remain frozen for both component replays.
"""

from numbers import Integral
from typing import Any

import numpy as np


def _vector(value: Any, name: str) -> np.ndarray:
    try:
        values = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite real nonempty vector") from exc
    if values.ndim != 1 or values.size == 0 or values.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a finite real nonempty vector")
    result = values.astype(np.longdouble)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    return result


def _rms(values: np.ndarray) -> np.longdouble:
    scale = np.max(np.abs(values))
    if not np.isfinite(scale) or scale == 0:
        return scale
    return scale * np.sqrt(np.mean(np.square(values / scale)))


def _number(value: Any) -> float | None:
    if not np.isfinite(value) or abs(value) > np.finfo(np.float64).max:
        return None
    converted = float(value)
    if converted == 0 and value != 0:
        return None
    return converted


def _db(numerator: np.longdouble, denominator: np.longdouble) -> float | None:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or numerator <= 0 or denominator <= 0:
        return None
    return _number(20.0 * (np.log10(numerator) - np.log10(denominator)))


def measure_harmonic_components(
    clean: Any,
    noise: Any,
    mixed_output: Any,
    clean_replayed: Any,
    noise_replayed: Any,
    *,
    start_sample: int = 0,
) -> dict:
    """Measure no-alignment/no-normalization replay components on one window.

    ``clean_relative_error_rmse`` is RMS(clean_replayed-clean) / RMS(clean),
    not a dB quantity. It includes phase and timing errors. ``noise_attenuation_db``
    compares full residual-noise RMS, not its in-phase projection. A failed
    additivity check does not silently discard results: measurements are marked
    invalid for component interpretation, with the measured discrepancy retained.
    All undefined or unreportable scalar results are JSON null plus reasons.
    """
    names = ("clean", "noise", "mixed_output", "clean_replayed", "noise_replayed")
    arrays = [_vector(value, name) for name, value in zip(
        names, (clean, noise, mixed_output, clean_replayed, noise_replayed), strict=True
    )]
    if len({values.size for values in arrays}) != 1:
        raise ValueError("all five reference/output vectors must have equal lengths")
    if isinstance(start_sample, (bool, np.bool_)) or not isinstance(start_sample, Integral):
        raise ValueError("start_sample must be an integer")
    if not 0 <= start_sample < arrays[0].size:
        raise ValueError("start_sample must leave at least one sample")
    total_count = int(arrays[0].size)
    c, n, y, yc, yn = (values[int(start_sample):] for values in arrays)
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        clean_error = yc - c
        raw_error = y - c
        additive_error = y - (yc + yn)
        clean_rms, noise_rms, yc_rms, residual_rms = map(_rms, (c, n, yc, yn))
        clean_error_rms = _rms(clean_error)
        additivity_error = np.max(np.abs(additive_error))
        relative = _number(clean_error_rms / clean_rms) if clean_rms > 0 else None
        additive_number = _number(additivity_error)
        additive_ok = bool(additivity_error <= np.longdouble("1e-10")) if np.isfinite(additivity_error) else False
        result = {
            "schema_version": 1,
            "sample_rate_hz": 48000,
            "total_sample_count": total_count,
            "analysis_start_sample": int(start_sample),
            "analyzed_sample_count": int(c.size),
            "interpretation": "known_synthetic_reference_frozen_decision_replay_only_not_observational_source_separation",
            "alignment_applied": False,
            "output_gain_normalization_applied": False,
            "component_gain_fitting_applied": False,
            "component_interpretation_valid": additive_ok,
            "decomposition_additivity_max_abs_error": additive_number,
            "decomposition_additivity_within_1e_10": additive_ok,
            "clean_rms": _number(clean_rms),
            "input_noise_rms": _number(noise_rms),
            "clean_replayed_rms": _number(yc_rms),
            "clean_error_rmse": _number(clean_error_rms),
            "clean_relative_error_rmse": relative,
            "clean_level_change_db": _db(yc_rms, clean_rms),
            "residual_noise_rms": _number(residual_rms),
            "noise_attenuation_db": _db(noise_rms, residual_rms),
            "raw_mixture_error_rmse": _number(_rms(raw_error)),
            "undefined_metrics": {},
        }
    undefined = result["undefined_metrics"]
    for key, value in result.items():
        if value is None:
            undefined[key] = "outside_json_float_range"
    if clean_rms == 0:
        for key in ("clean_relative_error_rmse", "clean_level_change_db"):
            undefined[key] = "zero_clean_reference"
    elif yc_rms == 0:
        undefined["clean_level_change_db"] = "zero_replayed_clean_not_finite_db"
    if noise_rms == 0:
        undefined["noise_attenuation_db"] = "zero_input_noise"
    elif residual_rms == 0:
        undefined["noise_attenuation_db"] = "zero_residual_noise_not_finite_db"
    if not additive_ok:
        result["component_interpretation_invalid_reason"] = "component_replay_is_not_additive_within_1e_10"
    return result
