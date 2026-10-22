"""Reference-aware measurements for the M1 synthetic test bench.

The caller supplies a known clean reference, known additive noise, and the raw
processor output. Nothing is aligned, normalized, or gain-matched here. Error
reduction is NOT synonymous with noise attenuation: muting or attenuating the
whole recording can also change reference error. The optional linear-component
fit is a model diagnostic, not a perceptual-quality score or proof of removal.
"""

from __future__ import annotations

from numbers import Integral
from typing import Any

import numpy as np


def _vector(value: Any, name: str) -> np.ndarray:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite real one-dimensional vector") from exc
    if array.ndim != 1 or array.size == 0 or array.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a finite real nonempty one-dimensional vector")
    # Use extended precision where the platform provides it (Windows may alias
    # longdouble to float64). Unrepresentable intermediate results become null.
    result = array.astype(np.longdouble)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    return result


def _rms(value: np.ndarray) -> np.longdouble:
    scale = np.max(np.abs(value))
    if scale == 0 or not np.isfinite(scale):
        return scale
    return scale * np.sqrt(np.mean(np.square(value / scale)))


def _number(value: Any) -> float | None:
    if value is None or not np.isfinite(value):
        return None
    if abs(value) > np.finfo(np.float64).max:
        return None
    result = float(value)
    # A nonzero extended-precision value that underflows is not a measured zero.
    if result == 0 and value != 0:
        return None
    return result


def _ratio_db(numerator: np.longdouble, denominator: np.longdouble) -> float | None:
    if not np.isfinite(numerator) or not np.isfinite(denominator):
        return None
    if numerator <= 0 or denominator <= 0:
        return None
    return _number(20 * (np.log10(numerator) - np.log10(denominator)))


def _component_fit(clean: np.ndarray, noise: np.ndarray, output: np.ndarray) -> dict:
    """Fit output ~= clean_gain*clean + noise_gain*noise without altering output.

    Rank/conditioning is evaluated after column normalization: different signal
    amplitudes alone do not make two independent references unidentifiable.
    Correlation warns about linear overlap, not all possible spectral overlap.
    Same-frequency proportional references deliberately produce no gains.
    """
    clean_rms, noise_rms, output_rms = map(_rms, (clean, noise, output))
    result: dict[str, Any] = {
        "status": "no_active_reference",
        "reason": "both_references_are_silent",
        "interpretation": "linear_reference_model_diagnostic_only",
        "attenuation_interpretation": "in_phase_projection_only_not_total_noise_reduction",
        "clean_gain": None,
        "noise_gain": None,
        "noise_attenuation_db": None,
        "residual_rms": _number(output_rms),
        "condition_number": None,
        "reference_correlation_abs": None,
        "reference_overlap_warning": False,
    }
    active = [("clean", clean, clean_rms), ("noise", noise, noise_rms)]
    active = [item for item in active if item[2] > 0]
    if not active:
        return result

    design = np.column_stack([np.asarray(values / level, dtype=np.float64)
                              for _, values, level in active])
    if len(active) == 2:
        correlation = abs(np.mean((clean / clean_rms) * (noise / noise_rms)))
        correlation = min(np.longdouble(1), correlation)
        result["reference_correlation_abs"] = _number(correlation)
        result["reference_overlap_warning"] = bool(correlation >= 0.98)
        singular = np.linalg.svd(design, compute_uv=False)
        if singular[-1] <= singular[0] * 1e-12:
            result.update(status="unidentifiable", reason="collinear_references", residual_rms=None)
            return result
        condition = singular[0] / singular[-1]
        result["condition_number"] = _number(condition)
        # A PCM16 sum can make proportional original references appear weakly
        # independent through quantization alone. Do not fit gains to that.
        if condition > 1e3:
            result.update(status="unidentifiable", reason="ill_conditioned_references", residual_rms=None)
            return result
    else:
        result["condition_number"] = 1.0

    output_scale = np.max(np.abs(output))
    if output_scale == 0:
        coefficients = np.zeros(len(active))
    else:
        coefficients = np.linalg.lstsq(
            design, np.asarray(output / output_scale, dtype=np.float64), rcond=None
        )[0]
    prediction = np.zeros_like(output)
    for coefficient, (name, values, level) in zip(coefficients, active):
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            gain = np.longdouble(coefficient) * output_scale / level
            # Reconstruct from normalized references, so an unreportably large
            # gain applied to a tiny reference does not itself overflow.
            prediction += np.longdouble(coefficient) * output_scale * (values / level)
        result[f"{name}_gain"] = _number(gain)
        if name == "noise" and gain != 0:
            # Negative gain is phase inversion; attenuation concerns magnitude.
            result["noise_attenuation_db"] = _number(-20 * np.log10(abs(gain)))
    with np.errstate(over="ignore", invalid="ignore"):
        residual_rms = _number(_rms(output - prediction))
    result.update(
        status="identifiable" if len(active) == 2 else f"{active[0][0]}_only_reference",
        reason=None,
        residual_rms=residual_rms,
    )
    if any(result[f"{name}_gain"] is None for name, _, _ in active):
        result.update(status="numerically_unreportable", reason="gain_outside_json_float_range")
    elif residual_rms is None:
        result.update(status="numerically_unreportable", reason="residual_outside_json_float_range")
    if noise_rms > 0 and result["noise_gain"] == 0:
        result["noise_attenuation_undefined_reason"] = "zero_fitted_noise_gain_not_finite_db"
    elif noise_rms > 0:
        noise_index = [item[0] for item in active].index("noise")
        if abs(coefficients[noise_index]) <= 1e-12 * max(1, np.max(np.abs(coefficients))):
            result["noise_attenuation_db"] = None
            result["noise_attenuation_undefined_reason"] = "fitted_noise_below_numerical_resolution"
    return result


def measure_output(
    clean: Any,
    noise: Any,
    output: Any,
    *,
    start_sample: int = 0,
    sample_rate_hz: int = 48000,
) -> dict:
    """Measure raw output error against known additive synthetic references.

    ``clean + noise`` defines the input. ``start_sample`` excludes an explicitly
    chosen settling interval from every metric, including levels and clip counts.
    The reported clip counts mean samples at or beyond normalized full scale;
    they cannot establish whether an earlier device actually clipped. Undefined
    dB values are JSON null with reasons, never artificial +/-infinity. No scalar
    is an overall pass/fail, perceptual-quality, or real-recording denoising claim.
    """
    arrays = [_vector(value, name) for value, name in
              ((clean, "clean"), (noise, "noise"), (output, "output"))]
    if len({array.size for array in arrays}) != 1:
        raise ValueError("clean, noise, and output must have equal lengths")
    if isinstance(sample_rate_hz, (bool, np.bool_)) or not isinstance(sample_rate_hz, Integral):
        raise ValueError("sample_rate_hz must be the integer 48000 for M1")
    if sample_rate_hz != 48000:
        raise ValueError("sample_rate_hz must be 48000 for M1")
    if isinstance(start_sample, (bool, np.bool_)) or not isinstance(start_sample, Integral):
        raise ValueError("start_sample must be an integer")
    if not 0 <= start_sample < arrays[0].size:
        raise ValueError("start_sample must leave at least one sample")

    total_count = int(arrays[0].size)
    c, n, y = [array[int(start_sample):] for array in arrays]
    with np.errstate(over="ignore", invalid="ignore"):
        mixture, error = c + n, y - c
    c_rms, n_rms, y_rms, error_rms = map(_rms, (c, n, y, error))
    result: dict[str, Any] = {
        "schema_version": 1,
        "sample_rate_hz": int(sample_rate_hz),
        "total_sample_count": total_count,
        "analysis_start_sample": int(start_sample),
        "analyzed_sample_count": int(c.size),
        "alignment_applied": False,
        "output_gain_normalization_applied": False,
        "clean_rms": _number(c_rms),
        "input_noise_rms": _number(n_rms),
        "input_rms": _number(_rms(mixture)),
        "output_rms": _number(y_rms),
        "output_error_rmse": _number(error_rms),
        "input_snr_db": _ratio_db(c_rms, n_rms),
        "output_snr_db": _ratio_db(c_rms, error_rms),
        "reference_error_reduction_db": _ratio_db(n_rms, error_rms),
        "input_peak_abs": _number(np.max(np.abs(mixture))),
        "output_peak_abs": _number(np.max(np.abs(y))),
        "input_fullscale_sample_count": int(np.count_nonzero(np.abs(mixture) >= 1)),
        "output_fullscale_sample_count": int(np.count_nonzero(np.abs(y) >= 1)),
        "undefined_metrics": {},
        "component_fit": _component_fit(c, n, y),
    }
    for name in ("input_snr_db", "output_snr_db", "reference_error_reduction_db"):
        if result[name] is None:
            if name != "reference_error_reduction_db" and c_rms == 0:
                reason = "zero_clean_reference"
            elif name == "input_snr_db" and n_rms == 0:
                reason = "zero_input_noise"
            elif name == "reference_error_reduction_db" and n_rms == 0:
                reason = "zero_input_noise"
            elif error_rms == 0:
                reason = "zero_output_error_not_finite_db"
            else:
                reason = "outside_json_float_range"
            result["undefined_metrics"][name] = reason
    for name in ("clean_rms", "input_noise_rms", "input_rms", "output_rms",
                 "output_error_rmse", "input_peak_abs", "output_peak_abs"):
        if result[name] is None:
            result["undefined_metrics"][name] = "outside_json_float_range"
    return result
