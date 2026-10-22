"""Explicit offline suppression of persistent lines in a selected hum family.

This NumPy DSP path is not the causal baseline. It observes the whole file,
uses local prominence rather than global power share, requires four mutually
consistent harmonic orders, then applies one frozen, stereo-linked zero-phase
frequency mask. Persistence cannot distinguish wanted tones from interference.
No automatic noise classification, device I/O, normalization or clipping occurs.
"""

from dataclasses import asdict, dataclass
import copy
import hashlib
import json
import math
import numbers

import numpy as np


ALGORITHM = "offline_persistent_harmonic_mask_v1"
POLICY = {
    "offline_whole_file_lookahead": True,
    "maximum_seconds": 60,
    "analysis_window_seconds": 2,
    "analysis_hop_seconds": 1,
    "minimum_analysis_windows": 3,
    "analysis_window": "symmetric_hann",
    "analysis_zero_padding_minimum_factor": 4,
    "channel_evidence": "mean_channel_spectral_powers_not_mean_samples",
    "search_half_width": "max_1Hz_or_2percent_nominal_center",
    "local_floor_annulus_hz": [4.0, 10.0],
    "minimum_local_prominence_db": 10.0,
    "minimum_peak_amplitude": 1e-7,
    "minimum_window_persistence": 0.75,
    "maximum_frequency_mad_hz": 0.25,
    "minimum_consistent_orders": 4,
    "maximum_family_residual_hz": 0.5,
    "base_frequency_search_fraction": 0.02,
    "maximum_nominal_center_hz": 2000,
    "mask_flat_half_width_hz": 1.0,
    "mask_outer_half_width_hz": 3.0,
    "mask_transition": "raised_cosine_amplitude_gain",
    "overlapping_masks": "minimum_gain_not_product",
    "render": "shared_real_nonnegative_FFT_gain_per_channel",
    "padding": "at_least_2seconds_reflection_each_side_next_power_of_two",
    "phase": "zero_phase_offline_not_zero_boundary_effect",
    "wanted_tone_protection": "not_guaranteed_same_family_wanted_lines_also_removed",
    "no_accepted_orders_or_zero_depth": "exact_validated_float_copy",
    "output_handling": "reject_nonfinite_or_out_of_range_no_clip_or_gain_normalization",
    "replay": "frozen_evidence_and_mask_no_audio_reestimation",
}


def _real(value, name):
    if (not isinstance(value, numbers.Real) or isinstance(value, (bool, np.bool_))
            or not math.isfinite(value)):
        raise ValueError(f"{name} must be finite and real")
    return float(value)


@dataclass(frozen=True)
class CombConfig:
    sample_rate_hz: int = 48000
    nominal_hum_hz: float = 60.0
    max_attenuation_db: float = 12.0
    orders: tuple[int, ...] = tuple(range(1, 33))

    def __post_init__(self):
        if type(self.sample_rate_hz) is not int or self.sample_rate_hz not in (44100, 48000):
            raise ValueError("persistent comb supports native 44100 or 48000 Hz")
        nominal = _real(self.nominal_hum_hz, "nominal_hum_hz")
        depth = _real(self.max_attenuation_db, "max_attenuation_db")
        if nominal not in (50.0, 60.0) or not 0 <= depth <= 18:
            raise ValueError("select 50/60 Hz and depth in [0,18] dB")
        if (type(self.orders) not in (tuple, list) or not 1 <= len(self.orders) <= 32
                or any(type(order) is not int or not 1 <= order <= 32 for order in self.orders)
                or tuple(self.orders) != tuple(sorted(set(self.orders)))):
            raise ValueError("orders must be sorted unique integers in [1,32]")
        object.__setattr__(self, "orders", tuple(self.orders))
        object.__setattr__(self, "nominal_hum_hz", nominal)
        object.__setattr__(self, "max_attenuation_db", depth)

    def to_dict(self):
        record = asdict(self)
        record["orders"] = list(self.orders)
        return record


def _config(config):
    if config is None:
        return CombConfig()
    if type(config) is not CombConfig:
        raise ValueError("config must be CombConfig")
    return CombConfig(**config.to_dict())


def _audio(samples, config):
    raw = np.asarray(samples)
    if (raw.dtype.kind not in "fiu" or raw.ndim not in (1, 2)
            or (raw.ndim == 2 and raw.shape[1] not in (1, 2))
            or not 1 <= raw.shape[0] <= config.sample_rate_hz * POLICY["maximum_seconds"]):
        raise ValueError("audio must contain 1 sample through 60 seconds of real mono/stereo samples")
    result = np.array(raw, dtype=np.float64, copy=True)
    if not np.isfinite(result).all() or np.any(result < -1) or np.any(result >= 1):
        raise ValueError("audio must be finite normalized samples in [-1,1)")
    return result


def _starts(frames, rate):
    width = 2 * rate
    if frames < width:
        return []
    return list(range(0, frames - width + 1, rate))


def _search(order, config):
    nominal = order * config.nominal_hum_hz
    half = max(1.0, nominal * 0.02)
    return nominal - half, nominal + half


def _observations(source, config, starts):
    rate, width = config.sample_rate_hz, 2 * config.sample_rate_hz
    fft_frames = 1 << (4 * width - 1).bit_length()
    window = np.hanning(width)
    window_sum = float(window.sum())
    frequencies = np.fft.rfftfreq(fft_frames, 1 / rate)
    step = rate / fft_frames
    samples = source[:, None] if source.ndim == 1 else source
    collected = {order: [] for order in config.orders}
    for start in starts:
        spectrum = np.fft.rfft(samples[start:start + width] * window[:, None], n=fft_frames, axis=0)
        powers = np.mean(spectrum.real ** 2 + spectrum.imag ** 2, axis=1)
        for order in config.orders:
            lower, upper = _search(order, config)
            first = max(1, math.ceil(lower / step))
            last = min(powers.size - 2, math.floor(upper / step))
            peak = first + int(np.argmax(powers[first:last + 1]))
            left, middle, right = np.log(np.maximum(powers[peak - 1:peak + 2], np.finfo(float).tiny))
            denominator = float(left - 2 * middle + right)
            offset = 0.0 if denominator >= -1e-18 else float(.5 * (left - right) / denominator)
            frequency = (peak + float(np.clip(offset, -1, 1))) * step
            delta = np.abs(frequencies - frequency)
            floor = float(np.median(powers[(delta >= 4) & (delta <= 10)]))
            peak_power = float(powers[peak])
            amplitude = 2 * math.sqrt(peak_power) / window_sum
            prominence = 10 * math.log10(max(peak_power, 1e-300) / max(floor, 1e-300))
            eligible = (lower <= frequency <= upper and amplitude >= 1e-7 and prominence >= 10)
            collected[order].append((frequency, prominence, eligible))
    records = []
    for order, rows in collected.items():
        qualified = [row for row in rows if row[2]]
        frequency = float(np.median([row[0] for row in qualified])) if qualified else None
        prominence = float(np.median([row[1] for row in qualified])) if qualified else None
        mad = float(np.median([abs(row[0] - frequency) for row in qualified])) if qualified else None
        persistent = (len(qualified) / len(starts) >= .75 and mad is not None and mad <= .25)
        records.append({"order": order, "candidate_frequency_hz": frequency,
                        "eligible_windows": len(qualified), "total_windows": len(starts),
                        "median_prominence_db": prominence, "frequency_mad_hz": mad,
                        "persistent": persistent})
    return records


def _family(records, config):
    candidates = [row for row in records if row["persistent"]]
    lower, upper = config.nominal_hum_hz * .98, config.nominal_hum_hz * 1.02
    best, best_key, base = [], None, None
    for anchor in candidates:
        estimate = anchor["candidate_frequency_hz"] / anchor["order"]
        if not lower <= estimate <= upper:
            continue
        group = [row for row in candidates
                 if abs(row["candidate_frequency_hz"] - row["order"] * estimate) <= .5]
        score = (len(group), sum(min(row["median_prominence_db"], 60.0) for row in group),
                 -abs(estimate - config.nominal_hum_hz))
        if best_key is None or score > best_key:
            best, best_key, base = group, score, estimate
    if len(best) < 4:
        return None, set()
    for _ in range(2):
        base = sum(row["order"] * row["candidate_frequency_hz"] for row in best) / sum(
            row["order"] ** 2 for row in best)
        best = [row for row in candidates if abs(row["candidate_frequency_hz"] - row["order"] * base) <= .5]
        if len(best) < 4 or not lower <= base <= upper:
            return None, set()
    return float(base), {row["order"] for row in best}


def _finish(records, config):
    base, accepted = _family(records, config)
    result = copy.deepcopy(records)
    for row in result:
        row["accepted"] = row["order"] in accepted
        row["reason"] = ("persistent_consistent_family" if row["accepted"] else
                         "insufficient_persistence_or_stability" if not row["persistent"] else
                         "insufficient_or_inconsistent_family")
    return base, result


def _mask(frequencies, lines, depth):
    gain = np.ones(frequencies.size, dtype=np.float64)
    floor = 10 ** (-depth / 20)
    for row in lines:
        if not row["accepted"]:
            continue
        distance = np.abs(frequencies - row["candidate_frequency_hz"])
        shape = np.zeros_like(gain)
        shape[distance <= 1] = 1
        edge = (distance > 1) & (distance < 3)
        shape[edge] = .5 * (1 + np.cos(np.pi * (distance[edge] - 1) / 2))
        np.minimum(gain, 1 - (1 - floor) * shape, out=gain)
    return gain


def _render(source, config, lines):
    if config.max_attenuation_db == 0 or not any(row["accepted"] for row in lines):
        return source.copy()
    frames = source.shape[0]
    fft_frames = 1 << (frames + 4 * config.sample_rate_hz - 1).bit_length()
    left = (fft_frames - frames) // 2
    right = fft_frames - frames - left
    samples = source[:, None] if source.ndim == 1 else source
    padded = np.pad(samples, ((left, right), (0, 0)), mode="reflect")
    spectrum = np.fft.rfft(padded, axis=0)
    gain = _mask(np.fft.rfftfreq(fft_frames, 1 / config.sample_rate_hz), lines, config.max_attenuation_db)
    spectrum *= gain[:, None]
    result = np.fft.irfft(spectrum, n=fft_frames, axis=0)[left:left + frames].copy()
    if source.ndim == 1:
        result = result[:, 0]
    if not np.isfinite(result).all() or np.any(result < -1) or np.any(result >= 1):
        raise ValueError("persistent comb output exceeds normalized range; no clipping was applied")
    return result


def render_persistent_comb(samples, config=None):
    """Return output, descriptor and a JSON-safe frozen-mask trace.

    Lookahead removes causal acquisition delay but introduces offline boundary
    effects and pre/post ringing. A stationary wanted harmonic family is just as
    removable as hum. The function never labels a component as unwanted.
    """
    selected = _config(config)
    source = _audio(samples, selected)
    starts = _starts(source.shape[0], selected.sample_rate_hz)
    reason, base, lines = "persistent_family_mask", None, []
    if selected.max_attenuation_db == 0:
        reason = "zero_depth"
    elif len(starts) < 3:
        reason = "insufficient_analysis_windows"
    else:
        base, lines = _finish(_observations(source, selected, starts), selected)
        if base is None:
            reason = "no_consistent_persistent_family"
    trace = {"schema_version": 1, "algorithm": ALGORITHM, "config": selected.to_dict(),
             "policy": copy.deepcopy(POLICY), "frames": source.shape[0],
             "channels": 1 if source.ndim == 1 else source.shape[1],
             "analysis_windows": len(starts), "base_frequency_hz": base, "reason": reason,
             "lines": lines,
             "mixture_canonical_float64_sha256": hashlib.sha256(source.astype("<f8").tobytes()).hexdigest()}
    output = _render(source, selected, lines)
    descriptor = {"algorithm": ALGORITHM, "config": selected.to_dict(),
                  "accepted_orders": [row["order"] for row in lines if row["accepted"]],
                  "reason": reason, "policy": copy.deepcopy(POLICY),
                  "quality_approval": False, "real_time": False,
                  "trace_binding": "frozen_configuration_shape_and_mask_not_source_authentication"}
    json.dumps((descriptor, trace), allow_nan=False)
    return output, descriptor, trace


def _validate_trace(trace, config, source):
    keys = {"schema_version", "algorithm", "config", "policy", "frames", "channels", "analysis_windows",
            "base_frequency_hz", "reason", "lines", "mixture_canonical_float64_sha256"}
    if type(trace) is not dict or set(trace) != keys:
        raise ValueError("invalid persistent-comb trace fields")
    # JSON canonical comparisons preserve type distinctions such as true vs 1.
    equal = lambda a, b: json.dumps(a, sort_keys=True, allow_nan=False) == json.dumps(b, sort_keys=True, allow_nan=False)
    channels = 1 if source.ndim == 1 else source.shape[1]
    if (not equal(trace["schema_version"], 1) or trace["algorithm"] != ALGORITHM
            or not equal(trace["config"], config.to_dict()) or not equal(trace["policy"], POLICY)
            or not equal(trace["frames"], source.shape[0]) or not equal(trace["channels"], channels)
            or not equal(trace["analysis_windows"], len(_starts(source.shape[0], config.sample_rate_hz)))):
        raise ValueError("persistent-comb trace configuration or shape mismatch")
    digest = trace["mixture_canonical_float64_sha256"]
    if type(digest) is not str or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("invalid mixture hash")
    rows = trace["lines"]
    if type(rows) is not list:
        raise ValueError("invalid line records")
    if config.max_attenuation_db == 0 or trace["analysis_windows"] < 3:
        reason = "zero_depth" if config.max_attenuation_db == 0 else "insufficient_analysis_windows"
        if rows or trace["base_frequency_hz"] is not None or trace["reason"] != reason:
            raise ValueError("invalid bypass trace")
        return rows
    if len(rows) != len(config.orders):
        raise ValueError("incomplete line records")
    evidence = []
    row_keys = {"order", "candidate_frequency_hz", "eligible_windows", "total_windows",
                "median_prominence_db", "frequency_mad_hz", "persistent", "accepted", "reason"}
    for order, row in zip(config.orders, rows, strict=True):
        if type(row) is not dict or set(row) != row_keys or not equal(row["order"], order):
            raise ValueError("invalid line evidence")
        count = row["eligible_windows"]
        if (type(count) is not int or not 0 <= count <= trace["analysis_windows"]
                or not equal(row["total_windows"], trace["analysis_windows"])):
            raise ValueError("invalid line observation counts")
        if count == 0:
            if any(row[key] is not None for key in ("candidate_frequency_hz", "median_prominence_db", "frequency_mad_hz")):
                raise ValueError("unsupported line cannot have estimates")
            persistent = False
        else:
            frequency = _real(row["candidate_frequency_hz"], "frequency")
            prominence = _real(row["median_prominence_db"], "prominence")
            mad = _real(row["frequency_mad_hz"], "frequency MAD")
            lower, upper = _search(order, config)
            if not lower <= frequency <= upper or prominence < 10 or mad < 0:
                raise ValueError("line estimates violate policy")
            persistent = count / trace["analysis_windows"] >= .75 and mad <= .25
        if not equal(row["persistent"], persistent) or type(row["accepted"]) is not bool:
            raise ValueError("line acceptance flags violate policy")
        evidence.append({key: copy.deepcopy(value) for key, value in row.items() if key not in ("accepted", "reason")})
    base, expected = _finish(evidence, config)
    reason = "no_consistent_persistent_family" if base is None else "persistent_family_mask"
    if not equal(rows, expected) or not equal(trace["base_frequency_hz"], base) or trace["reason"] != reason:
        raise ValueError("trace family decision differs from frozen evidence")
    return rows


def replay_persistent_comb(samples, config, trace):
    """Apply an already frozen mask, e.g. to known clean/noise components.

    No audio analysis runs. Mixture hash identifies the selection source but is
    intentionally not matched against isolated replay components. This is not an
    authenticated source validator or proof of independent quality.
    """
    selected = _config(config)
    source = _audio(samples, selected)
    return _render(source, selected, _validate_trace(trace, selected, source))
