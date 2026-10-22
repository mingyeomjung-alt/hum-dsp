"""Experimental offline profile-supported bank with a whole-file amplitude veto.

The user, not this algorithm, declares a same-input noise-only interval. It is
not source separation: a mistaken profile or an equally loud overlapping wanted
tone can still remove wanted content. The veto uses future audio and is neither
causal nor a real-time adaptive protection claim. A growing noise tone is also
vetoed, deliberately trading removal for a conservative unchanged-file outcome.
"""

from dataclasses import asdict, dataclass
import copy
import hashlib
import math
import numbers

import numpy as np

from .audio import validate_audio
from .extended_bank import ExtendedBankConfig, render_extended_bank, replay_extended_bank

ALGORITHM = "profile_supported_whole_file_amplitude_veto_research_v1"
RATE = 48000
OBSERVATION = 96000
FFT_SIZE = 524288
PROBE = 24000
PROBE_HOP = 12000
GUARD_RATIO = 2.0
GUARD_EPSILON = 1e-12
POLICY = {
    "research_only": True,
    "profile_assumption": "user_declared_noise_only_not_verified",
    "profile_binding": "same_input_sample_interval",
    "profile_window_frames": OBSERVATION, "profile_hop_frames": RATE,
    "profile_window": "symmetric_hann", "profile_fft_frames": FFT_SIZE,
    "minimum_coherent_power_fraction": .02, "minimum_peak_amplitude": 1e-7,
    "search_half_width_hz_absolute": 1.0, "minimum_eligible_profile_windows": 1,
    "frequency_selection": "median_eligible_profile_estimates",
    "amplitude_estimator": "hann_weighted_sine_cosine_dc_least_squares",
    "amplitude_probe_frames": PROBE, "amplitude_probe_hop_frames": PROBE_HOP,
    "profile_reference": "maximum_profile_probe_amplitude",
    "guard_scan": "whole_file_including_future_and_final_edge",
    "guard_ratio": GUARD_RATIO, "guard_absolute_epsilon": GUARD_EPSILON,
    "guard_action": "reject_entire_order_if_any_probe_exceeds_reference",
    "guard_ratio_status": "engineering_choice_not_calibrated_guarantee",
    "causal": False, "source_separation": False,
    "equal_amplitude_wanted_overlap_protection": "not_guaranteed",
    "inner_engine": "unchanged_manual_extended_tonal_bank_research_v1",
    "replay": "frozen_linear_component_replay_no_profile_or_guard_reestimation",
    "hashes": "canonical_little_endian_float64_sample_bytes_not_wav_file_sha256",
}


def _real(value, name):
    if (not isinstance(value, numbers.Real) or isinstance(value, (bool, np.bool_))
            or not math.isfinite(value)):
        raise ValueError(f"{name} must be a finite real number")
    return float(value)


@dataclass(frozen=True)
class ProfileGuardConfig:
    sample_rate_hz: int = RATE
    nominal_hum_hz: float = 60.0
    orders: tuple[int, ...] = tuple(range(1, 33))
    max_attenuation_db: float = 12.0
    profile_start_sample: int | None = None
    profile_end_sample: int | None = None
    block_size_frames: int = 4096

    def __post_init__(self):
        orders = self.orders
        if (type(orders) not in (list, tuple) or len(orders) > 32
                or any(type(v) is not int or not 1 <= v <= 32 for v in orders)
                or tuple(orders) != tuple(sorted(set(orders)))):
            raise ValueError("orders must be sorted unique integers in [1,32], or empty")
        checked = ExtendedBankConfig(
            sample_rate_hz=self.sample_rate_hz, nominal_hum_hz=self.nominal_hum_hz,
            orders=orders or (1,), max_attenuation_db=self.max_attenuation_db,
            block_size_frames=self.block_size_frames)
        start, end = self.profile_start_sample, self.profile_end_sample
        if (start is None) != (end is None):
            raise ValueError("profile start/end must both be absent or both present")
        if start is not None and (type(start) is not int or type(end) is not int
                                  or start < 0 or end - start < OBSERVATION):
            raise ValueError("declared profile must span at least 96000 samples")
        object.__setattr__(self, "orders", tuple(orders))
        object.__setattr__(self, "nominal_hum_hz", checked.nominal_hum_hz)
        object.__setattr__(self, "max_attenuation_db", checked.max_attenuation_db)

    def to_dict(self):
        record = asdict(ProfileGuardConfig(**asdict(self)))
        record["orders"] = list(record["orders"])
        return record


def _config(config):
    if config is None:
        return ProfileGuardConfig()
    if type(config) is not ProfileGuardConfig:
        raise ValueError("config must be ProfileGuardConfig")
    return ProfileGuardConfig(**config.to_dict())


def _starts(start, end, width, hop):
    values = list(range(start, end - width + 1, hop))
    if not values or values[-1] != end - width:
        values.append(end - width)
    return values


def _sha(source):
    return hashlib.sha256(np.asarray(source, dtype="<f8").tobytes()).hexdigest()


def _profile_evidence(source, config):
    window = np.hanning(OBSERVATION)
    window_sum, square_sum = float(window.sum()), float(window @ window)
    result = {order: [] for order in config.orders}
    step = RATE / FFT_SIZE
    for start in _starts(config.profile_start_sample, config.profile_end_sample, OBSERVATION, RATE):
        part = source[start:start + OBSERVATION] * window
        power = float(part @ part / square_sum)
        spectrum = np.abs(np.fft.rfft(part, n=FFT_SIZE)) ** 2
        for order in config.orders:
            lower, upper = order * config.nominal_hum_hz - 1, order * config.nominal_hum_hz + 1
            bins = np.arange(math.ceil(lower / step - 1), math.floor(upper / step + 1) + 1)
            peak = int(bins[np.argmax(spectrum[bins])])
            amp = 2 * math.sqrt(float(spectrum[peak])) / window_sum
            fraction = float(np.clip(.5 * amp ** 2 / power, 0, 1)) if power > 0 else 0.
            frequency, status = None, "insufficient_concentration"
            if power == 0:
                amp, status = 0., "insufficient_energy"
            elif amp >= 1e-7 and fraction >= .02:
                left, middle, right = np.log(np.maximum(spectrum[peak - 1:peak + 2], np.finfo(float).tiny))
                denominator = float(left - 2 * middle + right)
                offset = 0. if denominator >= -1e-18 else float(.5 * (left - right) / denominator)
                estimated = (peak + float(np.clip(offset, -1, 1))) * step
                status = "eligible" if lower - .001 <= estimated <= upper + .001 else "out_of_band"
                if status == "eligible":
                    frequency = float(np.clip(estimated, lower, upper))
            result[order].append({"start_sample": start, "end_sample": start + OBSERVATION,
                                  "estimate_hz": frequency, "coherent_power_fraction": fraction,
                                  "peak_amplitude": amp, "status": status})
    return result


def _amplitude_probes(source, frequency, starts):
    time = np.arange(PROBE) / RATE
    basis = np.column_stack((np.sin(2 * np.pi * frequency * time),
                             np.cos(2 * np.pi * frequency * time), np.ones(PROBE)))
    weight = np.sqrt(np.hanning(PROBE))
    projection = np.linalg.pinv(basis * weight[:, None]) * weight[None, :]
    result = []
    for start in starts:
        coefficients = projection @ source[start:start + PROBE]
        amplitude = math.hypot(float(coefficients[0]), float(coefficients[1]))
        result.append({"start_sample": start, "end_sample": start + PROBE, "amplitude": amplitude})
    return result


def _decision(order, evidence, profile_probes, whole_probes):
    eligible = [record["estimate_hz"] for record in evidence if record["status"] == "eligible"]
    supported = bool(eligible)
    frequency = float(np.median(eligible)) if supported else None
    reference = max(record["amplitude"] for record in profile_probes) if supported else None
    peak = max(whole_probes, key=lambda record: record["amplitude"]) if supported else None
    rejected = bool(supported and peak["amplitude"] > GUARD_RATIO * reference + GUARD_EPSILON)
    ratio = (peak["amplitude"] / reference if reference > 0 else None) if supported else None
    return {"order": order, "profile_observations": evidence, "profile_supported": supported,
            "frequency_hz": frequency, "profile_probes": profile_probes,
            "whole_file_probes": whole_probes, "profile_reference_amplitude": reference,
            "whole_file_maximum_amplitude": peak["amplitude"] if supported else None,
            "whole_file_maximum_probe_start_sample": peak["start_sample"] if supported else None,
            "maximum_amplitude_ratio": ratio, "amplitude_guard_rejected": rejected,
            "approved": supported and not rejected,
            "reason": "profile_unsupported" if not supported else
                      "whole_file_amplitude_exceeds_profile" if rejected else "profile_supported_amplitude_consistent"}


def _inner(config, orders):
    return ExtendedBankConfig(sample_rate_hz=config.sample_rate_hz, nominal_hum_hz=config.nominal_hum_hz,
                              orders=orders, max_attenuation_db=config.max_attenuation_db,
                              block_size_frames=config.block_size_frames)


def _bypass_reason(config):
    return ("zero_depth" if config.max_attenuation_db == 0 else "no_selected_orders" if not config.orders
            else "no_profile_declared" if config.profile_start_sample is None else None)


def render_profile_guard(samples, config=None):
    """Return output, descriptor, and auditable frozen whole-file guard trace."""
    source, selected = validate_audio(samples), _config(config)
    if selected.profile_end_sample is not None and selected.profile_end_sample > source.size:
        raise ValueError("declared profile interval exceeds same input length")
    reason, records, nested, approved = _bypass_reason(selected), [], None, []
    if reason is None:
        evidence = _profile_evidence(source, selected)
        for order in selected.orders:
            eligible = [row["estimate_hz"] for row in evidence[order] if row["status"] == "eligible"]
            profile, whole = [], []
            if eligible:
                frequency = float(np.median(eligible))
                profile = _amplitude_probes(source, frequency, _starts(
                    selected.profile_start_sample, selected.profile_end_sample, PROBE, PROBE_HOP))
                whole = _amplitude_probes(source, frequency, _starts(0, source.size, PROBE, PROBE_HOP))
            records.append(_decision(order, evidence[order], profile, whole))
        approved = [record["order"] for record in records if record["approved"]]
        if not approved:
            reason = "no_profile_supported_amplitude_consistent_orders"
    output = source.copy()
    if approved:
        output, _, nested = render_extended_bank(source, _inner(selected, approved))
    trace = {"schema_version": 1, "algorithm": ALGORITHM, "config": selected.to_dict(),
             "policy": copy.deepcopy(POLICY), "source_frames": source.size,
             "input_canonical_float64_sha256": _sha(source),
             "profile_interval_canonical_float64_sha256": None if selected.profile_start_sample is None else
                 _sha(source[selected.profile_start_sample:selected.profile_end_sample]),
             "bands": records, "approved_orders": approved, "bypass_reason": reason,
             "extended_bank_trace": nested}
    description = copy.deepcopy({key: value for key, value in trace.items() if key != "extended_bank_trace"})
    description.update(processing_bypassed=not approved, samples_processed=source.size,
                       maximum_input_peak=float(np.max(np.abs(source))),
                       maximum_output_peak=float(np.max(np.abs(output))))
    return output, description, trace


def _same(actual, expected):
    if type(actual) is not type(expected):
        raise ValueError("profile guard trace value/type differs")
    if type(expected) is dict:
        if actual.keys() != expected.keys():
            raise ValueError("profile guard trace fields differ")
        for key in expected:
            _same(actual[key], expected[key])
    elif type(expected) is list:
        if len(actual) != len(expected):
            raise ValueError("profile guard trace schedule differs")
        for value, reference in zip(actual, expected, strict=True):
            _same(value, reference)
    elif type(expected) is float:
        if not math.isfinite(actual) or actual != expected:
            raise ValueError("profile guard trace decision differs")
    elif actual != expected:
        raise ValueError("profile guard trace decision differs")


def _validate_evidence(evidence, config, order):
    starts = _starts(config.profile_start_sample, config.profile_end_sample, OBSERVATION, RATE)
    if type(evidence) is not list or len(evidence) != len(starts):
        raise ValueError("profile support window schedule differs")
    for row, start in zip(evidence, starts, strict=True):
        if type(row) is not dict or set(row) != {"start_sample", "end_sample", "estimate_hz",
                                                  "coherent_power_fraction", "peak_amplitude", "status"}:
            raise ValueError("invalid profile support fields")
        _same(row["start_sample"], start)
        _same(row["end_sample"], start + OBSERVATION)
        if type(row["coherent_power_fraction"]) is not float or type(row["peak_amplitude"]) is not float:
            raise ValueError("profile evidence amplitudes and fractions must be floats")
        fraction, amplitude = _real(row["coherent_power_fraction"], "fraction"), _real(row["peak_amplitude"], "amplitude")
        if not 0 <= fraction <= 1 or amplitude < 0 or type(row["status"]) is not str:
            raise ValueError("invalid profile support bounds")
        eligible = amplitude >= 1e-7 and fraction >= .02
        frequency, status = row["estimate_hz"], row["status"]
        if frequency is not None:
            if type(frequency) is not float:
                raise ValueError("profile frequency must be a float")
            frequency = _real(frequency, "profile frequency")
            if not eligible or status != "eligible" or abs(frequency - order * config.nominal_hum_hz) > 1:
                raise ValueError("inconsistent eligible profile evidence")
        elif (status == "insufficient_energy" and (fraction != 0 or amplitude != 0)
              or status == "insufficient_concentration" and eligible
              or status == "out_of_band" and not eligible
              or status not in ("insufficient_energy", "insufficient_concentration", "out_of_band")):
            raise ValueError("inconsistent rejected profile evidence")


def _validate_probes(probes, starts):
    if type(probes) is not list or len(probes) != len(starts):
        raise ValueError("profile guard amplitude probe schedule differs")
    for record, start in zip(probes, starts, strict=True):
        if type(record) is not dict or set(record) != {"start_sample", "end_sample", "amplitude"}:
            raise ValueError("invalid profile guard probe fields")
        _same(record["start_sample"], start)
        _same(record["end_sample"], start + PROBE)
        if type(record["amplitude"]) is not float or _real(record["amplitude"], "probe amplitude") < 0:
            raise ValueError("probe amplitude cannot be negative")


def replay_profile_guard(samples, config, trace):
    """Validate stored decisions and replay linearly, without estimating evidence.

    Recorded hashes identify the original render, not replay components. They
    cannot authenticate forged evidence, and intentionally are not compared to
    component input hashes. Length, settings, policy, schedules and decisions
    remain strictly bound to the frozen trace.
    """
    if type(config) is not ProfileGuardConfig:
        raise ValueError("replay config must be ProfileGuardConfig")
    source, selected = validate_audio(samples), _config(config)
    if selected.profile_end_sample is not None and selected.profile_end_sample > source.size:
        raise ValueError("declared profile interval exceeds replay input length")
    fields = {"schema_version", "algorithm", "config", "policy", "source_frames",
              "input_canonical_float64_sha256", "profile_interval_canonical_float64_sha256",
              "bands", "approved_orders", "bypass_reason", "extended_bank_trace"}
    if type(trace) is not dict or set(trace) != fields:
        raise ValueError("invalid complete profile guard trace")
    for key, value in (("schema_version", 1), ("algorithm", ALGORITHM), ("config", selected.to_dict()),
                       ("policy", POLICY), ("source_frames", source.size)):
        _same(trace[key], value)
    for key in ("input_canonical_float64_sha256", "profile_interval_canonical_float64_sha256"):
        digest = trace[key]
        if digest is None and key.startswith("profile_") and selected.profile_start_sample is None:
            continue
        if type(digest) is not str or len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            raise ValueError("invalid canonical sample identity hash")
    if selected.profile_start_sample is None:
        _same(trace["profile_interval_canonical_float64_sha256"], None)
    reason, expected_records, approved = _bypass_reason(selected), [], []
    if type(trace["bands"]) is not list:
        raise ValueError("profile guard bands must be a list")
    if reason is None:
        if len(trace["bands"]) != len(selected.orders):
            raise ValueError("profile guard selected band count differs")
        for order, record in zip(selected.orders, trace["bands"], strict=True):
            if type(record) is not dict or not {"profile_observations", "profile_probes", "whole_file_probes"} <= record.keys():
                raise ValueError("missing profile support and amplitude evidence")
            evidence, profile, whole = record["profile_observations"], record["profile_probes"], record["whole_file_probes"]
            _validate_evidence(evidence, selected, order)
            supported = any(row["status"] == "eligible" for row in evidence)
            _validate_probes(profile, _starts(selected.profile_start_sample, selected.profile_end_sample,
                                              PROBE, PROBE_HOP) if supported else [])
            _validate_probes(whole, _starts(0, source.size, PROBE, PROBE_HOP) if supported else [])
            expected_records.append(_decision(order, evidence, profile, whole))
        approved = [record["order"] for record in expected_records if record["approved"]]
        if not approved:
            reason = "no_profile_supported_amplitude_consistent_orders"
    _same(trace["bands"], expected_records)
    _same(trace["approved_orders"], approved)
    _same(trace["bypass_reason"], reason)
    if not approved:
        _same(trace["extended_bank_trace"], None)
        return source.copy()
    return replay_extended_bank(source, _inner(selected, approved), trace["extended_bank_trace"])
