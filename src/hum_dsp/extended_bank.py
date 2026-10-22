"""Research-only manual extended tonal bank; independent NumPy/offline DSP.

This is NOT broadband denoising or a replacement for the shipped engine. Each
manually selected harmonic is independently observed, never phase locked. A
qualified wanted tone is suppressed just like a noise tone. Frozen replay
explains a single mixture run and does not certify perceptual improvement.
"""

from dataclasses import asdict, dataclass
import copy
import math
import numbers

import numpy as np

from .audio import validate_audio

ALGORITHM = "manual_extended_tonal_bank_research_v1"
SAMPLE_RATE_HZ = 48000
OBSERVATION_FRAMES = 96000
UPDATE_FRAMES = 4800
FFT_FRAMES = 524288
ATTACK_FRAMES = 4800
RELEASE_FRAMES = 12000
HOLD_FRAMES = 14400
MIN_COHERENT_POWER_FRACTION = 0.02
MIN_PEAK_AMPLITUDE = 1e-7
SEARCH_HALF_WIDTH_HZ = 1.0
MAX_SLEW_HZ_PER_S = 0.05
QUALIFICATION_ERROR_HZ = 0.15
POLICY = {
    "research_only": True, "sample_rate_hz": SAMPLE_RATE_HZ,
    "observation_frames": OBSERVATION_FRAMES, "update_frames": UPDATE_FRAMES,
    "fft_frames": FFT_FRAMES, "window": "symmetric_hann",
    "estimation_input": "shared_fft_original_trailing_input",
    "estimator": "log_power_quadratic_peak_global_coherent_power_fraction",
    "minimum_coherent_power_fraction": MIN_COHERENT_POWER_FRACTION,
    "minimum_peak_amplitude": MIN_PEAK_AMPLITUDE,
    "search_half_width_hz_absolute": SEARCH_HALF_WIDTH_HZ,
    "maximum_slew_hz_per_s_absolute": MAX_SLEW_HZ_PER_S,
    "qualification_error_hz_absolute": QUALIFICATION_ERROR_HZ,
    "required_qualified_events": 2, "qualification_hold_frames": HOLD_FRAMES,
    "attack_frames": ATTACK_FRAMES, "release_frames": RELEASE_FRAMES,
    "notch_pole_design_bandwidth_hz": 1.0,
    "mix_envelope": "raised_cosine_k_1_through_L_retarget_from_current_mix",
    "event_timing": "N_consumed_frames_affects_output_N_onward_including_EOF_event",
    "harmonic_phase_lock": False, "confidence_is_probability": False,
    "wanted_tone_protection": "not_guaranteed",
    "output_handling": "validate_range_no_clipping_normalization_or_limiter",
    "replay": "frozen_linear_time_varying_cascade_no_reestimation",
}


def _real(value, name):
    if (not isinstance(value, numbers.Real) or isinstance(value, (bool, np.bool_))
            or not math.isfinite(value)):
        raise ValueError(f"{name} must be a finite real number")
    return float(value)


@dataclass(frozen=True)
class ExtendedBankConfig:
    sample_rate_hz: int = SAMPLE_RATE_HZ
    nominal_hum_hz: float = 60.0
    orders: tuple[int, ...] = tuple(range(1, 33))
    max_attenuation_db: float = 12.0
    block_size_frames: int = 480

    def __post_init__(self):
        if type(self.sample_rate_hz) is not int or self.sample_rate_hz != SAMPLE_RATE_HZ:
            raise ValueError("extended bank requires 48000 Hz")
        nominal = _real(self.nominal_hum_hz, "nominal_hum_hz")
        depth = _real(self.max_attenuation_db, "max_attenuation_db")
        if nominal not in (50.0, 60.0) or not 0 <= depth <= 12:
            raise ValueError("nominal must be 50/60 Hz and depth must be in [0,12] dB")
        if (type(self.block_size_frames) is not int or not 1 <= self.block_size_frames <= 4096):
            raise ValueError("block_size_frames must be an integer in [1,4096]")
        if (type(self.orders) not in (tuple, list) or not 1 <= len(self.orders) <= 32
                or any(type(order) is not int or not 1 <= order <= 32 for order in self.orders)
                or tuple(self.orders) != tuple(sorted(set(self.orders)))):
            raise ValueError("orders must be nonempty sorted unique integers in [1,32]")
        if max(self.orders) * nominal > 2000:
            raise ValueError("selected nominal centers must not exceed 2000 Hz")
        object.__setattr__(self, "orders", tuple(self.orders))
        object.__setattr__(self, "nominal_hum_hz", nominal)
        object.__setattr__(self, "max_attenuation_db", depth)

    def to_dict(self):
        record = asdict(ExtendedBankConfig(**asdict(self)))
        record["orders"] = list(record["orders"])
        return record


def _coefficients(center, depth):
    center, depth = _real(center, "center"), _real(depth, "depth")
    if not 0 < center < SAMPLE_RATE_HZ / 2 or not 0 <= depth <= 12:
        raise ValueError("invalid finite-notch parameters")
    gain = 1 / (1 + math.tan(math.pi / SAMPLE_RATE_HZ))
    a = (1.0, -2 * gain * math.cos(2 * math.pi * center / SAMPLE_RATE_HZ), 2 * gain - 1)
    full = (gain, a[1], gain)
    residual = 10 ** (-depth / 20)
    b = tuple(residual * denominator + (1 - residual) * numerator
              for denominator, numerator in zip(a, full, strict=True))
    return b, a


class _Band:
    def __init__(self, order, config):
        self.order, self.depth = order, config.max_attenuation_db
        self.nominal = order * config.nominal_hum_hz
        self.lower, self.upper = self.nominal - 1, self.nominal + 1
        self.center, self.acquired = self.nominal, False
        self.first_acquired = self.last_qualified = None
        self.consecutive = 0
        self.mix = self.target = self.start_mix = 0.0
        self.ramp_total = self.ramp_position = 0
        self.s1 = self.s2 = 0.0
        self.b, self.a = _coefficients(self.center, self.depth)
        step = SAMPLE_RATE_HZ / FFT_FRAMES
        self.search = np.arange(math.ceil(self.lower / step - 1),
                                math.floor(self.upper / step + 1) + 1, dtype=np.int64)

    def advance(self, frames=1):
        if self.ramp_position < self.ramp_total:
            self.ramp_position = min(self.ramp_total, self.ramp_position + frames)
            proportion = .5 * (1 - math.cos(math.pi * self.ramp_position / self.ramp_total))
            self.mix = self.start_mix + (self.target - self.start_mix) * proportion
            if self.ramp_position == self.ramp_total:
                self.mix = self.target

    def request(self, target):
        if target != self.target:
            self.start_mix, self.target = self.mix, target
            self.ramp_total = ATTACK_FRAMES if target else RELEASE_FRAMES
            self.ramp_position = 0

    def snapshot(self, evidence=None):
        return {"order": self.order, "center_hz": self.center, "acquired": self.acquired,
                "first_acquired_sample": self.first_acquired, "last_qualified_sample": self.last_qualified,
                "consecutive_qualified": self.consecutive, "mix": self.mix, "target_mix": self.target,
                "ramp_start_mix": self.start_mix, "ramp_total_frames": self.ramp_total,
                "ramp_position_frames": self.ramp_position, "b": list(self.b), "a": list(self.a),
                "evidence": evidence}

    def observe(self, evidence, clock):
        if type(evidence) is not dict or set(evidence) != {"estimate_hz", "confidence", "amplitude", "status"}:
            raise ValueError("invalid extended-bank evidence fields")
        confidence = _real(evidence["confidence"], "confidence")
        amplitude = _real(evidence["amplitude"], "amplitude")
        estimate, status = evidence["estimate_hz"], evidence["status"]
        if not 0 <= confidence <= 1 or amplitude < 0 or type(status) is not str:
            raise ValueError("invalid extended-bank evidence bounds")
        if estimate is not None:
            estimate = _real(estimate, "estimate")
            if (not self.lower <= estimate <= self.upper or status != "eligible"
                    or confidence < MIN_COHERENT_POWER_FRACTION or amplitude < MIN_PEAK_AMPLITUDE):
                raise ValueError("eligible estimate lacks declared spectral evidence")
            if not self.acquired:
                self.center, self.acquired, self.first_acquired = estimate, True, clock
            else:
                limit = MAX_SLEW_HZ_PER_S * UPDATE_FRAMES / SAMPLE_RATE_HZ
                self.center += float(np.clip(estimate - self.center, -limit, limit))
            self.b, self.a = _coefficients(self.center, self.depth)
        elif status == "insufficient_energy":
            if amplitude != 0 or confidence != 0:
                raise ValueError("zero-energy evidence must be zero")
        elif status == "insufficient_concentration":
            if amplitude >= MIN_PEAK_AMPLITUDE and confidence >= MIN_COHERENT_POWER_FRACTION:
                raise ValueError("concentration rejection is inconsistent")
        elif status != "out_of_band" or confidence < MIN_COHERENT_POWER_FRACTION or amplitude < MIN_PEAK_AMPLITUDE:
            raise ValueError("missing estimate has inconsistent status")
        qualified = estimate is not None and abs(estimate - self.center) <= QUALIFICATION_ERROR_HZ
        if qualified:
            self.last_qualified, self.consecutive = clock, self.consecutive + 1
            if self.consecutive >= 2:
                self.request(1.0)
        else:
            self.consecutive = 0
            if self.last_qualified is not None and clock - self.last_qualified >= HOLD_FRAMES:
                self.request(0.0)
        return self.snapshot(copy.deepcopy(evidence))

    def process(self, source):
        result = np.empty_like(source)
        b0, b1, b2 = self.b
        _, a1, a2 = self.a
        s1, s2 = self.s1, self.s2
        for index, value in enumerate(source):
            dry = float(value)
            wet = b0 * dry + s1
            s1, s2 = b1 * dry - a1 * wet + s2, b2 * dry - a2 * wet
            self.advance()
            result[index] = dry if self.mix == 0 else wet if self.mix == 1 else dry + self.mix * (wet - dry)
        self.s1, self.s2 = s1, s2
        if not math.isfinite(s1) or not math.isfinite(s2) or not np.isfinite(result).all():
            raise ValueError("extended-bank nonfinite filter state/output")
        return result


def _estimate(band, powers, weighted_power, window_sum):
    if weighted_power <= 0:
        return {"estimate_hz": None, "confidence": 0.0, "amplitude": 0.0, "status": "insufficient_energy"}
    peak = int(band.search[np.argmax(powers[band.search])])
    amplitude = 2 * math.sqrt(float(powers[peak])) / window_sum
    fraction = .5 * amplitude ** 2 / weighted_power
    confidence = float(np.clip(fraction, 0, 1))
    record = {"estimate_hz": None, "confidence": confidence, "amplitude": amplitude,
              "status": "insufficient_concentration"}
    if amplitude < MIN_PEAK_AMPLITUDE or fraction < MIN_COHERENT_POWER_FRACTION:
        return record
    left, mid, right = map(float, np.log(np.maximum(powers[peak - 1:peak + 2], np.finfo(float).tiny)))
    denominator = left - 2 * mid + right
    offset = 0.0 if denominator >= -1e-18 else .5 * (left - right) / denominator
    frequency = (peak + float(np.clip(offset, -1, 1))) * SAMPLE_RATE_HZ / FFT_FRAMES
    if not band.lower - .001 <= frequency <= band.upper + .001:
        record["status"] = "out_of_band"
    else:
        record.update(estimate_hz=float(np.clip(frequency, band.lower, band.upper)), status="eligible")
    return record


def _config(config):
    if config is None:
        return ExtendedBankConfig()
    if type(config) is not ExtendedBankConfig:
        raise ValueError("config must be ExtendedBankConfig")
    return ExtendedBankConfig(**config.to_dict())


def _header(config, frames, bands):
    return {"schema_version": 1, "algorithm": ALGORITHM, "config": config.to_dict(),
            "policy": copy.deepcopy(POLICY), "total_frames": frames, "sample_index": 0,
            "bands": [band.snapshot() for band in bands]}


def _process_segment(source, bands, block):
    output = np.empty_like(source)
    for start in range(0, source.size, block):
        part = source[start:start + block]
        for band in bands:
            part = band.process(part)
        output[start:start + part.size] = part
    return output


def render_extended_bank(samples, config=None):
    """Return validated floats, descriptor and complete frozen-decision trace."""
    source, selected = validate_audio(samples), _config(config)
    bands = [_Band(order, selected) for order in selected.orders]
    trace = [_header(selected, source.size, bands)]
    output = source.copy()
    fft_count = 0
    if selected.max_attenuation_db != 0:
        window = np.hanning(OBSERVATION_FRAMES)
        window_sum, square_sum = float(np.sum(window)), float(np.dot(window, window))
        previous = 0
        for clock in range(OBSERVATION_FRAMES, source.size + 1, UPDATE_FRAMES):
            output[previous:clock] = _process_segment(source[previous:clock], bands, selected.block_size_frames)
            windowed = source[clock - OBSERVATION_FRAMES:clock] * window
            weighted_power = float(np.dot(windowed, windowed) / square_sum)
            powers = np.abs(np.fft.rfft(windowed, n=FFT_FRAMES)) ** 2
            if not math.isfinite(weighted_power) or not np.isfinite(powers).all():
                raise ValueError("nonfinite extended-bank spectral observation")
            snapshots = [band.observe(_estimate(band, powers, weighted_power, window_sum), clock) for band in bands]
            trace.append({"sample_index": clock, "bands": snapshots})
            fft_count += 1
            previous = clock
        output[previous:] = _process_segment(source[previous:], bands, selected.block_size_frames)
    output = validate_audio(output)
    description = {"algorithm": ALGORITHM, "config": selected.to_dict(), "policy": copy.deepcopy(POLICY),
                   "samples_processed": source.size, "shared_original_input_fft_count": fft_count,
                   "processing_bypassed": selected.max_attenuation_db == 0,
                   "maximum_input_peak": float(np.max(np.abs(source))),
                   "maximum_output_peak": float(np.max(np.abs(output))),
                   "bands": [band.snapshot() for band in bands]}
    return output, description, trace


def _same(actual, expected):
    if type(expected) is dict:
        if type(actual) is not dict or actual.keys() != expected.keys():
            raise ValueError("extended-bank trace fields differ")
        for key in expected:
            _same(actual[key], expected[key])
    elif type(expected) is list:
        if type(actual) is not list or len(actual) != len(expected):
            raise ValueError("extended-bank trace list differs")
        for item, reference in zip(actual, expected, strict=True):
            _same(item, reference)
    elif type(expected) is float:
        if not math.isclose(_real(actual, "trace value"), expected, abs_tol=1e-12, rel_tol=0):
            raise ValueError("extended-bank trace state disagrees with causal decisions")
    elif type(actual) is not type(expected) or actual != expected:
        raise ValueError("extended-bank trace value/type differs")


def replay_extended_bank(samples, config, trace):
    """Validate frozen clocks, gates, coefficients and policy; then replay no FFT.

    A trace is not authenticated evidence of the original input's content. Its
    state transitions are checked, but an intentionally self-consistent forged
    set of spectral evidence cannot be disproved without re-estimation.
    """
    if type(config) is not ExtendedBankConfig:
        raise ValueError("replay config must be ExtendedBankConfig")
    source, selected = validate_audio(samples), _config(config)
    if type(trace) is not list or not trace:
        raise ValueError("extended-bank trace must be a complete list")
    clocks = [] if selected.max_attenuation_db == 0 else list(range(OBSERVATION_FRAMES, source.size + 1, UPDATE_FRAMES))
    if len(trace) != 1 + len(clocks):
        raise ValueError("extended-bank trace observation schedule differs")
    validator = [_Band(order, selected) for order in selected.orders]
    _same(trace[0], _header(selected, source.size, validator))
    previous = 0
    for event, clock in zip(trace[1:], clocks, strict=True):
        if type(event) is not dict or set(event) != {"sample_index", "bands"} or type(event["bands"]) is not list:
            raise ValueError("invalid extended-bank observation fields")
        if len(event["bands"]) != len(validator):
            raise ValueError("extended-bank selected band count differs")
        expected = []
        for band, recorded in zip(validator, event["bands"], strict=True):
            if type(recorded) is not dict or "evidence" not in recorded:
                raise ValueError("extended-bank observation lacks evidence")
            band.advance(clock - previous)
            expected.append(band.observe(recorded["evidence"], clock))
        _same(event, {"sample_index": clock, "bands": expected})
        previous = clock
    bands = [_Band(order, selected) for order in selected.orders]
    output, previous = source.copy(), 0
    for event, clock in zip(trace[1:], clocks, strict=True):
        output[previous:clock] = _process_segment(source[previous:clock], bands, selected.block_size_frames)
        for band, record in zip(bands, event["bands"], strict=True):
            band.observe(record["evidence"], clock)
        previous = clock
    if selected.max_attenuation_db != 0:
        output[previous:] = _process_segment(source[previous:], bands, selected.block_size_frames)
    return validate_audio(output)
