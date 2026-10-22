"""Opt-in, independently qualified harmonic notches and frozen-decision replay.

One FFT of ORIGINAL trailing input drives all selected bands. The bands track
independently: they are not harmonically phase locked, and spectral evidence is
not evidence that a tone is unwanted. Each stage is a finite-depth, stable 1 Hz
pole-design notch followed by a causal raised-cosine dry/wet gate. No clipping,
normalization, limiter, lookahead, device I/O or automatic noise semantics occur.

At boundary N, N input samples have already been consumed. A recorded change
first affects output N. Replay freezes these mixture-derived decisions; it does
not re-estimate from isolated clean/noise references. Its linear decomposition
diagnostics describe this one frozen run, not the nonlinear adaptive system in
general, nor a blind reference-free quality score.
"""

from dataclasses import asdict, dataclass
import copy
import json
import math
import numbers

import numpy as np

from .audio import validate_audio


OBSERVATION_FRAMES = 96000
UPDATE_FRAMES = 4800
FFT_FRAMES = 524288
MAX_BLOCK_FRAMES = 4096
POLE_DESIGN_BANDWIDTH_HZ = 1.0
MIN_COHERENT_POWER_FRACTION = 0.02
MIN_PEAK_AMPLITUDE = 1e-7
ESTIMATE_EDGE_TOLERANCE_HZ = 0.001
QUALIFICATION_ERROR_HZ = 0.15
REQUIRED_QUALIFIED_EVENTS = 2
QUALIFICATION_HOLD_FRAMES = 14400
ATTACK_FRAMES = 4800
RELEASE_FRAMES = 12000
ALGORITHM = "selective_independent_harmonic_notches_v1"


def _finite_real(value: object) -> bool:
    return (isinstance(value, numbers.Real) and not isinstance(value, (bool, np.bool_))
            and math.isfinite(value))


def _real(value: object, name: str) -> float:
    if not _finite_real(value):
        raise ValueError(f"{name} must be a finite real number")
    return float(value)


@dataclass(frozen=True)
class HarmonicConfig:
    sample_rate_hz: int = 48000
    nominal_hum_hz: float = 50.0
    max_attenuation_db: float = 12.0
    orders: tuple[int, ...] = (1, 2, 3)
    tracking_half_width_hz: float = 1.0
    max_slew_hz_per_s: float = 0.05
    block_size_frames: int = 480

    def __post_init__(self) -> None:
        if type(self.sample_rate_hz) is not int or self.sample_rate_hz != 48000:
            raise ValueError("harmonic processing supports 48000 Hz only")
        if type(self.block_size_frames) is not int or not 1 <= self.block_size_frames <= MAX_BLOCK_FRAMES:
            raise ValueError("block_size_frames must be an integer in [1, 4096]")
        nominal = _real(self.nominal_hum_hz, "nominal_hum_hz")
        depth = _real(self.max_attenuation_db, "max_attenuation_db")
        width = _real(self.tracking_half_width_hz, "tracking_half_width_hz")
        slew = _real(self.max_slew_hz_per_s, "max_slew_hz_per_s")
        if nominal not in (50.0, 60.0):
            raise ValueError("nominal_hum_hz must be 50 or 60")
        if not 0.0 <= depth <= 12.0:
            raise ValueError("max_attenuation_db must be in [0, 12]")
        if not 0.0 <= width <= 1.0:
            raise ValueError("tracking_half_width_hz must be in [0, 1] base-frequency Hz")
        if not 0.0 <= slew <= 0.05:
            raise ValueError("max_slew_hz_per_s must be in [0, 0.05] base-frequency Hz/s")
        if (not isinstance(self.orders, (tuple, list)) or not 1 <= len(self.orders) <= 6
                or any(type(order) is not int or not 1 <= order <= 6 for order in self.orders)
                or tuple(self.orders) != tuple(sorted(set(self.orders)))):
            raise ValueError("orders must be sorted, unique integers in [1, 6], with 1..6 selected bands")
        object.__setattr__(self, "orders", tuple(self.orders))
        for name, value in (("nominal_hum_hz", nominal), ("max_attenuation_db", depth),
                            ("tracking_half_width_hz", width), ("max_slew_hz_per_s", slew)):
            object.__setattr__(self, name, value)

    def to_dict(self) -> dict:
        validated = asdict(HarmonicConfig(**asdict(self)))
        validated["orders"] = list(validated["orders"])
        return validated


def _coefficients(frequency_hz: float, attenuation_db: float) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Own finite-depth form of the same full-notch pole design as fixed M2."""
    if not _finite_real(frequency_hz) or not 0.0 < frequency_hz < 24000.0:
        raise ValueError("filter center must be finite and below Nyquist")
    if not _finite_real(attenuation_db) or not 0.0 <= attenuation_db <= 12.0:
        raise ValueError("filter attenuation must be in [0, 12]")
    gain = 1.0 / (1.0 + math.tan(math.pi * POLE_DESIGN_BANDWIDTH_HZ / 48000.0))
    middle = -2.0 * gain * math.cos(2.0 * math.pi * frequency_hz / 48000.0)
    a = (1.0, middle, 2.0 * gain - 1.0)
    full = (gain, middle, gain)
    center_gain = 10.0 ** (-attenuation_db / 20.0)
    b = tuple(center_gain * denominator + (1.0 - center_gain) * numerator
              for denominator, numerator in zip(a, full, strict=True))
    return b, a


def _safe(value: object):
    if value is None or type(value) in (bool, int, str):
        return value
    if _finite_real(value):
        return float(value)
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    return None


class _Band:
    def __init__(self, order: int, config: dict) -> None:
        self.order = order
        self.nominal = order * config["nominal_hum_hz"]
        self.lower = self.nominal - order * config["tracking_half_width_hz"]
        self.upper = self.nominal + order * config["tracking_half_width_hz"]
        self.slew = order * config["max_slew_hz_per_s"]
        self.depth = config["max_attenuation_db"]
        self.center = self.nominal
        self.b, self.a = _coefficients(self.center, self.depth)
        self.state1 = self.state2 = 0.0
        self.acquired = False
        self.first_acquired_sample = None
        self.maximum_slew = 0.0
        self.mix = self.target = self.ramp_start = 0.0
        self.ramp_total = self.ramp_position = 0
        self.last_qualified = None
        self.consecutive_qualified = 0
        step = 48000.0 / FFT_FRAMES
        self.search = np.arange(max(1, math.ceil(self.lower / step - 1.0)),
                                min(FFT_FRAMES // 2 - 1, math.floor(self.upper / step + 1.0)) + 1,
                                dtype=np.int64)

    def validate(self, clock: int) -> None:
        if not _finite_real(self.center) or not self.lower - 1e-12 <= self.center <= self.upper + 1e-12:
            raise ValueError("internal harmonic center escaped its selected band")
        if type(self.acquired) is not bool:
            raise ValueError("internal acquisition flag must be bool")
        if (not _finite_real(self.maximum_slew) or self.maximum_slew < 0.0
                or self.maximum_slew > self.slew + 1e-10):
            raise ValueError("internal harmonic slew is invalid")
        for value in (self.state1, self.state2):
            if not _finite_real(value):
                raise ValueError("internal harmonic filter state is nonfinite")
        for value in (self.mix, self.ramp_start):
            if not _finite_real(value) or not 0.0 <= value <= 1.0:
                raise ValueError("internal harmonic mix must be finite in [0, 1]")
        if not _finite_real(self.target) or self.target not in (0.0, 1.0):
            raise ValueError("internal harmonic target must be 0 or 1")
        if type(self.ramp_total) is not int or self.ramp_total not in (0, ATTACK_FRAMES, RELEASE_FRAMES):
            raise ValueError("internal harmonic ramp duration is invalid")
        if type(self.ramp_position) is not int or not 0 <= self.ramp_position <= self.ramp_total:
            raise ValueError("internal harmonic ramp position is invalid")
        expected_events = 0 if clock < OBSERVATION_FRAMES else 1 + (clock - OBSERVATION_FRAMES) // UPDATE_FRAMES
        if type(self.consecutive_qualified) is not int or not 0 <= self.consecutive_qualified <= expected_events:
            raise ValueError("internal qualification count is invalid")
        for value in (self.first_acquired_sample, self.last_qualified):
            if value is not None and (type(value) is not int or not OBSERVATION_FRAMES <= value <= clock
                                      or (value - OBSERVATION_FRAMES) % UPDATE_FRAMES != 0):
                raise ValueError("internal harmonic observation clock is invalid")
        if self.acquired != (self.first_acquired_sample is not None):
            raise ValueError("internal acquisition history is inconsistent")
        if self.consecutive_qualified and self.last_qualified is None:
            raise ValueError("internal qualification history is inconsistent")
        if (not isinstance(self.a, tuple) or not isinstance(self.b, tuple) or len(self.a) != 3 or len(self.b) != 3
                or not all(_finite_real(value) for value in (*self.a, *self.b))):
            raise ValueError("internal harmonic coefficients are invalid")
        _, a1, a2 = self.a
        if self.a[0] != 1.0 or not (abs(a2) < 1.0 and 1.0 + a1 + a2 > 0.0 and 1.0 - a1 + a2 > 0.0):
            raise ValueError("internal harmonic poles are not strictly stable")
        if (self.b, self.a) != _coefficients(self.center, self.depth):
            raise ValueError("internal coefficients disagree with the selected center")

    def advance_mix(self, count: int = 1) -> None:
        if self.ramp_position < self.ramp_total:
            self.ramp_position = min(self.ramp_total, self.ramp_position + count)
            if self.ramp_position == self.ramp_total:
                self.mix = self.target
            else:
                proportion = 0.5 * (1.0 - math.cos(math.pi * self.ramp_position / self.ramp_total))
                self.mix = self.ramp_start + (self.target - self.ramp_start) * proportion

    def process(self, source: np.ndarray) -> np.ndarray:
        output = np.empty_like(source)
        b0, b1, b2 = self.b
        _, a1, a2 = self.a
        s1, s2 = self.state1, self.state2
        for index, value in enumerate(source):
            dry = float(value)
            wet = b0 * dry + s1
            s1 = b1 * dry - a1 * wet + s2
            s2 = b2 * dry - a2 * wet
            self.advance_mix()
            output[index] = dry if self.mix == 0.0 else (wet if self.mix == 1.0 else dry + self.mix * (wet - dry))
        self.state1, self.state2 = s1, s2
        if not (math.isfinite(s1) and math.isfinite(s2) and np.isfinite(output).all()):
            raise ValueError("harmonic filter produced nonfinite output or state")
        return output

    def request(self, target: float) -> bool:
        if target == self.target:
            return False
        self.ramp_start = self.mix
        self.target = target
        self.ramp_total = ATTACK_FRAMES if target == 1.0 else RELEASE_FRAMES
        self.ramp_position = 0
        return True

    def snapshot(self, *, estimate=None, confidence=0.0, status="initial_dry", previous=None,
                 initial=False, qualified=False, changed=False, reason="initial_dry") -> dict:
        return {
            "order": self.order, "nominal_frequency_hz": self.nominal,
            "selected_lower_hz": self.lower, "selected_upper_hz": self.upper,
            "estimated_frequency_hz": estimate, "filter_frequency_hz": self.center,
            "previous_filter_frequency_hz": self.center if previous is None else previous,
            "confidence": confidence, "confidence_is_probability": False,
            "status": status, "initial_acquisition": initial, "acquired": self.acquired,
            "first_acquired_sample": self.first_acquired_sample,
            "maximum_post_acquisition_slew_hz_per_s": self.maximum_slew,
            "qualified": qualified, "last_qualified_sample": self.last_qualified,
            "consecutive_qualified": self.consecutive_qualified,
            "mix_before": self.mix, "mix": self.mix, "target_mix": self.target,
            "ramp_start_mix": self.ramp_start, "ramp_total_frames": self.ramp_total,
            "ramp_position": self.ramp_position, "target_changed": changed, "reason": reason,
            "a": list(self.a), "b": list(self.b),
        }

    def observe(self, estimate: float | None, confidence: float, status: str, clock: int) -> dict:
        if not _finite_real(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("estimator confidence must be finite in [0, 1]")
        if type(status) is not str or not status:
            raise ValueError("estimator status must be a nonempty string")
        if estimate is not None and (not _finite_real(estimate) or not self.lower <= estimate <= self.upper):
            raise ValueError("estimator frequency escaped selected harmonic band")
        if estimate is not None and confidence < MIN_COHERENT_POWER_FRACTION:
            raise ValueError("eligible harmonic estimate lacks the declared concentration evidence")
        previous = self.center
        initial = False
        if estimate is not None:
            estimate = float(estimate)
            if not self.acquired:
                self.center = estimate
                self.acquired = True
                self.first_acquired_sample = clock
                initial = True
                status = "initial_acquisition"
            else:
                elapsed = UPDATE_FRAMES / 48000.0
                limit = self.slew * elapsed
                difference = estimate - self.center
                self.center += float(np.clip(difference, -limit, limit))
                self.maximum_slew = max(self.maximum_slew, abs(self.center - previous) / elapsed)
                status = "tracking" if abs(difference) <= limit else "tracking_slew_limited"
            self.b, self.a = _coefficients(self.center, self.depth)
        qualified = estimate is not None and abs(estimate - self.center) <= self.order * QUALIFICATION_ERROR_HZ
        if qualified:
            self.last_qualified = clock
            self.consecutive_qualified += 1
            changed = self.request(1.0) if self.consecutive_qualified >= REQUIRED_QUALIFIED_EVENTS else False
            reason = "qualification_pending" if self.consecutive_qualified < REQUIRED_QUALIFIED_EVENTS else "qualified_wet_request" if changed else "qualified_wet_hold"
        else:
            self.consecutive_qualified = 0
            expired = self.last_qualified is not None and clock - self.last_qualified >= QUALIFICATION_HOLD_FRAMES
            changed = self.request(0.0) if expired else False
            reason = "qualification_expired_dry_request" if expired else "unqualified_hold"
        return self.snapshot(estimate=estimate, confidence=float(confidence), status=status, previous=previous,
                             initial=initial, qualified=qualified, changed=changed, reason=reason)


class SelectiveHarmonicSuppressor:
    """Causal block processor. Invalid public input never mutates state.

    Internal failures latch and raise; the failed call never returns a successful
    dry fallback. Reset is required. Floating overshoots are not silently clipped.
    """

    def __init__(self, config: HarmonicConfig) -> None:
        if type(config) is not HarmonicConfig:
            raise ValueError("config must be a validated HarmonicConfig")
        self._config = config.to_dict()
        self.reset()

    def reset(self) -> None:
        self._bands = [_Band(order, self._config) for order in self._config["orders"]]
        self._samples_processed = 0
        self._next_update = OBSERVATION_FRAMES
        self._trace = []
        self._fault = None
        self._fft_count = 0
        self._maximum_input_peak = self._maximum_output_peak = 0.0
        self._history_position = 0
        self._history = None
        self._window = None
        self._window_sum = self._window_square_sum = None
        if self._config["max_attenuation_db"] != 0.0:
            self._history = np.zeros(OBSERVATION_FRAMES, dtype=np.float64)
            self._window = np.hanning(OBSERVATION_FRAMES)
            self._window_sum = float(np.sum(self._window))
            self._window_square_sum = float(np.dot(self._window, self._window))
            self._trace.append(self._event([band.snapshot() for band in self._bands], "initial_dry"))

    def _event(self, bands: list[dict], kind: str) -> dict:
        return {"schema_version": 1, "algorithm": ALGORITHM, "sample_index": self._samples_processed,
                "time_seconds": self._samples_processed / 48000.0, "event_kind": kind,
                "config": copy.deepcopy(self._config), "bands": bands}

    def get_trace(self) -> list[dict]:
        result = copy.deepcopy(self._trace)
        for event in result:
            event["trace_total_frames"] = self._samples_processed
        return result

    def _current_band_descriptions(self) -> list[dict]:
        """Separate current numerical state from evidence at the last boundary.

        snapshot() defaults describe an initial trace EVENT, not a live status.
        Never mislabel an acquired/wet current state as initial_dry or invent a
        fresh confidence/estimate between observations. Retain the real observed
        event in a clearly named, independently copied last_observation field.
        """
        latest = next((event for event in reversed(self._trace)
                       if event["event_kind"] == "observation"), None)
        event_only = ("estimated_frequency_hz", "confidence", "confidence_is_probability", "status",
                      "qualified", "reason", "previous_filter_frequency_hz", "initial_acquisition",
                      "target_changed", "mix_before")
        descriptions = []
        for index, band in enumerate(self._bands):
            current = band.snapshot()
            for name in event_only:
                current.pop(name)
            current["last_observation"] = None
            if latest is not None:
                current["last_observation"] = {
                    "sample_index": latest["sample_index"], "time_seconds": latest["time_seconds"],
                    **copy.deepcopy(latest["bands"][index]),
                }
            descriptions.append(current)
        return descriptions

    def describe(self) -> dict:
        description = {
            "algorithm": ALGORITHM, "config": copy.deepcopy(self._config),
            "samples_processed": self._samples_processed, "processing_bypassed": self._config["max_attenuation_db"] == 0.0,
            "trace_event_count": len(self._trace), "shared_original_input_fft_count": self._fft_count,
            "fault": copy.deepcopy(self._fault),
            "maximum_input_peak": self._maximum_input_peak, "maximum_output_peak": self._maximum_output_peak,
            "bands": self._current_band_descriptions(),
            "policy": {
                "observation_frames": OBSERVATION_FRAMES, "update_frames": UPDATE_FRAMES, "fft_frames": FFT_FRAMES,
                "estimation_input": "one_shared_fft_of_original_input_not_cascade_outputs",
                "estimation_method": "hann_fft_log_power_quadratic_peak_independent_per_band",
                "harmonic_phase_lock": False, "minimum_coherent_power_fraction": MIN_COHERENT_POWER_FRACTION,
                "minimum_peak_amplitude": MIN_PEAK_AMPLITUDE, "estimate_edge_tolerance_hz": ESTIMATE_EDGE_TOLERANCE_HZ,
                "qualification_error_hz_per_order": QUALIFICATION_ERROR_HZ,
                "required_qualified_events": REQUIRED_QUALIFIED_EVENTS,
                "qualification_hold_frames": QUALIFICATION_HOLD_FRAMES,
                "attack_frames": ATTACK_FRAMES, "release_frames": RELEASE_FRAMES,
                "pole_design_bandwidth_hz": POLE_DESIGN_BANDWIDTH_HZ,
                "envelope": "raised_cosine_k_1_through_L_retarget_from_current_mix",
                "event_timing": "N_consumed_frames_affects_output_N_onward",
                "selection": "manual_orders_1_through_6_independent_spectral_evidence",
                "confidence": "heuristic_not_probability_or_noise_semantics",
                "wanted_tone_protection": "not_guaranteed_same_frequency_wanted_content_is_suppressed",
                "internal_failure": "latched_raise_no_successful_fallback_until_reset",
                "replay": "frozen_mixture_decisions_no_reestimation_linear_time_varying_cascade",
                "output_peak_handling": "measured_only_no_clipping_normalization_or_limiter",
                "physical_safety_or_inaudibility_guarantee": False,
            },
        }
        return _safe(description) if self._fault is not None else description

    def _validate_state(self) -> None:
        if type(self._samples_processed) is not int or self._samples_processed < 0:
            raise ValueError("internal sample clock is invalid")
        if type(self._next_update) is not int or self._next_update < OBSERVATION_FRAMES:
            raise ValueError("internal observation clock is invalid")
        if (not isinstance(self._bands, list) or len(self._bands) != len(self._config["orders"])
                or [band.order for band in self._bands] != self._config["orders"]):
            raise ValueError("internal selected bands are inconsistent")
        expected = 0 if self._samples_processed < OBSERVATION_FRAMES else 1 + (self._samples_processed - OBSERVATION_FRAMES) // UPDATE_FRAMES
        bypass = self._config["max_attenuation_db"] == 0.0
        if type(self._fft_count) is not int or self._fft_count != (0 if bypass else expected):
            raise ValueError("internal shared FFT clock is inconsistent")
        if not isinstance(self._trace, list) or len(self._trace) != (0 if bypass else expected + 1):
            raise ValueError("internal trace clock is inconsistent")
        if not bypass and self._next_update != OBSERVATION_FRAMES + expected * UPDATE_FRAMES:
            raise ValueError("internal next observation is inconsistent")
        for band in self._bands:
            band.validate(self._samples_processed)
            if bypass and (band.mix != 0.0 or band.target != 0.0 or band.acquired):
                raise ValueError("zero-depth processor must remain dry and unacquired")
        for peak in (self._maximum_input_peak, self._maximum_output_peak):
            if not _finite_real(peak) or peak < 0.0:
                raise ValueError("internal measured peak is invalid")
        if not bypass and (not isinstance(self._history, np.ndarray) or self._history.shape != (OBSERVATION_FRAMES,)
                           or not np.isfinite(self._history).all()
                           or type(self._history_position) is not int or not 0 <= self._history_position < OBSERVATION_FRAMES):
            raise ValueError("internal trailing history is invalid")

    def _append_history(self, block: np.ndarray) -> None:
        position = self._history_position
        first = min(block.size, OBSERVATION_FRAMES - position)
        self._history[position:position + first] = block[:first]
        if first < block.size:
            self._history[:block.size - first] = block[first:]
        self._history_position = (position + block.size) % OBSERVATION_FRAMES

    def _estimate_band(self, band: _Band, powers: np.ndarray, weighted_power: float) -> tuple:
        if weighted_power <= 0.0:
            return None, 0.0, "insufficient_energy_hold"
        peak = int(band.search[int(np.argmax(powers[band.search]))])
        amplitude = 2.0 * math.sqrt(float(powers[peak])) / self._window_sum
        fraction = 0.5 * amplitude**2 / weighted_power
        confidence = float(np.clip(fraction, 0.0, 1.0))
        if amplitude < MIN_PEAK_AMPLITUDE or fraction < MIN_COHERENT_POWER_FRACTION:
            return None, confidence, "insufficient_concentration_hold"
        left, middle, right = map(float, np.log(np.maximum(powers[peak - 1:peak + 2], np.finfo(float).tiny)))
        denominator = left - 2.0 * middle + right
        offset = 0.0 if denominator >= -1e-18 else 0.5 * (left - right) / denominator
        estimate = (peak + float(np.clip(offset, -1.0, 1.0))) * 48000.0 / FFT_FRAMES
        if not band.lower - ESTIMATE_EDGE_TOLERANCE_HZ <= estimate <= band.upper + ESTIMATE_EDGE_TOLERANCE_HZ:
            return None, confidence, "peak_outside_selected_band_hold"
        return float(np.clip(estimate, band.lower, band.upper)), confidence, "eligible_peak"

    def _observe(self) -> None:
        position = self._history_position
        ordered = np.concatenate((self._history[position:], self._history[:position]))
        windowed = ordered * self._window
        weighted_power = float(np.dot(windowed, windowed) / self._window_square_sum)
        spectrum = np.fft.rfft(windowed, n=FFT_FRAMES)
        powers = np.abs(spectrum) ** 2
        if not math.isfinite(weighted_power) or not np.isfinite(powers).all():
            raise ValueError("shared spectral observation is nonfinite")
        decisions = []
        for band in self._bands:
            estimate, confidence, status = self._estimate_band(band, powers, weighted_power)
            decisions.append(band.observe(estimate, confidence, status, self._samples_processed))
        self._trace.append(self._event(decisions, "observation"))
        self._fft_count += 1

    def _process_part(self, part: np.ndarray) -> np.ndarray:
        output = part
        for band in self._bands:
            output = band.process(output)
        return output

    def process_block(self, samples: object) -> np.ndarray:
        try:
            source = np.asarray(samples)
        except (ValueError, TypeError) as exc:
            raise ValueError("block must be a real mono vector") from exc
        if source.ndim != 1 or not 1 <= source.size <= MAX_BLOCK_FRAMES:
            raise ValueError("block must contain 1..4096 mono samples")
        block = validate_audio(source)
        if self._fault is not None:
            raise ValueError("harmonic processor is faulted; reset is required")
        output = np.empty_like(block)
        offset = 0
        bypass = self._config["max_attenuation_db"] == 0.0
        try:
            self._validate_state()
            while offset < block.size:
                count = block.size - offset if bypass else min(block.size - offset, self._next_update - self._samples_processed)
                part = block[offset:offset + count]
                result = part.copy() if bypass else self._process_part(part)
                if not np.isfinite(result).all():
                    raise ValueError("harmonic cascade produced nonfinite output")
                output[offset:offset + count] = result
                if not bypass:
                    self._append_history(part)
                self._maximum_input_peak = max(self._maximum_input_peak, float(np.max(np.abs(part))))
                self._maximum_output_peak = max(self._maximum_output_peak, float(np.max(np.abs(result))))
                self._samples_processed += count
                offset += count
                if not bypass and self._samples_processed == self._next_update:
                    self._observe()
                    self._next_update += UPDATE_FRAMES
            self._validate_state()
        except Exception as exc:
            self._fault = {"sample_index": _safe(self._samples_processed), "reason": "internal_processing_failure",
                           "exception_type": type(exc).__name__}
            raise ValueError("internal harmonic processing failed; reset is required") from exc
        return output


def render_harmonic(samples: object, config: HarmonicConfig | None = None) -> tuple[np.ndarray, dict, list[dict]]:
    """Fresh causal render of <=60 s. Returns unnormalized floats, descriptor, trace."""
    source = validate_audio(samples)
    selected = HarmonicConfig() if config is None else config
    engine = SelectiveHarmonicSuppressor(selected)
    output = np.empty_like(source)
    block = selected.block_size_frames
    for start in range(0, source.size, block):
        output[start:start + block] = engine.process_block(source[start:start + block])
    description, trace = engine.describe(), engine.get_trace()
    json.dumps((description, trace), allow_nan=False)
    return output, description, trace


def _same_record(actual: object, expected: object, name: str = "trace") -> None:
    """Strict structure, primitive types and finite numerics; no NaN/bool coercion."""
    if isinstance(expected, dict):
        if type(actual) is not dict or actual.keys() != expected.keys():
            raise ValueError(f"{name} has invalid fields")
        for key in expected:
            _same_record(actual[key], expected[key], f"{name}.{key}")
    elif isinstance(expected, list):
        if type(actual) is not list or len(actual) != len(expected):
            raise ValueError(f"{name} has invalid list structure")
        for index, (item, reference) in enumerate(zip(actual, expected, strict=True)):
            _same_record(item, reference, f"{name}[{index}]")
    elif type(expected) is float:
        if not _finite_real(actual) or not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"{name} is inconsistent with frozen causal decisions")
    elif type(actual) is not type(expected) or actual != expected:
        raise ValueError(f"{name} has an invalid value or type")


def _apply_recorded_observation(engine: SelectiveHarmonicSuppressor, event: dict) -> dict:
    if type(event) is not dict or type(event.get("bands")) is not list or len(event["bands"]) != len(engine._bands):
        raise ValueError("invalid harmonic replay bands")
    decisions = []
    for band, recorded in zip(engine._bands, event["bands"], strict=True):
        if type(recorded) is not dict:
            raise ValueError("invalid harmonic replay band record")
        try:
            estimate, confidence, status = (recorded["estimated_frequency_hz"], recorded["confidence"], recorded["status"])
        except KeyError as exc:
            raise ValueError("incomplete harmonic replay evidence") from exc
        if estimate is None and status not in ("insufficient_energy_hold", "insufficient_concentration_hold", "peak_outside_selected_band_hold"):
            raise ValueError("invalid missing-estimate replay status")
        decisions.append(band.observe(estimate, confidence, status, engine._samples_processed))
    return engine._event(decisions, "observation")


def _validate_replay_trace(trace: object, config: HarmonicConfig, frames: int) -> list[dict]:
    if type(trace) is not list:
        raise ValueError("harmonic replay trace must be a list")
    bypass = config.max_attenuation_db == 0.0
    clocks = [] if bypass else [0, *range(OBSERVATION_FRAMES, frames + 1, UPDATE_FRAMES)]
    if len(trace) != len(clocks):
        raise ValueError("harmonic replay requires the complete exact observation schedule")
    if bypass:
        return []
    validator = SelectiveHarmonicSuppressor(config)
    expected_initial = validator.get_trace()[0]
    expected_initial["trace_total_frames"] = frames
    _same_record(trace[0], expected_initial)
    previous = 0
    for event, clock in zip(trace[1:], clocks[1:], strict=True):
        if type(event) is not dict or type(event.get("sample_index")) is not int or event["sample_index"] != clock:
            raise ValueError("harmonic replay observation clock is invalid")
        for band in validator._bands:
            band.advance_mix(clock - previous)
        validator._samples_processed = clock
        expected = _apply_recorded_observation(validator, event)
        expected["trace_total_frames"] = frames
        _same_record(event, expected)
        for band in validator._bands:
            band.validate(clock)
        previous = clock
    return copy.deepcopy(trace)


def render_harmonic_replay(samples: object, config: HarmonicConfig, trace: object) -> np.ndarray:
    """Replay validated frozen mixture decisions without FFT or re-estimation.

    The trace must match this configuration AND exact frame count, including an
    observation exactly at EOF. References must be the same decoded PCM boundary
    as the mixture. Changing the references does not prove adaptive decisions
    would have remained the same on those references in a separate run.
    """
    source = validate_audio(samples)
    if type(config) is not HarmonicConfig:
        raise ValueError("config must be a validated HarmonicConfig")
    selected = HarmonicConfig(**config.to_dict())
    events = _validate_replay_trace(trace, selected, source.size)
    if selected.max_attenuation_db == 0.0:
        return source.copy()
    engine = SelectiveHarmonicSuppressor(selected)
    output = np.empty_like(source)
    offset = 0
    for event in events[1:]:
        stop = event["sample_index"]
        output[offset:stop] = engine._process_part(source[offset:stop])
        engine._samples_processed = stop
        _apply_recorded_observation(engine, event)
        offset = stop
    if offset < source.size:
        output[offset:] = engine._process_part(source[offset:])
    if not np.isfinite(output).all():
        raise ValueError("harmonic replay produced nonfinite output")
    return output
