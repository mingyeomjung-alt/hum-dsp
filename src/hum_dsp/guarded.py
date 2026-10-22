"""M4 causal qualification gate and click-reducing dry/wet transitions.

The frozen M3 tracker still sees every input sample and its filter is unchanged.
This wrapper does not decide whether a tone is unwanted, guarantee inaudibility,
limit peaks, normalize gain, or hide internal processing failures. In particular,
wanted content at the selected frequency remains confusable with hum.

At a boundary N, events describe the state AFTER consuming N frames and BEFORE
output frame N. A ramp starts at its current mix, at position zero. Output frame
N uses k=1 in start + (target-start)*(1-cos(pi*k/L))/2. Position advances once per
sample, k is capped at L, and the Lth frame reaches the target exactly. Thus an
event never changes earlier output, and retargeting never jumps the envelope.
Unchanged targets do not restart a ramp. A soft OFF takes 12000 frames (250 ms),
not immediate muting; the raw tracker remains warm. Re-enabling requires two
fresh qualified observations. Reset restores the constructor's enabled flag.
"""

import copy
import json
import math
import numbers

import numpy as np

from .audio import validate_audio
from .config import ProcessorConfig
from .filter import MAX_BLOCK_FRAMES
from .tracking import OBSERVATION_FRAMES, UPDATE_FRAMES, TrackedHumSuppressor


QUALIFICATION_ERROR_HZ = 0.15
REQUIRED_QUALIFIED_EVENTS = 2
QUALIFICATION_HOLD_FRAMES = 14400
ATTACK_FRAMES = 4800
RELEASE_FRAMES = 12000
RAW_BAND_TOLERANCE_HZ = 1e-12


def _finite_real(value: object) -> bool:
    return (isinstance(value, numbers.Real) and not isinstance(value, (bool, np.bool_))
            and math.isfinite(value))


def _safe_audit_value(value: object):
    """Keep failure diagnostics JSON-safe without hiding a failure as audio."""
    if value is None or type(value) in (bool, int, str):
        return value
    if _finite_real(value):
        return float(value)
    if isinstance(value, dict):
        return {str(key): _safe_audit_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_audit_value(item) for item in value]
    return None


class GuardedHumSuppressor:
    """Streaming M4 wrapper, causal and independent of caller partitioning.

    Public input validation is transactional: invalid blocks do not change any
    state. An internal tracker failure is different: it latches an explicit fault,
    returns no output for that call, and prohibits further processing until reset.
    Successful earlier parts of the failed call may have consumed input; describe
    reports those counts. This is NOT a successful fallback to unprocessed audio.
    """

    def __init__(self, config: ProcessorConfig, *, enabled: bool = True) -> None:
        if type(config) is not ProcessorConfig:
            raise ValueError("config must be a validated ProcessorConfig")
        if type(enabled) is not bool:
            raise ValueError("enabled must be a bool")
        self._config = ProcessorConfig(**config.to_dict()).to_dict()
        self._initial_enabled = enabled
        self.reset()

    def reset(self) -> None:
        """Clear fault/history, restart dry and restore constructor enabled state."""
        self._raw = TrackedHumSuppressor(ProcessorConfig(**self._config))
        self._requested_enabled = self._initial_enabled
        self._samples_processed = 0
        self._next_update = OBSERVATION_FRAMES
        self._mix = 0.0
        self._target_mix = 0.0
        self._ramp_start_mix = 0.0
        self._ramp_total_frames = 0
        self._ramp_position = 0
        self._last_qualified_sample = None
        self._consecutive_qualified = 0
        self._tracking_trace = []
        self._guard_trace = []
        self._fault = None
        self._maximum_input_peak = 0.0
        self._maximum_raw_peak = 0.0
        self._maximum_output_peak = 0.0
        self._append_event("initial_dry", qualified=False, target_changed=False, mix_before=0.0)

    def get_tracking_trace(self) -> list[dict]:
        """Independent, unmodified M3 event snapshots, not guard decisions."""
        return copy.deepcopy(self._tracking_trace)

    def get_guard_trace(self) -> list[dict]:
        """Independent JSON-safe boundary decisions, including initial/toggle events."""
        return copy.deepcopy(self._guard_trace)

    def describe(self) -> dict:
        try:
            raw_description = self._raw.describe()
        except Exception as exc:
            if self._fault is None:
                raise
            raw_description = {"unavailable_due_to_fault": type(exc).__name__}
        description = {
            "algorithm": "qualified_m3_raised_cosine_dry_wet_guard_v1",
            "config": dict(self._config),
            "samples_processed": self._samples_processed,
            "requested_enabled": self._requested_enabled,
            "processing_bypassed": self._config["max_attenuation_db"] == 0.0,
            "mix": self._mix,
            "target_mix": self._target_mix,
            "ramp_start_mix": self._ramp_start_mix,
            "ramp_total_frames": self._ramp_total_frames,
            "ramp_position": self._ramp_position,
            "last_qualified_sample": self._last_qualified_sample,
            "consecutive_qualified": self._consecutive_qualified,
            "tracking_event_count": len(self._tracking_trace),
            "guard_event_count": len(self._guard_trace),
            "fault": copy.deepcopy(self._fault),
            "maximum_input_peak": self._maximum_input_peak,
            "maximum_raw_peak": self._maximum_raw_peak,
            "maximum_output_peak": self._maximum_output_peak,
            "peak_note": "Measured absolute peaks only; no clipping, limiting, gain normalization or audibility guarantee",
            "policy": {
                "initial_mix": 0.0,
                "qualification_frequency_error_hz": QUALIFICATION_ERROR_HZ,
                "required_consecutive_qualified_events": REQUIRED_QUALIFIED_EVENTS,
                "qualification_hold_frames": QUALIFICATION_HOLD_FRAMES,
                "qualification_hold_evaluated_at": "global_M3_observation_boundaries_only",
                "attack_frames": ATTACK_FRAMES,
                "release_frames": RELEASE_FRAMES,
                "envelope": "raised_cosine_k_1_through_L_retarget_from_current_mix",
                "envelope_sample_convention": "At boundary N, frame N uses k=1; Lth emitted frame reaches target exactly",
                "soft_off": "250_ms_release_not_emergency_mute_raw_tracker_remains_warm",
                "reenable": "requires_two_fresh_qualified_events",
                "internal_failure": "latched_fault_no_output_return_until_reset",
                "wanted_tone_protection": "not_guaranteed_same_frequency_content_is_confusable",
            },
            "raw_tracker": raw_description,
        }
        return _safe_audit_value(description) if self._fault is not None else description

    def _validate_state(self) -> None:
        """Internal invariants, checked inside the explicit failure boundary."""
        if type(self._requested_enabled) is not bool or type(self._initial_enabled) is not bool:
            raise ValueError("internal enabled state must be bool")
        for value in (self._mix, self._ramp_start_mix):
            if not _finite_real(value) or not 0.0 <= value <= 1.0:
                raise ValueError("internal mix state must be finite in [0, 1]")
        if not _finite_real(self._target_mix) or self._target_mix not in (0.0, 1.0):
            raise ValueError("internal target mix must be 0 or 1")
        if type(self._ramp_total_frames) is not int or self._ramp_total_frames not in (0, ATTACK_FRAMES, RELEASE_FRAMES):
            raise ValueError("internal ramp length is invalid")
        if type(self._ramp_position) is not int or not 0 <= self._ramp_position <= self._ramp_total_frames:
            raise ValueError("internal ramp position is invalid")
        if type(self._samples_processed) is not int or self._samples_processed < 0:
            raise ValueError("internal sample counter is invalid")
        if type(self._next_update) is not int or self._next_update < OBSERVATION_FRAMES:
            raise ValueError("internal update counter is invalid")
        if not isinstance(self._tracking_trace, list) or not isinstance(self._guard_trace, list):
            raise ValueError("internal audit container is invalid")
        bypass = self._config["max_attenuation_db"] == 0.0
        expected_events = (0 if bypass or self._samples_processed < OBSERVATION_FRAMES
                           else 1 + (self._samples_processed - OBSERVATION_FRAMES) // UPDATE_FRAMES)
        expected_update = OBSERVATION_FRAMES + expected_events * UPDATE_FRAMES
        if len(self._tracking_trace) != expected_events or (not bypass and self._next_update != expected_update):
            raise ValueError("internal observation clock is inconsistent")
        if (type(self._consecutive_qualified) is not int or self._consecutive_qualified < 0
                or self._consecutive_qualified > expected_events):
            raise ValueError("internal qualification counter is invalid")
        last = self._last_qualified_sample
        if last is not None and (type(last) is not int or not OBSERVATION_FRAMES <= last <= self._samples_processed
                                 or (last - OBSERVATION_FRAMES) % UPDATE_FRAMES != 0):
            raise ValueError("internal last-qualified clock is invalid")
        if self._consecutive_qualified > 0 and last is None:
            raise ValueError("internal qualification streak has no observation")
        if not self._requested_enabled and (last is not None or self._consecutive_qualified != 0 or self._target_mix != 0.0):
            raise ValueError("disabled state must not retain an active qualification")
        if bypass and (self._mix != 0.0 or self._target_mix != 0.0 or self._ramp_start_mix != 0.0):
            raise ValueError("zero-depth state must remain exactly dry")
        for value in (self._maximum_input_peak, self._maximum_raw_peak, self._maximum_output_peak):
            if not _finite_real(value) or value < 0.0:
                raise ValueError("internal measured peak is invalid")

    def _validate_raw_event(self, event: object) -> None:
        if type(event) is not dict:
            raise ValueError("raw tracker event must be a dictionary")
        json.dumps(event, allow_nan=False)
        if type(event["sample_index"]) is not int or event["sample_index"] != self._samples_processed:
            raise ValueError("raw tracker event has invalid boundary")
        lower = self._config["nominal_hum_hz"] - self._config["tracking_half_width_hz"]
        upper = self._config["nominal_hum_hz"] + self._config["tracking_half_width_hz"]
        for name in ("filter_frequency_hz", "estimated_frequency_hz"):
            value = event[name]
            if name == "estimated_frequency_hz" and value is None:
                continue
            if (not _finite_real(value)
                    or not lower - RAW_BAND_TOLERANCE_HZ <= value <= upper + RAW_BAND_TOLERANCE_HZ):
                raise ValueError("raw tracker event frequency is outside the configured band")
        confidence = event["confidence"]
        if not _finite_real(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("raw tracker event has invalid heuristic confidence")
        if type(event["status"]) is not str or not event["status"].strip():
            raise ValueError("raw tracker event has invalid status")
        descriptor = self._raw.describe()
        json.dumps(descriptor, allow_nan=False)
        if (type(descriptor["samples_processed"]) is not int
                or descriptor["samples_processed"] != self._samples_processed
                or descriptor["config"] != self._config):
            raise ValueError("raw tracker descriptor disagrees with guarded clock/config")
        if descriptor["filter_frequency_hz"] != event["filter_frequency_hz"]:
            raise ValueError("raw tracker descriptor frequency disagrees with event")
        a = descriptor["a"]
        b = descriptor["b"]
        if (not isinstance(a, (list, tuple)) or len(a) != 3
                or not isinstance(b, (list, tuple)) or len(b) != 3
                or not all(_finite_real(value) for value in (*a, *b))):
            raise ValueError("raw tracker coefficients are invalid")
        # Exact second-order real-coefficient Jury stability conditions for a0=1.
        if a[0] != 1.0 or not (abs(a[2]) < 1.0 and 1.0 + a[1] + a[2] > 0.0 and 1.0 - a[1] + a[2] > 0.0):
            raise ValueError("raw tracker denominator is not strictly stable")

    def _append_event(self, reason: str, *, qualified: bool, target_changed: bool, mix_before: float) -> None:
        self._guard_trace.append({
            "sample_index": self._samples_processed,
            "mix_before": mix_before,
            "target_mix": self._target_mix,
            "ramp_start_mix": self._ramp_start_mix,
            "ramp_total_frames": self._ramp_total_frames,
            "ramp_position": self._ramp_position,
            "reason": reason,
            "requested_enabled": self._requested_enabled,
            "qualified": qualified,
            "last_qualified_sample": self._last_qualified_sample,
            "consecutive_qualified": self._consecutive_qualified,
            "target_changed": target_changed,
        })

    def _request_target(self, target: float) -> bool:
        if target == self._target_mix:
            return False
        self._ramp_start_mix = self._mix
        self._target_mix = target
        self._ramp_total_frames = ATTACK_FRAMES if target == 1.0 else RELEASE_FRAMES
        self._ramp_position = 0
        return True

    def set_enabled(self, enabled: bool) -> None:
        """Request a soft release/requalification, not an immediate audio mute."""
        if type(enabled) is not bool:
            raise ValueError("enabled must be a bool")
        if self._fault is not None:
            raise ValueError("guarded processor is faulted; reset is required")
        if enabled == self._requested_enabled:
            return
        before = self._mix
        self._requested_enabled = enabled
        self._consecutive_qualified = 0
        self._last_qualified_sample = None
        changed = self._request_target(0.0)
        self._append_event("reenabled_awaiting_fresh_qualification" if enabled else "requested_soft_off",
                           qualified=False, target_changed=changed, mix_before=before)

    def _observe(self, event: dict) -> None:
        before = self._mix
        estimate = event["estimated_frequency_hz"]
        center = event["filter_frequency_hz"]
        qualified = (estimate is not None and math.isfinite(estimate)
                     and abs(estimate - center) <= QUALIFICATION_ERROR_HZ)
        if not self._requested_enabled:
            self._consecutive_qualified = 0
            self._last_qualified_sample = None
            changed = self._request_target(0.0)
            reason = "requested_disabled"
        elif qualified:
            self._last_qualified_sample = self._samples_processed
            self._consecutive_qualified += 1
            if self._consecutive_qualified >= REQUIRED_QUALIFIED_EVENTS:
                changed = self._request_target(1.0)
                reason = "qualified_wet_request" if changed else "qualified_wet_hold"
            else:
                changed = False
                reason = "qualification_pending"
        else:
            self._consecutive_qualified = 0
            expired = (self._last_qualified_sample is not None
                       and self._samples_processed - self._last_qualified_sample >= QUALIFICATION_HOLD_FRAMES)
            changed = self._request_target(0.0) if expired else False
            reason = "qualification_expired_dry_request" if expired else "unqualified_hold"
        self._append_event(reason, qualified=qualified, target_changed=changed, mix_before=before)

    def _mix_part(self, source: np.ndarray, filtered: np.ndarray) -> np.ndarray:
        output = np.empty_like(source)
        for index, (dry, wet) in enumerate(zip(source, filtered, strict=True)):
            if self._ramp_position < self._ramp_total_frames:
                self._ramp_position += 1
                if self._ramp_position == self._ramp_total_frames:
                    self._mix = self._target_mix
                else:
                    proportion = 0.5 * (1.0 - math.cos(math.pi * self._ramp_position / self._ramp_total_frames))
                    self._mix = self._ramp_start_mix + (self._target_mix - self._ramp_start_mix) * proportion
            if self._mix == 0.0:
                output[index] = dry
            elif self._mix == 1.0:
                output[index] = wet
            else:
                output[index] = float(dry) + self._mix * (float(wet) - float(dry))
        return output

    def _latch_fault(self, exc: Exception) -> None:
        self._fault = {"sample_index": _safe_audit_value(self._samples_processed), "reason": "internal_processing_failure",
                       "exception_type": type(exc).__name__}
        if not isinstance(self._guard_trace, list):
            self._guard_trace = []
        self._append_event("latched_internal_fault", qualified=False, target_changed=False, mix_before=self._mix)
        self._guard_trace[-1] = _safe_audit_value(self._guard_trace[-1])

    def process_block(self, samples: object) -> np.ndarray:
        try:
            source = np.asarray(samples)
        except (TypeError, ValueError) as exc:
            raise ValueError("block must be a real mono vector") from exc
        if source.ndim != 1 or not 1 <= source.size <= MAX_BLOCK_FRAMES:
            raise ValueError("block must contain 1..4096 mono samples")
        block = validate_audio(source)
        if self._fault is not None:
            raise ValueError("guarded processor is faulted; reset is required")
        output = np.empty_like(block)
        offset = 0
        bypass = self._config["max_attenuation_db"] == 0.0
        try:
            self._validate_state()
            while offset < block.size:
                count = block.size - offset if bypass else min(block.size - offset, self._next_update - self._samples_processed)
                part = block[offset:offset + count]
                filtered = self._raw.process_block(part)
                if (not isinstance(filtered, np.ndarray) or filtered.shape != part.shape
                        or filtered.dtype.kind not in "fiu" or not np.isfinite(filtered).all()):
                    raise ValueError("raw tracker returned invalid output")
                result = part.copy() if bypass else self._mix_part(part, filtered)
                if not np.isfinite(result).all():
                    raise ValueError("guard produced nonfinite output")
                output[offset:offset + count] = result
                self._maximum_input_peak = max(self._maximum_input_peak, float(np.max(np.abs(part))))
                self._maximum_raw_peak = max(self._maximum_raw_peak, float(np.max(np.abs(filtered))))
                self._maximum_output_peak = max(self._maximum_output_peak, float(np.max(np.abs(result))))
                self._samples_processed += count
                offset += count
                if not bypass and self._samples_processed == self._next_update:
                    raw_events = self._raw.get_trace()
                    if len(raw_events) != len(self._tracking_trace) + 1:
                        raise ValueError("raw tracker event clock disagrees with guard")
                    event = raw_events[-1]
                    self._validate_raw_event(event)
                    self._tracking_trace.append(copy.deepcopy(event))
                    self._observe(event)
                    self._next_update += UPDATE_FRAMES
            self._validate_state()
        except Exception as exc:
            self._latch_fault(exc)
            raise ValueError("internal guarded processing failed; reset is required") from exc
        return output


def process_guarded_audio(samples: object, config: ProcessorConfig | None = None, *, return_trace: bool = False):
    """Fresh guarded processing; traces return as (output, guard, raw_tracking).

    Depth zero is an exact bypass with no FFT or raw tracking events. No automatic
    clipping or gain changes are applied; callers must reject PCM16 overshoot.
    """
    if type(return_trace) is not bool:
        raise ValueError("return_trace must be a bool")
    audio = validate_audio(samples)
    engine = GuardedHumSuppressor(ProcessorConfig() if config is None else config)
    block_size = engine.describe()["config"]["block_size_frames"]
    output = np.empty_like(audio)
    for start in range(0, audio.size, block_size):
        stop = min(start + block_size, audio.size)
        output[start:stop] = engine.process_block(audio[start:stop])
    return (output, engine.get_guard_trace(), engine.get_tracking_trace()) if return_trace else output
