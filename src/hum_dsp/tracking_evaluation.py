"""Truth-aware tracking measurements; never imported by the tracking engine."""

import math

import numpy as np

from .audio import SAMPLE_RATE_HZ
from .fixtures import FixtureSpec


def _truth_hz(spec: FixtureSpec, sample_indices: np.ndarray) -> np.ndarray:
    if spec.noise_kind == "fixed":
        return np.full(sample_indices.shape, spec.nominal_hum_hz)
    if spec.noise_kind == "drift":
        time = sample_indices / SAMPLE_RATE_HZ
        return spec.nominal_hum_hz - spec.direction + spec.direction * 0.05 * np.clip(time - 2.0, 0, 40)
    raise ValueError("no ground-truth noise frequency for this fixture")


def measure_tracking(trace: list, spec: FixtureSpec, *, frames: int) -> dict:
    """Include every applied-center sample after the fixed 2s boundary.

    An event's sample_index counts consumed samples, with its center applying
    from that zero-based output index onward. Estimates are compared to current
    causal time, not the middle of their historical observation window.
    """
    if type(frames) is not int or frames <= 0:
        raise ValueError("frames must be a positive integer")
    lower, upper = spec.nominal_hum_hz - 1, spec.nominal_hum_hz + 1
    last_sample = -1
    acquired_at = None
    max_slew = 0.0
    prior_event = None
    valid_estimate_events = []
    for event in trace:
        count = event["sample_index"]
        if type(count) is not int or not 0 <= count <= frames or count <= last_sample:
            raise ValueError("tracking event is duplicated, out of order or in the future")
        center = event["filter_frequency_hz"]
        estimate = event["estimated_frequency_hz"]
        confidence = event["confidence"]
        if isinstance(center, bool) or not math.isfinite(center) or not lower - 1e-12 <= center <= upper + 1e-12:
            raise ValueError("tracked filter center is outside the declared range")
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("tracking confidence must be a finite bounded heuristic")
        if estimate is not None:
            if isinstance(estimate, bool) or not math.isfinite(estimate) or not lower - 1e-12 <= estimate <= upper + 1e-12:
                raise ValueError("estimate is outside the declared range")
            if count < 96000:
                raise ValueError("estimate was acquired before the declared history existed")
            if acquired_at is None:
                acquired_at = count
            valid_estimate_events.append(event)
        if prior_event is not None and acquired_at is not None and prior_event["sample_index"] >= acquired_at:
            elapsed = (count - prior_event["sample_index"]) / SAMPLE_RATE_HZ
            rate = abs(center - prior_event["filter_frequency_hz"]) / elapsed
            max_slew = max(max_slew, rate)
        prior_event = event
        last_sample = count
    after = [event for event in trace if event["sample_index"] >= 96000]
    expected_samples = set(range(96000, frames + 1, 4800))
    observed_samples = {event["sample_index"] for event in trace}
    complete_schedule = observed_samples == expected_samples
    valid_on_schedule = sum(event["sample_index"] in expected_samples for event in valid_estimate_events)
    result = {
        "first_acquired_sample": acquired_at,
        "first_acquisition_seconds": None if acquired_at is None else acquired_at / SAMPLE_RATE_HZ,
        "post_boundary_event_count": len(after),
        "valid_estimate_event_count": len(valid_estimate_events),
        "missing_estimate_event_count": sum(event["estimated_frequency_hz"] is None for event in after),
        "expected_event_count": len(expected_samples),
        "missing_scheduled_event_count": len(expected_samples - observed_samples),
        "unexpected_event_count": len(observed_samples - expected_samples),
        "event_schedule_complete": complete_schedule,
        "valid_estimate_event_fraction": None if not expected_samples else valid_on_schedule / len(expected_samples),
        "post_acquisition_max_center_slew_hz_per_s": max_slew,
        "frequency_truth_available": spec.noise_kind != "none",
        "current_time_estimate_max_error_hz": None,
        "all_post_boundary_samples_applied_center_max_error_hz": None,
        "all_post_boundary_samples_applied_center_mean_error_hz": None,
        "frequency_target_met": None,
    }
    if spec.noise_kind == "none" or frames <= 96000:
        return result
    indices = np.arange(96000, frames, dtype=np.int64)
    applied = np.full(indices.size, spec.nominal_hum_hz)
    # Fill held intervals too: missing estimates cannot hide their error.
    previous_start, previous_center = 96000, spec.nominal_hum_hz
    for event in trace:
        count, center = event["sample_index"], event["filter_frequency_hz"]
        if count < 96000:
            previous_center = center
            continue
        begin, end = max(previous_start, 96000), min(count, frames)
        if end > begin:
            applied[begin - 96000:end - 96000] = previous_center
        previous_start, previous_center = count, center
    if previous_start < frames:
        applied[max(previous_start, 96000) - 96000:] = previous_center
    errors = np.abs(applied - _truth_hz(spec, indices))
    if valid_estimate_events:
        event_indices = np.array([event["sample_index"] for event in valid_estimate_events])
        event_estimates = np.array([event["estimated_frequency_hz"] for event in valid_estimate_events])
        estimate_max = float(np.max(np.abs(event_estimates - _truth_hz(spec, event_indices))))
    else:
        estimate_max = None
    maximum = float(np.max(errors))
    result.update(
        current_time_estimate_max_error_hz=estimate_max,
        all_post_boundary_samples_applied_center_max_error_hz=maximum,
        all_post_boundary_samples_applied_center_mean_error_hz=float(np.mean(errors)),
        frequency_target_met=complete_schedule and acquired_at is not None and acquired_at <= 96000 and estimate_max is not None and estimate_max <= 0.3 and maximum <= 0.3 and max_slew <= 0.05 + 1e-10,
    )
    return result
