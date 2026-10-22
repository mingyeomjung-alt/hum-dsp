import json

import numpy as np
import pytest

from hum_dsp.fixtures import fixture_specs
from hum_dsp.tracking import TrackedHumSuppressor
from hum_dsp.config import ProcessorConfig
from hum_dsp.tracking_evaluation import measure_tracking


def _spec(kind="stationary_target", nominal=50):
    return next(spec for spec in fixture_specs() if spec.kind == kind and spec.nominal_hum_hz == nominal)


def _event(sample, estimate=50.0, center=50.0):
    return {"sample_index": sample, "estimated_frequency_hz": estimate, "filter_frequency_hz": center, "confidence": 0.9, "status": "test_event"}


def test_sparse_trace_cannot_claim_complete_coverage_or_success():
    result = measure_tracking([_event(96000)], _spec(), frames=288000)
    assert result["expected_event_count"] == 41
    assert result["missing_scheduled_event_count"] == 40
    assert result["valid_estimate_event_fraction"] == pytest.approx(1 / 41)
    assert result["event_schedule_complete"] is False
    assert result["frequency_target_met"] is False


def test_complete_static_track_is_measured_at_all_samples():
    events = [_event(sample) for sample in range(96000, 288001, 4800)]
    result = measure_tracking(events, _spec(), frames=288000)
    assert result["event_schedule_complete"] is True
    assert result["frequency_target_met"] is True
    assert result["first_acquisition_seconds"] == 2
    assert result["all_post_boundary_samples_applied_center_max_error_hz"] == 0


def test_missing_estimates_do_not_disappear_from_denominator():
    events = [_event(sample, estimate=50 if sample == 96000 else None) for sample in range(96000, 288001, 4800)]
    result = measure_tracking(events, _spec(), frames=288000)
    assert result["missing_estimate_event_count"] == 40
    assert result["valid_estimate_event_fraction"] == pytest.approx(1 / 41)
    assert result["all_post_boundary_samples_applied_center_max_error_hz"] == 0


def test_held_frequency_error_includes_every_sample_not_only_update_times():
    spec = _spec("moving_target")
    events = [_event(sample, estimate=spec.nominal_hum_hz, center=spec.nominal_hum_hz) for sample in range(96000, 44 * 48000 + 1, 4800)]
    result = measure_tracking(events, spec, frames=44 * 48000)
    assert result["all_post_boundary_samples_applied_center_max_error_hz"] == pytest.approx(1)
    assert result["frequency_target_met"] is False


def test_estimates_are_compared_to_current_time_not_two_second_window_midpoint():
    spec = _spec("moving_target")
    # At t=12s current truth is 49.5Hz; window midpoint truth is 49.45Hz.
    events = [_event(12 * 48000, estimate=49.45, center=49.45)]
    result = measure_tracking(events, spec, frames=13 * 48000)
    assert result["current_time_estimate_max_error_hz"] == pytest.approx(0.05)


def test_excessive_post_acquisition_slew_cannot_pass():
    events = [_event(sample, center=50 if sample == 96000 else 50.5) for sample in range(96000, 144001, 4800)]
    result = measure_tracking(events, _spec(), frames=144000)
    assert result["post_acquisition_max_center_slew_hz_per_s"] == pytest.approx(5)
    assert result["frequency_target_met"] is False


@pytest.mark.parametrize("events", [
    [_event(96000), _event(96000)],
    [_event(100000), _event(96000)],
    [_event(96001)],
    [_event(95999)],
    [_event(96000, estimate=51.1)],
    [_event(96000, center=49 - 1e-4)],
])
def test_invalid_or_unscheduled_event_cannot_be_success(events):
    try:
        result = measure_tracking(events, _spec(), frames=96000)
    except ValueError:
        return
    assert result["frequency_target_met"] is not True


def test_silence_has_no_fabricated_frequency_truth():
    spec = next(spec for spec in fixture_specs() if spec.case_id == "synthetic_silence")
    result = measure_tracking([_event(96000, estimate=None)], spec, frames=96000)
    assert result["frequency_truth_available"] is False
    assert result["frequency_target_met"] is None
    json.dumps(result, allow_nan=False)


def test_frozen_candidate_parameters_match_engine_metadata():
    from pathlib import Path
    candidate = json.loads((Path(__file__).parents[1] / "src/hum_dsp/m3_candidate.json").read_text(encoding="utf-8"))
    metadata = TrackedHumSuppressor(ProcessorConfig()).describe()
    for spec_key, metadata_key in (
        ("history_samples", "observation_frames"),
        ("update_interval_samples", "update_frames"),
        ("fft_size", "fft_frames"),
        ("coherent_power_fraction_min", "minimum_coherent_power_fraction"),
        ("estimated_peak_amplitude_min", "minimum_peak_amplitude"),
        ("estimate_edge_tolerance_hz", "estimate_edge_tolerance_hz"),
    ):
        assert candidate[spec_key] == metadata[metadata_key]
    assert metadata["pole_design_bandwidth_hz"] == 1
