"""Independent envelope reconstruction and transition-metric checks."""

import math

import numpy as np
import pytest

from hum_dsp.safety_evaluation import measure_guard, reconstruct_mix


def _event(at, start=0.0, target=1.0, length=4, *, changed=True):
    return {
        "sample_index": at,
        "target_changed": changed,
        "ramp_start_mix": start,
        "target_mix": target,
        "ramp_total_frames": length,
    }


def _raised(start, target, length):
    values = start + (target - start) * (1 - np.cos(np.pi * np.arange(1, length + 1) / length)) / 2
    values[-1] = target
    return values


def test_attack_uses_k_one_at_boundary_and_reaches_exact_final_endpoint():
    mix, hard = reconstruct_mix([_event(2)], 10)
    expected = np.concatenate((np.zeros(2), _raised(0, 1, 4), np.ones(4)))
    np.testing.assert_allclose(mix, expected, rtol=0, atol=1e-15)
    assert mix[2] == pytest.approx((1 - math.cos(math.pi / 4)) / 2)
    assert mix[5] == 1.0
    np.testing.assert_array_equal(hard, np.concatenate((np.zeros(2), np.ones(8))))


def test_release_retargets_from_current_mix_without_changing_emitted_frames():
    prefix, _ = reconstruct_mix([_event(0)], 2)
    mix, hard = reconstruct_mix([_event(0), _event(2, 0.5, 0.0)], 8)
    expected = np.concatenate((prefix, _raised(0.5, 0, 4), np.zeros(2)))
    np.testing.assert_allclose(mix, expected, rtol=0, atol=1e-15)
    assert mix[5] == 0.0
    np.testing.assert_array_equal(hard, [1, 1, 0, 0, 0, 0, 0, 0])


def test_duplicate_clock_observations_do_not_restart_an_unchanged_ramp():
    events = [_event(0, changed=False), _event(0), _event(2, changed=False), _event(2, 0.5, 0.0)]
    actual = reconstruct_mix(events, 9)
    expected = reconstruct_mix([_event(0), _event(2, 0.5, 0.0)], 9)
    for left, right in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(left, right)


def test_multiple_retargets_at_same_clock_emit_no_intermediate_frame():
    mix, hard = reconstruct_mix([_event(0), _event(2, 0.5, 0), _event(2, 0.5, 1)], 8)
    expected = np.concatenate((_raised(0, 1, 4)[:2], _raised(0.5, 1, 4), np.ones(2)))
    np.testing.assert_allclose(mix, expected, rtol=0, atol=1e-15)
    np.testing.assert_array_equal(hard, np.ones(8))


def test_event_at_exclusive_end_does_not_modify_prior_samples():
    baseline = reconstruct_mix([_event(0)], 8)
    actual = reconstruct_mix([_event(0), _event(8, 1.0, 0.0)], 8)
    for left, right in zip(actual, baseline, strict=True):
        np.testing.assert_array_equal(left, right)


@pytest.mark.parametrize("frames", [0, -1, 1.5, True])
def test_invalid_frame_count_is_rejected(frames):
    with pytest.raises(ValueError):
        reconstruct_mix([], frames)


@pytest.mark.parametrize("events", [
    [_event(-1)], [_event(9)], [_event(0.5)], [_event(True)],
    [_event(3, changed=False), _event(2, changed=False)],
    [_event(3, changed=False), _event(2)],
])
def test_invalid_or_nonmonotonic_clock_is_rejected_even_for_unchanged_events(events):
    with pytest.raises(ValueError):
        reconstruct_mix(events, 8)


@pytest.mark.parametrize("field,value", [
    ("ramp_start_mix", float("nan")), ("ramp_start_mix", float("inf")),
    ("ramp_start_mix", -0.1), ("ramp_start_mix", 1.1),
    ("target_mix", float("nan")), ("target_mix", float("inf")), ("target_mix", 0.5),
    ("ramp_total_frames", 0), ("ramp_total_frames", -1),
    ("ramp_total_frames", 4.0), ("ramp_total_frames", True),
])
def test_invalid_envelope_values_are_rejected(field, value):
    event = _event(0)
    event[field] = value
    with pytest.raises(ValueError):
        reconstruct_mix([event], 8)


@pytest.mark.parametrize("events", [[_event(0, 0.1)], [_event(0), _event(2, 0.9, 0)]])
def test_retarget_state_cannot_jump_from_reconstructed_current_envelope(events):
    with pytest.raises(ValueError, match="jump"):
        reconstruct_mix(events, 8)


def test_entirely_dry_metrics_have_zero_residual_and_no_transition_power():
    source = np.linspace(-0.2, 0.2, 128)
    raw = source * 0.25
    report = measure_guard(source, raw, source.copy(), [_event(0, changed=False)])
    assert report["reconstruction_max_error"] == 0
    assert report["mix_min"] == report["mix_max"] == 0
    assert report["maximum_mix_sample_step"] == 0
    assert report["final_dry_from_sample"] == 0
    assert report["no_processing_entire_record"] is True
    assert report["maximum_residual_first_difference"] == 0
    assert report["whole_record_error_from_input_rmse"] == 0
    assert report["transition_residual_500_to_20000hz_max_window_power"] is None
    assert report["hard_switch_residual_500_to_20000hz_max_window_power"] is None


def test_final_dry_is_none_when_mix_remains_wet_at_last_sample():
    source = np.zeros(16)
    raw = np.full(16, 0.5)
    events = [_event(2)]
    mix, _ = reconstruct_mix(events, source.size)
    report = measure_guard(source, raw, mix * 0.5, events)
    assert report["final_dry_from_sample"] is None
    assert report["no_processing_entire_record"] is False
    assert report["reconstruction_max_error"] == 0


def test_final_dry_marks_first_exact_zero_after_release_not_event_time():
    events = [_event(0), _event(6, 1.0, 0.0)]
    mix, _ = reconstruct_mix(events, 16)
    report = measure_guard(np.zeros(16), np.full(16, 0.5), 0.5 * mix, events)
    assert report["final_dry_from_sample"] == 9
    assert mix[8] > 0
    np.testing.assert_array_equal(mix[9:], np.zeros(7))


def test_smooth_transition_metrics_compare_same_events_against_hard_switch():
    frames = 26_000
    source = np.zeros(frames)
    raw = np.full(frames, 0.5)
    events = [_event(3000, length=4800), _event(11000, 1.0, 0.0, length=12000)]
    mix, hard = reconstruct_mix(events, frames)
    guarded = 0.5 * mix
    report = measure_guard(source, raw, guarded, events)
    hard_report = measure_guard(source, raw, 0.5 * hard, events)
    assert report["target_change_count"] == 2
    assert report["reconstruction_max_error"] == 0
    assert report["final_dry_from_sample"] == 22999
    assert report["maximum_mix_sample_step"] <= math.pi / (2 * 4800)
    assert report["maximum_residual_first_difference"] < report["hard_switch_maximum_residual_first_difference"]
    assert report["hard_switch_maximum_residual_first_difference"] == 0.5
    assert report["transition_residual_500_to_20000hz_max_window_power"] < report["hard_switch_residual_500_to_20000hz_max_window_power"]
    assert hard_report["maximum_residual_first_difference"] == report["hard_switch_maximum_residual_first_difference"]
    assert hard_report["transition_residual_500_to_20000hz_max_window_power"] == report["hard_switch_residual_500_to_20000hz_max_window_power"]
    assert report["whole_record_error_from_input_rmse"] == pytest.approx(np.sqrt(np.mean(guarded**2)))
    assert "not_perceptual" in report["metric_scope"]


@pytest.mark.parametrize("bad_signals", [
    (np.array([]), np.array([]), np.array([])),
    (np.zeros((4, 1)), np.zeros((4, 1)), np.zeros((4, 1))),
    (np.zeros(4), np.zeros(3), np.zeros(4)),
    (np.zeros(4), np.zeros(4), np.zeros(5)),
    (np.array([np.nan]), np.zeros(1), np.zeros(1)),
    (np.zeros(1), np.array([np.inf]), np.zeros(1)),
    (np.zeros(1), np.zeros(1), np.array([-np.inf])),
])
def test_measurement_rejects_nonfinite_or_incompatible_signals(bad_signals):
    with pytest.raises(ValueError):
        measure_guard(*bad_signals, [])


def test_single_frame_metrics_do_not_require_a_difference_pair():
    report = measure_guard(np.zeros(1), np.zeros(1), np.zeros(1), [])
    assert report["maximum_residual_first_difference"] == 0
    assert report["hard_switch_maximum_residual_first_difference"] == 0
    assert report["no_processing_entire_record"] is True
