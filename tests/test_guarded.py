"""M4 guard policy, causal transitions and explicit-failure contracts."""

import json
import math

import numpy as np
import pytest

from hum_dsp.config import ProcessorConfig
from hum_dsp.guarded import GuardedHumSuppressor, process_guarded_audio
from hum_dsp.tracking import TrackedHumSuppressor


def _tone(frequency=50.0, seconds=2.5, amplitude=0.05):
    t = np.arange(round(48000 * seconds)) / 48000.0
    return amplitude * np.sin(2 * np.pi * frequency * t + 0.37)


def _feed(engine, source, lengths=(4096,)):
    parts = []
    offset = 0
    index = 0
    while offset < source.size:
        stop = min(source.size, offset + lengths[index % len(lengths)])
        parts.append(engine.process_block(source[offset:stop]))
        offset = stop
        index += 1
    return np.concatenate(parts)


def _fake_event(engine, estimate, *, center=50.0):
    return {"sample_index": engine._samples_processed, "estimated_frequency_hz": estimate,
            "filter_frequency_hz": center}


def test_description_and_traces_are_independent_json_safe_snapshots():
    engine = GuardedHumSuppressor(ProcessorConfig())
    _feed(engine, _tone())
    description = engine.describe()
    guard = engine.get_guard_trace()
    tracker = engine.get_tracking_trace()
    for value in (description, guard, tracker):
        assert json.loads(json.dumps(value, allow_nan=False)) == value
    assert guard[0]["sample_index"] == 0
    assert guard[0]["target_mix"] == 0
    assert guard[1]["sample_index"] == 96000
    assert guard[1]["target_mix"] == 0
    assert guard[2]["sample_index"] == 100800
    assert guard[2]["target_mix"] == 1
    assert guard[2]["ramp_total_frames"] == 4800
    assert guard[2]["ramp_position"] == 0
    assert description["mix"] == 1
    assert description["samples_processed"] == 120000
    assert description["raw_tracker"]["samples_processed"] == 120000
    description["config"]["nominal_hum_hz"] = 999
    guard[0]["reason"] = "mutated"
    tracker[0]["confidence"] = 999
    assert engine.describe()["config"]["nominal_hum_hz"] == 50
    assert engine.get_guard_trace()[0]["reason"] == "initial_dry"
    assert engine.get_tracking_trace()[0]["confidence"] != 999


@pytest.mark.parametrize("config", [None, {}, True, 50, "50"])
def test_constructor_requires_validated_configuration(config):
    with pytest.raises(ValueError):
        GuardedHumSuppressor(config)


@pytest.mark.parametrize("enabled", [None, 0, 1, "yes", np.bool_(True)])
def test_enabled_flag_is_strict(enabled):
    with pytest.raises(ValueError):
        GuardedHumSuppressor(ProcessorConfig(), enabled=enabled)
    engine = GuardedHumSuppressor(ProcessorConfig())
    before = engine.describe()
    with pytest.raises(ValueError):
        engine.set_enabled(enabled)
    assert engine.describe() == before


def test_configuration_is_revalidated_and_snapshotted():
    config = ProcessorConfig()
    engine = GuardedHumSuppressor(config)
    before = engine.describe()
    object.__setattr__(config, "nominal_hum_hz", 60.0)
    assert engine.describe() == before
    object.__setattr__(config, "max_attenuation_db", True)
    with pytest.raises(ValueError):
        GuardedHumSuppressor(config)


def test_output_and_both_traces_are_bit_identical_across_irregular_partitions():
    source = _tone(49.4, 2.47) + _tone(333, 2.47, 0.03)
    expected = GuardedHumSuppressor(ProcessorConfig())
    actual = GuardedHumSuppressor(ProcessorConfig())
    np.testing.assert_array_equal(_feed(actual, source, (1, 7, 479, 4096, 13, 1024, 480)), _feed(expected, source))
    assert actual.describe() == expected.describe()
    assert actual.get_guard_trace() == expected.get_guard_trace()
    assert actual.get_tracking_trace() == expected.get_tracking_trace()


def test_one_sample_crosses_acquisition_and_qualification_without_backdating():
    source = _tone(50.4, 2.201)
    expected, guard, tracking = process_guarded_audio(source, return_trace=True)
    engine = GuardedHumSuppressor(ProcessorConfig())
    prefix = _feed(engine, source[:95990])
    tail = _feed(engine, source[95990:], (1,))
    np.testing.assert_array_equal(np.concatenate((prefix, tail)), expected)
    np.testing.assert_array_equal(expected[:100800], source[:100800])
    assert engine.get_guard_trace() == guard
    assert engine.get_tracking_trace() == tracking


def test_causal_prefix_is_independent_of_future_signal():
    prefix = _tone(50.7, 2.23)
    suffix = np.random.default_rng(15).uniform(-0.5, 0.5, 9999)
    before, guard, tracking = process_guarded_audio(prefix, return_trace=True)
    complete, full_guard, full_tracking = process_guarded_audio(np.concatenate((prefix, suffix)), return_trace=True)
    np.testing.assert_array_equal(before, complete[:prefix.size])
    assert guard == [event for event in full_guard if event["sample_index"] <= prefix.size]
    assert tracking == [event for event in full_tracking if event["sample_index"] <= prefix.size]


def test_ramp_exact_sample_convention_and_retarget_from_current_mix():
    engine = GuardedHumSuppressor(ProcessorConfig())
    assert engine._request_target(1.0)
    dry = np.zeros(2000)
    wet = np.ones(2000)
    actual = engine._mix_part(dry, wet)
    expected = np.array([0.5 * (1 - math.cos(math.pi * k / 4800)) for k in range(1, 2001)])
    np.testing.assert_array_equal(actual, expected)
    current = engine.describe()["mix"]
    assert engine._request_target(0.0)
    assert engine.describe()["mix"] == current
    assert engine.describe()["ramp_start_mix"] == current
    assert not engine._request_target(0.0)
    released = engine._mix_part(np.zeros(12000), np.ones(12000))
    assert released[0] == current + (0 - current) * 0.5 * (1 - math.cos(math.pi / 12000))
    assert released[-1] == 0
    assert np.all(np.diff(released) <= 0)


def test_soft_off_releases_in_exactly_12000_frames_keeps_tracker_warm_and_requalifies():
    source = _tone(seconds=3.0)
    engine = GuardedHumSuppressor(ProcessorConfig())
    _feed(engine, source[:120000])
    before_tracking = len(engine.get_tracking_trace())
    assert engine.describe()["mix"] == 1
    engine.set_enabled(False)
    assert engine.describe()["mix"] == 1
    assert engine.describe()["target_mix"] == 0
    released = _feed(engine, source[120000:132000])
    assert engine.describe()["mix"] == 0
    assert released[-1] == source[131999]
    assert len(engine.get_tracking_trace()) > before_tracking
    assert engine.describe()["consecutive_qualified"] == 0
    dry = _feed(engine, source[132000:134400])
    np.testing.assert_array_equal(dry, source[132000:134400])
    engine.set_enabled(True)
    _feed(engine, source[134400:139200])
    assert engine.describe()["consecutive_qualified"] == 1
    assert engine.describe()["target_mix"] == 0
    _feed(engine, source[139200:144000])
    assert engine.describe()["consecutive_qualified"] == 2
    assert engine.describe()["target_mix"] == 1
    assert engine.describe()["mix"] == 0


def test_unqualified_eligible_observations_do_not_refresh_hold_clock():
    engine = GuardedHumSuppressor(ProcessorConfig())
    for boundary in (96000, 100800):
        engine._samples_processed = boundary
        engine._observe(_fake_event(engine, 50.0))
    assert engine.describe()["target_mix"] == 1
    for boundary in (105600, 110400):
        engine._samples_processed = boundary
        engine._observe(_fake_event(engine, 50.3))
        assert engine.describe()["target_mix"] == 1
        assert engine.describe()["last_qualified_sample"] == 100800
        assert engine.describe()["consecutive_qualified"] == 0
    engine._samples_processed = 115200
    engine._observe(_fake_event(engine, 50.3))
    assert engine.describe()["target_mix"] == 0
    assert engine.get_guard_trace()[-1]["reason"] == "qualification_expired_dry_request"


def test_new_qualified_event_resets_timeout_but_broken_streak_needs_two_to_activate():
    engine = GuardedHumSuppressor(ProcessorConfig())
    for boundary, estimate in ((96000, 50), (100800, None), (105600, 50)):
        engine._samples_processed = boundary
        engine._observe(_fake_event(engine, estimate))
        assert engine.describe()["target_mix"] == 0
    engine._samples_processed = 110400
    engine._observe(_fake_event(engine, 50))
    assert engine.describe()["target_mix"] == 1
    for boundary, estimate in ((115200, None), (120000, None), (124800, 50), (129600, None), (134400, None)):
        engine._samples_processed = boundary
        engine._observe(_fake_event(engine, estimate))
        assert engine.describe()["target_mix"] == 1
    assert engine.describe()["last_qualified_sample"] == 124800


def test_reset_restores_fresh_constructor_state_and_output():
    source = _tone(60.6, 2.23)
    config = ProcessorConfig(nominal_hum_hz=60)
    engine = GuardedHumSuppressor(config, enabled=True)
    first = _feed(engine, source)
    engine.set_enabled(False)
    engine.reset()
    assert engine.describe() == GuardedHumSuppressor(config).describe()
    np.testing.assert_array_equal(_feed(engine, source), first)
    disabled = GuardedHumSuppressor(config, enabled=False)
    disabled.set_enabled(True)
    disabled.reset()
    assert disabled.describe()["requested_enabled"] is False


def test_zero_depth_is_exact_bypass_without_fft(monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("depth zero must not estimate or run FFT")

    monkeypatch.setattr(TrackedHumSuppressor, "_estimate", unexpected)
    monkeypatch.setattr(np.fft, "rfft", unexpected)
    source = _tone(50.8, 3.1).astype(np.float32)
    original = source.copy()
    output, guard, tracking = process_guarded_audio(source, ProcessorConfig(max_attenuation_db=0), return_trace=True)
    np.testing.assert_array_equal(output, source.astype(np.float64))
    np.testing.assert_array_equal(source, original)
    assert not np.shares_memory(output, source)
    assert tracking == []
    assert len(guard) == 1
    assert guard[0]["target_mix"] == 0


@pytest.mark.parametrize("samples", [[], 0.0, [[0.1]], [True], [0.1j], ["0.1"], np.array([0.1], dtype=object),
                                     [float("nan")], [float("inf")], [1.0], [-1.001], np.zeros(4097)])
def test_invalid_input_is_transactional_at_acquisition_boundary(samples):
    source = _tone(seconds=2.01)
    engine = GuardedHumSuppressor(ProcessorConfig())
    control = GuardedHumSuppressor(ProcessorConfig())
    _feed(engine, source[:95999])
    _feed(control, source[:95999])
    before = engine.describe()
    events = engine.get_guard_trace()
    with pytest.raises(ValueError):
        engine.process_block(samples)
    assert engine.describe() == before
    assert engine.get_guard_trace() == events
    np.testing.assert_array_equal(engine.process_block(source[95999:]), control.process_block(source[95999:]))


@pytest.mark.parametrize("mode", ["raise", "nan", "shape", "complex"])
def test_internal_failure_is_latched_not_successful_dry_fallback(monkeypatch, mode):
    engine = GuardedHumSuppressor(ProcessorConfig())

    def broken(part):
        if mode == "raise":
            raise RuntimeError("injected internal failure")
        if mode == "nan":
            return np.full_like(part, np.nan)
        if mode == "shape":
            return np.zeros(part.size + 1)
        return part.astype(np.complex128)

    monkeypatch.setattr(engine._raw, "process_block", broken)
    with pytest.raises(ValueError, match="reset is required"):
        engine.process_block(np.zeros(10))
    assert engine.describe()["fault"]["reason"] == "internal_processing_failure"
    assert engine.get_guard_trace()[-1]["reason"] == "latched_internal_fault"
    with pytest.raises(ValueError, match="reset is required"):
        engine.process_block(np.zeros(10))
    with pytest.raises(ValueError, match="reset is required"):
        engine.set_enabled(False)
    engine.reset()
    assert engine.describe()["fault"] is None
    np.testing.assert_array_equal(engine.process_block(np.zeros(10)), np.zeros(10))


def test_raw_peak_overshoot_is_not_clipped_or_normalized(monkeypatch):
    engine = GuardedHumSuppressor(ProcessorConfig())
    engine._mix = engine._target_mix = 1.0
    monkeypatch.setattr(engine._raw, "process_block", lambda block: np.full_like(block, 1.25))
    output = engine.process_block(np.zeros(10))
    np.testing.assert_array_equal(output, np.full(10, 1.25))
    assert engine.describe()["maximum_output_peak"] == 1.25


def test_internal_out_of_band_estimator_cannot_retune_and_activate_guard(monkeypatch):
    engine = GuardedHumSuppressor(ProcessorConfig())
    source = _tone(seconds=2)
    _feed(engine, source[:-1])
    monkeypatch.setattr(engine._raw, "_estimate", lambda: (500.0, 1.0, "eligible_peak"))
    with pytest.raises(ValueError, match="reset is required"):
        engine.process_block(source[-1:])
    assert engine.describe()["fault"] is not None
    assert engine.describe()["target_mix"] == 0
    assert engine.get_tracking_trace() == []
    assert not any(event["qualified"] for event in engine.get_guard_trace())
    with pytest.raises(ValueError, match="reset is required"):
        engine.process_block([0.0])


@pytest.mark.parametrize("field,value", [
    ("sample_index", 96000.0), ("sample_index", True), ("sample_index", 96001),
    ("filter_frequency_hz", 500.0), ("filter_frequency_hz", True),
    ("filter_frequency_hz", 48.999999999998), ("filter_frequency_hz", 51.000000000002),
    ("estimated_frequency_hz", 500.0), ("estimated_frequency_hz", True),
    ("estimated_frequency_hz", float("nan")), ("estimated_frequency_hz", float("inf")),
    ("confidence", True), ("confidence", -0.001), ("confidence", 1.001),
    ("confidence", float("nan")), ("confidence", "0.5"),
    ("status", ""), ("status", "  "), ("status", None), ("status", 1),
])
def test_invalid_raw_boundary_event_latches_before_qualification(monkeypatch, field, value):
    engine = GuardedHumSuppressor(ProcessorConfig())
    source = _tone(seconds=2)
    _feed(engine, source[:-1])
    original_trace = engine._raw.get_trace

    def corrupted_trace():
        events = original_trace()
        events[-1][field] = value
        return events

    monkeypatch.setattr(engine._raw, "get_trace", corrupted_trace)
    with pytest.raises(ValueError, match="reset is required"):
        engine.process_block(source[-1:])
    assert engine.describe()["fault"] is not None
    assert engine.get_tracking_trace() == []
    assert not any(event["qualified"] for event in engine.get_guard_trace())
    json.dumps(engine.get_guard_trace(), allow_nan=False)


@pytest.mark.parametrize("field,value", [
    ("_mix", 1.5), ("_mix", -0.1), ("_mix", float("nan")), ("_mix", True),
    ("_ramp_start_mix", 1.5), ("_ramp_start_mix", float("inf")),
    ("_target_mix", 0.5), ("_target_mix", True), ("_target_mix", float("nan")),
    ("_ramp_total_frames", 1), ("_ramp_total_frames", 4800.0), ("_ramp_total_frames", True),
    ("_ramp_position", -1), ("_ramp_position", 1), ("_ramp_position", True),
    ("_requested_enabled", 1), ("_initial_enabled", 1),
    ("_samples_processed", -1), ("_samples_processed", 0.0), ("_samples_processed", True),
    ("_next_update", 95999), ("_next_update", 96001), ("_next_update", 96000.0),
    ("_consecutive_qualified", -1), ("_consecutive_qualified", 1), ("_consecutive_qualified", True),
    ("_last_qualified_sample", 0), ("_last_qualified_sample", 96000),
    ("_maximum_output_peak", float("nan")), ("_maximum_raw_peak", -1.0),
])
def test_private_state_corruption_is_latched_and_auditable_not_returned_as_audio(field, value):
    engine = GuardedHumSuppressor(ProcessorConfig())
    setattr(engine, field, value)
    with pytest.raises(ValueError, match="reset is required"):
        engine.process_block([0.0])
    assert engine.describe()["fault"] is not None
    json.dumps(engine.get_guard_trace(), allow_nan=False)
    json.dumps(engine.describe(), allow_nan=False)
    with pytest.raises(ValueError, match="reset is required"):
        engine.process_block([0.0])


def test_public_validation_still_precedes_internal_fault_latching():
    engine = GuardedHumSuppressor(ProcessorConfig())
    engine._mix = float("nan")
    events = engine.get_guard_trace()
    with pytest.raises(ValueError, match="finite"):
        engine.process_block([float("nan")])
    assert engine._fault is None
    assert engine.get_guard_trace() == events
    with pytest.raises(ValueError, match="reset is required"):
        engine.process_block([0.0])


@pytest.mark.parametrize("field,values", [
    ("a", [1.0, -2.0, 1.0]), ("a", [1.0, 0.0, 1.1]),
    ("a", [1.0, float("nan"), 0.9]), ("a", [True, 0.0, 0.9]),
    ("b", [1.0, float("inf"), 1.0]), ("b", [1.0, 0.0]),
])
def test_invalid_or_unstable_raw_coefficients_are_latched_at_boundary(monkeypatch, field, values):
    engine = GuardedHumSuppressor(ProcessorConfig())
    source = _tone(seconds=2)
    _feed(engine, source[:-1])
    original_describe = engine._raw.describe

    def corrupted_description():
        description = original_describe()
        description[field] = values
        return description

    monkeypatch.setattr(engine._raw, "describe", corrupted_description)
    with pytest.raises(ValueError, match="reset is required"):
        engine.process_block(source[-1:])
    assert engine.describe()["fault"] is not None
    assert engine.get_tracking_trace() == []
    json.dumps(engine.describe(), allow_nan=False)


@pytest.mark.parametrize("flag", [None, 1, "yes", np.bool_(False)])
def test_wrapper_trace_flag_is_strict(flag):
    with pytest.raises(ValueError):
        process_guarded_audio([0.0], return_trace=flag)
