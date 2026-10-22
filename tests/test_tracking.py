"""M3 candidate-v1 causal clock, finite-depth filter and tracker contracts."""

import json

import numpy as np
import pytest

from hum_dsp.config import ProcessorConfig
from hum_dsp.filter import process_audio
from hum_dsp.tracking import TrackedHumSuppressor, process_tracked_audio


def _tone(frequency=50.0, seconds=3.1, amplitude=0.05):
    time = np.arange(round(48000 * seconds)) / 48000.0
    return amplitude * np.sin(2 * np.pi * frequency * time + 0.37)


def _stream(source, lengths, config=None):
    engine = TrackedHumSuppressor(ProcessorConfig() if config is None else config)
    parts = []
    offset = 0
    index = 0
    while offset < source.size:
        count = lengths[index % len(lengths)]
        stop = min(offset + count, source.size)
        parts.append(engine.process_block(source[offset:stop]))
        offset = stop
        index += 1
    return np.concatenate(parts), engine


def test_description_and_trace_are_independent_json_safe_snapshots():
    _, engine = _stream(_tone(), [4096])
    description = engine.describe()
    events = engine.get_trace()
    assert json.loads(json.dumps(description, allow_nan=False)) == description
    assert json.loads(json.dumps(events, allow_nan=False)) == events
    assert description["observation_seconds"] == 2
    assert description["update_seconds"] == 0.1
    assert description["pole_design_bandwidth_hz"] == 1
    assert description["before_acquisition"] == "fixed_nominal_notch"
    assert description["explicit_delay_samples"] == 0
    assert events[0]["sample_index"] == 96000
    events[0]["confidence"] = 999
    description["config"]["nominal_hum_hz"] = 999
    description["b"][0] = 999
    assert engine.get_trace()[0]["confidence"] != 999
    assert engine.describe()["config"]["nominal_hum_hz"] == 50
    assert engine.describe()["b"][0] != 999


@pytest.mark.parametrize("config", [None, {}, True, 50, "50"])
def test_constructor_requires_valid_config(config):
    with pytest.raises(ValueError):
        TrackedHumSuppressor(config)


def test_config_snapshot_and_guard_bypass_validation():
    config = ProcessorConfig()
    engine = TrackedHumSuppressor(config)
    previous = engine.describe()
    object.__setattr__(config, "nominal_hum_hz", 60.0)
    assert engine.describe() == previous
    object.__setattr__(config, "nominal_hum_hz", 51.0)
    with pytest.raises(ValueError):
        TrackedHumSuppressor(config)


@pytest.mark.parametrize("samples", [[], 0.0, [[0.1]], [True], [0.1j], ["0.1"],
    np.array([0.1], dtype=object), [float("nan")], [float("inf")], [1.0], [-1.001], np.zeros(4097)])
def test_invalid_blocks_are_transactional_even_at_acquisition_boundary(samples):
    prefix = _tone(seconds=2)[:-1]
    _, engine = _stream(prefix, [4096])
    _, reference = _stream(prefix, [4096])
    before = engine.describe()
    with pytest.raises(ValueError):
        engine.process_block(samples)
    assert engine.describe() == before
    assert engine.get_trace() == []
    probe = _tone(seconds=0.01)
    np.testing.assert_array_equal(engine.process_block(probe), reference.process_block(probe))
    assert engine.get_trace() == reference.get_trace()


def test_global_clock_output_and_trace_are_bit_identical_for_irregular_partitions():
    source = _tone(49.4, seconds=2.37) + _tone(333, seconds=2.37, amplitude=0.03)
    expected, reference = _stream(source, [4096])
    actual, engine = _stream(source, [1, 7, 479, 4096, 13, 1024, 480])
    np.testing.assert_array_equal(actual, expected)
    assert engine.get_trace() == reference.get_trace()
    assert engine.describe() == reference.describe()


def test_one_sample_partition_crosses_acquisition_identically():
    # Most prefix is streamed in ordinary blocks, then every boundary sample.
    source = _tone(50.4, seconds=2.001)
    expected, events = process_tracked_audio(source, return_trace=True)
    _, engine = _stream(source[:95990], [4096])
    tail = np.concatenate([engine.process_block([value]) for value in source[95990:]])
    np.testing.assert_array_equal(tail, expected[95990:])
    assert engine.get_trace() == events


def test_causal_prefix_does_not_depend_on_future_samples():
    prefix = _tone(50.7, seconds=2.23)
    suffix = np.random.default_rng(15).uniform(-0.5, 0.5, 9999)
    before, events = process_tracked_audio(prefix, return_trace=True)
    complete, complete_events = process_tracked_audio(np.concatenate([prefix, suffix]), return_trace=True)
    np.testing.assert_array_equal(before, complete[:prefix.size])
    assert events == [event for event in complete_events if event["sample_index"] <= prefix.size]


def test_prefix_before_first_update_matches_frozen_m2_filter_exactly():
    source = _tone(49.7, seconds=2)
    np.testing.assert_array_equal(process_tracked_audio(source), process_audio(source))


def test_reset_reproduces_output_and_clears_acquisition_trace():
    source = _tone(60.6, seconds=2.17)
    config = ProcessorConfig(nominal_hum_hz=60)
    first, engine = _stream(source, [4096], config)
    engine.reset()
    assert engine.get_trace() == []
    assert engine.describe()["acquired"] is False
    assert engine.describe()["filter_frequency_hz"] == 60
    assert engine.describe()["samples_processed"] == 0
    parts = [engine.process_block(source[start:start + 4096]) for start in range(0, source.size, 4096)]
    np.testing.assert_array_equal(np.concatenate(parts), first)


@pytest.mark.parametrize("nominal", [50, 60])
@pytest.mark.parametrize("offset", [-1.0, -0.7, 0, 0.7, 1.0])
def test_stationary_peaks_acquire_within_band_and_estimation_error(nominal, offset):
    _, events = process_tracked_audio(_tone(nominal + offset, seconds=2.2), ProcessorConfig(nominal_hum_hz=nominal), return_trace=True)
    assert events[0]["initial_acquisition"] is True
    for event in events:
        assert abs(event["estimated_frequency_hz"] - (nominal + offset)) < 0.005
        assert nominal - 1 <= event["filter_frequency_hz"] <= nominal + 1
        # A zero-padded FFT bin still has finite picket-fence amplitude error.
        assert event["confidence"] > 0.98


@pytest.mark.parametrize("nominal", [50, 60])
def test_silence_and_far_out_of_band_tone_do_not_acquire(nominal):
    for source in [np.zeros(48000 * 2 + 4800), _tone(nominal + 3, seconds=2.1)]:
        output, events = process_tracked_audio(source, ProcessorConfig(nominal_hum_hz=nominal), return_trace=True)
        assert np.isfinite(output).all()
        assert all(event["estimated_frequency_hz"] is None for event in events)
        assert all(event["filter_frequency_hz"] == nominal for event in events)
        assert all(event["acquired"] is False for event in events)


def test_broadband_noise_does_not_acquire():
    source = np.random.default_rng(901).uniform(-0.1, 0.1, 48000 * 3)
    _, events = process_tracked_audio(source, return_trace=True)
    assert all(event["estimated_frequency_hz"] is None for event in events)


@pytest.mark.parametrize("depth", [0, 3, 12])
def test_non_mutating_input_and_depth_zero_exact_identity(depth):
    source = _tone(50.8, seconds=2.1).astype(np.float32)
    original = source.copy()
    output = process_tracked_audio(source, ProcessorConfig(max_attenuation_db=depth))
    np.testing.assert_array_equal(source, original)
    assert not np.shares_memory(output, source)
    assert output.dtype == np.float64
    if depth == 0:
        np.testing.assert_array_equal(output, source.astype(np.float64))


def test_explicit_off_does_not_observe_estimate_or_acquire_past_two_seconds(monkeypatch):
    def unexpected_estimation(self):
        raise AssertionError("OFF must not estimate a frequency")

    monkeypatch.setattr(TrackedHumSuppressor, "_estimate", unexpected_estimation)
    source = _tone(50.8, seconds=3.1)
    output, engine = _stream(source, [4096, 7, 480], ProcessorConfig(max_attenuation_db=0))
    np.testing.assert_array_equal(output, source)
    assert engine.get_trace() == []
    assert engine.describe()["first_acquired_sample"] is None
    assert engine.describe()["acquired"] is False
    assert engine.describe()["tracking_enabled"] is False
    assert engine.describe()["processing_bypassed"] is True
    assert engine.describe()["samples_processed"] == source.size
    assert np.count_nonzero(engine._history) == 0


def test_slew_bound_applies_after_initial_acquisition_even_when_peak_jumps():
    source = np.concatenate([_tone(49.2, seconds=3), _tone(50.8, seconds=3)])
    _, events = process_tracked_audio(source, return_trace=True)
    assert events[0]["initial_acquisition"] is True
    previous = events[0]
    for event in events[1:]:
        delta = abs(event["filter_frequency_hz"] - previous["filter_frequency_hz"])
        elapsed = (event["sample_index"] - previous["sample_index"]) / 48000
        assert delta <= 0.05 * elapsed + 1e-12
        assert 49 <= event["filter_frequency_hz"] <= 51
        previous = event


@pytest.mark.parametrize("nominal", [50, 60])
def test_drifting_peak_noise_only_beats_fixed_same_bandwidth_baseline(nominal):
    # Generic slow chirp, independent of the engine's observation state.
    time = np.arange(48000 * 8) / 48000
    source = 0.05 * np.sin(2 * np.pi * ((nominal - 0.7) * time + 0.025 * time**2) + 0.37)
    config = ProcessorConfig(nominal_hum_hz=nominal)
    tracked = process_tracked_audio(source, config)
    fixed = process_audio(source, config)
    start = 48000 * 3
    reduction = -10 * np.log10(np.mean(tracked[start:]**2) / np.mean(source[start:]**2))
    assert reduction >= 6
    assert np.mean(tracked[start:]**2) < np.mean(fixed[start:]**2)


@pytest.mark.parametrize("samples", [[], [1.0], [True], [float("nan")], [[0.0]], np.zeros(48000 * 60 + 1)])
def test_wrapper_rejects_invalid_complete_input(samples):
    with pytest.raises(ValueError):
        process_tracked_audio(samples)


@pytest.mark.parametrize("flag", [None, 1, "yes"])
def test_trace_flag_must_be_bool(flag):
    with pytest.raises(ValueError):
        process_tracked_audio([0.0], return_trace=flag)
