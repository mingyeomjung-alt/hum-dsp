"""Selective harmonic DSP: own kernels, causal gates, and frozen-run diagnostics."""

import copy
import json
import math

import numpy as np
import pytest
from scipy import signal

from hum_dsp.harmonics import (
    ATTACK_FRAMES, FFT_FRAMES, OBSERVATION_FRAMES, RELEASE_FRAMES, UPDATE_FRAMES,
    HarmonicConfig, SelectiveHarmonicSuppressor, _Band, _coefficients,
    render_harmonic, render_harmonic_replay,
)


def _tone(frequency=50.0, seconds=2.5, amplitude=0.05):
    t = np.arange(round(48000 * seconds)) / 48000.0
    return amplitude * np.sin(2 * np.pi * frequency * t + 0.37)


def _feed(engine, source, sizes=(4096,)):
    parts, offset, index = [], 0, 0
    while offset < source.size:
        stop = min(source.size, offset + sizes[index % len(sizes)])
        parts.append(engine.process_block(source[offset:stop]))
        offset, index = stop, index + 1
    return np.concatenate(parts)


@pytest.fixture(scope="module")
def rendered():
    config = HarmonicConfig(block_size_frames=4096)
    clean = _tone(705.0, 3.4, 0.11)
    noise = sum(_tone(f, 3.4, 0.035) for f in (50.0, 100.0, 150.0))
    mixture = clean + noise
    output, description, trace = render_harmonic(mixture, config)
    return config, clean, noise, mixture, output, description, trace


@pytest.mark.parametrize("kwargs", [
    {"sample_rate_hz": 16000}, {"sample_rate_hz": True},
    {"nominal_hum_hz": 55}, {"nominal_hum_hz": True}, {"nominal_hum_hz": float("nan")},
    {"max_attenuation_db": -0.01}, {"max_attenuation_db": 12.01}, {"max_attenuation_db": float("inf")},
    {"tracking_half_width_hz": -0.01}, {"tracking_half_width_hz": 1.01}, {"tracking_half_width_hz": True},
    {"max_slew_hz_per_s": -0.01}, {"max_slew_hz_per_s": 0.051}, {"max_slew_hz_per_s": "0.05"},
    {"block_size_frames": 0}, {"block_size_frames": 4097}, {"block_size_frames": True},
    {"orders": ()}, {"orders": (0,)}, {"orders": (7,)}, {"orders": (1, 1)},
    {"orders": (2, 1)}, {"orders": (True,)}, {"orders": (1.0,)}, {"orders": "123"},
])
def test_configuration_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        HarmonicConfig(**kwargs)


def test_configuration_json_round_trip_and_immutable_snapshot():
    config = HarmonicConfig(orders=[2, 4, 6], nominal_hum_hz=60, tracking_half_width_hz=0, max_slew_hz_per_s=0)
    assert config.orders == (2, 4, 6)
    assert HarmonicConfig(**json.loads(json.dumps(config.to_dict()))) == config
    engine = SelectiveHarmonicSuppressor(config)
    object.__setattr__(config, "orders", (99,))
    assert engine.describe()["config"]["orders"] == [2, 4, 6]
    with pytest.raises(ValueError):
        config.to_dict()


@pytest.mark.parametrize("config", [None, {}, True, 50, "50"])
def test_constructor_requires_revalidated_config(config):
    with pytest.raises(ValueError):
        SelectiveHarmonicSuppressor(config)


@pytest.mark.parametrize("frequency", [49.0, 50.0, 60.0, 100.0, 180.0, 366.0])
def test_own_coefficients_match_independent_scipy_full_notch_reference(frequency):
    b, a = _coefficients(frequency, 12.0)
    full_b, full_a = signal.iirnotch(frequency, frequency / 1.0, fs=48000)
    gain = 10.0 ** (-12.0 / 20.0)
    expected = gain * full_a + (1.0 - gain) * full_b
    np.testing.assert_allclose(a, full_a, rtol=0, atol=2e-16)
    np.testing.assert_allclose(b, expected, rtol=0, atol=2e-16)
    assert np.all(np.abs(np.roots(a)) < 1.0)
    _, response = signal.freqz(b, a, worN=np.array([2 * np.pi * frequency / 48000]))
    assert -20 * np.log10(abs(response[0])) == pytest.approx(12.0, abs=1e-6)


@pytest.mark.parametrize("order", [1, 2, 3, 6])
def test_own_direct_form_filter_kernel_matches_scipy_lfilter(order):
    band = _Band(order, HarmonicConfig().to_dict())
    band.mix = band.target = 1.0
    x = np.random.default_rng(51).normal(0, 0.03, 4096)
    expected = signal.lfilter(band.b, band.a, x)
    actual = band.process(x)
    # SciPy's C arithmetic groups operations differently; this is an independent
    # numerical reference, not a bit-identical partitioning assertion.
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)


@pytest.mark.parametrize("bad", [[], np.zeros(4097), [[0.1]], [float("nan")], [float("inf")], [1.0], [-1.01],
                                   [True], [1j], ["0.1"], [object()]])
def test_invalid_public_blocks_are_transactional(bad):
    engine = SelectiveHarmonicSuppressor(HarmonicConfig())
    before, trace = engine.describe(), engine.get_trace()
    with pytest.raises(ValueError):
        engine.process_block(bad)
    assert engine.describe() == before
    assert engine.get_trace() == trace
    assert engine.describe()["fault"] is None


def test_all_selected_bands_share_exactly_one_original_input_fft_per_update(monkeypatch):
    calls = []
    original = np.fft.rfft
    def counted(array, n):
        calls.append((array.copy(), n))
        return original(array, n=n)
    monkeypatch.setattr(np.fft, "rfft", counted)
    source = _tone(50, 2.3) + _tone(100, 2.3) + _tone(150, 2.3)
    _, description, trace = render_harmonic(source, HarmonicConfig(orders=(1, 2, 3, 4, 5, 6), block_size_frames=4096))
    assert len(calls) == 4
    assert description["shared_original_input_fft_count"] == 4
    assert all(n == FFT_FRAMES for _, n in calls)
    np.testing.assert_array_equal(calls[0][0], source[:OBSERVATION_FRAMES] * np.hanning(OBSERVATION_FRAMES))
    np.testing.assert_array_equal(calls[1][0], source[UPDATE_FRAMES:OBSERVATION_FRAMES + UPDATE_FRAMES] * np.hanning(OBSERVATION_FRAMES))
    assert [b["target_mix"] for b in trace[-1]["bands"]] == [1, 1, 1, 0, 0, 0]


def test_cascade_uses_original_not_already_filtered_input_for_all_band_evidence(rendered):
    _, _, _, _, _, description, trace = rendered
    first = trace[1]["bands"]
    assert all(b["initial_acquisition"] for b in first)
    assert all(b["confidence"] >= 0.02 for b in first)
    assert [b["first_acquired_sample"] for b in description["bands"]] == [96000, 96000, 96000]


def test_zero_depth_exact_identity_without_fft_history_or_trace(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("bypass must not perform FFT")
    monkeypatch.setattr(np.fft, "rfft", forbidden)
    x = _tone(100, 2.3)
    config = HarmonicConfig(max_attenuation_db=0, block_size_frames=4096)
    engine = SelectiveHarmonicSuppressor(config)
    np.testing.assert_array_equal(_feed(engine, x), x)
    assert engine._history is None
    assert engine.get_trace() == []
    y, description, trace = render_harmonic(x, config)
    np.testing.assert_array_equal(y, x)
    np.testing.assert_array_equal(render_harmonic_replay(x, config, trace), x)
    assert description["shared_original_input_fft_count"] == 0


def test_block_partition_exactness_including_observation_and_ramp_boundaries(rendered):
    config, _, _, x, expected, _, expected_trace = rendered
    engine = SelectiveHarmonicSuppressor(config)
    actual = _feed(engine, x, (13, 2047, 1, 4096, 377))
    np.testing.assert_array_equal(actual, expected)
    assert engine.get_trace() == expected_trace


def test_initial_dry_then_two_observations_and_exact_raised_cosine_ramp(rendered):
    _, _, _, x, y, _, trace = rendered
    np.testing.assert_array_equal(y[:100800], x[:100800])
    assert all(b["target_mix"] == 0 for b in trace[1]["bands"])
    assert all(b["target_mix"] == 1 and b["ramp_position"] == 0 and b["ramp_total_frames"] == ATTACK_FRAMES
               for b in trace[2]["bands"])
    assert all(b["mix"] == 1 for b in trace[3]["bands"])
    assert trace[2]["sample_index"] == 100800
    assert trace[-1]["trace_total_frames"] == x.size


@pytest.mark.parametrize("source", [np.zeros(110400), np.full(110400, 0.1), _tone(63, 2.3), _tone(705, 2.3)])
def test_absent_target_bands_remain_exactly_dry(source):
    y, description, _ = render_harmonic(source, HarmonicConfig(block_size_frames=4096))
    np.testing.assert_array_equal(y, source)
    assert all(b["target_mix"] == b["mix"] == 0 for b in description["bands"])


def test_orders_without_fundamental_are_supported_and_only_present_band_activates():
    x = _tone(100.3, 2.4)
    _, description, trace = render_harmonic(x, HarmonicConfig(orders=(2, 3), block_size_frames=4096))
    assert [b["order"] for b in description["bands"]] == [2, 3]
    assert description["bands"][0]["filter_frequency_hz"] == pytest.approx(100.3, abs=0.001)
    assert description["bands"][0]["target_mix"] == 1
    assert description["bands"][1]["target_mix"] == 0
    assert trace[1]["bands"][0]["selected_lower_hz"] == 98
    assert trace[1]["bands"][1]["selected_upper_hz"] == 153


def test_tracking_slew_scales_with_order_without_changing_one_hz_poles(monkeypatch):
    engine = SelectiveHarmonicSuppressor(HarmonicConfig(orders=(1, 3), block_size_frames=4096))
    def estimates(band, powers, weighted_power):
        offset = 0.8 if engine._samples_processed == OBSERVATION_FRAMES else -0.8
        return band.nominal + band.order * offset, 0.5, "eligible_peak"
    monkeypatch.setattr(engine, "_estimate_band", estimates)
    _feed(engine, _tone(seconds=2.3))
    trace = engine.get_trace()
    for event in trace[2:]:
        for b in event["bands"]:
            assert abs(b["filter_frequency_hz"] - b["previous_filter_frequency_hz"]) <= b["order"] * 0.005 + 1e-12
    for b in engine.describe()["bands"]:
        assert b["maximum_post_acquisition_slew_hz_per_s"] <= b["order"] * 0.05 + 1e-10
        # Pole radius does not scale with order or selected search range.
        assert b["a"][2] == pytest.approx(_coefficients(50, 12)[1][2], abs=0)


def test_same_frequency_wanted_tone_is_harmed_not_misreported_as_preserved():
    x = _tone(100, 3.0)
    y, description, _ = render_harmonic(x, HarmonicConfig(orders=(2,), block_size_frames=4096))
    ratio = np.sqrt(np.mean(y[-24000:] ** 2)) / np.sqrt(np.mean(x[-24000:] ** 2))
    assert ratio < 0.3
    assert description["policy"]["wanted_tone_protection"].startswith("not_guaranteed")
    assert description["policy"]["harmonic_phase_lock"] is False


def test_trace_and_descriptor_are_independent_json_snapshots(rendered):
    config, _, _, x, _, _, _ = rendered
    engine = SelectiveHarmonicSuppressor(config)
    _feed(engine, x[:100810])
    trace, description = engine.get_trace(), engine.describe()
    assert json.loads(json.dumps((trace, description), allow_nan=False)) == [trace, description]
    trace[0]["bands"][0]["mix"] = 0.7
    description["config"]["orders"].clear()
    assert engine.get_trace()[0]["bands"][0]["mix"] == 0
    assert engine.describe()["config"]["orders"] == [1, 2, 3]


def test_description_separates_current_ramp_from_actual_last_observation(rendered):
    config, _, _, x, _, _, _ = rendered
    engine = SelectiveHarmonicSuppressor(config)
    _feed(engine, x[:100810])
    description = engine.describe()
    actual_event = engine.get_trace()[-1]
    assert actual_event["sample_index"] == 100800
    assert description["samples_processed"] == 100810
    for current, observed in zip(description["bands"], actual_event["bands"], strict=True):
        assert current["acquired"] is True
        assert current["mix"] > 0
        assert current["ramp_position"] == 10
        for event_only in ("status", "reason", "estimated_frequency_hz", "confidence", "qualified", "mix_before"):
            assert event_only not in current
        last = current["last_observation"]
        assert last["sample_index"] == 100800
        assert last["time_seconds"] == actual_event["time_seconds"]
        assert {key: value for key, value in last.items() if key not in ("sample_index", "time_seconds")} == observed
        assert last["mix"] == 0
        assert last["ramp_position"] == 0
        assert last["qualified"] is True
        assert last["status"] in ("tracking", "tracking_slew_limited")
        assert last["reason"] == "qualified_wet_request"
    description["bands"][0]["last_observation"]["status"] = "forged"
    assert engine.describe()["bands"][0]["last_observation"]["status"] != "forged"


@pytest.mark.parametrize("depth", [0.0, 12.0])
def test_description_has_no_fabricated_observation_before_first_fft(depth):
    engine = SelectiveHarmonicSuppressor(HarmonicConfig(max_attenuation_db=depth))
    for band in engine.describe()["bands"]:
        assert band["last_observation"] is None
        assert "status" not in band and "confidence" not in band
    engine.process_block(np.zeros(100))
    assert all(band["last_observation"] is None for band in engine.describe()["bands"])


def test_frozen_replay_exact_reconstruction_and_linear_clean_noise_decomposition(rendered):
    config, clean, noise, mixture, y, _, trace = rendered
    reproduced = render_harmonic_replay(mixture, config, trace)
    preserved = render_harmonic_replay(clean, config, trace)
    residual = render_harmonic_replay(noise, config, trace)
    np.testing.assert_array_equal(reproduced, y)
    np.testing.assert_allclose(preserved + residual, y, rtol=0, atol=1e-12)
    # Even off-band wanted audio has phase/transient error. This test verifies
    # decomposition, not a fixture-specific perceptual preservation threshold.
    assert np.sqrt(np.mean((preserved[-48000:] - clean[-48000:]) ** 2)) > 0
    assert np.sqrt(np.mean(residual[-48000:] ** 2)) < 0.3 * np.sqrt(np.mean(noise[-48000:] ** 2))


def test_frozen_replay_never_runs_fft_or_estimates(monkeypatch, rendered):
    config, _, _, x, expected, _, trace = rendered
    def forbidden(*args, **kwargs):
        raise AssertionError("replay must not estimate")
    monkeypatch.setattr(np.fft, "rfft", forbidden)
    monkeypatch.setattr(SelectiveHarmonicSuppressor, "_estimate_band", forbidden)
    np.testing.assert_array_equal(render_harmonic_replay(x, config, trace), expected)


@pytest.mark.parametrize("mutation", [
    "drop_event", "duplicate_event", "clock_bool", "clock_shift", "nan", "infinite", "order_bool", "wrong_order",
    "outside_center", "outside_estimate", "confidence_bool", "confidence_negative", "eligible_low_confidence",
    "ramp_length", "ramp_position", "ramp_start", "target", "mix", "total_frames", "extra_field", "config",
    "missing_estimate_status", "missing_initial", "wrong_coefficients", "initial_acquired_bool",
])
def test_frozen_replay_refuses_forged_incomplete_or_noncausal_trace(mutation, rendered):
    config, _, _, x, _, _, original = rendered
    trace = copy.deepcopy(original)
    band = trace[2]["bands"][0]
    if mutation == "drop_event": trace.pop()
    elif mutation == "duplicate_event": trace.append(copy.deepcopy(trace[-1]))
    elif mutation == "clock_bool": trace[1]["sample_index"] = True
    elif mutation == "clock_shift": trace[1]["sample_index"] += 1
    elif mutation == "nan": band["filter_frequency_hz"] = float("nan")
    elif mutation == "infinite": band["confidence"] = float("inf")
    elif mutation == "order_bool": band["order"] = True
    elif mutation == "wrong_order": band["order"] = 6
    elif mutation == "outside_center": band["filter_frequency_hz"] = 70
    elif mutation == "outside_estimate": band["estimated_frequency_hz"] = 70
    elif mutation == "confidence_bool": band["confidence"] = True
    elif mutation == "confidence_negative": band["confidence"] = -1
    elif mutation == "eligible_low_confidence": band["confidence"] = 0.0
    elif mutation == "ramp_length": band["ramp_total_frames"] = 4801
    elif mutation == "ramp_position": band["ramp_position"] = -1
    elif mutation == "ramp_start": band["ramp_start_mix"] = 0.5
    elif mutation == "target": band["target_mix"] = 2.0
    elif mutation == "mix": band["mix"] = 0.5
    elif mutation == "total_frames": trace[0]["trace_total_frames"] += 1
    elif mutation == "extra_field": band["fabricated"] = 1
    elif mutation == "config": trace[0]["config"]["orders"] = [1]
    elif mutation == "missing_estimate_status": band["estimated_frequency_hz"] = None
    elif mutation == "missing_initial": trace[0]["event_kind"] = "observation"
    elif mutation == "wrong_coefficients": band["b"][0] += 0.01
    elif mutation == "initial_acquired_bool": trace[0]["bands"][0]["acquired"] = 0
    with pytest.raises(ValueError):
        render_harmonic_replay(x, config, trace)


def test_replay_refuses_different_frame_count_even_with_same_observation_schedule(rendered):
    config, _, _, x, _, _, trace = rendered
    with pytest.raises(ValueError):
        render_harmonic_replay(x[:-1], config, trace)


@pytest.mark.parametrize("corruption", ["mix_nan", "mix_above_one", "state_inf", "center_outside", "coefficients", "history_nan", "clock_bool"])
def test_internal_numerical_faults_latch_and_reset(corruption):
    config = HarmonicConfig(block_size_frames=4096)
    engine = SelectiveHarmonicSuppressor(config)
    band = engine._bands[0]
    if corruption == "mix_nan": band.mix = float("nan")
    elif corruption == "mix_above_one": band.mix = 1.5
    elif corruption == "state_inf": band.state1 = float("inf")
    elif corruption == "center_outside": band.center = 500.0
    elif corruption == "coefficients": band.a = (1.0, 0.0, 2.0)
    elif corruption == "history_nan": engine._history[0] = float("nan")
    elif corruption == "clock_bool": engine._samples_processed = True
    with pytest.raises(ValueError, match="internal harmonic"):
        engine.process_block(np.zeros(100))
    assert engine.describe()["fault"] is not None
    json.dumps(engine.describe(), allow_nan=False)
    with pytest.raises(ValueError, match="faulted"):
        engine.process_block(np.zeros(100))
    engine.reset()
    assert engine.describe()["fault"] is None
    np.testing.assert_array_equal(engine.process_block(np.zeros(100)), np.zeros(100))


@pytest.mark.parametrize("estimate,confidence", [(500.0, 1.0), (float("nan"), 1.0), (50.0, 0.0), (True, 1.0), (50.0, True)])
def test_corrupt_estimator_never_returns_successful_audio(monkeypatch, estimate, confidence):
    engine = SelectiveHarmonicSuppressor(HarmonicConfig(block_size_frames=4096))
    monkeypatch.setattr(engine, "_estimate_band", lambda *args: (estimate, confidence, "eligible_peak"))
    with pytest.raises(ValueError, match="internal harmonic"):
        _feed(engine, _tone(seconds=2.0))
    assert engine.describe()["fault"] is not None


def test_reset_reproduces_identical_output_and_trace(rendered):
    config, _, _, x, expected, _, trace = rendered
    engine = SelectiveHarmonicSuppressor(config)
    _feed(engine, x)
    engine.reset()
    np.testing.assert_array_equal(_feed(engine, x), expected)
    assert engine.get_trace() == trace


def test_exact_eof_observation_is_required_and_replayed_before_no_future_samples():
    config = HarmonicConfig(orders=(1,), block_size_frames=4096)
    x = _tone(seconds=2.0)
    y, _, trace = render_harmonic(x, config)
    assert [event["sample_index"] for event in trace] == [0, OBSERVATION_FRAMES]
    np.testing.assert_array_equal(y, x)
    np.testing.assert_array_equal(render_harmonic_replay(x, config, trace), y)
    with pytest.raises(ValueError):
        render_harmonic_replay(x, config, trace[:-1])


def test_qualified_loss_returns_exact_dry_after_history_hold_and_release():
    x = _tone(seconds=6.5)
    x[round(3 * 48000):] = 0.0
    y, description, trace = render_harmonic(x, HarmonicConfig(orders=(1,), block_size_frames=4096))
    assert any(event["bands"][0]["target_mix"] == 1.0 for event in trace)
    assert description["bands"][0]["mix"] == 0.0
    assert any(event["bands"][0]["reason"] == "qualification_expired_dry_request" for event in trace)
    np.testing.assert_array_equal(y[-12000:], x[-12000:])
    assert description["policy"]["release_frames"] == RELEASE_FRAMES


def test_weak_harmonic_below_fixed_concentration_threshold_is_not_silently_filtered():
    x = _tone(50, 2.4, 0.1) + _tone(100, 2.4, 0.001)
    _, description, _ = render_harmonic(x, HarmonicConfig(block_size_frames=4096))
    assert description["bands"][0]["target_mix"] == 1.0
    assert description["bands"][1]["target_mix"] == 0.0
    assert description["bands"][2]["target_mix"] == 0.0
