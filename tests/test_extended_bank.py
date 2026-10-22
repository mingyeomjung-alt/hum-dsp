"""Independent research-bank contracts, not perceptual quality approval."""

import copy
import json

import numpy as np
import pytest
from scipy import signal

from hum_dsp.extended_bank import (
    ATTACK_FRAMES, OBSERVATION_FRAMES, UPDATE_FRAMES, ExtendedBankConfig, _Band,
    _coefficients, render_extended_bank, replay_extended_bank,
)


def tone(frequency, seconds=2.5, amplitude=.04):
    t = np.arange(round(seconds * 48000)) / 48000
    return amplitude * np.sin(2 * np.pi * frequency * t + .37)


@pytest.fixture(scope="module")
def rendered():
    config = ExtendedBankConfig(orders=(1, 21, 24), block_size_frames=4096)
    desired = tone(713, 2.5, .08)
    noise = tone(1260, 2.5, .035) + tone(1440, 2.5, .035)
    output, description, trace = render_extended_bank(desired + noise, config)
    return config, desired, noise, output, description, trace


@pytest.mark.parametrize("changes", [
    {"sample_rate_hz": 44100}, {"sample_rate_hz": True}, {"nominal_hum_hz": 55},
    {"nominal_hum_hz": float("nan")}, {"nominal_hum_hz": True},
    {"orders": ()}, {"orders": (0,)}, {"orders": (33,)}, {"orders": (1, 1)},
    {"orders": (2, 1)}, {"orders": (True,)}, {"orders": (1.,)}, {"orders": "123"},
    {"max_attenuation_db": -1}, {"max_attenuation_db": 13},
    {"max_attenuation_db": True}, {"max_attenuation_db": float("inf")},
    {"block_size_frames": 0}, {"block_size_frames": 4097}, {"block_size_frames": True},
])
def test_invalid_configuration(changes):
    with pytest.raises(ValueError):
        ExtendedBankConfig(**changes)


def test_config_roundtrip():
    config = ExtendedBankConfig(orders=[2, 21, 32], nominal_hum_hz=50)
    assert config.orders == (2, 21, 32)
    assert ExtendedBankConfig(**json.loads(json.dumps(config.to_dict()))) == config
    object.__setattr__(config, "orders", (99,))
    with pytest.raises(ValueError):
        config.to_dict()


@pytest.mark.parametrize("center", [49, 60, 100, 1260, 1440, 1920])
def test_coefficients_match_independent_scipy(center):
    b, a = _coefficients(center, 12)
    full_b, full_a = signal.iirnotch(center, center, fs=48000)
    residual = 10 ** (-12 / 20)
    # Different libm argument grouping may differ by one binary64 ulp.
    np.testing.assert_allclose(a, full_a, rtol=0, atol=4e-16)
    np.testing.assert_allclose(b, residual * full_a + (1 - residual) * full_b, rtol=0, atol=4e-16)
    assert np.max(np.abs(np.roots(a))) < 1
    _, response = signal.freqz(b, a, worN=np.array([2 * np.pi * center / 48000]))
    assert -20 * np.log10(abs(response[0])) == pytest.approx(12, abs=1e-6)


def test_own_filter_matches_scipy():
    band = _Band(24, ExtendedBankConfig(orders=(24,)))
    band.mix = band.target = 1.
    source = np.random.default_rng(13).normal(0, .03, 4096)
    np.testing.assert_allclose(band.process(source), signal.lfilter(band.b, band.a, source), rtol=0, atol=1e-12)


def test_absolute_not_order_scaled_bounds_and_slew():
    band = _Band(24, ExtendedBankConfig(orders=(24,)))
    assert (band.lower, band.upper) == (1439., 1441.)
    evidence = {"estimate_hz": 1440., "confidence": .5, "amplitude": .1, "status": "eligible"}
    band.observe(evidence, OBSERVATION_FRAMES)
    evidence["estimate_hz"] = 1440.5
    record = band.observe(evidence, OBSERVATION_FRAMES + UPDATE_FRAMES)
    assert band.center == pytest.approx(1440.005, abs=1e-12)
    assert record["consecutive_qualified"] == 0
    assert record["target_mix"] == 0


def test_causal_two_event_attack_and_exact_frozen_replay(rendered, monkeypatch):
    config, desired, noise, output, description, trace = rendered
    assert len(trace) == 7
    assert trace[1]["sample_index"] == OBSERVATION_FRAMES
    for event in trace[1:3]:
        assert event["bands"][1]["consecutive_qualified"] == (event["sample_index"] - OBSERVATION_FRAMES) // UPDATE_FRAMES + 1
    assert trace[1]["bands"][1]["target_mix"] == 0
    assert trace[2]["bands"][1]["target_mix"] == 1
    assert trace[2]["bands"][1]["ramp_total_frames"] == ATTACK_FRAMES
    np.testing.assert_array_equal(output[:100800], (desired + noise)[:100800])
    assert description["shared_original_input_fft_count"] == 6
    monkeypatch.setattr(np.fft, "rfft", lambda *args, **kwargs: pytest.fail("replay must not estimate"))
    np.testing.assert_array_equal(replay_extended_bank(desired + noise, config, trace), output)


def test_frozen_replay_additivity(rendered):
    config, desired, noise, output, _, trace = rendered
    isolated = replay_extended_bank(desired, config, trace) + replay_extended_bank(noise, config, trace)
    np.testing.assert_allclose(output, isolated, rtol=0, atol=3e-13)


def test_block_partition_invariant():
    source = tone(1440, seconds=2.2)
    first, _, trace1 = render_extended_bank(source, ExtendedBankConfig(orders=(24,), block_size_frames=37))
    second, _, trace2 = render_extended_bank(source, ExtendedBankConfig(orders=(24,), block_size_frames=4096))
    np.testing.assert_array_equal(first, second)
    trace1[0]["config"]["block_size_frames"] = 4096
    assert trace1 == trace2


def test_zero_depth_exact_no_fft(monkeypatch):
    monkeypatch.setattr(np.fft, "rfft", lambda *args, **kwargs: pytest.fail("bypass must not estimate"))
    source = tone(1440)
    config = ExtendedBankConfig(max_attenuation_db=0)
    output, description, trace = render_extended_bank(source, config)
    np.testing.assert_array_equal(output, source)
    np.testing.assert_array_equal(replay_extended_bank(source, config, trace), source)
    assert description["processing_bypassed"] is True
    assert len(trace) == 1


def test_silence_remains_dry():
    source = np.zeros(OBSERVATION_FRAMES)
    output, _, trace = render_extended_bank(source, ExtendedBankConfig(orders=(1, 24)))
    np.testing.assert_array_equal(output, source)
    assert all(not band["acquired"] for band in trace[-1]["bands"])
    assert all(band["evidence"]["status"] == "insufficient_energy" for band in trace[-1]["bands"])


@pytest.mark.parametrize("tamper", [
    "clock", "bool_clock", "frames", "config", "policy", "coefficient", "mix", "ramp",
    "acquired", "count", "nan", "bool_confidence", "missing", "extra", "order", "estimate",
    "confidence", "evidence_status", "initial_band", "eof_missing",
])
def test_replay_rejects_tampering(rendered, tamper):
    config, desired, noise, _, _, original = rendered
    trace = copy.deepcopy(original)
    band = trace[2]["bands"][1]
    if tamper == "clock": trace[2]["sample_index"] += 1
    elif tamper == "bool_clock": trace[2]["sample_index"] = True
    elif tamper == "frames": trace[0]["total_frames"] -= 1
    elif tamper == "config": trace[0]["config"]["orders"] = [1, 21]
    elif tamper == "policy": trace[0]["policy"]["minimum_coherent_power_fraction"] = .01
    elif tamper == "coefficient": band["b"][0] += .001
    elif tamper == "mix": band["mix"] = 1.
    elif tamper == "ramp": band["ramp_position_frames"] += 1
    elif tamper == "acquired": band["acquired"] = False
    elif tamper == "count": band["consecutive_qualified"] = 99
    elif tamper == "nan": band["evidence"]["amplitude"] = float("nan")
    elif tamper == "bool_confidence": band["evidence"]["confidence"] = True
    elif tamper == "missing": del band["a"]
    elif tamper == "extra": band["other"] = 1
    elif tamper == "order": band["order"] = 24
    elif tamper == "estimate": band["evidence"]["estimate_hz"] = 1460.
    elif tamper == "confidence": band["evidence"]["confidence"] = .01
    elif tamper == "evidence_status": band["evidence"]["status"] = "insufficient_concentration"
    elif tamper == "initial_band": trace[0]["bands"][0]["center_hz"] += .1
    elif tamper == "eof_missing": trace.pop()
    with pytest.raises(ValueError):
        replay_extended_bank(desired + noise, config, trace)


def test_exact_eof_observation_required():
    source = tone(60, seconds=2)
    config = ExtendedBankConfig(orders=(1,))
    output, _, trace = render_extended_bank(source, config)
    assert trace[-1]["sample_index"] == source.size
    np.testing.assert_array_equal(replay_extended_bank(source, config, trace), output)
    with pytest.raises(ValueError):
        replay_extended_bank(source, config, trace[:-1])


@pytest.mark.parametrize("bad", [[], [[.1]], [float("nan")], [float("inf")], [True], [1j], [1.], [-1.01]])
def test_audio_validation(bad):
    with pytest.raises(ValueError):
        render_extended_bank(bad)


def test_wanted_same_frequency_tone_not_semantically_protected():
    # This input is deliberately labelled wanted. Spectral selection cannot know
    # that label, and attenuation is an explicit limitation rather than a pass.
    source = tone(1440, seconds=3)
    output, description, _ = render_extended_bank(source, ExtendedBankConfig(orders=(24,)))
    before = np.sqrt(np.mean(source[-12000:] ** 2))
    after = np.sqrt(np.mean(output[-12000:] ** 2))
    assert 20 * np.log10(before / after) > 10
    assert description["policy"]["wanted_tone_protection"] == "not_guaranteed"


def test_output_overshoot_is_rejected_not_hidden(monkeypatch):
    monkeypatch.setattr(_Band, "process", lambda self, x: np.full_like(x, 1.1))
    with pytest.raises(ValueError, match="no automatic normalization"):
        render_extended_bank(np.zeros(100), ExtendedBankConfig(orders=(1,)))
