"""Declared-profile research guard contracts; not a listening-quality claim."""

import copy
import hashlib
import json

import numpy as np
import pytest

from hum_dsp.profile_guard import (
    OBSERVATION, PROBE, ProfileGuardConfig, _amplitude_probes, _starts,
    render_profile_guard, replay_profile_guard,
)


def tone(frequency, seconds=3, amplitude=.04, phase=.37):
    return amplitude * np.sin(2 * np.pi * frequency * np.arange(round(seconds * 48000)) / 48000 + phase)


def declared(**changes):
    settings = {"profile_start_sample": 0, "profile_end_sample": OBSERVATION}
    settings.update(changes)
    return ProfileGuardConfig(**settings)


@pytest.fixture(scope="module")
def stationary():
    config = declared(orders=(1, 2))
    noise = tone(60)
    desired = tone(713, amplitude=.015)
    desired[:OBSERVATION] = 0
    output, descriptor, trace = render_profile_guard(noise + desired, config)
    return config, noise, desired, output, descriptor, trace


@pytest.mark.parametrize("changes", [
    {"sample_rate_hz": 44100}, {"sample_rate_hz": True}, {"nominal_hum_hz": True},
    {"nominal_hum_hz": 55}, {"nominal_hum_hz": float("nan")},
    {"orders": (2, 1)}, {"orders": (1, 1)}, {"orders": (True,)},
    {"orders": (1.,)}, {"orders": (33,)}, {"orders": "123"},
    {"max_attenuation_db": -1}, {"max_attenuation_db": 13},
    {"max_attenuation_db": True}, {"max_attenuation_db": float("inf")},
    {"profile_start_sample": None}, {"profile_end_sample": None},
    {"profile_start_sample": True}, {"profile_end_sample": 96000.},
    {"profile_start_sample": -1}, {"profile_end_sample": 95999},
    {"block_size_frames": 0}, {"block_size_frames": True}, {"block_size_frames": 4097},
])
def test_invalid_config(changes):
    with pytest.raises(ValueError):
        declared(**changes)


def test_roundtrip_and_mutation_validation():
    config = ProfileGuardConfig(orders=[1, 3], nominal_hum_hz=50)
    assert config.orders == (1, 3)
    assert ProfileGuardConfig(**json.loads(json.dumps(config.to_dict()))) == config
    object.__setattr__(config, "profile_end_sample", 2)
    with pytest.raises(ValueError):
        config.to_dict()


@pytest.mark.parametrize("bad", [[], [[.1]], [True], [1j], [float("nan")], [1.], [-1.01]])
def test_invalid_audio(bad):
    with pytest.raises(ValueError):
        render_profile_guard(bad)


def test_interval_must_fit_input():
    with pytest.raises(ValueError, match="exceeds"):
        render_profile_guard(np.zeros(95999), declared())


@pytest.mark.parametrize("changes,reason", [
    ({"orders": ()}, "no_selected_orders"),
    ({"profile_start_sample": None, "profile_end_sample": None}, "no_profile_declared"),
    ({"max_attenuation_db": 0}, "zero_depth"),
])
def test_explicit_bypass_exact_and_no_estimation(monkeypatch, changes, reason):
    monkeypatch.setattr(np.fft, "rfft", lambda *a, **k: pytest.fail("bypass cannot run FFT"))
    config, source = declared(**changes), tone(60, seconds=2)
    output, descriptor, trace = render_profile_guard(source, config)
    np.testing.assert_array_equal(output, source)
    np.testing.assert_array_equal(replay_profile_guard(source, config, trace), source)
    assert descriptor["bypass_reason"] == reason
    assert descriptor["processing_bypassed"] is True
    assert trace["bands"] == [] and trace["extended_bank_trace"] is None


def test_profile_support_exactly_one_window_allowed_and_unsupported_order_dry(stationary):
    _, _, _, _, descriptor, trace = stationary
    first, second = trace["bands"]
    assert len(first["profile_observations"]) == 1
    assert first["profile_supported"] is True
    assert first["frequency_hz"] == pytest.approx(60, abs=.002)
    assert first["profile_reference_amplitude"] == pytest.approx(.04, abs=1e-6)
    assert first["approved"] is True
    assert second["profile_supported"] is False
    assert second["reason"] == "profile_unsupported"
    assert trace["approved_orders"] == [1]
    assert descriptor["policy"]["minimum_eligible_profile_windows"] == 1


def test_stationary_noise_attenuates_and_replay_is_exact_no_fft(stationary, monkeypatch):
    config, noise, desired, output, _, trace = stationary
    assert np.sqrt(np.mean(output[-12000:] ** 2)) < np.sqrt(np.mean((noise + desired)[-12000:] ** 2)) * .6
    monkeypatch.setattr(np.fft, "rfft", lambda *a, **k: pytest.fail("replay cannot run FFT"))
    monkeypatch.setattr(np.linalg, "pinv", lambda *a, **k: pytest.fail("replay cannot estimate amplitude"))
    np.testing.assert_array_equal(replay_profile_guard(noise + desired, config, trace), output)


def test_frozen_component_replay_additive_and_not_hash_authenticated(stationary):
    config, noise, desired, output, _, trace = stationary
    combined = replay_profile_guard(noise, config, trace) + replay_profile_guard(desired, config, trace)
    np.testing.assert_allclose(combined, output, rtol=0, atol=3e-13)
    assert hashlib.sha256(np.asarray(noise, dtype="<f8").tobytes()).hexdigest() != trace["input_canonical_float64_sha256"]


def test_hashes_identify_canonical_whole_input_and_selected_interval(stationary):
    _, noise, desired, _, descriptor, trace = stationary
    source = noise + desired
    assert trace["input_canonical_float64_sha256"] == hashlib.sha256(np.asarray(source, dtype="<f8").tobytes()).hexdigest()
    assert trace["profile_interval_canonical_float64_sha256"] == hashlib.sha256(np.asarray(source[:96000], dtype="<f8").tobytes()).hexdigest()
    assert descriptor["input_canonical_float64_sha256"] == trace["input_canonical_float64_sha256"]


def test_strong_later_overlap_vetoes_entire_order_before_overlap():
    source = tone(60, seconds=3, amplitude=.02)
    source[96000:] += tone(60, seconds=1, amplitude=.10)
    output, descriptor, trace = render_profile_guard(source, declared(orders=(1,)))
    np.testing.assert_array_equal(output, source)
    record = trace["bands"][0]
    assert record["amplitude_guard_rejected"] is True
    assert record["whole_file_maximum_amplitude"] > .10
    assert record["maximum_amplitude_ratio"] > 5
    assert record["whole_file_maximum_probe_start_sample"] >= 96000
    assert descriptor["processing_bypassed"] is True
    assert descriptor["policy"]["causal"] is False


def test_same_veto_applies_to_growing_noise_not_semantic_source_separation():
    source = tone(60, amplitude=.02)
    source[96000:] *= 4
    output, _, trace = render_profile_guard(source, declared(orders=(1,)))
    np.testing.assert_array_equal(output, source)
    assert trace["bands"][0]["reason"] == "whole_file_amplitude_exceeds_profile"


def test_equal_amplitude_overlap_is_not_guaranteed_protected():
    source = tone(60, amplitude=.02)
    source[96000:] += tone(60, seconds=1, amplitude=.01)
    _, descriptor, trace = render_profile_guard(source, declared(orders=(1,)))
    assert trace["approved_orders"] == [1]
    assert descriptor["policy"]["equal_amplitude_wanted_overlap_protection"] == "not_guaranteed"


def test_selected_profile_not_assumed_whole_file_and_non_grid_edges():
    source = tone(60.3, seconds=3.3)
    config = ProfileGuardConfig(orders=(1,), profile_start_sample=7000, profile_end_sample=110111)
    _, _, trace = render_profile_guard(source, config)
    record = trace["bands"][0]
    assert [row["start_sample"] for row in record["profile_observations"]] == [7000, 14111]
    assert record["profile_probes"][0]["start_sample"] == 7000
    assert record["profile_probes"][-1]["end_sample"] == 110111
    assert record["whole_file_probes"][-1]["end_sample"] == source.size
    assert record["frequency_hz"] == pytest.approx(60.3, abs=.003)


def test_amplitude_least_squares_matches_known_tone_plus_dc():
    source = tone(60.39, seconds=.5, amplitude=.073) + .12
    records = _amplitude_probes(source, 60.39, [0])
    assert records == [{"start_sample": 0, "end_sample": PROBE,
                        "amplitude": pytest.approx(.073, abs=1e-14)}]
    assert _starts(2, 100, 20, 30) == [2, 32, 62, 80]


def test_silent_profile_bypasses_later_tones():
    source = tone(60)
    source[:96000] = 0
    output, _, trace = render_profile_guard(source, declared(orders=(1,)))
    np.testing.assert_array_equal(output, source)
    assert trace["bands"][0]["profile_observations"][0]["status"] == "insufficient_energy"


@pytest.mark.parametrize("tamper", [
    "schema", "config", "policy", "frames", "bool_frames", "missing", "extra",
    "identity", "profile_hash", "order", "approved", "reason", "frequency", "reference",
    "ratio", "maximum", "location", "guard", "support", "observation_clock", "amplitude_nan",
    "amplitude_bool", "amplitude_int", "fraction_int", "frequency_int", "probe_clock",
    "probe_count", "observation_status", "nested",
])
def test_replay_rejects_inconsistent_trace(stationary, tamper):
    config, noise, desired, _, _, frozen = stationary
    trace = copy.deepcopy(frozen)
    band = trace["bands"][0]
    if tamper == "schema": trace["schema_version"] = True
    elif tamper == "config": trace["config"]["max_attenuation_db"] = 3.
    elif tamper == "policy": trace["policy"]["guard_ratio"] = 3.
    elif tamper == "frames": trace["source_frames"] += 1
    elif tamper == "bool_frames": trace["source_frames"] = True
    elif tamper == "missing": del trace["source_frames"]
    elif tamper == "extra": trace["other"] = None
    elif tamper == "identity": trace["input_canonical_float64_sha256"] = "not_a_hash"
    elif tamper == "profile_hash": trace["profile_interval_canonical_float64_sha256"] = None
    elif tamper == "order": band["order"] = 2
    elif tamper == "approved": trace["approved_orders"] = [1, 2]
    elif tamper == "reason": band["reason"] = "profile_unsupported"
    elif tamper == "frequency": band["frequency_hz"] += .1
    elif tamper == "reference": band["profile_reference_amplitude"] *= 2
    elif tamper == "ratio": band["maximum_amplitude_ratio"] = .1
    elif tamper == "maximum": band["whole_file_maximum_amplitude"] *= 2
    elif tamper == "location": band["whole_file_maximum_probe_start_sample"] += 1
    elif tamper == "guard": band["amplitude_guard_rejected"] = True
    elif tamper == "support": band["profile_supported"] = False
    elif tamper == "observation_clock": band["profile_observations"][0]["start_sample"] = True
    elif tamper == "amplitude_nan": band["whole_file_probes"][0]["amplitude"] = float("nan")
    elif tamper == "amplitude_bool": band["profile_probes"][0]["amplitude"] = True
    elif tamper == "amplitude_int": band["profile_probes"][0]["amplitude"] = 0
    elif tamper == "fraction_int": band["profile_observations"][0]["coherent_power_fraction"] = 1
    elif tamper == "frequency_int": band["profile_observations"][0]["estimate_hz"] = 60
    elif tamper == "probe_clock": band["whole_file_probes"][0]["end_sample"] += 1
    elif tamper == "probe_count": band["whole_file_probes"].pop()
    elif tamper == "observation_status": band["profile_observations"][0]["status"] = "insufficient_energy"
    elif tamper == "nested": trace["extended_bank_trace"][0]["policy"]["required_qualified_events"] = 1
    with pytest.raises(ValueError):
        replay_profile_guard(noise + desired, config, trace)


def test_no_profile_short_file_is_supported_but_tampered_bypass_rejected():
    source = np.array([.01, -.01])
    config = ProfileGuardConfig(profile_start_sample=None, profile_end_sample=None)
    output, _, trace = render_profile_guard(source, config)
    np.testing.assert_array_equal(output, source)
    trace["bands"] = [{}]
    with pytest.raises(ValueError):
        replay_profile_guard(source, config, trace)


def test_default_api_never_assumes_first_two_seconds_are_noise_only(monkeypatch):
    source = tone(60)
    monkeypatch.setattr(np.fft, "rfft", lambda *a, **k: pytest.fail("undeclared profile cannot estimate"))
    output, descriptor, trace = render_profile_guard(source)
    np.testing.assert_array_equal(output, source)
    assert descriptor["bypass_reason"] == "no_profile_declared"
    assert trace["profile_interval_canonical_float64_sha256"] is None
    np.testing.assert_array_equal(replay_profile_guard(source, ProfileGuardConfig(), trace), source)


def test_replay_requires_config_type(stationary):
    _, noise, desired, _, _, trace = stationary
    with pytest.raises(ValueError):
        replay_profile_guard(noise + desired, {}, trace)

