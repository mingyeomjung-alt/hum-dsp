"""Independent public contracts for the opt-in, nonsemantic offline comb.

These checks establish deterministic signal behavior, not listening quality,
natural-noise separation, or permission to replace the existing default engine.
"""

import copy
import json

import numpy as np
import pytest

from hum_dsp.persistent_comb import (
    CombConfig, render_persistent_comb, replay_persistent_comb,
)


def _tone(frequency, rate=48000, seconds=5, amplitude=.035, phase=.23):
    time = np.arange(round(rate * seconds), dtype=np.float64) / rate
    return amplitude * np.sin(2 * np.pi * frequency * time + phase)


def _parts(rate=48000, nominal=60):
    # Four stationary, independently phased lines meet the declared minimum
    # comb evidence; the distant wanted line is not a selected noise component.
    noise = sum(_tone(nominal * order, rate, phase=.27 * order)
                for order in (1, 2, 3, 4))
    wanted = _tone(733.25, rate, amplitude=.06, phase=.71)
    return wanted, noise


def _rms(values):
    return float(np.sqrt(np.mean(np.asarray(values) ** 2)))


def _amplitude(values, frequency, rate):
    # Independent sine/cosine least-squares measurement in the central region,
    # not the engine's own FFT evidence or a fitted output gain/alignment.
    selected = np.asarray(values)[rate:-rate]
    time = np.arange(len(selected), dtype=np.float64) / rate
    basis = np.column_stack((np.sin(2 * np.pi * frequency * time),
                             np.cos(2 * np.pi * frequency * time),
                             np.ones(len(selected))))
    coefficients = np.linalg.lstsq(basis, selected, rcond=None)[0]
    return float(np.hypot(coefficients[0], coefficients[1]))


@pytest.fixture(scope="module")
def rendered():
    config = CombConfig(orders=(1, 2, 3, 4), max_attenuation_db=12)
    wanted, noise = _parts()
    source = wanted + noise
    output, descriptor, trace = render_persistent_comb(source, config)
    return config, wanted, noise, source, output, descriptor, trace


@pytest.mark.parametrize("changes", [
    {"sample_rate_hz": 8000}, {"sample_rate_hz": 47999},
    {"sample_rate_hz": True}, {"sample_rate_hz": 48000.0},
    {"nominal_hum_hz": 55}, {"nominal_hum_hz": True},
    {"nominal_hum_hz": float("nan")}, {"nominal_hum_hz": float("inf")},
    {"max_attenuation_db": -1}, {"max_attenuation_db": 18.01},
    {"max_attenuation_db": True}, {"max_attenuation_db": float("nan")},
    {"max_attenuation_db": float("inf")},
    {"orders": ()}, {"orders": (0, 1, 2, 3)}, {"orders": (1, 2, 3, 33)},
    {"orders": (1, 2, 3, 3)}, {"orders": (4, 3, 2, 1)},
    {"orders": (1, 2, 3, True)}, {"orders": (1, 2, 3, 4.0)},
    {"orders": "1234"}, {"orders": None},
])
def test_configuration_rejects_invalid_values(changes):
    with pytest.raises((TypeError, ValueError)):
        CombConfig(**changes)


@pytest.mark.parametrize("depth", [0, 12, 18])
@pytest.mark.parametrize("rate", [44100, 48000])
def test_declared_configuration_boundaries(depth, rate):
    config = CombConfig(sample_rate_hz=rate, nominal_hum_hz=50,
                        max_attenuation_db=depth, orders=(1, 4, 17, 32))
    assert config.sample_rate_hz == rate
    assert config.max_attenuation_db == depth
    assert tuple(config.orders) == (1, 4, 17, 32)


@pytest.mark.parametrize("bad", [None, False, 3, {}, "manual"])
def test_non_config_objects_are_rejected(bad):
    # None is the documented default for render, but never a replay config.
    if bad is not None:
        with pytest.raises((TypeError, ValueError)):
            render_persistent_comb(np.zeros(4 * 48000), bad)
    with pytest.raises((TypeError, ValueError)):
        replay_persistent_comb(np.zeros(4 * 48000), bad, {})


@pytest.mark.parametrize("kind", [
    "empty", "too_long", "scalar", "three_channels", "three_dimensions",
    "bool", "complex", "object", "nan", "infinity", "positive_endpoint",
    "below_negative_endpoint",
])
def test_audio_validation_applies_even_on_bypass(kind):
    source = np.zeros(4 * 48000, dtype=np.float64)
    if kind == "empty": source = np.array([])
    elif kind == "too_long": source = np.broadcast_to(np.zeros(1), (60 * 48000 + 1,))
    elif kind == "scalar": source = np.array(.2)
    elif kind == "three_channels": source = np.zeros((4 * 48000, 3))
    elif kind == "three_dimensions": source = source[:, None, None]
    elif kind == "bool": source = source.astype(bool)
    elif kind == "complex": source = source.astype(complex)
    elif kind == "object": source = source.astype(object)
    elif kind == "nan": source[10] = np.nan
    elif kind == "infinity": source[10] = np.inf
    elif kind == "positive_endpoint": source[10] = 1
    else: source[10] = -1.0001
    with pytest.raises((TypeError, ValueError)):
        render_persistent_comb(source, CombConfig(max_attenuation_db=0))


def test_mutated_frozen_configuration_is_revalidated():
    config = CombConfig()
    object.__setattr__(config, "max_attenuation_db", 100)
    with pytest.raises((TypeError, ValueError)):
        render_persistent_comb(np.zeros(4 * 48000), config)


@pytest.mark.parametrize("layout", ["mono_vector", "mono_matrix", "stereo"])
def test_zero_depth_is_exact_copy_without_transform(monkeypatch, layout):
    left = _tone(60, seconds=4)
    left[::19] = -0.0
    source = left if layout == "mono_vector" else left[:, None]
    if layout == "stereo": source = np.column_stack((left, -left))
    saved = source.copy()

    def forbidden(*args, **kwargs):
        pytest.fail("zero depth must not estimate or transform audio")

    monkeypatch.setattr(np.fft, "rfft", forbidden)
    monkeypatch.setattr(np.fft, "irfft", forbidden)
    config = CombConfig(max_attenuation_db=0)
    output, descriptor, trace = render_persistent_comb(source, config)
    replayed = replay_persistent_comb(source, config, trace)
    np.testing.assert_array_equal(output, saved)
    np.testing.assert_array_equal(replayed, saved)
    np.testing.assert_array_equal(np.signbit(output), np.signbit(saved))
    assert output.shape == saved.shape
    assert not np.shares_memory(output, source)
    np.testing.assert_array_equal(source, saved)
    json.dumps((descriptor, trace), allow_nan=False)


@pytest.mark.parametrize("source_kind", ["silence", "single_wanted_tone", "three_lines"])
def test_insufficient_comb_evidence_returns_exact_dry(source_kind):
    source = np.zeros(4 * 48000)
    if source_kind == "single_wanted_tone": source = _tone(240, seconds=4)
    elif source_kind == "three_lines":
        source = sum(_tone(f, seconds=4) for f in (60, 120, 180))
    output, descriptor, trace = render_persistent_comb(source, CombConfig())
    np.testing.assert_array_equal(output, source)
    np.testing.assert_array_equal(replay_persistent_comb(source, CombConfig(), trace), source)
    assert output.shape == source.shape
    json.dumps((descriptor, trace), allow_nan=False)


def test_short_lived_comb_does_not_meet_whole_file_persistence():
    _, noise = _parts()
    source = np.zeros_like(noise)
    source[:48000] = noise[:48000]
    output, _, _ = render_persistent_comb(source, CombConfig(orders=(1, 2, 3, 4)))
    np.testing.assert_array_equal(output, source)


@pytest.mark.parametrize("frames", [1, 48000, 2 * 48000, 4 * 48000 - 1])
def test_too_short_for_three_observations_is_explicit_exact_bypass(frames):
    source = sum(_tone(60 * order, seconds=4, phase=.19 * order)
                 for order in (1, 2, 3, 4))[:frames]
    config = CombConfig(orders=(1, 2, 3, 4))
    output, descriptor, trace = render_persistent_comb(source, config)
    np.testing.assert_array_equal(output, source)
    assert descriptor["reason"] == "insufficient_analysis_windows"
    assert trace["reason"] == "insufficient_analysis_windows"
    assert trace["lines"] == []
    np.testing.assert_array_equal(replay_persistent_comb(source, config, trace), source)


def test_four_harmonics_receive_substantial_local_attenuation(rendered):
    _, _, _, source, output, _, _ = rendered
    for frequency in (60, 120, 180, 240):
        before = _amplitude(source, frequency, 48000)
        after = _amplitude(output, frequency, 48000)
        assert 20 * np.log10(before / after) >= 10
    assert output.shape == source.shape
    assert np.isfinite(output).all()
    assert np.max(np.abs(output)) < 1


def test_frozen_replay_preserves_offband_wanted_component(rendered):
    config, wanted, _, _, _, _, trace = rendered
    replayed = replay_persistent_comb(wanted, config, trace)
    central = slice(48000, -48000)
    relative_error = _rms(replayed[central] - wanted[central]) / _rms(wanted[central])
    assert relative_error < .002
    assert abs(20 * np.log10(_rms(replayed[central]) / _rms(wanted[central]))) < .01


def test_render_and_replay_do_not_mutate_inputs_or_trace(rendered):
    config, _, _, source, expected, descriptor, trace = rendered
    input_copy, original_trace = source.copy(), copy.deepcopy(trace)
    output, second_descriptor, second_trace = render_persistent_comb(source, config)
    np.testing.assert_array_equal(output, expected)
    assert second_descriptor == descriptor
    assert second_trace == trace
    np.testing.assert_array_equal(replay_persistent_comb(source, config, trace), expected)
    np.testing.assert_array_equal(source, input_copy)
    assert trace == original_trace
    json.dumps((descriptor, trace), allow_nan=False)


def test_frozen_mixture_replay_additivity_and_no_reestimation(rendered, monkeypatch):
    config, wanted, noise, _, expected, _, trace = rendered
    # Applying a mask requires FFT, but does not require estimating another
    # center; replace only the estimation helper when the module exposes it.
    import hum_dsp.persistent_comb as module
    for name in ("_observations",):
        if hasattr(module, name):
            monkeypatch.setattr(module, name, lambda *args, **kwargs: pytest.fail("replay re-estimated evidence"))
    desired_out = replay_persistent_comb(wanted, config, trace)
    noise_out = replay_persistent_comb(noise, config, trace)
    np.testing.assert_allclose(desired_out + noise_out, expected, rtol=0, atol=2e-12)


@pytest.mark.parametrize("bad_trace", [None, [], {}, {"schema_version": True}, "trace"])
def test_malformed_replay_trace_rejected(rendered, bad_trace):
    config, _, _, source, _, _, _ = rendered
    with pytest.raises((TypeError, ValueError)):
        replay_persistent_comb(source, config, bad_trace)


def test_replay_binds_config_length_and_channel_layout(rendered):
    config, _, _, source, _, _, trace = rendered
    wrong = CombConfig(orders=(1, 2, 3, 4), max_attenuation_db=6)
    with pytest.raises((TypeError, ValueError)):
        replay_persistent_comb(source, wrong, trace)
    with pytest.raises((TypeError, ValueError)):
        replay_persistent_comb(source[:-1], config, trace)
    with pytest.raises((TypeError, ValueError)):
        replay_persistent_comb(np.column_stack((source, source)), config, trace)


@pytest.mark.parametrize("tamper", [
    "schema_bool", "algorithm", "config", "policy", "frames", "channels", "clock",
    "hash", "hash_type", "missing", "extra", "base", "reason", "missing_line",
    "order", "order_bool", "count", "count_bool", "total_windows", "frequency",
    "frequency_nan", "frequency_bool", "prominence", "mad_negative", "mad_bool",
    "persistent_bool", "accepted_bool", "acceptance_flip", "line_reason", "line_extra",
])
def test_frozen_trace_rejects_schema_and_decision_tampering(rendered, tamper):
    config, _, _, source, _, _, original = rendered
    trace = copy.deepcopy(original)
    row = trace["lines"][0]
    if tamper == "schema_bool": trace["schema_version"] = True
    elif tamper == "algorithm": trace["algorithm"] = "other"
    elif tamper == "config": trace["config"]["max_attenuation_db"] = 6
    elif tamper == "policy": trace["policy"]["minimum_local_prominence_db"] = 1
    elif tamper == "frames": trace["frames"] += 1
    elif tamper == "channels": trace["channels"] = True
    elif tamper == "clock": trace["analysis_windows"] += 1
    elif tamper == "hash": trace["mixture_canonical_float64_sha256"] = "z" * 64
    elif tamper == "hash_type": trace["mixture_canonical_float64_sha256"] = None
    elif tamper == "missing": del trace["reason"]
    elif tamper == "extra": trace["trusted"] = True
    elif tamper == "base": trace["base_frequency_hz"] += .1
    elif tamper == "reason": trace["reason"] = "zero_depth"
    elif tamper == "missing_line": trace["lines"].pop()
    elif tamper == "order": row["order"] = 2
    elif tamper == "order_bool": row["order"] = True
    elif tamper == "count": row["eligible_windows"] = 99
    elif tamper == "count_bool": row["eligible_windows"] = True
    elif tamper == "total_windows": row["total_windows"] += 1
    elif tamper == "frequency": row["candidate_frequency_hz"] = 66
    elif tamper == "frequency_nan": row["candidate_frequency_hz"] = np.nan
    elif tamper == "frequency_bool": row["candidate_frequency_hz"] = True
    elif tamper == "prominence": row["median_prominence_db"] = 9.9
    elif tamper == "mad_negative": row["frequency_mad_hz"] = -.1
    elif tamper == "mad_bool": row["frequency_mad_hz"] = False
    elif tamper == "persistent_bool": row["persistent"] = 1
    elif tamper == "accepted_bool": row["accepted"] = 1
    elif tamper == "acceptance_flip": row["accepted"] = False
    elif tamper == "line_reason": row["reason"] = "safe"
    else: row["extra"] = 1
    with pytest.raises((TypeError, ValueError)):
        replay_persistent_comb(source, config, trace)


@pytest.mark.parametrize("rate,nominal", [(44100, 50), (48000, 60)])
def test_native_rate_and_antiphase_stereo_do_not_erase_detection(rate, nominal):
    wanted, noise = _parts(rate, nominal)
    source = np.column_stack((wanted + noise, -(wanted + noise)))
    config = CombConfig(sample_rate_hz=rate, nominal_hum_hz=nominal,
                        orders=(1, 2, 3, 4))
    output, _, trace = render_persistent_comb(source, config)
    assert output.shape == source.shape
    for frequency in (nominal, 2 * nominal, 3 * nominal, 4 * nominal):
        before = _amplitude(source[:, 0], frequency, rate)
        after = _amplitude(output[:, 0], frequency, rate)
        assert 20 * np.log10(before / after) > 10
    np.testing.assert_allclose(output[:, 0], -output[:, 1], rtol=0, atol=2e-12)
    np.testing.assert_array_equal(replay_persistent_comb(source, config, trace), output)


def test_wanted_harmonic_comb_is_not_semantically_protected():
    # These four deliberately wanted notes are numerically identical to a hum
    # comb. Their suppression is a documented limitation, NOT a quality pass.
    wanted = sum(_tone(60 * order, phase=.13 * order) for order in (1, 2, 3, 4))
    output, descriptor, _ = render_persistent_comb(wanted, CombConfig(orders=(1, 2, 3, 4)))
    assert 20 * np.log10(_rms(wanted[48000:-48000]) / _rms(output[48000:-48000])) > 10
    assert descriptor != {}
