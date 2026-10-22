"""Preregistered H1 mathematical inventory; not a processor performance test."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from hum_dsp.audio import SAMPLE_RATE_HZ
from hum_dsp.harmonic_fixtures import HFixture, harmonic_fixtures


@pytest.fixture(scope="module")
def cases():
    return harmonic_fixtures()


@pytest.fixture(scope="module")
def by_name(cases):
    return {case.case_id.removeprefix("synthetic_h1_"): case for case in cases}


def test_frozen_inventory(cases):
    assert isinstance(cases, tuple)
    assert len(cases) == 12
    assert tuple(case.case_id for case in cases) == tuple(f"synthetic_h1_{name}" for name in (
        "compound_stationary_noise_only_50", "compound_stationary_noise_only_60",
        "compound_stationary_mixture_50", "compound_stationary_mixture_60",
        "fundamental_only_50", "missing_fundamental_60", "clean_program_50",
        "neighboring_desired_50", "wanted_harmonic_50", "silence_50",
        "abrupt_end_50", "drift_compound_50",
    ))


@pytest.mark.parametrize("index", range(12))
def test_arrays_metadata_readonly_and_bounded(cases, index):
    case = cases[index]
    assert isinstance(case, HFixture)
    assert case.nominal_hum_hz in (50.0, 60.0)
    assert isinstance(case.kind, str) and case.kind
    assert "not real-world validation" in case.notes
    for samples in (case.clean, case.noise, case.input_audio):
        assert samples.dtype == np.dtype(np.float64)
        assert samples.ndim == 1
        assert samples.size == 8 * SAMPLE_RATE_HZ
        assert np.isfinite(samples).all()
        assert np.max(np.abs(samples)) < 1
        assert not samples.flags.writeable
        with pytest.raises(ValueError):
            samples[0] = 1.0
    np.testing.assert_array_equal(case.input_audio, case.clean + case.noise)
    with pytest.raises(FrozenInstanceError):
        case.kind = "changed"


@pytest.mark.parametrize("nominal", (50, 60))
def test_stationary_harmonic_rms_and_phases(by_name, nominal):
    case = by_name[f"compound_stationary_noise_only_{nominal}"]
    time = np.arange(case.input_audio.size, dtype=np.float64) / SAMPLE_RATE_HZ
    expected = np.zeros_like(time)
    for order, phase in enumerate((0.37, 1.13, -0.61), start=1):
        expected += np.sqrt(2) * 0.012 * np.sin(2 * np.pi * (nominal * order) * time + phase)
    np.testing.assert_array_equal(case.noise, expected)
    np.testing.assert_array_equal(case.clean, np.zeros_like(time))
    assert np.sqrt(np.mean(case.noise ** 2)) == pytest.approx(np.sqrt(3) * 0.012, abs=1e-14)
    assert case.kind == "stationary_noise_only"


@pytest.mark.parametrize("nominal", (50, 60))
def test_mixtures_share_owned_program_and_noise(by_name, nominal):
    mixture = by_name[f"compound_stationary_mixture_{nominal}"]
    noise_only = by_name[f"compound_stationary_noise_only_{nominal}"]
    np.testing.assert_array_equal(mixture.noise, noise_only.noise)
    np.testing.assert_array_equal(mixture.clean, by_name["clean_program_50"].clean)
    assert np.sqrt(np.mean(mixture.clean ** 2)) == pytest.approx(np.sqrt((0.06 ** 2 + 0.04 ** 2 + 0.02 ** 2) / 2), abs=1e-14)


def test_absent_harmonics_are_really_absent(by_name):
    for name, nominal, present in (
        ("fundamental_only_50", 50, (1,)), ("missing_fundamental_60", 60, (2, 3)),
    ):
        samples = by_name[name].noise
        time = np.arange(samples.size, dtype=np.float64) / SAMPLE_RATE_HZ
        for order in (1, 2, 3):
            amplitude = 2 * abs(np.mean(samples * np.exp(-2j * np.pi * nominal * order * time)))
            assert amplitude == pytest.approx(np.sqrt(2) * 0.012 if order in present else 0.0, abs=1e-13)


def test_semantic_limits_are_not_preservation_claims(by_name):
    wanted = by_name["wanted_harmonic_50"]
    neighbor = by_name["neighboring_desired_50"]
    for case in (wanted, neighbor):
        assert np.count_nonzero(case.noise) == 0
        assert "KNOWN SEMANTIC LIMIT" in case.notes
    assert wanted.kind == "wanted_harmonic_limit"
    assert neighbor.kind == "neighboring_desired_limit"
    assert "not an exact-dry" in neighbor.notes
    assert "waveform-identical" in wanted.notes


def test_negative_controls_and_quantization_boundary(by_name):
    for name in ("clean_program_50", "silence_50"):
        case = by_name[name]
        assert np.count_nonzero(case.noise) == 0
        assert "exactly dry" in case.notes
    assert np.count_nonzero(by_name["silence_50"].input_audio) == 0
    # Tiny nonzero math is not converted to fake infinite attenuation. At the
    # shared PCM16 input boundary these values become exactly silent instead.
    sub_lsb = by_name["fundamental_only_50"].input_audio * 1e-8
    assert np.count_nonzero(sub_lsb) > 0
    assert np.count_nonzero(np.rint(sub_lsb * 32768).astype("<i2")) == 0


def test_compound_offset_fade_and_exact_tail(by_name):
    case = by_name["abrupt_end_50"]
    original = by_name["compound_stationary_noise_only_50"].noise
    start, end = 192000, 192960
    np.testing.assert_array_equal(case.noise[:start], original[:start])
    expected = original[start:end] * 0.5 * (1 + np.cos(np.pi * np.arange(960) / 960))
    np.testing.assert_array_equal(case.noise[start:end], expected)
    assert np.count_nonzero(case.noise[end:]) == 0
    assert "192960" in case.notes


def test_drift_is_phase_continuous_integrated_frequency(by_name):
    case = by_name["drift_compound_50"]
    time = np.arange(case.noise.size, dtype=np.float64) / SAMPLE_RATE_HZ
    cycles = 49.8 * time + 0.5 * 0.02 * time ** 2
    expected = sum((np.sqrt(2) * 0.012 * np.sin(2 * np.pi * order * cycles + phase)
                    for order, phase in enumerate((0.37, 1.13, -0.61), start=1)), start=np.zeros_like(time))
    np.testing.assert_array_equal(case.noise, expected)
    assert np.max(np.abs(np.diff(case.noise))) < 0.002


def test_generation_is_deterministic(cases):
    for first, second in zip(cases, harmonic_fixtures(), strict=True):
        assert first.case_id == second.case_id
        assert first.kind == second.kind
        assert first.notes == second.notes
        for name in ("clean", "noise", "input_audio"):
            np.testing.assert_array_equal(getattr(first, name), getattr(second, name))
