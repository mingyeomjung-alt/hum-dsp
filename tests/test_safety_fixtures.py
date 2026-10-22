"""Frozen mathematical contract for M4 safety cases, not engine performance."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from hum_dsp.audio import SAMPLE_RATE_HZ
from hum_dsp.safety_fixtures import SafetyFixture, safety_fixtures


@pytest.fixture(scope="module")
def cases():
    return safety_fixtures()


@pytest.fixture(scope="module")
def by_id(cases):
    return {case.case_id.removeprefix("synthetic_m4_"): case for case in cases}


def test_case_inventory_is_frozen(cases):
    assert isinstance(cases, tuple)
    assert len(cases) == 10
    assert tuple(case.case_id for case in cases) == tuple(
        f"synthetic_m4_{suffix}" for suffix in (
            "tone_ends_50", "tone_appears_60", "soft_toggle_50", "inband_step_50",
            "silence", "impulse", "desired_tone_50", "below_pcm_floor_60", "dc",
            "nearby_desired_60",
        )
    )
    assert all(isinstance(case, SafetyFixture) for case in cases)


@pytest.mark.parametrize("index", range(10))
def test_arrays_and_metadata_contract(cases, index):
    case = cases[index]
    assert case.nominal_hum_hz in (50.0, 60.0)
    assert type(case.expected_final_dry) is bool
    assert type(case.expect_exact_dry_everywhere) is bool
    assert "synthetic" in case.notes
    assert "not real-world validation" in case.notes
    assert isinstance(case.controls, tuple)
    assert list(case.controls) == sorted(case.controls)
    assert all(type(sample) is int and type(enabled) is bool for sample, enabled in case.controls)
    for samples in (case.clean, case.noise, case.input_audio):
        assert samples.ndim == 1
        assert samples.dtype == np.dtype(np.float64)
        assert not samples.flags.writeable
        assert np.isfinite(samples).all()
        assert np.max(np.abs(samples)) < 1.0
        assert len(samples) == len(case.input_audio)
    np.testing.assert_array_equal(case.input_audio, case.clean + case.noise)
    with pytest.raises(ValueError):
        case.input_audio[0] = 0.0
    with pytest.raises(FrozenInstanceError):
        case.nominal_hum_hz = 60.0


def test_duration_contract(by_id):
    for name, case in by_id.items():
        seconds = 10 if name in ("tone_ends_50", "tone_appears_60") else 12 if name in ("soft_toggle_50", "inband_step_50") else 8
        assert case.input_audio.size == seconds * SAMPLE_RATE_HZ


def test_ending_tone_has_exact_tail_and_raised_cosine(by_id):
    case = by_id["tone_ends_50"]
    end = 4 * SAMPLE_RATE_HZ + 960
    assert case.noise_end_sample == end
    assert case.expected_final_dry is True
    assert np.count_nonzero(case.noise[end:]) == 0
    time = np.arange(case.noise.size, dtype=np.float64) / SAMPLE_RATE_HZ
    tone = np.sqrt(2.0) * 0.02 * np.sin(2.0 * np.pi * (50.0 * time) + 0.37)
    start = 4 * SAMPLE_RATE_HZ
    np.testing.assert_array_equal(case.noise[:start], tone[:start])
    envelope = 0.5 * (1.0 + np.cos(np.pi * np.arange(960) / 960))
    np.testing.assert_array_equal(case.noise[start:end], tone[start:end] * envelope)
    assert abs(case.noise[end - 1] - case.noise[end]) < 1e-6


def test_appearing_tone_is_silent_before_fade(by_id):
    case = by_id["tone_appears_60"]
    start = 3 * SAMPLE_RATE_HZ
    end = start + 960
    assert np.count_nonzero(case.noise[:start + 1]) == 0
    time = np.arange(case.noise.size, dtype=np.float64) / SAMPLE_RATE_HZ
    tone = np.sqrt(2.0) * 0.02 * np.sin(2.0 * np.pi * (60.0 * time) + 0.37)
    envelope = 0.5 * (1.0 - np.cos(np.pi * np.arange(960) / 960))
    np.testing.assert_array_equal(case.noise[start:end], tone[start:end] * envelope)
    np.testing.assert_array_equal(case.noise[end:], tone[end:])


def test_controls_and_disable_interval_are_sample_exact(by_id):
    case = by_id["soft_toggle_50"]
    assert case.controls == ((192000, False), (288000, True))
    assert case.disable_interval == (204000, 288000)
    assert case.expected_final_dry is False
    for name, other in by_id.items():
        if name != "soft_toggle_50":
            assert other.controls == ()
            assert other.disable_interval is None


def test_frequency_step_is_phase_continuous(by_id):
    case = by_id["inband_step_50"]
    boundary = 5 * SAMPLE_RATE_HZ
    amplitude = np.sqrt(2.0) * 0.02
    time = np.arange(case.noise.size, dtype=np.float64) / SAMPLE_RATE_HZ
    cycles = 50.8 * np.minimum(time, 5.0) + 49.2 * np.maximum(time - 5.0, 0.0)
    expected = amplitude * np.sin(2.0 * np.pi * cycles + 0.37)
    np.testing.assert_array_equal(case.noise, expected)
    # First difference remains bounded by the largest sinusoid slope on either
    # side, including the event boundary; there is no introduced phase spike.
    assert np.max(np.abs(np.diff(case.noise[boundary - 3:boundary + 4]))) <= amplitude * 2.0 * np.pi * 50.8 / SAMPLE_RATE_HZ
    assert case.expected_final_dry is True


def test_negative_controls_and_semantic_limit_are_explicit(by_id):
    for name in ("silence", "impulse", "below_pcm_floor_60", "dc", "nearby_desired_60"):
        assert by_id[name].expect_exact_dry_everywhere is True
        assert by_id[name].expected_final_dry is True
    impulse = by_id["impulse"]
    assert np.count_nonzero(impulse.clean) == 1
    assert impulse.clean[3 * SAMPLE_RATE_HZ] == 0.5
    np.testing.assert_array_equal(by_id["dc"].clean, np.full(8 * SAMPLE_RATE_HZ, 0.1))
    desired = by_id["desired_tone_50"]
    assert np.count_nonzero(desired.noise) == 0
    assert desired.expected_final_dry is False
    assert desired.expect_exact_dry_everywhere is False
    assert "NOT a preservation pass criterion" in desired.notes
    # A wanted 50 Hz tone has the exact waveform of the unwanted continuous
    # noise case.  Those desired/noise labels cannot be inferred from samples.
    np.testing.assert_array_equal(desired.clean, by_id["soft_toggle_50"].noise[:desired.clean.size])


def test_below_pcm_floor_decodes_to_exact_zero(by_id):
    case = by_id["below_pcm_floor_60"]
    assert np.any(case.input_audio != 0.0)
    assert np.max(np.abs(case.input_audio)) <= 1e-9
    pcm16 = np.clip(np.rint(case.input_audio * 32768.0), -32768, 32767).astype("<i2")
    assert np.count_nonzero(pcm16) == 0
    assert "decoded PCM16 input is all zero" in case.notes


def test_generation_is_deterministic(cases):
    repeated = safety_fixtures()
    for left, right in zip(cases, repeated, strict=True):
        assert left.case_id == right.case_id
        assert left.controls == right.controls
        assert left.notes == right.notes
        assert left.noise_end_sample == right.noise_end_sample
        np.testing.assert_array_equal(left.clean, right.clean)
        np.testing.assert_array_equal(left.noise, right.noise)
        np.testing.assert_array_equal(left.input_audio, right.input_audio)
