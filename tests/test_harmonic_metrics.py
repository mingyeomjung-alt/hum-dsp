"""Known-reference replay metrics preserve phase error and undefined values."""

import json

import numpy as np
import pytest

from hum_dsp.harmonic_metrics import measure_harmonic_components


def test_perfect_additive_identity_is_not_noise_removal():
    clean = np.array([0.1, -0.2, 0.3, -0.4])
    noise = np.array([-0.1, 0.1, 0.05, 0.1])
    result = measure_harmonic_components(clean, noise, clean + noise, clean, noise)
    assert result["decomposition_additivity_within_1e_10"] is True
    assert result["component_interpretation_valid"] is True
    assert result["decomposition_additivity_max_abs_error"] < 1e-16
    assert result["clean_error_rmse"] == 0
    assert result["clean_relative_error_rmse"] == 0
    assert result["clean_level_change_db"] == 0
    assert result["noise_attenuation_db"] == 0
    assert result["raw_mixture_error_rmse"] == pytest.approx(result["input_noise_rms"])
    assert "not_observational_source_separation" in result["interpretation"]
    assert result["alignment_applied"] is False
    assert result["component_gain_fitting_applied"] is False
    assert result["output_gain_normalization_applied"] is False
    json.dumps(result, allow_nan=False)


def test_full_rms_attenuation_and_clean_damage_are_independent():
    clean = np.sin(np.arange(1000) * 0.03) * 0.1
    noise = np.cos(np.arange(1000) * 0.07) * 0.02
    yc, yn = 0.8 * clean, 0.25 * noise
    result = measure_harmonic_components(clean, noise, yc + yn, yc, yn)
    assert result["clean_relative_error_rmse"] == pytest.approx(0.2)
    assert result["clean_level_change_db"] == pytest.approx(20 * np.log10(0.8))
    assert result["noise_attenuation_db"] == pytest.approx(20 * np.log10(4))
    assert result["residual_noise_rms"] == pytest.approx(np.sqrt(np.mean(yn ** 2)))


def test_phase_inversion_has_no_level_change_but_large_preservation_error():
    clean = np.array([0.2, -0.2, 0.2, -0.2])
    noise = np.array([0.03, 0.03, -0.03, -0.03])
    yc, yn = -clean, -noise
    result = measure_harmonic_components(clean, noise, yc + yn, yc, yn)
    assert result["clean_level_change_db"] == 0
    assert result["clean_relative_error_rmse"] == pytest.approx(2)
    assert result["noise_attenuation_db"] == 0


def test_quadrature_noise_is_not_falsely_removed_by_projection():
    time = np.arange(48000) / 48000
    noise = 0.02 * np.sin(2 * np.pi * 100 * time)
    residual = 0.02 * np.cos(2 * np.pi * 100 * time)
    zeros = np.zeros_like(time)
    result = measure_harmonic_components(zeros, noise, residual, zeros, residual)
    assert abs(result["noise_attenuation_db"]) < 1e-12
    assert result["residual_noise_rms"] == pytest.approx(0.02 / np.sqrt(2))


def test_same_frequency_oracle_replay_not_ill_conditioned_fit():
    clean = np.sin(np.arange(1000) * 0.2) * 0.02
    noise = 2 * clean
    yc, yn = 0.5 * clean, 0.5 * noise
    result = measure_harmonic_components(clean, noise, yc + yn, yc, yn)
    assert result["component_interpretation_valid"] is True
    assert result["clean_relative_error_rmse"] == pytest.approx(0.5)
    assert result["noise_attenuation_db"] == pytest.approx(20 * np.log10(2))


def test_silence_uses_null_reasons_not_infinity():
    zeros = np.zeros(100)
    result = measure_harmonic_components(zeros, zeros, zeros, zeros, zeros)
    assert result["noise_attenuation_db"] is None
    assert result["clean_relative_error_rmse"] is None
    assert result["clean_level_change_db"] is None
    assert result["undefined_metrics"] == {
        "clean_relative_error_rmse": "zero_clean_reference",
        "clean_level_change_db": "zero_clean_reference",
        "noise_attenuation_db": "zero_input_noise",
    }
    assert result["raw_mixture_error_rmse"] == 0
    json.dumps(result, allow_nan=False)


def test_zero_residual_and_muted_clean_have_explicit_nonfinite_db_reasons():
    clean = np.ones(10) * 0.2
    noise = np.ones(10) * 0.01
    zeros = np.zeros(10)
    result = measure_harmonic_components(clean, noise, zeros, zeros, zeros)
    assert result["clean_relative_error_rmse"] == 1
    assert result["clean_level_change_db"] is None
    assert result["noise_attenuation_db"] is None
    assert result["undefined_metrics"]["noise_attenuation_db"] == "zero_residual_noise_not_finite_db"
    assert result["undefined_metrics"]["clean_level_change_db"] == "zero_replayed_clean_not_finite_db"


def test_sub_lsb_does_not_assume_pcm16_quantization():
    noise = np.ones(20) * 1e-12
    zeros = np.zeros(20)
    result = measure_harmonic_components(zeros, noise, noise / 2, zeros, noise / 2)
    assert result["noise_attenuation_db"] == pytest.approx(20 * np.log10(2))
    assert result["input_noise_rms"] == pytest.approx(1e-12)
    quantized = np.rint(noise * 32768) / 32768
    silence_result = measure_harmonic_components(zeros, quantized, quantized, zeros, quantized)
    assert silence_result["noise_attenuation_db"] is None
    assert silence_result["undefined_metrics"]["noise_attenuation_db"] == "zero_input_noise"


def test_additivity_failure_is_flagged_without_fitted_repair():
    clean = np.ones(100) * 0.1
    noise = np.ones(100) * 0.01
    result = measure_harmonic_components(clean, noise, clean + noise + 0.02, clean, noise)
    assert result["decomposition_additivity_max_abs_error"] == pytest.approx(0.02)
    assert result["decomposition_additivity_within_1e_10"] is False
    assert result["component_interpretation_valid"] is False
    assert result["component_interpretation_invalid_reason"] == "component_replay_is_not_additive_within_1e_10"
    assert result["raw_mixture_error_rmse"] == pytest.approx(0.03)


def test_additivity_threshold_is_absolute_and_explicit():
    zeros = np.zeros(10)
    low = measure_harmonic_components(zeros, zeros, np.full(10, 0.9e-10), zeros, zeros)
    high = measure_harmonic_components(zeros, zeros, np.full(10, 1.1e-10), zeros, zeros)
    assert low["decomposition_additivity_within_1e_10"] is True
    assert high["decomposition_additivity_within_1e_10"] is False


def test_window_is_same_for_all_vectors_and_keeps_global_count():
    clean = np.array([0.5, 0.2, 0.2])
    noise = np.array([0.1, 0.01, 0.01])
    yc = np.array([0.0, 0.2, 0.2])
    yn = noise * 0.5
    result = measure_harmonic_components(clean, noise, yc + yn, yc, yn, start_sample=1)
    assert result["total_sample_count"] == 3
    assert result["analysis_start_sample"] == 1
    assert result["analyzed_sample_count"] == 2
    assert result["clean_error_rmse"] == 0
    assert result["noise_attenuation_db"] == pytest.approx(20 * np.log10(2))


@pytest.mark.parametrize("bad", ([], [[0.1]], [np.nan], [np.inf], [1j], [True], ["0.1"], None))
def test_invalid_vectors_refused_in_every_position(bad):
    for index in range(5):
        arguments = [[0.1]] * 5
        arguments[index] = bad
        with pytest.raises(ValueError):
            measure_harmonic_components(*arguments)


def test_mismatched_lengths_refused():
    with pytest.raises(ValueError, match="equal lengths"):
        measure_harmonic_components([0], [0], [0, 0], [0], [0])


@pytest.mark.parametrize("start", (-1, 2, 1.1, True, np.bool_(False), "0", None))
def test_invalid_start_sample_refused(start):
    with pytest.raises(ValueError):
        measure_harmonic_components([0, 1], [0, 1], [0, 1], [0, 1], [0, 1], start_sample=start)


def test_large_finite_values_remain_json_safe_even_when_intermediates_overflow():
    huge = np.array([1e308, -1e308])
    result = measure_harmonic_components(huge, huge, -huge, huge, huge)
    assert result["component_interpretation_valid"] is False
    assert result["decomposition_additivity_max_abs_error"] is None
    json.dumps(result, allow_nan=False)


def test_rms_avoids_naive_square_overflow_and_underflow():
    for level in (1e200, 1e-200):
        clean = np.array([level, -level])
        noise = np.zeros(2)
        result = measure_harmonic_components(clean, noise, clean, clean, noise)
        assert result["clean_rms"] == level
        assert result["clean_relative_error_rmse"] == 0
        assert result["clean_level_change_db"] == 0
        json.dumps(result, allow_nan=False)
