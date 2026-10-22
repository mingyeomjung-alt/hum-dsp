"""Raw-reference and anti-gaming checks; these tests do not test a filter."""

import json

import numpy as np
import pytest

from hum_dsp.metrics import measure_output


def references():
    time = np.arange(48000) / 48000
    return 0.4 * np.sin(2 * np.pi * 440 * time), 0.1 * np.sin(2 * np.pi * 60 * time)


def test_exact_bypass_has_zero_error_reduction_and_unity_gains():
    clean, noise = references()
    measured = measure_output(clean, noise, clean + noise)
    assert measured["reference_error_reduction_db"] == pytest.approx(0, abs=1e-12)
    assert measured["output_error_rmse"] == pytest.approx(np.sqrt(np.mean(noise**2)))
    assert measured["input_snr_db"] == pytest.approx(measured["output_snr_db"])
    assert measured["component_fit"]["clean_gain"] == pytest.approx(1)
    assert measured["component_fit"]["noise_gain"] == pytest.approx(1)
    assert measured["component_fit"]["residual_rms"] < 1e-14
    assert measured["alignment_applied"] is False
    assert measured["output_gain_normalization_applied"] is False


def test_muting_is_not_success_and_infinite_attenuation_is_not_reported():
    clean, noise = references()
    measured = measure_output(clean, noise, np.zeros_like(clean))
    assert measured["output_error_rmse"] == pytest.approx(np.sqrt(np.mean(clean**2)))
    assert measured["reference_error_reduction_db"] < 0
    assert measured["output_snr_db"] == pytest.approx(0)
    assert measured["component_fit"]["clean_gain"] == 0
    assert measured["component_fit"]["noise_gain"] == 0
    assert measured["component_fit"]["noise_attenuation_db"] is None


def test_global_gain_drop_cannot_be_hidden_by_normalization():
    clean, noise = references()
    measured = measure_output(clean, noise, 0.5 * (clean + noise))
    expected_error = 0.5 * noise - 0.5 * clean
    assert measured["output_error_rmse"] == pytest.approx(np.sqrt(np.mean(expected_error**2)))
    assert measured["reference_error_reduction_db"] < 0
    assert measured["component_fit"]["clean_gain"] == pytest.approx(0.5)
    assert measured["component_fit"]["noise_attenuation_db"] == pytest.approx(6.020599913)
    assert measured["component_fit"]["interpretation"] == "linear_reference_model_diagnostic_only"


def test_known_noise_reduction_and_unmodelled_residual_are_separate():
    clean, noise = references()
    residual = 0.01 * np.cos(2 * np.pi * 1000 * np.arange(clean.size) / 48000)
    measured = measure_output(clean, noise, clean + 0.25 * noise + residual)
    fit = measured["component_fit"]
    assert fit["noise_attenuation_db"] == pytest.approx(12.041199826)
    assert fit["residual_rms"] == pytest.approx(np.sqrt(np.mean(residual**2)))
    assert measured["reference_error_reduction_db"] < fit["noise_attenuation_db"]


def test_perfect_clean_has_null_infinite_snr_with_reason():
    clean, noise = references()
    measured = measure_output(clean, noise, clean)
    assert measured["output_error_rmse"] == 0
    assert measured["output_snr_db"] is None
    assert measured["reference_error_reduction_db"] is None
    assert measured["undefined_metrics"]["output_snr_db"] == "zero_output_error_not_finite_db"
    json.dumps(measured, allow_nan=False)


@pytest.mark.parametrize("output_level", [0.0, 0.1])
def test_silent_references_have_no_component_or_snr_claim(output_level):
    zero = np.zeros(100)
    measured = measure_output(zero, zero, zero + output_level)
    assert measured["input_snr_db"] is None
    assert measured["output_snr_db"] is None
    assert measured["reference_error_reduction_db"] is None
    assert measured["component_fit"]["status"] == "no_active_reference"
    assert measured["component_fit"]["residual_rms"] == pytest.approx(output_level)
    json.dumps(measured, allow_nan=False)


def test_clean_only_and_noise_only_are_explicit():
    clean, noise = references()
    zeros = np.zeros_like(clean)
    clean_only = measure_output(clean, zeros, 0.8 * clean)
    assert clean_only["input_snr_db"] is None
    assert clean_only["reference_error_reduction_db"] is None
    assert clean_only["component_fit"]["status"] == "clean_only_reference"
    assert clean_only["component_fit"]["clean_gain"] == pytest.approx(0.8)
    assert clean_only["component_fit"]["noise_gain"] is None
    noise_only = measure_output(zeros, noise, 0.5 * noise)
    assert noise_only["output_snr_db"] is None
    assert noise_only["reference_error_reduction_db"] == pytest.approx(6.020599913)
    assert noise_only["component_fit"]["status"] == "noise_only_reference"
    assert noise_only["component_fit"]["noise_gain"] == pytest.approx(0.5)


def test_collinear_same_frequency_references_do_not_claim_component_separation():
    clean, _ = references()
    measured = measure_output(clean, 0.25 * clean, 0.9 * clean)
    fit = measured["component_fit"]
    assert fit["status"] == "unidentifiable"
    assert fit["reason"] == "collinear_references"
    assert fit["reference_overlap_warning"] is True
    assert fit["clean_gain"] is None
    assert fit["noise_gain"] is None
    assert fit["noise_attenuation_db"] is None
    json.dumps(measured, allow_nan=False)


def test_near_collinear_references_are_ill_conditioned():
    clean, independent = references()
    noise = clean + independent * 1e-5
    fit = measure_output(clean, noise, clean + noise)["component_fit"]
    assert fit["status"] == "unidentifiable"
    assert fit["reason"] == "ill_conditioned_references"
    assert fit["condition_number"] > 1e4


def test_negative_fitted_gain_reports_magnitude_attenuation():
    clean, noise = references()
    fit = measure_output(clean, noise, clean - 0.5 * noise)["component_fit"]
    assert fit["noise_gain"] == pytest.approx(-0.5)
    assert fit["noise_attenuation_db"] == pytest.approx(6.020599913)


def test_start_sample_applies_to_all_metrics_and_fullscale_counts():
    clean = np.array([0, 0, 0.2, -0.2, 0.2])
    noise = np.array([0, 0, 0, 0, 0])
    output = np.array([9, -9, 1, -1, 0.2])
    measured = measure_output(clean, noise, output, start_sample=np.int64(2))
    assert measured["total_sample_count"] == 5
    assert measured["analyzed_sample_count"] == 3
    assert measured["output_peak_abs"] == 1
    assert measured["output_fullscale_sample_count"] == 2
    assert measured["input_fullscale_sample_count"] == 0
    assert measured["output_error_rmse"] == pytest.approx(np.sqrt((0.8**2 + 0.8**2) / 3))


@pytest.mark.parametrize("bad", [[], [[0, 1]], [np.nan], [np.inf], [-np.inf],
                                 [1 + 0j], [True, False], ["0", "1"], [object()]])
@pytest.mark.parametrize("argument", [0, 1, 2])
def test_invalid_vectors_rejected(bad, argument):
    values = [[0.1, 0.2]] * 3
    values[argument] = bad
    with pytest.raises(ValueError):
        measure_output(*values)


def test_mismatched_lengths_rejected():
    with pytest.raises(ValueError, match="equal lengths"):
        measure_output([0, 0], [0], [0, 0])


@pytest.mark.parametrize("start", [-1, 2, 3, 1.0, True, "0", None])
def test_invalid_start_rejected(start):
    with pytest.raises(ValueError):
        measure_output([0, 1], [0, 1], [0, 1], start_sample=start)


@pytest.mark.parametrize("rate", [0, 16000, 44100, 48000.0, True, "48000", None])
def test_invalid_m1_sample_rate_rejected(rate):
    with pytest.raises(ValueError):
        measure_output([0, 1], [0, 1], [0, 1], sample_rate_hz=rate)


def test_finite_extreme_inputs_still_produce_strict_json():
    measured = measure_output([1e308, -1e308], [1e308, -1e308], [-1e308, 1e308])
    assert measured["output_error_rmse"] is None
    assert measured["input_peak_abs"] is None
    json.dumps(measured, allow_nan=False)


def test_unreportably_large_fitted_gain_still_produces_strict_json():
    measured = measure_output([1e-300, -1e-300], [0, 0], [1e308, -1e308])
    assert measured["component_fit"]["status"] == "numerically_unreportable"
    assert measured["component_fit"]["clean_gain"] is None
    json.dumps(measured, allow_nan=False)


def test_no_delay_compensation_is_hidden():
    clean, noise = references()
    measured = measure_output(clean, noise, np.roll(clean, 200))
    assert measured["output_error_rmse"] > 0.1
    assert measured["alignment_applied"] is False
