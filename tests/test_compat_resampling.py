"""Reference, timing, boundary, and bounded compatibility-resampler tests."""

import json

import numpy as np
import pytest
from scipy.signal import firwin, resample_poly

from hum_dsp.compat_resampling import convert_sample_rate


@pytest.mark.parametrize("source_rate,target_rate,up,down", [(44100, 48000, 160, 147), (48000, 44100, 147, 160)])
@pytest.mark.parametrize("frames", [1, 2, 13, 147, 160, 4095, 4096, 4097, 12017])
@pytest.mark.parametrize("channels", [None, 1, 2])
def test_matches_independent_scipy_reference(source_rate, target_rate, up, down, frames, channels):
    rng = np.random.default_rng(frames)
    shape = (frames,) if channels is None else (frames, channels)
    source = rng.uniform(-0.35, 0.35, size=shape)
    actual, metadata = convert_sample_rate(source, source_rate, target_rate)
    # Independent SciPy coefficient generator and convolution use exactly the
    # declared longer FIR, not SciPy resample_poly's shorter default filter.
    fir = firwin(2 * 32 * max(up, down) + 1, 1 / max(up, down), window=("kaiser", 5.0))
    expected = resample_poly(source, up, down, axis=0, window=fir, padtype="constant")
    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)
    assert len(actual) == (frames * up + down - 1) // down
    assert metadata["filter_tap_count"] == 10241
    assert metadata["maximum_input_taps_per_output_frame"] <= 70
    assert metadata["outside_input_boundary"] == "zero_extension"
    assert metadata["automatic_gain_normalization_applied"] is False
    assert metadata["clipping_applied"] is False
    assert len(metadata["filter_coefficients_sha256_float64_le"]) == 64
    json.dumps(metadata, allow_nan=False)


@pytest.mark.parametrize("source_rate,target_rate", [(44100, 48000), (48000, 44100)])
@pytest.mark.parametrize("frames", [1, 3, 147, 160, 1021, 44101])
def test_round_trip_original_frame_count_is_explicit(source_rate, target_rate, frames):
    source = np.sin(np.arange(frames) * 0.01) * 0.1
    forward, _ = convert_sample_rate(source, source_rate, target_rate)
    inverse_default = (len(forward) * source_rate + target_rate - 1) // target_rate
    if inverse_default - frames > 1:
        # 48 ->44.1 ->48 is not a product path; two-ceil math can add2 frames.
        with pytest.raises(ValueError, match="one fewer"):
            convert_sample_rate(forward, target_rate, source_rate, target_frames=frames)
        return
    returned, report = convert_sample_rate(forward, target_rate, source_rate, target_frames=frames)
    assert len(returned) == frames
    assert report["discarded_final_frames"] in (0, 1)
    assert report["explicit_target_frames"] is True


@pytest.mark.parametrize("source_rate,target_rate", [(44100, 48000), (48000, 44100)])
def test_impulse_time_origin_and_interior_alignment(source_rate, target_rate):
    # 10 ms is integral on both clocks; centered FIR must not introduce delay.
    source = np.zeros(source_rate // 10)
    source[source_rate // 100] = 0.5
    converted, report = convert_sample_rate(source, source_rate, target_rate)
    assert np.argmax(np.abs(converted)) == target_rate // 100
    assert report["linear_phase_fir_delay_compensated"] is True
    assert report["output_time_origin_seconds"] == 0


@pytest.mark.parametrize("source_rate,target_rate", [(44100, 48000), (48000, 44100)])
def test_dc_gain_interior_and_zero_extended_boundary_are_exposed(source_rate, target_rate):
    converted, metadata = convert_sample_rate(np.full(source_rate, 0.25), source_rate, target_rate)
    # Finite polyphase filters have phase-dependent ripple, not hidden gain fits.
    assert np.max(np.abs(converted[100:-100] - 0.25)) < 0.0002
    assert abs(np.mean(converted[100:-100]) - 0.25) < 1e-7
    assert abs(converted[-1] - 0.25) > 1e-3
    assert metadata["boundary_transients_removed"] is False
    assert metadata["gain_fitting_applied"] is False


@pytest.mark.parametrize("frequency", [50.0, 100.0])
@pytest.mark.parametrize("source_rate,target_rate", [(44100, 48000), (48000, 44100)])
def test_low_frequency_amplitude_phase_and_round_trip(frequency, source_rate, target_rate):
    phase = 0.43
    source_time = np.arange(source_rate * 2) / source_rate
    source = 0.2 * np.sin(2 * np.pi * frequency * source_time + phase)
    converted, _ = convert_sample_rate(source, source_rate, target_rate)
    target_time = np.arange(len(converted)) / target_rate
    ideal = 0.2 * np.sin(2 * np.pi * frequency * target_time + phase)
    interior = slice(100, -100)
    relative_rmse = np.linalg.norm(converted[interior] - ideal[interior]) / np.linalg.norm(ideal[interior])
    assert relative_rmse < 0.001
    returned, _ = convert_sample_rate(converted, target_rate, source_rate, target_frames=len(source))
    round_trip_rmse = np.linalg.norm(returned[interior] - source[interior]) / np.linalg.norm(source[interior])
    assert round_trip_rmse < 0.002
    # A quadrature projection detects phase error without fitting the waveform.
    quadrature = np.cos(2 * np.pi * frequency * source_time[interior] + phase)
    assert abs(2 * np.mean((returned[interior] - source[interior]) * quadrature)) < 2e-6


def test_48_to_44100_rejects_above_output_nyquist_with_declared_transition_limit():
    # 23.9 kHz lies in the declared longer FIR's stop band.  This is not a
    # claim of brick-wall rejection immediately above22.05 kHz.
    time = np.arange(48000 * 2) / 48000
    source = 0.2 * np.sin(2 * np.pi * 23900 * time)
    converted, _ = convert_sample_rate(source, 48000, 44100)
    source_rms = np.sqrt(np.mean(source[100:-100] ** 2))
    converted_rms = np.sqrt(np.mean(converted[100:-100] ** 2))
    assert 20 * np.log10(converted_rms / source_rms) < -45


def test_44100_to_48_preserves_passband_without_strong_interpolation_image():
    time = np.arange(44100) / 44100
    source = 0.2 * np.sin(2 * np.pi * 1000 * time)
    converted, _ = convert_sample_rate(source, 44100, 48000)
    target_time = np.arange(48000) / 48000
    # The 44100−1000 image aliases to 4900 Hz on the target clock.
    interior = slice(100, -100)
    wanted = 2 * abs(np.mean(converted[interior] * np.exp(-2j * np.pi * 1000 * target_time[interior])))
    image = 2 * abs(np.mean(converted[interior] * np.exp(-2j * np.pi * 4900 * target_time[interior])))
    assert abs(wanted - 0.2) < 0.001
    assert 20 * np.log10(image / wanted) < -50


@pytest.mark.parametrize("source_rate,target_rate", [(44100, 48000), (48000, 44100)])
def test_same_stereo_kernel_preserves_channel_ratio_and_does_not_mix(source_rate, target_rate):
    time = np.arange(10021) / source_rate
    first = 0.25 * np.sin(2 * np.pi * 211 * time)
    source = np.column_stack((first, -0.37 * first))
    stereo, metadata = convert_sample_rate(source, source_rate, target_rate)
    mono, _ = convert_sample_rate(first, source_rate, target_rate)
    # NumPy's reduction path can differ by a few ulps for 1 versus2 channels;
    # time mapping, coefficients, and channel independence remain identical.
    np.testing.assert_allclose(stereo[:, 0], mono, rtol=0, atol=3e-16)
    np.testing.assert_allclose(stereo[:, 1], -0.37 * mono, rtol=0, atol=3e-16)
    silent_right, _ = convert_sample_rate(np.column_stack((first, np.zeros_like(first))), source_rate, target_rate)
    assert np.count_nonzero(silent_right[:, 1]) == 0
    assert metadata["same_kernel_and_time_mapping_for_every_channel"] is True
    assert metadata["channel_mixing_applied"] is False


@pytest.mark.parametrize("rate", [44100, 48000])
@pytest.mark.parametrize("shape", [(100,), (100, 1), (100, 2)])
def test_equal_rate_is_exact_independent_copy(rate, shape):
    source = np.linspace(-1, 0.99, np.prod(shape)).reshape(shape)
    converted, metadata = convert_sample_rate(source, rate, rate)
    np.testing.assert_array_equal(converted, source)
    assert not np.shares_memory(converted, source)
    assert metadata["identity_copy"] is True
    assert metadata["filter_tap_count"] == 0
    assert metadata["filter_coefficients_sha256_float64_le"] is None


@pytest.mark.parametrize("source", [[], np.empty((0, 2)), [np.nan], [np.inf], [-np.inf], [-1.01], [1.0], [True], [1j], ["0.1"], np.zeros((2, 3)), np.zeros((1, 1, 1)), [[0.1], [0.2, 0.3]]])
def test_invalid_audio_refused(source):
    with pytest.raises(ValueError):
        convert_sample_rate(source, 44100, 48000)


@pytest.mark.parametrize("bad_rate", [True, 44100.0, 8000, 96000, "44100", None])
@pytest.mark.parametrize("position", [0, 1])
def test_invalid_rates_refused(bad_rate, position):
    rates = [44100, 48000]
    rates[position] = bad_rate
    with pytest.raises(ValueError):
        convert_sample_rate([0.1], *rates)


@pytest.mark.parametrize("target_frames", [True, 10.0, 0, -1, 2, 11, 12, "10"])
def test_invalid_explicit_lengths_refused(target_frames):
    # ceil(9 *160/147)=10; only 10 or9 are accepted.
    with pytest.raises(ValueError):
        convert_sample_rate(np.zeros(9), 44100, 48000, target_frames=target_frames)


def test_equal_rate_does_not_allow_length_crop():
    with pytest.raises(ValueError):
        convert_sample_rate(np.zeros(9), 48000, 48000, target_frames=8)


@pytest.mark.parametrize("rate", [44100, 48000])
def test_excess_duration_refused_before_filtering(rate):
    with pytest.raises(ValueError, match="60 seconds"):
        convert_sample_rate(np.zeros(rate * 60 + 1), rate, 48000)


def test_one_frame_trim_matches_default_prefix_and_never_pads():
    source = np.linspace(-0.2, 0.2, 401)
    default, _ = convert_sample_rate(source, 44100, 48000)
    trimmed, metadata = convert_sample_rate(source, 44100, 48000, target_frames=len(default) - 1)
    np.testing.assert_array_equal(trimmed, default[:-1])
    assert metadata["discarded_final_frames"] == 1


def test_peak_overshoot_is_not_normalized_or_clipped():
    source = np.tile(np.array([-0.99, -0.99, 0.99, 0.99]), 200)
    converted, metadata = convert_sample_rate(source, 44100, 48000)
    assert np.max(np.abs(converted)) > 1
    assert metadata["peak_limiting_applied"] is False
    assert metadata["automatic_gain_normalization_applied"] is False


def test_coefficient_hash_is_reproducible_and_direction_specific():
    first, a = convert_sample_rate(np.zeros(3), 44100, 48000)
    _, b = convert_sample_rate(np.zeros(7), 44100, 48000)
    _, reverse = convert_sample_rate(first, 48000, 44100, target_frames=3)
    assert a["filter_coefficients_sha256_float64_le"] == b["filter_coefficients_sha256_float64_le"]
    assert a["filter_coefficients_sha256_float64_le"] != reverse["filter_coefficients_sha256_float64_le"]
    assert a["maximum_output_chunk_frames"] == 1024
    assert a["temporary_array_payload_upper_bytes_excluding_input_output"] == 4194304


def test_longest_supported_native_input_has_exact_duration_without_audio_expansion(monkeypatch):
    import hum_dsp.compat_resampling as implementation

    original_take = implementation.np.take
    largest_index_block = 0

    def bounded_take(array, indices, *args, **kwargs):
        nonlocal largest_index_block
        assert indices.ndim == 2
        assert indices.shape[0] <= 1024
        assert indices.shape[1] <= 70
        largest_index_block = max(largest_index_block, indices.size)
        return original_take(array, indices, *args, **kwargs)

    monkeypatch.setattr(implementation.np, "take", bounded_take)
    converted, metadata = convert_sample_rate(np.zeros(44100 * 60), 44100, 48000)
    assert converted.shape == (48000 * 60,)
    assert np.count_nonzero(converted) == 0
    assert largest_index_block <= 1024 * 70
    assert metadata["output_frames"] == 2880000


@pytest.mark.parametrize("channels", [None, 1, 2])
def test_read_only_noncontiguous_audio_is_not_mutated_or_reordered(channels):
    base = np.column_stack((np.linspace(-0.1, 0.1, 201), np.linspace(0.2, -0.2, 201)))
    source = base[::-2, 0] if channels is None else base[::-2, :1] if channels == 1 else base[::-2, ::-1]
    before = source.copy()
    source.flags.writeable = False
    actual, metadata = convert_sample_rate(source, 44100, 48000)
    fir = firwin(10241, 1 / 160, window=("kaiser", 5.0))
    expected = resample_poly(before, 160, 147, axis=0, window=fir, padtype="constant")
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(source, before)
    assert metadata["validated_contiguous_input_storage_bytes_excluded_from_chunk_bound"] == before.nbytes
    assert metadata["returned_output_storage_bytes_excluded_from_chunk_bound"] == actual.nbytes
