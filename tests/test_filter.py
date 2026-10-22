"""M2 causal filter arithmetic, input contract and streaming regressions."""

import json
import math

import numpy as np
import pytest

from hum_dsp.config import ProcessorConfig
from hum_dsp.filter import FixedHumSuppressor, process_audio


def _response(description, frequency):
    z = np.exp(-2j * np.pi * np.asarray(frequency) / 48000.0)
    b, a = description["b"], description["a"]
    return (b[0] + b[1] * z + b[2] * z**2) / (a[0] + a[1] * z + a[2] * z**2)


@pytest.mark.parametrize("frequency", [50.0, 60.0])
@pytest.mark.parametrize("depth", [0.0, 3.0, 6.0, 12.0])
def test_design_center_depth_unity_endpoints_and_stable_poles(frequency, depth):
    engine = FixedHumSuppressor(ProcessorConfig(nominal_hum_hz=frequency, max_attenuation_db=depth))
    description = engine.describe()
    assert abs(_response(description, frequency)) == pytest.approx(10 ** (-depth / 20), abs=2e-8)
    np.testing.assert_allclose(abs(_response(description, np.array([0.0, 24000.0]))), [1.0, 1.0], atol=1e-10)
    assert np.max(np.abs(np.roots(description["a"]))) < 1.0
    assert np.max(np.abs(_response(description, np.linspace(0, 24000, 48001)))) <= 1.0 + 1e-9


def test_description_is_json_safe_fixed_and_not_a_mutation_handle():
    engine = FixedHumSuppressor(ProcessorConfig())
    description = engine.describe()
    assert json.loads(json.dumps(description, allow_nan=False)) == description
    assert description["tracking_enabled"] is False
    assert description["pole_design_bandwidth_hz"] == 1.0
    assert description["explicit_delay_samples"] == 0
    assert "not zero" in description["delay_note"]
    description["b"][0] = 999
    description["config"]["max_attenuation_db"] = 0
    assert engine.describe()["b"][0] != 999
    assert engine.describe()["config"]["max_attenuation_db"] == 12


def test_fixed_filter_snapshots_caller_configuration():
    config = ProcessorConfig()
    engine = FixedHumSuppressor(config)
    previous = engine.describe()
    object.__setattr__(config, "nominal_hum_hz", 60.0)
    assert engine.describe() == previous


@pytest.mark.parametrize("config", [None, {}, True, "50", 50])
def test_constructor_requires_config(config):
    with pytest.raises(ValueError):
        FixedHumSuppressor(config)


def test_constructor_revalidates_guard_bypassed_configuration():
    config = ProcessorConfig()
    object.__setattr__(config, "nominal_hum_hz", 51.0)
    with pytest.raises(ValueError):
        FixedHumSuppressor(config)


@pytest.mark.parametrize("frequency", [50.0, 60.0])
@pytest.mark.parametrize("depth", [0.0, 3.0, 12.0])
def test_own_scalar_kernel_matches_independent_scipy_lfilter(frequency, depth):
    signal = pytest.importorskip("scipy.signal")
    config = ProcessorConfig(nominal_hum_hz=frequency, max_attenuation_db=depth)
    source = np.random.default_rng(55).uniform(-0.8, 0.8, 10003)
    description = FixedHumSuppressor(config).describe()
    reference = signal.lfilter(description["b"], description["a"], source)
    np.testing.assert_allclose(process_audio(source, config), reference, rtol=1e-10, atol=1e-11)


@pytest.mark.parametrize("frequency", [50.0, 60.0])
def test_full_notch_coefficients_agree_with_independent_scipy_design(frequency):
    signal = pytest.importorskip("scipy.signal")
    description = FixedHumSuppressor(ProcessorConfig(nominal_hum_hz=frequency)).describe()
    reference_b, reference_a = signal.iirnotch(frequency, frequency / 1.0, fs=48000.0)
    center_gain = 10 ** (-12 / 20)
    expected_b = center_gain * reference_a + (1 - center_gain) * reference_b
    np.testing.assert_allclose(description["b"], expected_b, atol=1e-15, rtol=1e-15)
    np.testing.assert_allclose(description["a"], reference_a, atol=1e-15, rtol=1e-15)


@pytest.mark.parametrize("block_size", [1, 7, 479, 480, 1024, 4096])
def test_process_audio_is_bit_identical_across_block_sizes_including_remainder(block_size):
    source = np.random.default_rng(19).uniform(-0.7, 0.7, 10007)
    expected = process_audio(source, ProcessorConfig(block_size_frames=4096))
    actual = process_audio(source, ProcessorConfig(block_size_frames=block_size))
    np.testing.assert_array_equal(actual, expected)


def test_arbitrary_block_partitions_preserve_state_and_count():
    source = np.random.default_rng(91).uniform(-0.7, 0.7, 12001)
    config = ProcessorConfig()
    engine = FixedHumSuppressor(config)
    blocks = []
    start = 0
    for length in [1, 13, 4096, 7, 480, 1024, 4096, 2284]:
        stop = min(start + length, source.size)
        if stop > start:
            blocks.append(engine.process_block(source[start:stop]))
        start = stop
    assert start == source.size
    np.testing.assert_array_equal(np.concatenate(blocks), process_audio(source, config))
    assert engine.describe()["samples_processed"] == source.size


def test_reset_reproduces_zero_state_output_and_resets_counter():
    engine = FixedHumSuppressor(ProcessorConfig())
    source = np.random.default_rng(46).uniform(-0.7, 0.7, 4096)
    first = engine.process_block(source)
    engine.process_block(source)
    assert engine.describe()["samples_processed"] == 8192
    engine.reset()
    assert engine.describe()["samples_processed"] == 0
    np.testing.assert_array_equal(engine.process_block(source), first)


@pytest.mark.parametrize("samples", [
    [], 0.0, [[0.1]], [True], [0.1j], ["0.1"], np.array([0.1], dtype=object),
    [float("nan")], [float("inf")], [-float("inf")], [1.0], [-1.001], np.zeros(4097),
])
def test_invalid_block_does_not_advance_state_or_counter(samples):
    engine = FixedHumSuppressor(ProcessorConfig())
    reference = FixedHumSuppressor(ProcessorConfig())
    prefix = np.linspace(-0.8, 0.8, 1000)
    probe = np.linspace(0.8, -0.8, 1000)
    engine.process_block(prefix)
    reference.process_block(prefix)
    before = engine.describe()
    with pytest.raises(ValueError):
        engine.process_block(samples)
    assert engine.describe() == before
    np.testing.assert_array_equal(engine.process_block(probe), reference.process_block(probe))


@pytest.mark.parametrize("depth", [0.0, 12.0])
def test_input_arrays_are_never_mutated_or_shared_with_output(depth):
    source = np.linspace(-0.8, 0.8, 1000, dtype=np.float32)
    original = source.copy()
    result = FixedHumSuppressor(ProcessorConfig(max_attenuation_db=depth)).process_block(source)
    np.testing.assert_array_equal(source, original)
    assert result.dtype == np.float64
    assert not np.shares_memory(source, result)


def test_zero_depth_is_exact_identity_for_all_supported_real_input_types():
    for source in [np.array([-1, 0], dtype=np.int16), [-1.0, 0.0, np.nextafter(1.0, 0.0)], np.linspace(-0.8, 0.8, 1000, dtype=np.float32)]:
        np.testing.assert_array_equal(process_audio(source, ProcessorConfig(max_attenuation_db=0)), np.asarray(source, dtype=np.float64))


def test_zero_input_stays_exactly_zero():
    np.testing.assert_array_equal(process_audio(np.zeros(7001)), np.zeros(7001))


def test_no_future_lookahead_prefix_does_not_depend_on_suffix():
    prefix = np.random.default_rng(71).uniform(-0.6, 0.6, 1007)
    suffix = np.random.default_rng(72).uniform(-0.6, 0.6, 7000)
    np.testing.assert_array_equal(process_audio(prefix), process_audio(np.concatenate([prefix, suffix]))[:prefix.size])


def test_impulse_has_immediate_feedthrough_but_no_output_before_impulse():
    source = np.zeros(4096)
    source[1000] = 0.5
    description = FixedHumSuppressor(ProcessorConfig()).describe()
    actual = process_audio(source)
    np.testing.assert_array_equal(actual[:1000], np.zeros(1000))
    assert actual[1000] == description["b"][0] * 0.5
    assert np.any(actual[1001:] != 0.0)


def test_filter_does_not_silently_clip_or_normalize_float_output():
    # A transition can produce a peak above 1 even for a stable filter whose
    # steady-state response is <=1. The WAV boundary must handle this explicitly.
    source = np.concatenate([np.full(24000, 0.99999), np.full(24000, -0.99999)])
    result = process_audio(source)
    assert np.isfinite(result).all()
    assert np.max(np.abs(result)) > 1.0
    assert result[0] == pytest.approx(source[0] * FixedHumSuppressor(ProcessorConfig()).describe()["b"][0])


@pytest.mark.parametrize("frequency", [50.0, 60.0])
def test_fixed_steady_tone_reaches_finite_twelve_db_attenuation(frequency):
    sample_index = np.arange(48000 * 5, dtype=np.float64)
    tone = 0.25 * np.sin(2 * np.pi * frequency * sample_index / 48000)
    result = process_audio(tone, ProcessorConfig(nominal_hum_hz=frequency))
    source_rms = np.sqrt(np.mean(tone[-48000:] ** 2))
    output_rms = np.sqrt(np.mean(result[-48000:] ** 2))
    attenuation = -20 * math.log10(output_rms / source_rms)
    assert attenuation == pytest.approx(12.0, abs=0.001)


def test_process_audio_rejects_over_sixty_seconds():
    with pytest.raises(ValueError):
        process_audio(np.zeros(48000 * 60 + 1))


def test_process_audio_sixty_second_boundary_and_exact_bypass():
    source = np.zeros(48000 * 60, dtype=np.float64)
    actual = process_audio(source, ProcessorConfig(max_attenuation_db=0))
    assert actual.dtype == np.float64
    np.testing.assert_array_equal(actual, source)


@pytest.mark.parametrize("samples", [[], [1.0], [True], [float("nan")], [[0.0]]])
def test_process_audio_validates_complete_input(samples):
    with pytest.raises(ValueError):
        process_audio(samples)
