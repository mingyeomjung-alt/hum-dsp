"""Continuous phase, boundary, strict-input and bounded-state FIR contracts."""

import hashlib
import json

import numpy as np
import pytest
from scipy.signal import firwin, resample_poly

from hum_dsp.compat_resampling import _kernel, convert_sample_rate
from hum_dsp.stream_resampling import StreamingResampler


def render(source, source_rate=44100, target_rate=48000, splits=(4096,), target_frames=None):
    engine = StreamingResampler(source_rate, target_rate, len(source), source.shape[1], target_frames)
    blocks = []
    start = turn = 0
    while start < len(source):
        end = min(len(source), start + splits[turn % len(splits)])
        output = engine.process_block(source[start:end])
        assert output.ndim == 2 and output.dtype == np.float64
        assert len(output) <= 4608
        blocks.append(output)
        start, turn = end, turn + 1
    blocks.append(engine.finish())
    return np.concatenate(blocks), engine


@pytest.mark.parametrize("rates", [(44100, 48000), (48000, 44100)])
@pytest.mark.parametrize("frames", [1, 2, 3, 31, 64, 147, 160, 4095, 4096, 4097, 13019])
@pytest.mark.parametrize("channels", [1, 2])
def test_exact_legacy_arithmetic_and_arbitrary_boundaries(rates, frames, channels):
    source = np.random.default_rng(frames).uniform(-0.25, 0.25, (frames, channels))
    expected, legacy = convert_sample_rate(source, *rates)
    actual, engine = render(source, *rates, splits=(1, 17, 4096, 2, 113))
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(np.rint(actual * 32768), np.rint(expected * 32768))
    report = engine.describe()
    assert report["finished"] is True
    assert report["input_frames_received"] == frames
    assert report["output_frames_emitted"] == len(expected)
    assert report["output_frames_expected"] == len(expected)
    assert report["retained_input_frames"] == 0
    assert report["filter_coefficients_sha256_float64_le"] == legacy["filter_coefficients_sha256_float64_le"]
    assert report["maximum_retained_overlap_frames"] <= report["maximum_input_taps_per_output_frame"]
    assert report["maximum_combined_input_buffer_frames"] <= report["input_buffer_frame_upper_bound"]
    assert report["global_next_output_index"] == len(expected)
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("rates", [(44100, 48000), (48000, 44100)])
def test_exact_impulses_at_caller_boundaries_with_no_padding(rates):
    source = np.zeros((16403, 2))
    positions = [0, 31, 4095, 4096, 4097, 8191, 8192, 12288, 16402]
    source[positions, 0] = 0.5
    source[:, 1] = -source[:, 0]
    expected, _ = convert_sample_rate(source, *rates)
    actual, _ = render(source, *rates)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual[:, 0], -actual[:, 1])


@pytest.mark.parametrize("frames", [1, 2, 147, 160, 44101, 100003])
def test_native_round_trip_exact_frame_count_and_batch_pcm(frames):
    time = np.arange(frames) / 44100
    source = np.column_stack((0.2 * np.sin(2 * np.pi * 50 * time),
                              0.15 * np.sin(2 * np.pi * 501 * time)))
    forward, _ = render(source)
    actual, engine = render(forward, 48000, 44100, splits=(29, 3072, 4096), target_frames=frames)
    reference_forward, _ = convert_sample_rate(source, 44100, 48000)
    expected, _ = convert_sample_rate(reference_forward, 48000, 44100, target_frames=frames)
    np.testing.assert_array_equal(actual, expected)
    assert len(actual) == frames
    assert engine.describe()["discarded_final_frames"] in (0, 1)


def test_lookahead_emission_and_only_true_eof_extension():
    source = np.arange(100, dtype=np.float64).reshape(-1, 1) / 1000
    engine = StreamingResampler(44100, 48000, 100)
    assert engine.process_block(source[:1]).shape == (0, 1)
    before = engine.describe()
    with pytest.raises(ValueError, match="all declared"):
        engine.finish()
    assert engine.describe() == before
    middle = engine.process_block(source[1:])
    assert 0 < len(middle) < engine.output_frames
    tail = engine.finish()
    assert len(tail) > 0
    expected, _ = convert_sample_rate(source, 44100, 48000)
    np.testing.assert_array_equal(np.concatenate((middle, tail)), expected)
    assert engine.describe()["caller_block_boundaries_zero_padded"] is False


@pytest.mark.parametrize("rate", [44100, 48000])
def test_equal_rate_is_exact_independent_copy_without_kernel(rate):
    source = np.random.default_rng(83).uniform(-0.99, 0.99, (10003, 2))
    actual, engine = render(source, rate, rate, splits=(1, 4096, 73))
    np.testing.assert_array_equal(actual, source)
    assert not np.shares_memory(source, actual)
    report = engine.describe()
    assert report["filter_tap_count"] == 0
    assert report["retained_input_frames"] == 0
    assert report["maximum_combined_input_buffer_frames"] == 0
    assert report["filter_coefficients_sha256_float64_le"] is None


@pytest.mark.parametrize("kwargs", [
    {"source_rate_hz": True}, {"source_rate_hz": 32000}, {"source_rate_hz": 44100.0},
    {"target_rate_hz": 96000}, {"target_rate_hz": False},
    {"input_frames": 0}, {"input_frames": -1}, {"input_frames": 44100 * 600 + 1},
    {"input_frames": True}, {"input_frames": 50.0},
    {"channels": 0}, {"channels": 3}, {"channels": True}, {"channels": 1.0},
    {"target_frames": 0}, {"target_frames": 2}, {"target_frames": True},
    {"target_frames": 110.0},
])
def test_invalid_constructor(kwargs):
    args = {"source_rate_hz": 44100, "target_rate_hz": 48000, "input_frames": 100}
    args.update(kwargs)
    with pytest.raises(ValueError):
        StreamingResampler(**args)


def test_duration_limit_and_one_final_frame_override_are_exact():
    for source, target in ((44100, 48000), (48000, 44100)):
        count = source * 600
        engine = StreamingResampler(source, target, count)
        assert engine.output_frames == target * 600
        assert engine.describe()["maximum_duration_seconds"] == 600
    with pytest.raises(ValueError):
        StreamingResampler(48000, 48000, 100, target_frames=99)
    with pytest.raises(ValueError):
        StreamingResampler(48000, 44100, 100, target_frames=90)
    assert StreamingResampler(48000, 44100, 100, target_frames=91).output_frames == 91


@pytest.mark.parametrize("samples", [
    [], np.empty((0, 1)), np.zeros(2), np.zeros((4, 2)), np.zeros((4097, 1)),
    np.zeros((4, 1, 1)), [[True]], [[1.0]], [[-1.00001]], [[float("nan")]],
    [[float("inf")]], [[complex(0.1, 0)]], [["0.1"]], [[object()]], [[0.1], [0.2, 0.3]],
])
def test_invalid_blocks_are_transactional(samples):
    engine = StreamingResampler(44100, 48000, 100)
    engine.process_block(np.zeros((1, 1)))
    before = engine.describe()
    with pytest.raises(ValueError):
        engine.process_block(samples)
    assert engine.describe() == before
    engine.process_block(np.zeros((99, 1)))
    engine.finish()


def test_overfeed_premature_finish_and_repeated_finish_are_not_internal_faults():
    engine = StreamingResampler(44100, 48000, 100)
    before = engine.describe()
    with pytest.raises(ValueError, match="exceeds"):
        engine.process_block(np.zeros((101, 1)))
    with pytest.raises(ValueError, match="all declared"):
        engine.finish()
    assert engine.describe() == before
    engine.process_block(np.zeros((100, 1)))
    engine.finish()
    done = engine.describe()
    with pytest.raises(ValueError, match="finished"):
        engine.finish()
    with pytest.raises(ValueError, match="finished"):
        engine.process_block(np.zeros((1, 1)))
    assert engine.describe() == done


def test_fault_latches_internal_computation_and_corrupt_clock(monkeypatch):
    engine = StreamingResampler(44100, 48000, 100)
    def broken(stop):
        raise ArithmeticError("owned injected failure")
    monkeypatch.setattr(engine, "_render", broken)
    with pytest.raises(ValueError, match="internal"):
        engine.process_block(np.zeros((100, 1)))
    assert engine.describe()["fault"]["exception_type"] == "ArithmeticError"
    with pytest.raises(ValueError, match="faulted"):
        engine.finish()
    other = StreamingResampler(44100, 48000, 100)
    other._received = -1
    with pytest.raises(ValueError, match="internal"):
        other.process_block(np.zeros((1, 1)))
    assert other.describe()["fault"] is not None


def test_strided_input_not_mutated_and_report_cannot_mutate_state():
    backing = np.random.default_rng(7).uniform(-0.4, 0.4, (800, 4))
    before = backing.copy()
    source = backing[::2, ::2]
    actual, engine = render(source, splits=(13, 89))
    expected, _ = convert_sample_rate(source, 44100, 48000)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(backing, before)
    report = engine.describe()
    report["input_frames"] = -10
    assert engine.describe()["input_frames"] == len(source)


def test_output_overshoot_is_not_silently_clipped_or_normalized():
    source = np.full((1000, 1), -0.99)
    source[500:] = 0.99
    actual, _ = render(source, splits=(311, 4096))
    expected, _ = convert_sample_rate(source, 44100, 48000)
    assert np.max(np.abs(actual)) > 1.0
    np.testing.assert_array_equal(actual, expected)


def test_actual_61_second_continuous_clock_and_bounded_overlap():
    # An actual >60 s input, not a patched counter. Independent SciPy full-file
    # convolution is only a test oracle; no SciPy is imported by runtime code.
    rate, target = 44100, 48000
    count = rate * 61 + 19
    time = np.arange(count) / rate
    source = (0.17 * np.sin(2 * np.pi * 57.3 * time)).reshape(-1, 1)
    source[rate * 60 - 1:rate * 60 + 2] += 0.2
    actual, engine = render(source, rate, target, splits=(4096, 3071, 129))
    coefficients = firwin(10241, 1 / 160, window=("kaiser", 5.0))
    expected = resample_poly(source, 160, 147, axis=0, window=coefficients, padtype="constant")
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)
    report = engine.describe()
    assert report["input_frames_received"] == count > rate * 60
    assert report["output_frames_emitted"] == len(expected) > target * 60
    assert report["global_next_polyphase_index"] == (len(expected) * 147 + 5120) % 160
    assert report["maximum_retained_overlap_frames"] <= 65
    assert report["maximum_combined_input_buffer_frames"] <= 4163
    assert report["retained_input_frames"] == 0
    frozen, _ = _kernel(160, 147)
    assert report["filter_coefficients_sha256_float64_le"] == hashlib.sha256(
        frozen.astype("<f8").tobytes()).hexdigest()
