"""Continuous blocks retain H2's exact signal behavior without whole-file state."""

import copy
import json

import numpy as np
import pytest

from hum_dsp.band_control import BandControlConfig, render_band_control, render_band_control_replay
from hum_dsp.stream_bands import MAX_STREAM_FRAMES, OBSERVATION_FRAMES, UPDATE_FRAMES, StreamBandControl


def _audio(seconds=3.3, nominal=50):
    t = np.arange(round(48000 * seconds)) / 48000.0
    return sum(0.025 * np.sin(2 * np.pi * f * t + phase)
               for f, phase in ((nominal, .37), (2 * nominal, 1.13), (3 * nominal, -.61), (731, .2)))


def _feed(engine, source, sizes=(4096,)):
    output, offset, index = [], 0, 0
    while offset < len(source):
        stop = min(len(source), offset + sizes[index % len(sizes)])
        output.append(engine.process_block(source[offset:stop]))
        offset, index = stop, index + 1
    return np.concatenate(output)


@pytest.mark.parametrize("depths", [(6, 3, 0), (12, 12, 12), (0, 0, 0)])
@pytest.mark.parametrize("nominal", [50, 60])
def test_mono_is_bit_exact_h2_and_arbitrary_partitions(depths, nominal):
    source = _audio(nominal=nominal)
    config = BandControlConfig(nominal_hum_hz=nominal, band_depths_db=depths, block_size_frames=4096)
    expected, old_description, _ = render_band_control(source, config)
    events = []
    engine = StreamBandControl(config, trace_sink=events.append)
    actual = _feed(engine, source[:, None], (13, 2047, 1, 4096, 377))[:, 0]
    np.testing.assert_array_equal(actual, expected)
    description = engine.describe()
    assert description["samples_processed"] == len(source)
    assert description["shared_original_input_fft_count"] == old_description["shared_original_input_fft_count"]
    assert description["trace_event_count"] == len(events)
    assert description["retained_trace_records"] == bool(events)
    assert description["retained_audio_frames"] == (OBSERVATION_FRAMES if events else 0)
    assert [event["sample_index"] for event in events] == (
        [0, *range(OBSERVATION_FRAMES, len(source) + 1, UPDATE_FRAMES)] if any(depths) else [])
    assert description["latest_event"] == (events[-1] if events else None)
    json.dumps(description, allow_nan=False)


@pytest.mark.parametrize("reference", [0, 1])
def test_stereo_matches_reference_frozen_h2_replay(reference):
    x = _audio()
    other = 0.7 * _audio() + .01 * np.sin(2 * np.pi * 249 * np.arange(x.size) / 48000)
    source = np.column_stack((x, other))
    config = BandControlConfig(band_depths_db=(9, 3, 6), block_size_frames=4096)
    _, _, bundle = render_band_control(source[:, reference], config)
    expected = np.column_stack([render_band_control_replay(source[:, ch], config, bundle)
                                for ch in range(2)])
    engine = StreamBandControl(config, channels=2, reference_channel_index=reference)
    actual = _feed(engine, source, (4096, 31, 480, 7))
    np.testing.assert_array_equal(actual, expected)
    assert engine.describe()["reference_channel_index"] == reference
    assert len(engine.describe()["channel_bands"]) == 2


def test_antiphase_is_not_summed_away_and_independent_states():
    source = _audio()
    engine = StreamBandControl(BandControlConfig(band_depths_db=(12, 12, 12)), channels=2)
    output = _feed(engine, np.column_stack((source, -source)))
    np.testing.assert_array_equal(output[:, 1], -output[:, 0])
    assert all(band["target_mix"] == 1 for band in engine.describe()["bands"])
    assert engine._bands[0][0] is not engine._bands[1][0]


def test_one_original_reference_fft_per_observation(monkeypatch):
    source = np.column_stack((_audio(2.1), -_audio(2.1)))
    original, calls = np.fft.rfft, []
    def counted(samples, n):
        calls.append(samples.copy())
        return original(samples, n=n)
    monkeypatch.setattr(np.fft, "rfft", counted)
    engine = StreamBandControl(BandControlConfig(band_depths_db=(12, 12, 12)), channels=2,
                               reference_channel_index=1)
    _feed(engine, source, (4096, 2001, 1))
    assert len(calls) == 2
    window = np.hanning(OBSERVATION_FRAMES)
    np.testing.assert_array_equal(calls[0], source[:OBSERVATION_FRAMES, 1] * window)
    np.testing.assert_array_equal(calls[1], source[UPDATE_FRAMES:OBSERVATION_FRAMES + UPDATE_FRAMES, 1] * window)


def test_boundary_event_affects_next_sample_and_eof_is_observed():
    source = _audio(2.1)
    events = []
    engine = StreamBandControl(BandControlConfig(), trace_sink=events.append)
    _feed(engine, source[:OBSERVATION_FRAMES, None])
    assert events[-1]["sample_index"] == OBSERVATION_FRAMES
    assert events[-1]["bands"][0]["target_mix"] == 0
    _feed(engine, source[OBSERVATION_FRAMES:, None], (4096, 703, 1))
    assert events[-1]["sample_index"] == OBSERVATION_FRAMES + UPDATE_FRAMES
    assert events[-1]["bands"][0]["target_mix"] == 1
    assert events[-1]["bands"][0]["mix"] == 0
    one = engine.process_block(np.array([[.025]]))
    assert engine.describe()["bands"][0]["ramp_position"] == 1
    assert np.isfinite(one).all()


def test_zero_depth_is_exact_identity_without_fft_history_or_events(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("bypass must not compute FFT")
    monkeypatch.setattr(np.fft, "rfft", forbidden)
    events = []
    engine = StreamBandControl(BandControlConfig(band_depths_db=(0, 0, 0)), channels=2,
                               reference_channel_index=1, trace_sink=events.append)
    source = np.array([[-1, 32767 / 32768], [.4, -.2], [0, 0]])
    np.testing.assert_array_equal(engine.process_block(source), source)
    assert events == [] and engine._history is None and engine._window is None
    assert engine.describe()["latest_event"] is None


@pytest.mark.parametrize("bad", [None, {}, True, 50, "50"])
def test_constructor_config_invalid(bad):
    with pytest.raises(ValueError):
        StreamBandControl(bad)


@pytest.mark.parametrize("kwargs", [
    {"channels": 0}, {"channels": 3}, {"channels": True}, {"channels": 1.0},
    {"reference_channel_index": -1}, {"reference_channel_index": 1},
    {"reference_channel_index": True}, {"trace_sink": []},
])
def test_constructor_shape_and_sink_invalid(kwargs):
    with pytest.raises(ValueError):
        StreamBandControl(BandControlConfig(), **kwargs)


def test_tampered_frozen_config_revalidated():
    config = BandControlConfig()
    object.__setattr__(config, "band_depths_db", (99, 0, 0))
    with pytest.raises(ValueError):
        StreamBandControl(config)


@pytest.mark.parametrize("bad", [[], [0.1], [[True]], [[1j]], [[".1"]], [[object()]],
    np.zeros((0, 1)), np.zeros((4097, 1)), np.zeros((3, 2)), np.zeros((2, 1, 1)),
    [[np.nan]], [[np.inf]], [[1.0]], [[-1.01]]])
def test_bad_blocks_never_mutate_state(bad):
    events = []
    engine = StreamBandControl(BandControlConfig(), trace_sink=events.append)
    before, before_events = engine.describe(), copy.deepcopy(events)
    with pytest.raises(ValueError):
        engine.process_block(bad)
    assert engine.describe() == before and events == before_events


def test_engine_never_retains_or_exposes_mutable_event_history():
    sink_events = []
    def sink(event):
        sink_events.append(copy.deepcopy(event))
        event["bands"].clear()
    engine = StreamBandControl(BandControlConfig(), trace_sink=sink)
    _feed(engine, _audio()[:, None])
    snapshot = engine.describe()
    assert snapshot["retained_trace_records"] == 1
    assert len(snapshot["latest_event"]["bands"]) == 2
    snapshot["latest_event"]["bands"].clear()
    snapshot["bands"].clear()
    assert len(engine.describe()["latest_event"]["bands"]) == 2
    assert not any(isinstance(value, list) and value and isinstance(value[0], dict)
                   for name, value in vars(engine).items() if name != "_bands")
    assert sink_events[-1] == engine.describe()["latest_event"]


@pytest.mark.parametrize("fault", ["filter", "clock", "sample_clock", "ring", "fft", "nonfinite_fft", "peak"])
def test_internal_failures_latch_without_successful_fallback(fault, monkeypatch):
    engine = StreamBandControl(BandControlConfig())
    source = np.zeros((10, 1))
    if fault == "filter":
        engine._bands[0][0].state1 = np.nan
    elif fault == "clock":
        engine._next_update += 1
    elif fault == "sample_clock":
        engine._samples_processed = "invalid"
    elif fault == "ring":
        engine._history[0] = np.nan
    elif fault == "peak":
        engine._maximum_output_peak = np.inf
    else:
        _feed(engine, _audio(2)[:-1, None])
        source = np.array([[0.]])
        if fault == "fft":
            def failure(*args, **kwargs):
                raise RuntimeError("injected FFT failure")
            monkeypatch.setattr(np.fft, "rfft", failure)
        else:
            monkeypatch.setattr(np.fft, "rfft", lambda *args, **kwargs: np.full(262145, np.nan))
    with pytest.raises(ValueError, match="internal continuous"):
        engine.process_block(source)
    assert engine.describe()["fault"] is not None
    json.dumps(engine.describe(), allow_nan=False)
    with pytest.raises(ValueError, match="faulted"):
        engine.process_block(np.zeros((1, 1)))


def test_sink_failure_latches_and_initial_sink_failure_refuses_construction():
    def fail(event):
        raise RuntimeError("injected sink failure")
    with pytest.raises(ValueError, match="initialization"):
        StreamBandControl(BandControlConfig(), trace_sink=fail)
    def later(event):
        if event["sample_index"]:
            raise RuntimeError("injected later sink failure")
    engine = StreamBandControl(BandControlConfig(), trace_sink=later)
    with pytest.raises(ValueError, match="internal continuous"):
        _feed(engine, _audio(2)[:, None])
    assert engine.describe()["fault"] is not None
    with pytest.raises(ValueError, match="faulted"):
        engine.process_block(np.zeros((1, 1)))


def test_nonfinite_stage_output_latches_without_clipping_or_dry_success(monkeypatch):
    engine = StreamBandControl(BandControlConfig())
    monkeypatch.setattr(engine._bands[0][0], "process", lambda source: np.full_like(source, np.inf))
    with pytest.raises(ValueError, match="internal continuous"):
        engine.process_block(np.zeros((10, 1)))
    assert engine.describe()["fault"] is not None


def test_input_is_not_aliased_and_readonly_noncontiguous_blocks_work():
    engine = StreamBandControl(BandControlConfig(band_depths_db=(0, 0, 0)), channels=2)
    backing = np.arange(48, dtype=np.float32).reshape(12, 4) / 100
    source = backing[::2, ::2]
    source.flags.writeable = False
    before = source.copy()
    output = engine.process_block(source)
    assert output.dtype == np.float64
    np.testing.assert_array_equal(source, before)
    np.testing.assert_array_equal(output, before)
    output[:] = .1
    np.testing.assert_array_equal(source, before)


def test_trace_sink_cannot_reenter_processing_or_mutate_stream_clock():
    holder, errors = {}, []
    def sink(event):
        if event["sample_index"]:
            try:
                holder["engine"].process_block(np.zeros((1, 1)))
            except ValueError as exc:
                errors.append(str(exc))
    engine = StreamBandControl(BandControlConfig(), trace_sink=sink)
    holder["engine"] = engine
    _feed(engine, _audio(2.1)[:, None])
    assert len(errors) == 2 and all("recursively" in error for error in errors)
    assert engine.describe()["samples_processed"] == 100800
    assert engine.describe()["fault"] is None


def test_actual_61_seconds_has_no_60_second_reset_and_fixed_retained_memory():
    # Actual >60s processing, not a manually advanced clock. Feed one generated
    # periodic block repeatedly rather than retaining any whole-file input.
    events = 0
    def count_event(event):
        nonlocal events
        events += 1
    engine = StreamBandControl(BandControlConfig(band_depths_db=(6, 0, 0)), trace_sink=count_event)
    block = .05 * np.sin(2 * np.pi * 50 * np.arange(3840) / 48000 + .37)
    before_60 = None
    frames = 61 * 48000
    for offset in range(0, frames, block.size):
        count = min(block.size, frames - offset)
        result = engine.process_block(block[:count, None])
        assert result.shape == (count, 1)
        if offset < 60 * 48000 <= offset + count:
            before_60 = engine.describe()
    description = engine.describe()
    assert description["samples_processed"] == frames
    assert before_60["bands"][0]["first_acquired_sample"] == OBSERVATION_FRAMES
    assert description["bands"][0]["first_acquired_sample"] == OBSERVATION_FRAMES
    assert description["bands"][0]["mix"] == 1
    assert events == 1 + 1 + (frames - OBSERVATION_FRAMES) // UPDATE_FRAMES
    assert description["retained_trace_records"] == 1
    assert description["retained_audio_frames"] == OBSERVATION_FRAMES
    assert engine._history.nbytes + engine._window.nbytes == 2 * OBSERVATION_FRAMES * 8


def test_ten_minute_cap_accepts_exact_limit_then_rejects_without_mutation():
    engine = StreamBandControl(BandControlConfig(band_depths_db=(0, 0, 0)))
    # Bypass is O(n) copy only and really consumes the entire permitted clock.
    source = np.zeros((4096, 1))
    for offset in range(0, MAX_STREAM_FRAMES, 4096):
        engine.process_block(source[:min(4096, MAX_STREAM_FRAMES - offset)])
    before = engine.describe()
    assert before["samples_processed"] == MAX_STREAM_FRAMES
    with pytest.raises(ValueError, match="ten-minute"):
        engine.process_block(np.zeros((1, 1)))
    assert engine.describe() == before
