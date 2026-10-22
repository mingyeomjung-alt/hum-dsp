"""Offline manual per-band depths; unchanged H1 evidence and strict replay."""

import copy
from dataclasses import FrozenInstanceError
import json

import numpy as np
import pytest

from hum_dsp.band_control import (
    BandControlConfig, render_band_control, render_band_control_replay,
)
from hum_dsp.harmonics import HarmonicConfig, render_harmonic


def _tone(frequency, seconds=2.5, amplitude=0.04):
    time = np.arange(round(seconds * 48000)) / 48000.0
    return amplitude * np.sin(2 * np.pi * frequency * time + 0.37)


@pytest.fixture(scope="module")
def rendered():
    clean = _tone(419, 2.6, 0.07)
    noise = sum(_tone(frequency, 2.6, 0.025) for frequency in (50, 100, 150))
    mixture = clean + noise
    config = BandControlConfig(band_depths_db=(6, 3, 0), block_size_frames=4096)
    output, description, trace = render_band_control(mixture, config)
    return config, clean, noise, mixture, output, description, trace


@pytest.mark.parametrize("kwargs", [
    {"sample_rate_hz": 16000}, {"sample_rate_hz": True}, {"sample_rate_hz": 48000.0},
    {"nominal_hum_hz": True}, {"nominal_hum_hz": 51}, {"nominal_hum_hz": "50"},
    {"nominal_hum_hz": float("nan")}, {"nominal_hum_hz": float("inf")},
    {"block_size_frames": 0}, {"block_size_frames": 4097}, {"block_size_frames": True},
    {"block_size_frames": 480.0},
    {"band_depths_db": ()}, {"band_depths_db": (1, 2)}, {"band_depths_db": (1, 2, 3, 4)},
    {"band_depths_db": "630"}, {"band_depths_db": {1, 2, 3}},
    {"band_depths_db": (True, 3, 0)}, {"band_depths_db": (6, np.bool_(False), 0)},
    {"band_depths_db": (-0.1, 3, 0)}, {"band_depths_db": (6, 12.1, 0)},
    {"band_depths_db": (6, 3, float("nan"))}, {"band_depths_db": (float("inf"), 3, 0)},
    {"band_depths_db": (6, "3", 0)}, {"band_depths_db": (6, 3, 1j)},
])
def test_config_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        BandControlConfig(**kwargs)


def test_config_is_immutable_validated_and_json_compatible():
    config = BandControlConfig(nominal_hum_hz=60, band_depths_db=[0, np.float64(3), 6])
    assert config.band_depths_db == (0.0, 3.0, 6.0)
    assert config.orders == (2, 3)
    assert config.max_attenuation_db == 6.0
    assert json.loads(json.dumps(config.to_dict())) == {
        "sample_rate_hz": 48000, "nominal_hum_hz": 60.0,
        "band_depths_db": [0.0, 3.0, 6.0], "block_size_frames": 480,
        "orders": [2, 3], "max_attenuation_db": 6.0,
    }
    with pytest.raises(FrozenInstanceError):
        config.nominal_hum_hz = 50
    object.__setattr__(config, "band_depths_db", (6, 3, 99))
    with pytest.raises(ValueError):
        config.to_dict()
    with pytest.raises(ValueError):
        render_band_control([0.0], config)


@pytest.mark.parametrize("config", [True, {}, "config", HarmonicConfig()])
def test_render_requires_the_new_config(config):
    with pytest.raises(ValueError):
        render_band_control([0.0], config)


@pytest.mark.parametrize("bad", [[], [[0.1]], [True], ["0.1"], [1j], [float("nan")],
                                  [float("inf")], [1.0], [-1.01], [object()]])
def test_invalid_audio_is_rejected(bad):
    with pytest.raises(ValueError):
        render_band_control(bad)


def test_uniform_twelve_db_matches_unchanged_h1_exactly():
    source = sum(_tone(frequency, 2.6, 0.04) for frequency in (50, 100, 150, 419))
    expected, _, _ = render_harmonic(source, HarmonicConfig(block_size_frames=4096))
    actual, description, bundle = render_band_control(
        source, BandControlConfig(band_depths_db=(12, 12, 12), block_size_frames=4096))
    np.testing.assert_array_equal(actual, expected)
    assert description["processing_mode"].startswith("offline_two_pass")
    assert description["streaming_api"] is False
    assert [stage["order"] for stage in bundle["stages"]] == [1, 2, 3]


def test_lower_depth_reduces_known_wanted_100_hz_damage():
    wanted = _tone(100, 3.0, 0.05)
    lower, _, _ = render_band_control(wanted, BandControlConfig(band_depths_db=(0, 3, 0), block_size_frames=4096))
    deeper, _, _ = render_band_control(wanted, BandControlConfig(band_depths_db=(0, 12, 0), block_size_frames=4096))
    settled = slice(2 * 48000, None)
    shallow_error = np.sqrt(np.mean((lower[settled] - wanted[settled])**2))
    deep_error = np.sqrt(np.mean((deeper[settled] - wanted[settled])**2))
    assert shallow_error < deep_error
    assert shallow_error > 0.0  # Reduced damage is not semantic protection.


def test_second_band_off_preserves_only_100_hz_exactly():
    wanted = _tone(100)
    actual, _, bundle = render_band_control(wanted, BandControlConfig(band_depths_db=(6, 0, 6), block_size_frames=4096))
    np.testing.assert_array_equal(actual, wanted)
    assert [stage["order"] for stage in bundle["stages"]] == [1, 3]
    assert all(event["bands"][0]["target_mix"] == 0 for stage in bundle["stages"] for event in stage["trace"])


def test_all_zero_exact_identity_without_fft_or_trace(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("all-zero bypass must not perform FFT")
    monkeypatch.setattr(np.fft, "rfft", forbidden)
    source = _tone(100)
    config = BandControlConfig(band_depths_db=(0, 0, 0))
    output, description, bundle = render_band_control(source, config)
    np.testing.assert_array_equal(output, source)
    assert bundle["analysis_config"] is None
    assert bundle["analysis_trace"] == []
    assert bundle["stages"] == []
    assert description["analysis"] is None
    assert description["shared_original_input_fft_count"] == 0
    np.testing.assert_array_equal(render_band_control_replay(source, config, bundle), source)


def test_block_partition_config_parity_and_original_shared_analysis(rendered, monkeypatch):
    config, _, _, source, expected, _, bundle = rendered
    calls = []
    fft = np.fft.rfft
    def counted(array, n):
        calls.append(array.copy())
        return fft(array, n=n)
    monkeypatch.setattr(np.fft, "rfft", counted)
    actual, description, alternate = render_band_control(
        source, BandControlConfig(band_depths_db=config.band_depths_db, block_size_frames=257))
    np.testing.assert_array_equal(actual, expected)
    assert len(calls) == description["shared_original_input_fft_count"] == 7
    np.testing.assert_array_equal(calls[0], source[:96000] * np.hanning(96000))
    for first, second in zip(bundle["analysis_trace"], alternate["analysis_trace"], strict=True):
        first = copy.deepcopy(first)
        first["config"]["block_size_frames"] = 257
        assert first == second


def test_frozen_replay_exact_reconstruction_additivity_and_no_fft(rendered, monkeypatch):
    config, clean, noise, mixture, expected, _, bundle = rendered
    def forbidden(*args, **kwargs):
        raise AssertionError("frozen replay must not perform FFT")
    monkeypatch.setattr(np.fft, "rfft", forbidden)
    replayed = render_band_control_replay(mixture, config, bundle)
    clean_replay = render_band_control_replay(clean, config, bundle)
    noise_replay = render_band_control_replay(noise, config, bundle)
    np.testing.assert_array_equal(replayed, expected)
    np.testing.assert_allclose(clean_replay + noise_replay, expected, atol=1e-10, rtol=0)
    decoded_bundle = json.loads(json.dumps(bundle, allow_nan=False))
    np.testing.assert_array_equal(render_band_control_replay(mixture, config, decoded_bundle), expected)


@pytest.mark.parametrize("mutation", [
    lambda b: b.update(extra="not allowed"),
    lambda b: b.update(schema_version=True),
    lambda b: b.update(algorithm="other"),
    lambda b: b.update(source_frames=b["source_frames"] - 1),
    lambda b: b["config"]["band_depths_db"].__setitem__(0, 7.0),
    lambda b: b["config"]["band_depths_db"].__setitem__(0, True),
    lambda b: b["config"]["orders"].append(3),
    lambda b: b["analysis_config"].update(max_attenuation_db=6.0),
    lambda b: b["analysis_trace"].pop(),
    lambda b: b["analysis_trace"][0].update(trace_total_frames=0),
    lambda b: b["analysis_trace"][1].update(sample_index=96001),
    lambda b: b["analysis_trace"][1]["bands"][0].update(confidence=float("nan")),
    lambda b: b["analysis_trace"][1]["bands"][0].update(confidence=True),
    lambda b: b["analysis_trace"][1]["bands"][0].update(estimated_frequency_hz=250.0),
    lambda b: b["stages"].pop(),
    lambda b: b["stages"][0].update(order=2),
    lambda b: b["stages"][0].update(attenuation_db=3.0),
    lambda b: b["stages"][0]["config"].update(max_attenuation_db=7.0),
    lambda b: b["stages"][0]["trace"][0]["bands"][0]["b"].__setitem__(0, 0.0),
    lambda b: b["stages"][0]["trace"][2]["bands"][0].update(target_mix=0.0),
    lambda b: b["stages"][0]["trace"][2]["bands"][0].update(ramp_total_frames=0),
    lambda b: b["stages"][0]["trace"].append(copy.deepcopy(b["stages"][0]["trace"][-1])),
])
def test_malformed_or_tampered_bundle_is_rejected(rendered, mutation):
    config, _, _, source, _, _, bundle = rendered
    changed = copy.deepcopy(bundle)
    mutation(changed)
    with pytest.raises(ValueError):
        render_band_control_replay(source, config, changed)


@pytest.mark.parametrize("bad_bundle", [None, [], {}, True, "trace"])
def test_invalid_bundle_is_rejected(rendered, bad_bundle):
    config, _, _, source, _, _, _ = rendered
    with pytest.raises(ValueError):
        render_band_control_replay(source, config, bad_bundle)


def test_replay_rejects_depth_mutation_and_different_source_length(rendered):
    config, _, _, source, _, _, bundle = rendered
    with pytest.raises(ValueError):
        render_band_control_replay(source, BandControlConfig(band_depths_db=(6, 4, 0), block_size_frames=4096), bundle)
    with pytest.raises(ValueError):
        render_band_control_replay(source[:-1], config, bundle)
    assert bundle == copy.deepcopy(bundle)


def test_zero_trace_is_strict_too():
    config = BandControlConfig(band_depths_db=(0, 0, 0))
    source = np.zeros(10)
    _, _, bundle = render_band_control(source, config)
    bundle["analysis_trace"].append({})
    with pytest.raises(ValueError):
        render_band_control_replay(source, config, bundle)
