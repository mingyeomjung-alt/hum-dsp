"""Independent input-contract tests; no devices or private audio are used."""

from dataclasses import FrozenInstanceError
import json
import wave

import numpy as np
import pytest

from hum_dsp.audio import read_pcm16_wav, validate_audio, write_pcm16_wav
from hum_dsp.config import ProcessorConfig


SAMPLE_RATE_HZ = 48_000
MAX_FRAMES = 60 * SAMPLE_RATE_HZ
PCM16_STEP = 1.0 / 32_768


def _write_wave(path, *, rate=48_000, channels=1, width=2, frames=64):
    """Create a small independently encoded WAV, not the production writer."""
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(channels)
        stream.setsampwidth(width)
        stream.setframerate(rate)
        stream.writeframes(bytes(frames * channels * width))


def test_config_defaults_are_explicit_and_json_serializable():
    config = ProcessorConfig()
    expected = {
        "sample_rate_hz": 48_000,
        "nominal_hum_hz": 50.0,
        "max_attenuation_db": 12.0,
        "tracking_half_width_hz": 1.0,
        "max_slew_hz_per_s": 0.05,
        "block_size_frames": 480,
    }
    assert config.to_dict() == expected
    assert json.loads(json.dumps(config.to_dict(), allow_nan=False)) == expected


def test_config_is_frozen_and_export_is_not_a_mutation_handle():
    config = ProcessorConfig()
    with pytest.raises(FrozenInstanceError):
        config.nominal_hum_hz = 60.0
    exported = config.to_dict()
    exported["nominal_hum_hz"] = 60.0
    assert config.nominal_hum_hz == 50.0


@pytest.mark.parametrize(
    "settings",
    [
        {"nominal_hum_hz": 60.0},
        {"max_attenuation_db": 0.0},
        {"max_attenuation_db": 12.0},
        {"tracking_half_width_hz": 0.001},
        {"max_slew_hz_per_s": 0.001},
        {"block_size_frames": 1},
        {"block_size_frames": 4096},
    ],
)
def test_config_accepts_documented_boundaries(settings):
    assert ProcessorConfig(**settings).to_dict().items() >= settings.items()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("sample_rate_hz", 8_000),
        ("sample_rate_hz", 44_100),
        ("sample_rate_hz", 48_000.5),
        ("nominal_hum_hz", 49.0),
        ("nominal_hum_hz", 50.1),
        ("nominal_hum_hz", 61.0),
        ("max_attenuation_db", -0.001),
        ("max_attenuation_db", 12.001),
        ("tracking_half_width_hz", 0.0),
        ("tracking_half_width_hz", -0.1),
        ("tracking_half_width_hz", 1.001),
        ("max_slew_hz_per_s", 0.0),
        ("max_slew_hz_per_s", -0.01),
        ("max_slew_hz_per_s", 0.05001),
        ("block_size_frames", 0),
        ("block_size_frames", 4097),
        ("block_size_frames", 480.0),
        ("block_size_frames", 1.5),
    ],
)
def test_config_rejects_out_of_contract_values(field, value):
    with pytest.raises((TypeError, ValueError)):
        ProcessorConfig(**{field: value})


@pytest.mark.parametrize(
    "field",
    [
        "sample_rate_hz",
        "nominal_hum_hz",
        "max_attenuation_db",
        "tracking_half_width_hz",
        "max_slew_hz_per_s",
        "block_size_frames",
    ],
)
@pytest.mark.parametrize("value", [True, False, float("nan"), float("inf"), -float("inf"), None, "50"])
def test_config_rejects_booleans_nonfinite_and_nonnumeric_values(field, value):
    with pytest.raises((TypeError, ValueError)):
        ProcessorConfig(**{field: value})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("sample_rate_hz", 44_100),
        ("nominal_hum_hz", 51.0),
        ("max_attenuation_db", float("nan")),
        ("block_size_frames", False),
    ],
)
def test_config_export_revalidates_even_if_frozen_guard_was_bypassed(field, value):
    config = ProcessorConfig()
    object.__setattr__(config, field, value)
    with pytest.raises((TypeError, ValueError)):
        config.to_dict()


def test_audio_validation_returns_float64_without_normalizing():
    source = np.array([-1.0, -0.5, 0.0, 0.25, 0.5], dtype=np.float32)
    validated = validate_audio(source)
    assert isinstance(validated, np.ndarray)
    assert validated.dtype == np.float64
    assert validated.ndim == 1
    np.testing.assert_array_equal(validated, source.astype(np.float64))


def test_audio_validation_accepts_real_sequence_and_half_open_range():
    source = [-1.0, 0.0, np.nextafter(1.0, 0.0)]
    np.testing.assert_array_equal(validate_audio(source), np.array(source, dtype=np.float64))


@pytest.mark.parametrize(
    "samples",
    [
        [],
        np.array([], dtype=np.float64),
        0.0,
        np.array(0.0),
        [[0.1], [0.2]],
        np.zeros((2, 2)),
        True,
        [True, False],
        np.array([False, True], dtype=bool),
        np.array([0.1], dtype=object),
        ["0.1"],
        [0.1 + 0.0j],
        np.array([0.1 + 0.2j]),
        [float("nan")],
        [float("inf")],
        [-float("inf")],
        [1.0],
        [-1.000001],
    ],
)
def test_audio_validation_rejects_invalid_samples(samples):
    with pytest.raises((TypeError, ValueError)):
        validate_audio(samples)


def test_audio_validation_enforces_sixty_second_limit():
    assert validate_audio(np.zeros(MAX_FRAMES)).shape == (MAX_FRAMES,)
    with pytest.raises((TypeError, ValueError)):
        validate_audio(np.zeros(MAX_FRAMES + 1))


def test_read_independent_pcm16_wave_preserves_signed_scaling(tmp_path):
    path = tmp_path / "signed_pcm.wav"
    encoded = np.array([-32768, -16384, -1, 0, 1, 16384, 32767], dtype="<i2")
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(SAMPLE_RATE_HZ)
        stream.writeframes(encoded.tobytes())
    decoded = read_pcm16_wav(path)
    assert decoded.dtype == np.float64
    np.testing.assert_array_equal(decoded, encoded.astype(np.float64) / 32768.0)


@pytest.mark.parametrize(
    "settings",
    [
        {"rate": 8_000},
        {"rate": 44_100},
        {"channels": 2},
        {"width": 1},
        {"width": 3},
        {"width": 4},
        {"frames": 0},
        {"frames": MAX_FRAMES + 1},
    ],
)
def test_reader_rejects_wrong_format_empty_or_excessive_duration(tmp_path, settings):
    path = tmp_path / "unsupported.wav"
    _write_wave(path, **settings)
    with pytest.raises(ValueError):
        read_pcm16_wav(path)


@pytest.mark.parametrize("payload", [b"", b"not a WAV file", b"RIFF\x24\x00\x00\x00WA"])
def test_reader_normalizes_malformed_header_failures_to_value_error(tmp_path, payload):
    path = tmp_path / "malformed.wav"
    path.write_bytes(payload)
    with pytest.raises(ValueError):
        read_pcm16_wav(path)


@pytest.mark.parametrize("format_tag", [3, 6, 7])
def test_reader_rejects_non_pcm_wave_encodings(tmp_path, format_tag):
    path = tmp_path / "non_pcm.wav"
    _write_wave(path)
    contents = bytearray(path.read_bytes())
    assert contents[12:16] == b"fmt "
    contents[20:22] = format_tag.to_bytes(2, "little")
    path.write_bytes(contents)
    with pytest.raises(ValueError):
        read_pcm16_wav(path)


@pytest.mark.parametrize("removed_bytes", [1, 2, 64, 128])
def test_reader_rejects_truncated_declared_frame_payload(tmp_path, removed_bytes):
    path = tmp_path / "truncated.wav"
    _write_wave(path, frames=64)
    contents = path.read_bytes()
    path.write_bytes(contents[:-removed_bytes])
    with pytest.raises(ValueError):
        read_pcm16_wav(path)


def test_writer_sets_strict_format_and_has_bounded_quantization(tmp_path):
    path = tmp_path / "result.wav"
    source = np.concatenate(
        (
            np.array([-1.0, -PCM16_STEP, 0.0, PCM16_STEP, np.nextafter(1.0, 0.0)]),
            np.random.default_rng(731).uniform(-1.0, 1.0, 1024),
        )
    )
    write_pcm16_wav(path, source)
    with wave.open(str(path), "rb") as stream:
        assert stream.getnchannels() == 1
        assert stream.getsampwidth() == 2
        assert stream.getframerate() == SAMPLE_RATE_HZ
        assert stream.getcomptype() == "NONE"
        assert stream.getnframes() == source.size
    decoded = read_pcm16_wav(path)
    assert np.max(np.abs(source - decoded)) <= PCM16_STEP
    assert decoded[0] == -1.0
    assert decoded[-1] < 1.0


def test_writer_refuses_existing_output_unless_explicitly_overwritten(tmp_path):
    path = tmp_path / "existing.wav"
    write_pcm16_wav(path, np.array([0.125, -0.25]))
    original_bytes = path.read_bytes()
    with pytest.raises(FileExistsError):
        write_pcm16_wav(path, np.array([0.75]))
    assert path.read_bytes() == original_bytes
    write_pcm16_wav(path, np.array([0.75]), overwrite=True)
    np.testing.assert_array_equal(read_pcm16_wav(path), np.array([0.75]))


@pytest.mark.parametrize(
    "samples",
    [[], [1.0], [-1.0001], [float("nan")], [float("inf")], [True], [0.1j], [[0.1]]],
)
def test_writer_rejects_invalid_audio_before_creating_output(tmp_path, samples):
    path = tmp_path / "must_not_exist.wav"
    with pytest.raises((TypeError, ValueError)):
        write_pcm16_wav(path, samples)
    assert not path.exists()


def test_invalid_replacement_preserves_existing_output(tmp_path):
    path = tmp_path / "preserved.wav"
    write_pcm16_wav(path, np.array([0.25]))
    original_bytes = path.read_bytes()
    with pytest.raises((TypeError, ValueError)):
        write_pcm16_wav(path, [float("nan")], overwrite=True)
    assert path.read_bytes() == original_bytes


def test_writer_rejects_excessive_duration_before_creating_output(tmp_path):
    path = tmp_path / "too_long.wav"
    with pytest.raises((TypeError, ValueError)):
        write_pcm16_wav(path, np.zeros(MAX_FRAMES + 1))
    assert not path.exists()
