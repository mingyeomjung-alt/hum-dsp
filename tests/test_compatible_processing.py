"""Strict native WAV compatibility and frozen linked-channel regression tests."""

import hashlib
import json
import struct
import wave

import numpy as np
import pytest

from hum_dsp import compatible_processing as compatible
from hum_dsp.band_control import BandControlConfig
from hum_dsp.band_processing import process_band_wav


def _write(path, codes, rate=48000, width=2):
    array = np.asarray(codes)
    if array.ndim == 1:
        array = array[:, None]
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(array.shape[1])
        wav.setsampwidth(width)
        wav.setframerate(rate)
        if width == 2:
            wav.writeframes(array.astype("<i2").tobytes())
        else:
            wav.writeframes(bytes(array.shape[0] * array.shape[1] * width))
    return path


def _codes(signal):
    return np.rint(np.asarray(signal) * 32768).astype("<i2")


def _signal(rate=48000, seconds=3.4):
    time = np.arange(round(rate * seconds)) / rate
    return .08 * np.sin(2 * np.pi * 50 * time + .37) + .02 * np.sin(2 * np.pi * 430 * time)


def _read_codes(path):
    with wave.open(str(path), "rb") as wav:
        return np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").reshape(-1, wav.getnchannels())


def _write_extensible(path, channels, mask, rate=48000):
    guid = b"\x01\x00\x00\x00\x00\x00\x10\x00\x80\x00\x00\xaa\x00\x38\x9b\x71"
    fmt = struct.pack("<HHIIHHHHI16s", 0xFFFE, channels, rate, rate * channels * 2,
                      channels * 2, 16, 22, 16, mask, guid)
    payload = np.arange(20 * channels, dtype="<i2").tobytes()
    body = b"WAVEfmt " + struct.pack("<I", len(fmt)) + fmt + b"data" + struct.pack("<I", len(payload)) + payload
    path.write_bytes(b"RIFF" + struct.pack("<I", len(body)) + body)
    return path


@pytest.mark.parametrize("channels,mask", [(1, 0), (1, 4), (2, 0), (2, 3)])
@pytest.mark.parametrize("rate", [44100, 48000])
def test_canonical_extensible_layout_admitted_and_output_mask_not_claimed_copied(tmp_path, channels, mask, rate):
    source = _write_extensible(tmp_path / "extensible.wav", channels, mask, rate)
    samples, metadata = compatible.read_compatible_wav(source)
    assert samples.shape == (20, channels)
    assert metadata["wav_format_tag"] == 0xFFFE
    assert metadata["channel_mask"] == mask
    assert metadata["channel_order"] == (["front_center"] if channels == 1 else ["front_left", "front_right"])
    report = compatible.process_compatible_wav(source, tmp_path / "out", BandControlConfig(band_depths_db=(0, 0, 0)))
    assert report["input"]["channel_mask"] == mask
    assert report["output"]["wav_format_tag"] == 1
    assert report["output"]["channel_mask"] is None
    np.testing.assert_array_equal(_read_codes(tmp_path / "out" / "processed.wav"), _read_codes(source))


@pytest.mark.parametrize("channels,mask", [(1, 1), (1, 2), (1, 8), (2, 12), (2, 48), (2, 4), (2, 0x80000000)])
def test_noncanonical_extensible_layout_rejected_before_output(tmp_path, channels, mask):
    source = _write_extensible(tmp_path / "extensible.wav", channels, mask)
    with pytest.raises(ValueError, match="channel masks"):
        compatible.process_compatible_wav(source, tmp_path / "out")
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("rate", [44100, 48000])
@pytest.mark.parametrize("channels", [1, 2])
def test_supported_reader_preserves_native_shape_and_codes(tmp_path, rate, channels):
    codes = np.arange(800 * channels, dtype=np.int16).reshape(800, channels)
    path = _write(tmp_path / "source.wav", codes, rate)
    actual, metadata = compatible.read_compatible_wav(path)
    assert actual.shape == codes.shape
    np.testing.assert_array_equal(actual * 32768, codes)
    assert metadata["sample_rate_hz"] == rate
    assert metadata["channels"] == channels
    assert metadata["frames"] == 800


@pytest.mark.parametrize("rate", [44100, 48000])
def test_zero_depth_all_pcm_codes_native_stereo_exact_and_no_engine(tmp_path, monkeypatch, rate):
    codes = np.arange(-32768, 32768, dtype=np.int32).astype(np.int16)
    stereo = np.column_stack((codes, codes[::-1]))
    source = _write(tmp_path / "all_codes.wav", stereo, rate)
    original_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    def forbidden(*args, **kwargs):
        raise AssertionError("zero-depth processing must not resample, replay or FFT")
    monkeypatch.setattr(compatible, "convert_sample_rate", forbidden)
    monkeypatch.setattr(compatible, "render_band_control", forbidden)
    monkeypatch.setattr(compatible, "render_band_control_replay", forbidden)
    report = compatible.process_compatible_wav(source, tmp_path / "out", BandControlConfig(band_depths_db=(0, 0, 0)))
    np.testing.assert_array_equal(_read_codes(tmp_path / "out" / "processed.wav"), stereo)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == original_sha
    assert report["processing_bypassed"] is True
    assert report["algorithm"]["shared_original_input_fft_count"] == 0
    trace = json.loads((tmp_path / "out" / "compatible.json").read_text())
    assert trace["band_control_bundle"] == {}
    assert trace["stereo_policy"]["reference_channel_index"] is None
    assert report["output"]["sample_rate_hz"] == rate
    assert report["output"]["channels"] == 2


@pytest.mark.parametrize("rate", [44100, 48000])
def test_identical_stereo_channels_remain_exactly_identical(tmp_path, rate):
    mono = _codes(_signal(rate))
    source = _write(tmp_path / "same.wav", np.column_stack((mono, mono)), rate)
    report = compatible.process_compatible_wav(source, tmp_path / "out")
    output = _read_codes(tmp_path / "out" / "processed.wav")
    np.testing.assert_array_equal(output[:, 0], output[:, 1])
    assert output.shape == (mono.size, 2)
    assert report["stereo_policy"]["reference_channel_index"] == 0
    assert report["algorithm"]["shared_original_input_fft_count"] > 0
    assert report["reference_quality_metrics"] is None


@pytest.mark.parametrize("rate", [44100, 48000])
def test_opposite_polarity_not_summed_and_same_trace_filters(tmp_path, rate):
    mono = _codes(_signal(rate))
    source = _write(tmp_path / "inverse.wav", np.column_stack((mono, -mono)), rate)
    report = compatible.process_compatible_wav(source, tmp_path / "out")
    output = _read_codes(tmp_path / "out" / "processed.wav").astype(np.int64)
    assert np.max(np.abs(output[:, 0] + output[:, 1])) <= 1
    assert np.any(output[:, 0] != mono)
    assert report["algorithm"]["shared_original_input_fft_count"] > 0
    bundle = json.loads((tmp_path / "out" / "compatible.json").read_text())["band_control_bundle"]
    assert bundle["analysis_trace"][-1]["bands"][0]["mix"] > 0
    assert report["stereo_policy"]["same_trace_and_coefficients_all_channels"] is True
    assert report["stereo_policy"]["independent_channel_frequency_tracking"] is False


@pytest.mark.parametrize("rate", [44100, 48000])
def test_linked_filters_preserve_scaled_channel_relation_to_quantization_bound(tmp_path, rate):
    left = (_codes(_signal(rate)).astype(np.int32) // 2 * 2).astype(np.int16)
    right = (left.astype(np.int32) // 2).astype(np.int16)
    source = _write(tmp_path / "scaled.wav", np.column_stack((left, right)), rate)
    report = compatible.process_compatible_wav(source, tmp_path / "out")
    output = _read_codes(tmp_path / "out" / "processed.wav").astype(np.float64)
    assert np.max(np.abs(output[:, 1] - .5 * output[:, 0])) <= 1
    assert report["stereo_policy"]["reference_channel_index"] == 0


def test_louder_channel_fixed_before_analysis_and_no_crossfeed(tmp_path):
    signal = _codes(_signal())
    stereo = np.column_stack((np.zeros(signal.size, dtype=np.int16), signal))
    source = _write(tmp_path / "right.wav", stereo)
    report = compatible.process_compatible_wav(source, tmp_path / "out")
    output = _read_codes(tmp_path / "out" / "processed.wav")
    assert report["stereo_policy"]["reference_channel_index"] == 1
    assert np.count_nonzero(output[:, 0]) == 0
    assert np.any(output[:, 1] != signal)


def test_native_48000_mono_output_is_identical_to_unchanged_h2(tmp_path):
    source = _write(tmp_path / "mono.wav", _codes(_signal()))
    compatible.process_compatible_wav(source, tmp_path / "compatible")
    process_band_wav(source, tmp_path / "old")
    assert (tmp_path / "compatible" / "processed.wav").read_bytes() == (tmp_path / "old" / "processed.wav").read_bytes()


@pytest.mark.parametrize("rate", [44100, 48000])
def test_short_native_frame_count_restored_after_conversion(tmp_path, rate):
    source = _write(tmp_path / "short.wav", np.array([[0, 0], [100, -200], [0, 0]], dtype=np.int16), rate)
    report = compatible.process_compatible_wav(source, tmp_path / "out")
    assert report["output"]["frames"] == 3
    assert _read_codes(tmp_path / "out" / "processed.wav").shape == (3, 2)


@pytest.mark.parametrize("rate,channels,width", [(32000, 1, 2), (96000, 2, 2), (44100, 3, 2), (48000, 1, 3), (48000, 2, 4)])
def test_unsupported_header_rejected_before_output(tmp_path, rate, channels, width):
    source = _write(tmp_path / "bad.wav", np.zeros((10, channels)), rate, width)
    with pytest.raises(ValueError):
        compatible.process_compatible_wav(source, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_ieee_float_format_rejected(tmp_path):
    payload = struct.pack("<4f", 0, .1, -.1, 0)
    fmt = struct.pack("<HHIIHH", 3, 1, 48000, 192000, 4, 32)
    body = b"WAVEfmt " + struct.pack("<I", len(fmt)) + fmt + b"data" + struct.pack("<I", len(payload)) + payload
    source = tmp_path / "float.wav"
    source.write_bytes(b"RIFF" + struct.pack("<I", len(body)) + body)
    with pytest.raises(ValueError):
        compatible.process_compatible_wav(source, tmp_path / "out")
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("content", [b"", b"not a wave", b"RIFF\x20\x00\x00\x00WAVE"])
def test_invalid_container_rejected(tmp_path, content):
    source = tmp_path / "invalid.wav"
    source.write_bytes(content)
    with pytest.raises(ValueError):
        compatible.process_compatible_wav(source, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_truncated_payload_rejected(tmp_path):
    source = _write(tmp_path / "truncated.wav", np.zeros((10, 2), dtype=np.int16))
    source.write_bytes(source.read_bytes()[:-1])
    with pytest.raises(ValueError, match="truncated"):
        compatible.read_compatible_wav(source)


@pytest.mark.parametrize("channels", [1, 2])
def test_partial_pcm_frame_is_not_silently_rounded_down(tmp_path, channels):
    source = _write(tmp_path / "partial.wav", np.zeros((4, channels), dtype=np.int16))
    raw = bytearray(source.read_bytes())
    raw[40:44] = struct.pack("<I", 4 * channels * 2 + 1)
    raw.extend(b"\x00\x00")  # Odd-size data plus its RIFF pad byte.
    raw[4:8] = struct.pack("<I", len(raw) - 8)
    source.write_bytes(raw)
    with pytest.raises(ValueError, match="incomplete PCM16 frame"):
        compatible.read_compatible_wav(source)


@pytest.mark.parametrize("offset,format_code,value", [(28, "<I", 1), (32, "<H", 1)])
def test_inconsistent_fmt_alignment_or_byte_rate_rejected(tmp_path, offset, format_code, value):
    source = _write(tmp_path / "format.wav", np.zeros((4, 2), dtype=np.int16))
    raw = bytearray(source.read_bytes())
    payload = struct.pack(format_code, value)
    raw[offset:offset + len(payload)] = payload
    source.write_bytes(raw)
    with pytest.raises(ValueError, match="alignment or byte rate"):
        compatible.read_compatible_wav(source)


def test_bounded_unknown_metadata_chunks_accepted_but_not_copied(tmp_path):
    source = _write(tmp_path / "metadata.wav", np.zeros((8, 2), dtype=np.int16))
    raw = source.read_bytes()
    extra = b"JUNK" + struct.pack("<I", 3) + b"abc\x00"
    inserted = bytearray(raw[:36] + extra + raw[36:])
    inserted[4:8] = struct.pack("<I", len(inserted) - 8)
    source.write_bytes(inserted)
    report = compatible.process_compatible_wav(source, tmp_path / "out", BandControlConfig(band_depths_db=(0, 0, 0)))
    assert report["output"]["frames"] == 8
    assert b"JUNK" not in (tmp_path / "out" / "processed.wav").read_bytes()
    assert source.read_bytes() == inserted


def test_trailing_or_unbounded_chunk_metadata_rejected(tmp_path):
    source = _write(tmp_path / "trailing.wav", np.zeros((8, 1), dtype=np.int16))
    source.write_bytes(source.read_bytes() + b"outside RIFF")
    with pytest.raises(ValueError, match="trailing"):
        compatible.read_compatible_wav(source)


@pytest.mark.parametrize("rate", [44100, 48000])
def test_empty_and_over_60_second_headers_rejected(tmp_path, rate):
    empty = _write(tmp_path / "empty.wav", np.zeros((0, 1), dtype=np.int16), rate)
    with pytest.raises(ValueError):
        compatible.read_compatible_wav(empty)
    source = _write(tmp_path / "long.wav", np.zeros((1, 1), dtype=np.int16), rate)
    raw = bytearray(source.read_bytes())
    raw[40:44] = struct.pack("<I", (60 * rate + 1) * 2)
    source.write_bytes(raw)
    with pytest.raises(ValueError, match="60"):
        compatible.read_compatible_wav(source)


def test_container_size_bound_checked_before_decode(tmp_path, monkeypatch):
    source = _write(tmp_path / "size.wav", np.zeros((100, 1), dtype=np.int16))
    monkeypatch.setattr(compatible, "MAX_FILE_BYTES", 64)
    with pytest.raises(ValueError, match="32 MiB"):
        compatible.read_compatible_wav(source)


def test_existing_output_is_never_changed(tmp_path):
    source = _write(tmp_path / "source.wav", np.zeros((10, 1), dtype=np.int16))
    target = tmp_path / "out"
    target.mkdir()
    (target / "keep.txt").write_text("keep")
    with pytest.raises(FileExistsError):
        compatible.process_compatible_wav(source, target)
    assert list(path.name for path in target.iterdir()) == ["keep.txt"]


@pytest.mark.parametrize("value", [None, {}, 1, True])
def test_invalid_explicit_configuration_rejected_without_output(tmp_path, value):
    if value is None:
        value = object()
    source = _write(tmp_path / "source.wav", np.zeros((10, 1), dtype=np.int16))
    with pytest.raises(ValueError):
        compatible.process_compatible_wav(source, tmp_path / "out", value)
    assert not (tmp_path / "out").exists()


def test_tampered_frozen_config_revalidated(tmp_path):
    source = _write(tmp_path / "source.wav", np.zeros((10, 1), dtype=np.int16))
    config = BandControlConfig()
    object.__setattr__(config, "band_depths_db", (0, 0, float("nan")))
    with pytest.raises(ValueError):
        compatible.process_compatible_wav(source, tmp_path / "out", config)
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("bad_sample", [float("nan"), float("inf"), -1.01, 1.0, 1 - 1 / 131072])
def test_engine_invalid_or_unrepresentable_outputs_rejected_before_mkdir(tmp_path, monkeypatch, bad_sample):
    source = _write(tmp_path / "source.wav", np.zeros((10, 1), dtype=np.int16))
    def bad_render(samples, settings):
        return np.full_like(samples, bad_sample), {"shared_original_input_fft_count": 0, "analysis_trace_event_count": 0}, {}
    monkeypatch.setattr(compatible, "render_band_control", bad_render)
    with pytest.raises(ValueError):
        compatible.process_compatible_wav(source, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_nonfinite_metadata_rejected_before_any_output(tmp_path, monkeypatch):
    source = _write(tmp_path / "source.wav", np.zeros((10, 1), dtype=np.int16))
    original = compatible.convert_sample_rate
    def bad_conversion(*args, **kwargs):
        samples, metadata = original(*args, **kwargs)
        return samples, {**metadata, "bad": float("nan")}
    monkeypatch.setattr(compatible, "convert_sample_rate", bad_conversion)
    with pytest.raises(ValueError):
        compatible.process_compatible_wav(source, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_source_reread_hash_change_rejected_before_output(tmp_path, monkeypatch):
    source = _write(tmp_path / "source.wav", np.zeros((10, 1), dtype=np.int16))
    calls = []
    def changed(path):
        calls.append(path)
        return "first" if len(calls) == 1 else "changed"
    monkeypatch.setattr(compatible, "_input_sha256", changed)
    with pytest.raises(ValueError, match="changed"):
        compatible.process_compatible_wav(source, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_reports_basename_only_and_source_unchanged(tmp_path):
    source = _write(tmp_path / "sensitive-name.wav", _codes(_signal(seconds=.1)))
    before = source.read_bytes()
    report = compatible.process_compatible_wav(source, tmp_path / "out")
    assert source.read_bytes() == before
    for name in ("report.json", "compatible.json"):
        record = (tmp_path / "out" / name).read_text()
        assert str(tmp_path) not in record
        assert "\\\\" not in record
    assert report["input"]["file_name"] == source.name
    assert report["output"]["sample_rate_hz"] == 48000
    assert report["reference_quality_metrics"] is None
    assert report["device_io"] is False

