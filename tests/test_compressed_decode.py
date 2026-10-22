"""Bounded, opt-in native decoding and independent fail-closed boundary checks."""

from fractions import Fraction
import hashlib
from importlib import metadata as import_metadata
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from hum_dsp import _compressed_worker as worker
from hum_dsp import compressed_decode as decoder
from hum_dsp.precision_wav import FLOAT32, PrecisionWavReader, inspect_precision_wav


@pytest.fixture(autouse=True)
def _metadata_only_backend_for_parent_unit_tests(monkeypatch):
    # Parent boundary tests do not require a native decoder installation.
    # Real codec tests explicitly import/validate PyAV in _av() and skip when
    # optional support is absent. Tests of missing/old metadata override this.
    original = decoder.import_metadata.version
    monkeypatch.setattr(decoder.import_metadata, "version",
                        lambda name: "19.0.0" if name == "av" else original(name))


def _av():
    av = pytest.importorskip("av")
    if av.__version__.split(".")[0] != "19":
        pytest.skip("the optional decoder requires PyAV 19.x")
    return av


def _encode(path, kind, rate=48000, channels=1, frames=10000, *, codec=None, streams=1,
            layout=None, amplitude=.12):
    av = _av()
    container_format = "mp3" if kind == "mp3" else "mp4"
    codec = codec or ("libmp3lame" if kind == "mp3" else "aac")
    layout = layout or ("mono" if channels == 1 else "stereo")
    with path.open("xb") as handle:
        with av.open(handle, mode="w", format=container_format) as container:
            audio = []
            for _ in range(streams):
                stream = container.add_stream(codec, rate=rate)
                stream.layout = layout
                audio.append(stream)
            for start in range(0, frames, 4096):
                axis = np.arange(start, min(frames, start + 4096)) / rate
                values = np.array([amplitude * np.sin(2 * np.pi * (163 + 37 * channel) * axis)
                                   for channel in range(channels)], dtype=np.float32)
                for stream in audio:
                    frame = av.AudioFrame.from_ndarray(values, format="fltp", layout=layout)
                    frame.sample_rate, frame.pts = rate, start
                    for packet in stream.encode(frame):
                        container.mux(packet)
            for stream in audio:
                for packet in stream.encode(None):
                    container.mux(packet)
    return path


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _independent_native_decode(source, kind):
    av = _av()
    values = []
    with source.open("rb") as handle:
        with av.open(handle, mode="r", format="mp3" if kind == "mp3" else "mov") as container:
            for frame in container.decode(audio=0):
                values.append(frame.to_ndarray())
    return np.concatenate(values, axis=1).T


@pytest.mark.parametrize("kind", ["mp3", "m4a"])
@pytest.mark.parametrize("rate", [44100, 48000])
@pytest.mark.parametrize("channels", [1, 2])
def test_real_lossy_decode_same_rate_layout_native_float_and_source_unchanged(tmp_path, kind, rate, channels):
    source = _encode(tmp_path / ("source." + kind), kind, rate, channels)
    before = _digest(source)
    output = tmp_path / "decoded.wav"
    result = decoder.decode_compressed(source, output)
    assert _digest(source) == before
    assert result["source_sha256"] == before
    assert result["source_format"] == kind
    assert result["source_codec"] == ("mp3" if kind == "mp3" else "aac")
    assert result["source_container"] == ("mp3" if kind == "mp3" else "mov")
    assert result["source_bytes"] == source.stat().st_size
    assert result["lossy_decode"] is True
    assert result["decoder_delay_padding_handled_by_backend"] is True
    assert result["original_sample_identity_guaranteed"] is False
    assert result["native_decoder_os_sandbox"] is False
    assert result["network_protocols_enabled"] is False
    assert result["external_data_references_enabled"] is False
    assert result["decoder"]["name"] == "PyAV"
    assert result["decoder"]["version"].startswith("19.")
    assert result["decoder"]["ffmpeg_libraries"]["libavcodec"][0] >= 63
    metadata = inspect_precision_wav(output)
    assert metadata["sample_format"] == FLOAT32
    assert metadata["sample_rate_hz"] == rate
    assert metadata["channels"] == channels
    assert metadata["fact_frames"] == metadata["frames"]
    assert metadata["container_bytes"] == 58 + metadata["frames"] * channels * 4
    assert result["decoded"]["sha256"] == _digest(output)
    assert result["decoded"]["frames"] == metadata["frames"]
    with PrecisionWavReader(output) as reader:
        native = np.concatenate([values for _raw, values in reader])
    reference = _independent_native_decode(source, kind)
    np.testing.assert_array_equal(native, reference.astype(np.float64))
    assert metadata["frames"] == len(reference)
    assert result["duration_seconds"] == len(reference) / rate
    assert not list(tmp_path.glob("hum_decode_*"))


@pytest.mark.parametrize("kind", ["mp3", "m4a"])
def test_explicit_format_not_suffix_and_quiet_values_below_pcm16_are_preserved(tmp_path, kind):
    source = _encode(tmp_path / "upload.bin", kind, frames=7000, amplitude=1e-5)
    output = tmp_path / "decoded.wav"
    result = decoder.decode_compressed(source, output, source_format=kind)
    with PrecisionWavReader(output) as reader:
        values = np.concatenate([values for _raw, values in reader])
    assert np.any((np.abs(values) > 0) & (np.abs(values) < 1 / 32768))
    assert np.max(np.abs(values)) < 1 / 32768
    assert result["source_format"] == kind


@pytest.mark.parametrize("rate", [8000, 16000, 22050, 32000])
@pytest.mark.parametrize("kind", ["mp3", "m4a"])
def test_unsupported_rate_is_not_automatically_resampled(tmp_path, kind, rate):
    source = _encode(tmp_path / ("source." + kind), kind, rate)
    output = tmp_path / "decoded.wav"
    with pytest.raises(ValueError, match="44100 or 48000"):
        decoder.decode_compressed(source, output)
    assert not output.exists()
    assert source.exists()
    assert not list(tmp_path.glob("hum_decode_*"))


def test_aac_surround_layout_not_downmixed(tmp_path):
    source = _encode(tmp_path / "source.m4a", "m4a", channels=6, layout="5.1")
    with pytest.raises(ValueError, match="canonical mono or stereo"):
        decoder.decode_compressed(source, tmp_path / "decoded.wav")
    assert not (tmp_path / "decoded.wav").exists()


def test_multiple_audio_streams_rejected(tmp_path):
    source = _encode(tmp_path / "source.m4a", "m4a", streams=2)
    with pytest.raises(ValueError, match="exactly one"):
        decoder.decode_compressed(source, tmp_path / "decoded.wav")


def test_non_aac_m4a_rejected(tmp_path):
    source = _encode(tmp_path / "source.m4a", "m4a", codec="alac")
    with pytest.raises(ValueError, match="exactly one"):
        decoder.decode_compressed(source, tmp_path / "decoded.wav")


@pytest.mark.parametrize("kind", ["mp3", "m4a"])
def test_corrupt_container_fails_without_output_or_path_leak(tmp_path, kind):
    source = tmp_path / ("private_secret_name." + kind)
    source.write_bytes(b"not compressed audio\x00" * 20)
    with pytest.raises(ValueError) as caught:
        decoder.decode_compressed(source, tmp_path / "decoded.wav")
    assert "private_secret_name" not in str(caught.value)
    assert str(tmp_path) not in str(caught.value)
    assert not (tmp_path / "decoded.wav").exists()
    assert not list(tmp_path.glob("hum_decode_*"))


def test_truncated_aac_container_rejected(tmp_path):
    source = _encode(tmp_path / "source.m4a", "m4a")
    source.write_bytes(source.read_bytes()[:-15])
    with pytest.raises(ValueError):
        decoder.decode_compressed(source, tmp_path / "decoded.wav")


def test_existing_output_never_truncated_or_replaced(tmp_path):
    source = tmp_path / "source.mp3"
    source.write_bytes(b"invalid")
    output = tmp_path / "decoded.wav"
    output.write_bytes(b"precious existing data")
    with pytest.raises(ValueError, match="already exist"):
        decoder.decode_compressed(source, output)
    assert output.read_bytes() == b"precious existing data"


@pytest.mark.parametrize("source", ["https://example.com/x.mp3", "file:///x.mp3", "//server/share/x.mp3",
                                    "\\\\server\\share\\x.mp3", "", "bad\x00path.mp3"])
def test_remote_or_nonlocal_input_rejected_before_decoder(source, tmp_path, monkeypatch):
    monkeypatch.setattr(decoder, "decoder_status", lambda: pytest.fail("dependency must not be consulted"))
    with pytest.raises(ValueError, match="local filesystem"):
        decoder.decode_compressed(source, tmp_path / "decoded.wav")


@pytest.mark.parametrize("kind", ["wav", "mp4", "aac", "ogg", "MP3", ".mp3", "", None, 1, {}])
def test_explicit_unsupported_format_is_rejected(tmp_path, kind):
    source = tmp_path / "source.unknown"
    source.write_bytes(b"x")
    with pytest.raises(ValueError, match="only local MP3"):
        decoder.decode_compressed(source, tmp_path / "decoded.wav", source_format=kind)


@pytest.mark.parametrize("timeout", [0, -1, 121, float("nan"), float("inf"), True, "120", None])
def test_timeout_is_finite_bounded_and_not_bool(tmp_path, timeout):
    with pytest.raises(ValueError, match="timeout"):
        decoder.decode_compressed(tmp_path / "source.mp3", tmp_path / "decoded.wav", timeout_seconds=timeout)


def test_directory_input_and_missing_parent_rejected(tmp_path):
    with pytest.raises(ValueError, match="regular"):
        decoder.decode_compressed(tmp_path, tmp_path / "decoded.wav")
    source = tmp_path / "source.mp3"
    source.write_bytes(b"x")
    with pytest.raises(ValueError, match="parent directories must exist"):
        decoder.decode_compressed(source, tmp_path / "missing" / "decoded.wav")


def test_zero_input_rejected(tmp_path):
    source = tmp_path / "empty.mp3"
    source.touch()
    with pytest.raises(ValueError, match="1 byte to 32 MiB"):
        decoder.decode_compressed(source, tmp_path / "decoded.wav")


def test_large_source_rejected_by_fstat_without_reading_payload(tmp_path, monkeypatch):
    source = tmp_path / "source.mp3"
    with source.open("wb") as handle:
        handle.truncate(decoder.MAX_SOURCE_BYTES + 1)
    monkeypatch.setattr(decoder, "_worker_decode", lambda *args: pytest.fail("must not launch worker"))
    with pytest.raises(ValueError, match="32 MiB"):
        decoder.decode_compressed(source, tmp_path / "decoded.wav")


def test_dependency_metadata_does_not_import_native_libraries(monkeypatch):
    monkeypatch.setattr(decoder.import_metadata, "version", lambda name: "19.0.0")
    before = sys.modules.get("av")
    status = decoder.decoder_status()
    assert status == {"available": True, "dependency": "av", "required_version": ">=19,<20",
                      "installed_version": "19.0.0", "reason": None}
    assert sys.modules.get("av") is before


@pytest.mark.parametrize("version", ["18.0.0", "20.0.0", "19.0.0rc1", "19", "nonsense"])
def test_unsupported_decoder_version_not_accepted(monkeypatch, version):
    monkeypatch.setattr(decoder.import_metadata, "version", lambda name: version)
    assert decoder.decoder_status()["available"] is False
    assert decoder.decoder_status()["reason"] == "unsupported_dependency_version"


def test_missing_optional_backend_no_auto_install_or_network(tmp_path, monkeypatch):
    source = tmp_path / "source.mp3"
    source.write_bytes(b"x")

    def missing(name):
        raise import_metadata.PackageNotFoundError(name)

    monkeypatch.setattr(decoder.import_metadata, "version", missing)
    monkeypatch.setattr(decoder.subprocess, "Popen", lambda *a, **kw: pytest.fail("must not launch anything"))
    with pytest.raises(RuntimeError, match="optional"):
        decoder.decode_compressed(source, tmp_path / "decoded.wav")
    assert decoder.decoder_status()["reason"] == "optional_dependency_missing"
    assert not (tmp_path / "decoded.wav").exists()


def test_worker_launch_is_fixed_isolated_hidden_and_has_no_pipes(tmp_path, monkeypatch):
    source, output, status = tmp_path / "input.mp3", tmp_path / "decoded.wav", tmp_path / "status.json"
    observed = {}

    class Child:
        def __init__(self, command, **kwargs):
            observed.update(command=command, **kwargs)
            status.write_text('{"ok":false,"error":"corrupt_audio"}', encoding="ascii")

        def wait(self, timeout):
            observed["timeout"] = timeout
            return 1

    monkeypatch.setattr(decoder.subprocess, "Popen", Child)
    with pytest.raises(ValueError, match="corrupt or incomplete"):
        decoder._worker_decode(source, output, "mp3", status, 12.5)
    assert observed["command"][:3] == [sys.executable, "-I", "-c"]
    assert observed["command"][3] == decoder._BOOTSTRAP
    assert observed["command"][-4:] == [str(source), str(output), "mp3", str(status)]
    assert observed["shell"] is False
    assert observed["stdin"] == observed["stdout"] == observed["stderr"] == subprocess.DEVNULL
    assert observed["cwd"] == str(tmp_path)
    assert observed["timeout"] == 12.5
    if os.name == "nt":
        assert observed["creationflags"] & subprocess.CREATE_NO_WINDOW


def test_timeout_kills_and_reaps_child_and_cleans_owned_partial_only(tmp_path, monkeypatch):
    source = tmp_path / "input.mp3"
    source.write_bytes(b"source data")
    unrelated = tmp_path / "unrelated.wav"
    unrelated.write_bytes(b"keep")
    observed = {"waits": 0, "killed": False}

    class Child:
        def __init__(self, command, **kwargs):
            Path(command[-3]).write_bytes(b"owned partial")

        def wait(self, timeout):
            observed["waits"] += 1
            if observed["waits"] == 1:
                raise subprocess.TimeoutExpired("fixed worker", timeout)
            return -9

        def kill(self):
            observed["killed"] = True

    monkeypatch.setattr(decoder.subprocess, "Popen", Child)
    with pytest.raises(RuntimeError, match="time limit"):
        decoder.decode_compressed(source, tmp_path / "decoded.wav", timeout_seconds=.01)
    assert observed == {"waits": 2, "killed": True}
    assert source.read_bytes() == b"source data"
    assert unrelated.read_bytes() == b"keep"
    assert not list(tmp_path.glob("hum_decode_*"))
    assert not (tmp_path / "decoded.wav").exists()


@pytest.mark.parametrize("payload", [b"", b"[]", b"bad", b"{}", b"x" * 16385])
def test_invalid_or_oversized_worker_status_is_fail_closed(tmp_path, monkeypatch, payload):
    status = tmp_path / "status.json"

    class Child:
        def __init__(self, *args, **kwargs):
            status.write_bytes(payload)

        def wait(self, timeout):
            return 0

    monkeypatch.setattr(decoder.subprocess, "Popen", Child)
    with pytest.raises(ValueError):
        decoder._worker_decode(tmp_path / "input.mp3", tmp_path / "out.wav", "mp3", status, 1)


def _box(kind, payload):
    return struct.pack(">I4s", len(payload) + 8, kind) + payload


def _reference_container(entry=None, entries=1):
    entry = _box(b"url ", struct.pack(">I", 1)) if entry is None else entry
    dref = _box(b"dref", struct.pack(">II", 0, entries) + entry)
    nested = dref
    for kind in (b"dinf", b"minf", b"mdia", b"trak", b"moov"):
        nested = _box(kind, nested)
    return _box(b"ftyp", b"M4A \x00\x00\x00\x00") + nested + _box(b"mdat", b"audio")


def test_m4a_data_references_self_contained_only(tmp_path):
    source = tmp_path / "test.m4a"
    source.write_bytes(_reference_container())
    with source.open("rb") as handle:
        worker._validate_m4a_references(handle, source.stat().st_size)
        assert handle.tell() == 0


@pytest.mark.parametrize("entry", [_box(b"url ", struct.pack(">I", 0) + b"https://host/audio\x00"),
                                    _box(b"url ", struct.pack(">I", 0) + b"file:///private\x00"),
                                    _box(b"urn ", struct.pack(">I", 1)),
                                    _box(b"alis", b"\x00" * 4),
                                    _box(b"url ", struct.pack(">I", 3))])
def test_m4a_external_url_urn_alias_or_ambiguous_flags_rejected_without_native_open(tmp_path, entry):
    source = tmp_path / "test.m4a"
    source.write_bytes(_reference_container(entry))
    with source.open("rb") as handle:
        with pytest.raises(worker.DecodeError, match="invalid_container"):
            worker._validate_m4a_references(handle, source.stat().st_size)


@pytest.mark.parametrize("payload", [b"", _box(b"mdat", b"x"), _box(b"moov", b""),
                                      struct.pack(">I4s", 100, b"moov") + b"x",
                                      _reference_container() + b"trailing"])
def test_m4a_box_boundaries_fail_closed(tmp_path, payload):
    source = tmp_path / "test.m4a"
    source.write_bytes(payload)
    with source.open("rb") as handle:
        with pytest.raises(worker.DecodeError):
            worker._validate_m4a_references(handle, source.stat().st_size)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), -1.001, 1.0, 1.01])
def test_float_writer_refuses_nonfinite_or_headroom_instead_of_clipping(tmp_path, value):
    writer = worker._FloatWriter(tmp_path / "out.wav", 48000, 1)
    try:
        with pytest.raises(worker.DecodeError, match="invalid_decoded_samples"):
            writer.write(np.array([[value]], dtype=np.float32), np)
    finally:
        writer.close()


def test_float_writer_preserves_negative_zero_and_native_low_level_bits(tmp_path):
    source = np.array([[-0.0, 1e-8], [np.nextafter(np.float32(0), np.float32(1)), -.75]], dtype=np.float32)
    path = tmp_path / "out.wav"
    writer = worker._FloatWriter(path, 44100, 2)
    writer.write(source, np)
    writer.finish()
    assert path.read_bytes()[58:] == source.astype("<f4").tobytes()
    metadata = inspect_precision_wav(path)
    assert metadata["frames"] == metadata["fact_frames"] == 2
    assert metadata["sample_format"] == FLOAT32


def test_float_writer_checks_duration_and_byte_caps_before_writing(tmp_path, monkeypatch):
    writer = worker._FloatWriter(tmp_path / "out.wav", 44100, 2)
    try:
        monkeypatch.setattr(worker, "MAX_SECONDS", 0)
        with pytest.raises(worker.DecodeError, match="decode_limit_exceeded"):
            writer.write(np.zeros((1, 2), np.float32), np)
        monkeypatch.setattr(worker, "MAX_SECONDS", 600)
        monkeypatch.setattr(worker, "MAX_WAV_BYTES", 58)
        with pytest.raises(worker.DecodeError, match="decode_limit_exceeded"):
            writer.write(np.zeros((1, 2), np.float32), np)
        assert writer.frames == 0
        assert writer.handle.tell() == 58
    finally:
        writer.close()


def test_float_writer_bounds_one_write_and_rejects_empty_finish(tmp_path):
    writer = worker._FloatWriter(tmp_path / "out.wav", 48000, 1)
    try:
        with pytest.raises(worker.DecodeError):
            writer.write(np.zeros((4097, 1), np.float32), np)
        with pytest.raises(worker.DecodeError, match="empty_decoded_audio"):
            writer.finish()
    finally:
        writer.close()


def test_worker_rejects_rate_changes_corrupt_packets_timestamps_and_overshoot(tmp_path, monkeypatch):
    # Native decode is exercised above; this synthetic decoder boundary lets us
    # cover otherwise nondeterministic corrupt/rate/timestamp branches exactly.
    av = _av()
    source = _encode(tmp_path / "input.mp3", "mp3", frames=3000)
    real_open = av.open
    real_container = real_open(source.open("rb"), mode="r", format="mp3")
    frames = list(real_container.decode(audio=0))
    real_container.close()
    assert len(frames) > 1

    class Context:
        codec = type("Codec", (), {"canonical_name": "mp3"})()
        sample_rate = 48000
        layout = frames[0].layout
        options = {}

    class Stream:
        type = "audio"
        codec_context = Context()

    class Packet:
        is_corrupt = False

        def decode(self):
            return frames

    class Container:
        streams = [Stream()]

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def demux(self, stream):
            return [Packet()]

    calls = []

    def controlled_open(handle, **kwargs):
        calls.append((handle, kwargs))
        return Container()

    monkeypatch.setattr(av, "open", controlled_open)
    original_pts = frames[1].pts
    frames[1].pts += 14112000
    with pytest.raises(worker.DecodeError, match="discontinuous_timestamps"):
        worker._decode(source, tmp_path / "gap.wav", "mp3")
    assert calls[0][1]["format"] == "mp3"
    assert calls[0][1]["options"]["enable_drefs"] == "0"
    assert calls[0][1]["options"]["use_absolute_path"] == "0"
    assert calls[0][1]["options"]["protocol_whitelist"] == "pipe"
    assert calls[0][0].closed
    frames[1].pts = original_pts
    frames[0].sample_rate = 44100
    with pytest.raises(worker.DecodeError, match="unsupported_rate"):
        worker._decode(source, tmp_path / "rate.wav", "mp3")
    frames[0].sample_rate = 48000
    Packet.is_corrupt = True
    with pytest.raises(worker.DecodeError, match="corrupt_audio"):
        worker._decode(source, tmp_path / "corrupt.wav", "mp3")


def _fake_worker_success(source, output, kind, status, timeout):
    values = np.array([[0.125], [-0.0625]], dtype=np.float32)
    writer = worker._FloatWriter(output, 48000, 1)
    writer.write(values, np)
    writer.finish()
    return {"ok": True, "source_format": kind, "source_container": "mp3", "source_codec": "mp3",
            "source_bytes": source.stat().st_size, "source_sha256": _digest(source),
            "decoder": {"name": "PyAV", "version": "19.0.0", "ffmpeg_libraries": {"libavcodec": [63, 1, 102]}},
            "decoded": {"frames": 2, "sample_rate_hz": 48000, "channels": 1,
                        "sample_format": FLOAT32, "container_bytes": output.stat().st_size,
                        "sha256": _digest(output)},
            "duration_seconds": 2 / 48000, "lossy_decode": True,
            "decoder_delay_padding_handled_by_backend": True, "original_sample_identity_guaranteed": False}


def test_changed_original_is_not_published_and_child_uses_private_snapshot(tmp_path, monkeypatch):
    source = tmp_path / "source.mp3"
    source.write_bytes(b"original local input")

    def decode(snapshot, *args):
        assert snapshot != source
        assert snapshot.parent.parent == tmp_path
        assert snapshot.read_bytes() == b"original local input"
        result = _fake_worker_success(snapshot, *args)
        source.write_bytes(b"changed source")
        return result

    monkeypatch.setattr(decoder, "_worker_decode", decode)
    with pytest.raises(ValueError, match="changed during"):
        decoder.decode_compressed(source, tmp_path / "decoded.wav")
    assert source.read_bytes() == b"changed source"
    assert not (tmp_path / "decoded.wav").exists()
    assert not list(tmp_path.glob("hum_decode_*"))


@pytest.mark.parametrize("branch,key,value", [("decoded", "frames", 3), ("decoded", "channels", 2),
                                               ("decoded", "sample_rate_hz", 44100),
                                               ("decoded", "container_bytes", 12),
                                               ("decoded", "sha256", "0" * 64),
                                               ("decoded", "sample_format", "signed_PCM16_little_endian"),
                                               (None, "source_sha256", "0" * 64),
                                               (None, "source_bytes", True),
                                               (None, "duration_seconds", True),
                                               (None, "source_codec", "aac"),
                                               (None, "lossy_decode", False),
                                               ("decoder", "version", "18.0.0")])
def test_parent_validates_worker_claims_against_strict_wav_and_source(tmp_path, monkeypatch, branch, key, value):
    source = tmp_path / "source.mp3"
    source.write_bytes(b"compressed source placeholder")

    def inconsistent(*args):
        result = _fake_worker_success(*args)
        (result if branch is None else result[branch])[key] = value
        return result

    monkeypatch.setattr(decoder, "_worker_decode", inconsistent)
    with pytest.raises(ValueError):
        decoder.decode_compressed(source, tmp_path / "decoded.wav")
    assert not (tmp_path / "decoded.wav").exists()
    assert not list(tmp_path.glob("hum_decode_*"))


def test_parent_reads_native_samples_not_only_headers_and_hashes(tmp_path, monkeypatch):
    source = tmp_path / "source.mp3"
    source.write_bytes(b"source")

    def invalid_float(*args):
        result = _fake_worker_success(*args)
        output = args[1]
        with output.open("r+b") as handle:
            handle.seek(58)
            handle.write(struct.pack("<f", float("nan")))
        result["decoded"]["sha256"] = _digest(output)
        return result

    monkeypatch.setattr(decoder, "_worker_decode", invalid_float)
    with pytest.raises(ValueError, match="finite and in"):
        decoder.decode_compressed(source, tmp_path / "decoded.wav")
    assert not (tmp_path / "decoded.wav").exists()


def test_destination_created_during_decode_not_replaced(tmp_path, monkeypatch):
    source, output = tmp_path / "source.mp3", tmp_path / "decoded.wav"
    source.write_bytes(b"source")

    def racing(*args):
        result = _fake_worker_success(*args)
        output.write_bytes(b"new unrelated file")
        return result

    monkeypatch.setattr(decoder, "_worker_decode", racing)
    with pytest.raises(ValueError, match="already exist"):
        decoder.decode_compressed(source, output)
    assert output.read_bytes() == b"new unrelated file"
    assert not list(tmp_path.glob("hum_decode_*"))


@pytest.mark.parametrize("target_kind", ["source", "parent"])
def test_reparse_inputs_or_ancestors_are_rejected(tmp_path, monkeypatch, target_kind):
    source = tmp_path / "source.mp3"
    source.write_bytes(b"source")
    target = source if target_kind == "source" else tmp_path
    original_lstat = Path.lstat

    def reparse(path, *args, **kwargs):
        info = original_lstat(path, *args, **kwargs)
        if path == target:
            return SimpleNamespace(st_mode=info.st_mode, st_file_attributes=1024)
        return info

    monkeypatch.setattr(Path, "lstat", reparse)
    with pytest.raises(ValueError, match="reparse"):
        decoder.decode_compressed(source, tmp_path / "decoded.wav")


def test_start_failure_is_operational_and_sanitized(tmp_path, monkeypatch):
    def failed(*args, **kwargs):
        raise OSError("private interpreter path must not leak")

    monkeypatch.setattr(decoder.subprocess, "Popen", failed)
    with pytest.raises(RuntimeError, match="could not be started") as caught:
        decoder._worker_decode(tmp_path / "in.mp3", tmp_path / "out.wav", "mp3", tmp_path / "status.json", 1)
    assert "private" not in str(caught.value)


def test_float32_endpoint_rounding_is_refused_not_saturated(tmp_path):
    writer = worker._FloatWriter(tmp_path / "output.wav", 48000, 1)
    try:
        with pytest.raises(worker.DecodeError, match="invalid_decoded_samples"):
            writer.write(np.array([[1 - 1e-10]], dtype=np.float64), np)
        assert writer.frames == 0
    finally:
        writer.close()

