"""Explicit compressed import, transactional delivery and retained DSP tests."""

import hashlib
import json

import pytest

from hum_dsp.band_control import BandControlConfig
from hum_dsp import compressed_processing as processing
from hum_dsp.compressed_evaluation import _write_owned_compressed
from hum_dsp.precision_processing import process_precision_wav
from hum_dsp.precision_wav import inspect_precision_wav


@pytest.mark.parametrize("codec", ["mp3", "m4a"])
@pytest.mark.parametrize("depths", [(0, 0, 0), (6, 3, 0)])
def test_actual_import_keeps_decoded_format_and_frozen_engine(tmp_path, codec, depths):
    source = tmp_path / ("source." + codec)
    _write_owned_compressed(source, codec.upper(), 48000, .18, 2)
    original = hashlib.sha256(source.read_bytes()).hexdigest()
    target = tmp_path / "delivered"
    config = BandControlConfig(band_depths_db=depths, block_size_frames=4096)
    report = processing.process_compressed_file(source, target, config)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == original
    assert report["compressed_source"]["source_sha256"] == original
    assert report["compressed_source"]["lossy_decode"] is True
    assert report["compressed_source"]["original_sample_identity_guaranteed"] is False
    assert report["input"]["sample_format"] == "IEEE_FLOAT32_little_endian"
    assert report["output"]["frames"] == report["input"]["frames"]
    assert report["output"]["channels"] == 2
    assert report["output"]["sample_rate_hz"] == 48000
    assert report["compressed_import"]["comparison_A"].startswith("decoded_")
    assert report["compressed_import"]["decoder_process_is_OS_security_sandbox"] is False
    assert report["precision_io"]["PCM16_intermediate_used"] is False
    assert report["compressed_import"]["decoded_Float32_WAV_written_to_disk"] is True
    assert report["resource_limits"]["temporary_float_audio_written"] is True
    assert report["resource_limits"]["resampler_float_intermediate_written"] is False
    assert not any(item.startswith("MP3/AAC,") for item in report["limitations"])
    assert report["conversion_pipeline"][0]["sample_rate_conversion_applied"] is False
    expected = tmp_path / "reference"
    process_precision_wav(target / "decoded_input.wav", expected, config)
    assert (target / "processed.wav").read_bytes() == (expected / "processed.wav").read_bytes()
    assert (target / "long_trace.jsonl").read_bytes() == (expected / "long_trace.jsonl").read_bytes()
    assert json.loads((target / "report.json").read_text("utf-8")) == report
    if not any(depths):
        assert report["precision_io"]["raw_payload_identity"] is True
    assert not list(tmp_path.glob(".hum_compressed_*"))
    assert inspect_precision_wav(target / "decoded_input.wav")["frames"] > 0


def test_existing_results_never_replaced_or_decoded(tmp_path, monkeypatch):
    target = tmp_path / "result"
    target.mkdir()
    (target / "keep").write_text("keep", encoding="utf-8")
    from hum_dsp import compressed_decode
    monkeypatch.setattr(compressed_decode, "decode_compressed", lambda *a, **k: pytest.fail("No decode"))
    with pytest.raises(FileExistsError):
        processing.process_compressed_file(tmp_path / "absent.mp3", target)
    assert (target / "keep").read_text("utf-8") == "keep"


@pytest.mark.parametrize("value", [{}, "settings", object()])
def test_invalid_config_is_rejected_before_output(tmp_path, value):
    with pytest.raises(ValueError):
        processing.process_compressed_file(tmp_path / "input.mp3", tmp_path / "out", value)
    assert not (tmp_path / "out").exists()


def test_source_byte_limit_checked_without_full_read(tmp_path):
    source = tmp_path / "large.mp3"
    with source.open("xb") as stream:
        stream.truncate(processing.MAX_SOURCE_BYTES + 1)
    with pytest.raises(ValueError, match="32 MiB"):
        processing.source_digest(source)


@pytest.mark.parametrize("path", ["https://example.com/private.mp3", "//server/share/file.mp3", "\\\\server\\share\\file.mp3"])
def test_remote_source_refused_before_hash_or_decode(path):
    with pytest.raises(ValueError, match="local filesystem"):
        processing.source_digest(path)


def test_decode_failure_leaves_no_result_and_preserves_source(tmp_path, monkeypatch):
    source = tmp_path / "x.mp3"
    source.write_bytes(b"owned invalid compressed input")
    from hum_dsp import compressed_decode
    def fail(*args, **kwargs):
        raise ValueError("owned failure")
    monkeypatch.setattr(compressed_decode, "decode_compressed", fail)
    with pytest.raises(ValueError, match="owned failure"):
        processing.process_compressed_file(source, tmp_path / "out")
    assert source.read_bytes() == b"owned invalid compressed input"
    assert not (tmp_path / "out").exists()
    assert not list(tmp_path.glob(".hum_compressed_*"))
