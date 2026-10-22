"""Bounded owned fixtures and fail-closed H3e bookkeeping contracts."""

import json
import sys

import pytest

from hum_dsp import compressed_evaluation as evaluation


@pytest.mark.parametrize("codec,rate,channels", [("MP3", 44100, 1), ("MP3", 48000, 2), ("M4A", 44100, 2), ("M4A", 48000, 1)])
def test_owned_codec_fixture_decodes_to_expected_native_layout(tmp_path, codec, rate, channels):
    av = pytest.importorskip("av")
    target = tmp_path / ("owned.mp3" if codec == "MP3" else "owned.m4a")
    fixture = evaluation._write_owned_compressed(target, codec, rate, .11, channels)
    with av.open(str(target)) as container:
        streams = list(container.streams.audio)
        assert len(streams) == 1
        assert streams[0].codec_context.sample_rate == rate
        assert streams[0].codec_context.channels == channels
        assert sum(frame.samples for frame in container.decode(streams[0])) > 0
    assert fixture["preencode_frames"] == round(rate * .11)
    assert fixture["encoder_and_runtime_decoder_share_backend"] is True
    assert fixture["preencode_frame_identity_after_lossy_codec_claimed"] is False
    with pytest.raises(FileExistsError):
        evaluation._write_owned_compressed(target, codec, rate, .11, channels)


@pytest.mark.parametrize("codec,rate,seconds,channels,suffix", [
    ("FLAC", 48000, 1, 1, ".flac"), ("MP3", 32000, 1, 1, ".mp3"),
    ("MP3", 48000, 0, 1, ".mp3"), ("M4A", 48000, 3.5, 1, ".m4a"),
    ("M4A", 48000, 1, 3, ".m4a"), ("MP3", 48000, 1, 1, ".m4a"),
])
def test_invalid_fixture_refused_without_output(tmp_path, codec, rate, seconds, channels, suffix):
    target = tmp_path / ("invalid" + suffix)
    with pytest.raises(ValueError):
        evaluation._write_owned_compressed(target, codec, rate, seconds, channels)
    assert not target.exists()


def test_missing_optional_dependency_has_no_fabricated_evidence(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "av", None)
    with pytest.raises(RuntimeError, match="optional PyAV"):
        evaluation.evaluate_compressed(tmp_path / "result")
    assert not (tmp_path / "result").exists()


@pytest.mark.parametrize("codec", ["MP3", "M4A"])
def test_actual_short_active_bypass_case_matches_frozen_precision(tmp_path, codec):
    pytest.importorskip("av")
    case = evaluation._case(tmp_path, codec + "_48000_1ch", codec, 48000, 1)
    assert case["all_checks_met"] is True
    assert [run["label"] for run in case["runs"]] == ["active_6_3_0", "bypass_0_0_0"]
    assert all(run["checks"]["frozen_precision_output_container_exact"] for run in case["runs"])
    assert case["runs"][1]["checks"]["bypass_matches_decoded_payload_not_compressed_bytes"] is True
    json.dumps(case, allow_nan=False)


def test_existing_evaluation_never_overwritten(tmp_path, monkeypatch):
    marker = tmp_path / "keep"
    marker.write_text("keep", encoding="utf-8")
    monkeypatch.setattr(evaluation, "_preflight", lambda: pytest.fail("Target refused before preflight"))
    with pytest.raises(FileExistsError):
        evaluation.evaluate_compressed(tmp_path)
    assert marker.read_text(encoding="utf-8") == "keep"


def _fake_sources(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    for name in (*evaluation.CURRENT_SOURCES, "old.py"):
        (source / name).write_text(name, encoding="utf-8")
    frozen = {"old.py": evaluation._sha(source / "old.py")}
    monkeypatch.setattr(evaluation, "_preflight", lambda: (source, frozen, {"stage": "fixed"}))
    return source


def _fake_cases():
    return [{"case_id": name, "fixture": {}, "runs": [], "checks": {}, "all_checks_met": True}
            for name in evaluation.CASE_IDS]


def test_complete_bookkeeping_excludes_private_paths_and_quality_claims(tmp_path, monkeypatch):
    _fake_sources(tmp_path, monkeypatch)
    monkeypatch.setattr(evaluation, "_declared_cases", lambda target: _fake_cases())
    result = evaluation.evaluate_compressed(tmp_path / "result")
    assert result["case_count"] == 8 and result["all_declared_contract_checks_met"] is True
    assert result["perceptual_quality_verified"] is False
    assert result["decoder_independence_claimed"] is False
    assert result["codec_original_sample_identity_claimed"] is False
    assert str(tmp_path) not in json.dumps(result)


@pytest.mark.parametrize("failure", ["missing_case", "case_check", "old_source", "current_source"])
def test_incomplete_failed_or_changed_contract_cannot_pass(tmp_path, monkeypatch, failure):
    source = _fake_sources(tmp_path, monkeypatch)
    def execute(target):
        if failure == "missing_case":
            return []
        if failure in ("old_source", "current_source"):
            (source / ("old.py" if failure == "old_source" else "compressed_processing.py")).write_text("changed", encoding="utf-8")
        cases = _fake_cases()
        if failure == "case_check":
            cases[0]["all_checks_met"] = False
        return cases
    monkeypatch.setattr(evaluation, "_declared_cases", execute)
    if failure == "missing_case":
        with pytest.raises(ValueError, match="Incomplete"):
            evaluation.evaluate_compressed(tmp_path / "result")
    else:
        assert evaluation.evaluate_compressed(tmp_path / "result")["all_declared_contract_checks_met"] is False
