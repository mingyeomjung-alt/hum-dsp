"""Fixed H3c evaluation bookkeeping, no heavy long run in the unit suite."""

import json
import wave

import numpy as np
import pytest

from hum_dsp import long_compatible_evaluation as evaluation


@pytest.mark.parametrize("rate", [44100, 48000])
@pytest.mark.parametrize("channels,antiphase,identity", [(1, False, False), (2, False, False), (2, True, False), (1, False, True)])
def test_deterministic_owned_files_native_rate_and_channels(tmp_path, rate, channels, antiphase, identity):
    first, second = tmp_path / "a.wav", tmp_path / "b.wav"
    for path in (first, second):
        metadata = evaluation._write_owned(path, rate, .11, channels, antiphase=antiphase, identity=identity)
    assert first.read_bytes() == second.read_bytes()
    digest, native = evaluation._pcm(first)
    assert len(digest) == 64 and native["frames"] == round(rate * .11)
    assert native["channels"] == channels and native["sample_rate_hz"] == rate
    assert metadata["antiphase"] is antiphase
    if antiphase:
        with wave.open(str(first), "rb") as audio:
            codes = np.frombuffer(audio.readframes(audio.getnframes()), dtype="<i2").reshape(-1, 2)
        assert np.array_equal(codes[:, 0].astype(int), -codes[:, 1].astype(int))
    with pytest.raises(FileExistsError):
        evaluation._write_owned(first, rate, .11, channels)


@pytest.mark.parametrize("rate,seconds,channels", [(32000, 1, 1), (44100, 0, 1), (44100, 601, 1), (44100, 1, 3), (44100, .000001, 1)])
def test_invalid_owned_spec_leaves_no_output(tmp_path, rate, seconds, channels):
    path = tmp_path / "bad.wav"
    with pytest.raises(ValueError):
        evaluation._write_owned(path, rate, seconds, channels)
    assert not path.exists()


def test_existing_result_refused_before_preflight(tmp_path, monkeypatch):
    marker = tmp_path / "keep"
    marker.write_text("keep", encoding="utf-8")
    monkeypatch.setattr(evaluation, "_preflight", lambda: pytest.fail("Must refuse existing directory first"))
    with pytest.raises(FileExistsError):
        evaluation.evaluate_long_compatible(tmp_path)
    assert marker.read_text() == "keep"


def _fake_sources(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    names = ("stream_resampling.py", "long_compatible_processing.py", "long_compatible_evaluation.py", "h3c_policy.json", "old.py")
    for name in names:
        (source / name).write_text(name, encoding="utf-8")
    frozen = {"old.py": evaluation._sha(source / "old.py")}
    monkeypatch.setattr(evaluation, "_preflight", lambda: (source, frozen, {"stage": "fixed"}))
    return source


def test_orchestration_records_all_cases_and_hashes_without_private_paths(tmp_path, monkeypatch):
    _fake_sources(tmp_path, monkeypatch)
    cases = [{"case_id": name, "fixture": {}, "runs": [], "checks": {}, "all_checks_met": True} for name in evaluation.CASE_IDS]
    monkeypatch.setattr(evaluation, "_declared_cases", lambda target: cases)
    target = tmp_path / "result"
    result = evaluation.evaluate_long_compatible(target)
    assert result["case_count"] == 7 and result["all_declared_contract_checks_met"] is True
    assert result["policy"] == {"stage": "fixed"}
    assert result["perceptual_quality_verified"] is False
    raw = (target / "evaluation.json").read_text(encoding="utf-8")
    assert str(tmp_path) not in raw and str(tmp_path) not in (target / "report.html").read_text()
    assert json.loads(raw) == result


def test_missing_fixed_case_refuses_success_report(tmp_path, monkeypatch):
    _fake_sources(tmp_path, monkeypatch)
    monkeypatch.setattr(evaluation, "_declared_cases", lambda target: [])
    target = tmp_path / "result"
    with pytest.raises(ValueError, match="Incomplete"):
        evaluation.evaluate_long_compatible(target)
    assert not (target / "evaluation.json").exists()


@pytest.mark.parametrize("failure", ["case", "historical", "current"])
def test_failed_or_mutated_evidence_cannot_pass(tmp_path, monkeypatch, failure):
    source = _fake_sources(tmp_path, monkeypatch)
    def execute(target):
        if failure in ("historical", "current"):
            (source / ("old.py" if failure == "historical" else "stream_resampling.py")).write_text("changed", encoding="utf-8")
        return [{"case_id": name, "fixture": {}, "runs": [], "checks": {}, "all_checks_met": failure != "case"} for name in evaluation.CASE_IDS]
    monkeypatch.setattr(evaluation, "_declared_cases", execute)
    assert evaluation.evaluate_long_compatible(tmp_path / "result")["all_declared_contract_checks_met"] is False
