"""Independent fixture and bookkeeping tests for the fixed H3d evaluation."""

import json

import numpy as np
import pytest
from scipy.io import wavfile

from hum_dsp import precision_evaluation as evaluation


@pytest.mark.parametrize("fmt", ["PCM16", "PCM24", "Float32"])
@pytest.mark.parametrize("rate,channels", [(44100, 1), (48000, 2)])
def test_independent_fixture_native_format_and_oracle(tmp_path, fmt, rate, channels):
    first, second = tmp_path / "a.wav", tmp_path / "b.wav"
    fixture = evaluation._write_owned(first, fmt, rate, .11, channels)
    evaluation._write_owned(second, fmt, rate, .11, channels)
    assert first.read_bytes() == second.read_bytes()
    digest, metadata = evaluation._payload(first)
    assert len(digest) == 64
    assert metadata["bits_per_sample"] == fixture["bits_per_sample"]
    assert metadata["frames"] == round(rate * .11)
    assert evaluation._short_oracle(first, fixture)
    with pytest.raises(FileExistsError):
        evaluation._write_owned(first, fmt, rate, .11, channels)


def test_float_fixture_keeps_exact_sign_and_subnormal_bits(tmp_path):
    path = tmp_path / "signs.wav"
    evaluation._write_owned(path, "Float32", 48000, .11, 1, kind="float_bits")
    _, audio = wavfile.read(path)
    assert audio.view(np.uint32)[:8].tolist() == [0, 0x80000000, 0x3F000000, 0xBF000000, 0x3F7FFFFF, 0xBF800000, 1, 0x80000001]


def test_quiet_pcm24_fixture_contains_detail_below_pcm16_resolution(tmp_path):
    path = tmp_path / "quiet.wav"
    evaluation._write_owned(path, "PCM24", 48000, .11, 1, kind="quiet")
    _, left_aligned = wavfile.read(path)
    assert np.any(left_aligned != 0)
    assert np.max(np.abs(left_aligned.astype(np.int64))) < 32768
    assert np.all(np.rint(left_aligned.astype(float) / 2147483648 * 32768) == 0)


@pytest.mark.parametrize("fmt,rate,seconds,channels,kind", [
    ("PCM8", 48000, 1, 1, "sines"), ("PCM24", 32000, 1, 1, "sines"),
    ("PCM24", 48000, 0, 1, "sines"), ("Float32", 48000, 601, 1, "sines"),
    ("Float32", 48000, 1, 3, "sines"), ("PCM24", 48000, 1, 1, "float_bits"),
    ("Float32", 48000, 1, 1, "integer_endpoints"), ("PCM24", 48000, 1, 1, "unknown"),
    ("PCM24", 48000, 600, 2, "sines"), ("Float32", 48000, 600, 2, "sines"),
])
def test_invalid_or_oversize_fixture_refused_before_file_creation(tmp_path, fmt, rate, seconds, channels, kind):
    path = tmp_path / "invalid.wav"
    with pytest.raises(ValueError):
        evaluation._write_owned(path, fmt, rate, seconds, channels, kind=kind)
    assert not path.exists()


def test_one_actual_short_precision_case_has_JSON_safe_checks(tmp_path):
    case = evaluation._case(tmp_path, "active", "PCM24", 48000, 3.4, 1)
    assert case["all_checks_met"] is True
    json.dumps(case, allow_nan=False)


def test_existing_evaluation_never_overwrites(tmp_path, monkeypatch):
    marker = tmp_path / "keep"
    marker.write_text("keep", encoding="utf-8")
    monkeypatch.setattr(evaluation, "_preflight", lambda: pytest.fail("Existing target must be refused first"))
    with pytest.raises(FileExistsError):
        evaluation.evaluate_precision(tmp_path)
    assert marker.read_text() == "keep"


def _fake_sources(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    for name in ("precision_wav.py", "precision_processing.py", "precision_evaluation.py", "h3d_policy.json", "old.py"):
        (source / name).write_text(name, encoding="utf-8")
    frozen = {"old.py": evaluation._sha(source / "old.py")}
    monkeypatch.setattr(evaluation, "_preflight", lambda: (source, frozen, {"stage": "fixed"}))
    return source


def test_complete_evaluation_bookkeeping_and_private_path_exclusion(tmp_path, monkeypatch):
    _fake_sources(tmp_path, monkeypatch)
    cases = [{"case_id": name, "fixture": {}, "runs": [], "checks": {}, "all_checks_met": True} for name in evaluation.CASE_IDS]
    monkeypatch.setattr(evaluation, "_declared_cases", lambda target: cases)
    result = evaluation.evaluate_precision(tmp_path / "result")
    assert result["case_count"] == 15 and result["all_declared_contract_checks_met"] is True
    assert result["perceptual_quality_verified"] is False
    assert str(tmp_path) not in json.dumps(result)


@pytest.mark.parametrize("failure", ["missing_case", "case_check", "old_source", "current_source"])
def test_incomplete_failed_or_changed_evidence_cannot_pass(tmp_path, monkeypatch, failure):
    source = _fake_sources(tmp_path, monkeypatch)
    def execute(target):
        if failure == "missing_case":
            return []
        if failure in ("old_source", "current_source"):
            (source / ("old.py" if failure == "old_source" else "precision_wav.py")).write_text("changed", encoding="utf-8")
        return [{"case_id": name, "fixture": {}, "runs": [], "checks": {}, "all_checks_met": failure != "case_check"} for name in evaluation.CASE_IDS]
    monkeypatch.setattr(evaluation, "_declared_cases", execute)
    if failure == "missing_case":
        with pytest.raises(ValueError):
            evaluation.evaluate_precision(tmp_path / "result")
    else:
        assert evaluation.evaluate_precision(tmp_path / "result")["all_declared_contract_checks_met"] is False
