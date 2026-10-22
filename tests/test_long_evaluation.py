"""Lightweight H3b evaluation orchestration; full long cases are CLI opt-in."""

import json
import wave

import numpy as np
import pytest

from hum_dsp import long_evaluation as evaluation


def _events(frames):
    clocks = [0, *range(96000, frames + 1, 4800)]
    return [{"sample_index": clock, "time_seconds": clock / 48000.,
             "event_kind": "initial_dry" if clock == 0 else "observation",
             "bands": [{"first_acquired_sample": None if clock == 0 else 96000}]} for clock in clocks]


def _write_trace(path, events):
    path.write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")


@pytest.mark.parametrize("frames", [1, 95999, 96000, 100799, 100800, 60 * 48000, 125 * 48000])
def test_trace_summary_includes_exact_global_boundaries(tmp_path, frames):
    path = tmp_path / "events.jsonl"
    events = _events(frames)
    _write_trace(path, events)
    summary = evaluation._trace_summary(path, frames, True)
    assert summary["event_count"] == len(events)
    assert summary["last_sample_index"] == events[-1]["sample_index"]
    assert summary["checkpoint_seen"]["60_seconds"] == (frames >= 60 * 48000)
    assert summary["checkpoint_seen"]["120_seconds"] == (frames >= 120 * 48000)
    assert summary["acquisition_preserved_across_60_second_boundary"] is True


@pytest.mark.parametrize("mutation", ["missing", "reset", "time", "kind", "bool_clock", "extra"])
def test_invalid_or_reset_trace_is_refused(tmp_path, mutation):
    path = tmp_path / "events.jsonl"
    events = _events(100800)
    if mutation == "missing":
        events.pop()
    elif mutation == "reset":
        events[-1]["sample_index"] = 0
    elif mutation == "time":
        events[-1]["time_seconds"] = 0
    elif mutation == "kind":
        events[-1]["event_kind"] = "initial_dry"
    elif mutation == "bool_clock":
        events[0]["sample_index"] = False
    else:
        events.append(events[-1])
    _write_trace(path, events)
    with pytest.raises(ValueError):
        evaluation._trace_summary(path, 100800, True)


def test_empty_bypass_trace_is_required_and_has_zero_events(tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_text("", encoding="utf-8")
    assert evaluation._trace_summary(path, 600 * 48000, False)["event_count"] == 0
    _write_trace(path, _events(1))
    with pytest.raises(ValueError):
        evaluation._trace_summary(path, 600 * 48000, False)


def test_acquisition_reset_at_long_boundary_is_reported_not_hidden(tmp_path):
    path = tmp_path / "events.jsonl"
    events = _events(60 * 48000)
    events[-1]["bands"][0]["first_acquired_sample"] = 60 * 48000
    _write_trace(path, events)
    result = evaluation._trace_summary(path, 60 * 48000, True)
    assert result["acquisition_preserved_across_60_second_boundary"] is False


@pytest.mark.parametrize("channels", [1, 2])
@pytest.mark.parametrize("identity", [False, True])
def test_owned_fixture_is_deterministic_bounded_and_new_only(tmp_path, channels, identity):
    first, second = tmp_path / "one.wav", tmp_path / "two.wav"
    metadata = evaluation._write_owned(first, .11, channels, identity_codes=identity)
    evaluation._write_owned(second, .11, channels, identity_codes=identity)
    assert first.read_bytes() == second.read_bytes()
    digest, inspected = evaluation._pcm_digest(first)
    assert len(digest) == 64 and inspected["frames"] == 5280
    assert inspected["channels"] == channels and metadata["maximum_generation_block_frames"] == 4096
    with pytest.raises(FileExistsError):
        evaluation._write_owned(first, .11, channels)
    with wave.open(str(first), "rb") as wav:
        codes = np.frombuffer(wav.readframes(1), dtype="<i2")
    if identity:
        assert codes.tolist() == [7919 - 32768] * channels


@pytest.mark.parametrize("seconds,channels", [(0, 1), (601, 1), (1, 0), (1, 3)])
def test_bad_fixture_spec_does_not_create_file(tmp_path, seconds, channels):
    path = tmp_path / "bad.wav"
    with pytest.raises(ValueError):
        evaluation._write_owned(path, seconds, channels)
    assert not path.exists()


@pytest.mark.parametrize("channels", [1, 2])
def test_actual_short_file_path_matches_preserved_engines(tmp_path, channels):
    result = evaluation._declared_case(tmp_path, f"short_{channels}", 3.4, channels, legacy=True)
    assert result["all_declared_checks_met"] is True
    assert result["checks"]["short_48k_frozen_H3a_exact_PCM16_identity"] is True
    if channels == 1:
        assert result["checks"]["short_48k_mono_frozen_H2_exact_PCM16_identity"] is True
    assert result["runs"][0]["resource_limits"]["max_engine_block_frames_observed"] == 4096


def test_existing_evaluation_directory_is_never_overwritten(tmp_path, monkeypatch):
    marker = tmp_path / "keep.txt"
    marker.write_text("keep", encoding="utf-8")
    monkeypatch.setattr(evaluation, "_preflight", lambda: pytest.fail("existing target must fail before preflight"))
    with pytest.raises(FileExistsError):
        evaluation.evaluate_long(tmp_path)
    assert marker.read_text(encoding="utf-8") == "keep"


def test_evaluation_orchestration_writes_policy_hashes_and_no_private_paths(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    names = ("stream_bands.py", "long_processing.py", "long_evaluation.py", "h3b_policy.json")
    for name in (*names, "old.py"):
        (source / name).write_text(name, encoding="utf-8")
    preserved = {"old.py": evaluation._sha(source / "old.py")}
    policy = {"stage": "H3b_bounded_continuous_long_WAV", "frozen": True}
    monkeypatch.setattr(evaluation, "_preflight", lambda: (source, policy, preserved, "a" * 64))
    cases = [{"case_id": case_id, "runs": [], "checks": {}, "all_declared_checks_met": True}
             for case_id in evaluation.CASE_IDS]
    monkeypatch.setattr(evaluation, "_run_declared_cases", lambda target: cases)
    target = tmp_path / "evaluation"
    result = evaluation.evaluate_long(target)
    stored = json.loads((target / "evaluation.json").read_text(encoding="utf-8"))
    assert result == stored and stored["case_count"] == 5
    assert stored["policy"] == policy and stored["policy_sha256"] == evaluation._sha(source / "h3b_policy.json")
    assert stored["all_declared_contract_checks_met"] is True
    assert stored["perceptual_or_natural_noise_accuracy_verified"] is False
    assert str(tmp_path) not in (target / "evaluation.json").read_text(encoding="utf-8")
    assert "real 125-second" in (target / "report.html").read_text(encoding="utf-8")


def test_orchestration_refuses_missing_declared_case(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    for name in ("stream_bands.py", "long_processing.py", "long_evaluation.py", "h3b_policy.json"):
        (source / name).write_text("frozen", encoding="utf-8")
    monkeypatch.setattr(evaluation, "_preflight", lambda: (source, {}, {}, "a" * 64))
    monkeypatch.setattr(evaluation, "_run_declared_cases", lambda target: [])
    target = tmp_path / "evaluation"
    with pytest.raises(ValueError, match="complete fixed"):
        evaluation.evaluate_long(target)
    assert not (target / "evaluation.json").exists()
