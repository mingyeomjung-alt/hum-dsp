"""Cheap evaluator contracts; the full nine-case matrix is an explicit command."""

import io
import json
import os
from pathlib import Path
from types import SimpleNamespace
import wave

import numpy as np
import pytest

from hum_dsp import local_evaluation as evaluation
from hum_dsp.config import ProcessorConfig


def test_existing_directory_is_refused_before_any_evaluation(tmp_path, monkeypatch):
    sentinel = tmp_path / "keep.txt"
    sentinel.write_text("preserve", encoding="utf-8")
    monkeypatch.setattr(evaluation, "_preflight", lambda: pytest.fail("preflight should not run"))
    with pytest.raises(FileExistsError):
        evaluation.run_local_evaluation(tmp_path)
    assert sentinel.read_text(encoding="utf-8") == "preserve"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["keep.txt"]


@pytest.mark.parametrize("path", (None, "", "   ", 3, object()))
def test_empty_or_nonpath_destination_is_refused(path):
    with pytest.raises(ValueError):
        evaluation.run_local_evaluation(path)


def test_file_parent_is_refused_without_creating_a_result(tmp_path, monkeypatch):
    parent = tmp_path / "not_a_directory"
    parent.write_bytes(b"keep")
    monkeypatch.setattr(evaluation, "_preflight", lambda: {})
    with pytest.raises(OSError):
        evaluation.run_local_evaluation(parent / "new_result")
    assert parent.read_bytes() == b"keep"


def test_cli_uses_actual_entry_point_with_isolated_process_environment(tmp_path, monkeypatch):
    observed = {}
    original_pythonpath = os.environ.get("PYTHONPATH")

    def fake_run(command, **kwargs):
        observed.update(command=command, **kwargs)
        return SimpleNamespace(returncode=0, stdout="a private local path", stderr="")

    monkeypatch.setattr(evaluation.subprocess, "run", fake_run)
    config = ProcessorConfig(nominal_hum_hz=60, max_attenuation_db=12)
    record = evaluation._run_cli(tmp_path / "input.wav", tmp_path / "cli", "guarded", config)
    assert observed["command"][0] == evaluation.sys.executable
    assert observed["command"][1:3] == ["-c", "from hum_dsp.cli import main; raise SystemExit(main())"]
    assert observed["command"][3] == "process-guarded"
    assert "--hum-hz" in observed["command"] and "60" in observed["command"]
    assert observed["check"] is False
    assert observed["timeout"] == 120
    assert observed["capture_output"] is True
    assert str(evaluation._PROJECT_ROOT / "src") in observed["env"]["PYTHONPATH"]
    assert os.environ.get("PYTHONPATH") == original_pythonpath
    assert "private local path" not in json.dumps(record)
    assert record["raw_console_output_stored"] is False


@pytest.mark.parametrize("first,acquired,trace,mix,expected", (
    (96000, True, [{"estimated_frequency_hz": 50, "acquired": True}], 1.0, True),
    (None, False, [], 0.0, False),
    (10, True, [{"estimated_frequency_hz": 50, "acquired": True}], 1.0, False),
    (96000, True, [], 1.0, False),
    (96000, True, [{"estimated_frequency_hz": None, "acquired": True}], 1.0, False),
    (96000, True, [{"estimated_frequency_hz": 50, "acquired": True}], 0.0, False),
))
def test_guarded_nonzero_acquisition_requires_evidence_and_actual_mix(first, acquired, trace, mix, expected):
    report = {"config": {"max_attenuation_db": 12}, "algorithm": {"mix": mix, "raw_tracker": {"first_acquired_sample": first, "acquired": acquired}}}
    assert evaluation._acquisition("guarded", report, trace)["met"] is expected


def test_malformed_input_is_refused_and_temporary_directory_remains_empty(tmp_path):
    work = tmp_path / "jobs"
    work.mkdir()
    result = evaluation._malformed_case(work)
    assert result == {"refused": True, "temporary_jobs_cleaned": True, "successful_payload_created": False, "all_checks_met": True}
    assert list(work.iterdir()) == []


def test_one_cheap_bypass_case_tracks_actual_runner_and_cli_contract(tmp_path, monkeypatch):
    samples = np.rint(0.08 * 32768 * np.sin(2 * np.pi * 233 * np.arange(480) / 48000)).astype("<i2")
    payload = io.BytesIO()
    with wave.open(payload, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(48000)
        wav.writeframes(samples.tobytes())
    monkeypatch.setattr(evaluation, "_example_wav", lambda config: payload.getvalue())

    def fake_cli(source, target, mode, config):
        from hum_dsp.processing import process_wav
        process_wav(source, target, config)
        return {"exit_code": 0, "mocked_in_unit_test_only": True}

    monkeypatch.setattr(evaluation, "_run_cli", fake_cli)
    work = tmp_path / "jobs"
    work.mkdir()
    result = evaluation._case(tmp_path, work, "fixed", 50, 0)
    assert result["all_checks_met"] is True
    assert result["zero_depth"]["exact_pcm16_identity"] is True
    assert result["zero_depth"]["local_path_fft_call_count"] == 0
    assert result["frames"] == 480
    assert list(work.iterdir()) == []
    assert (tmp_path / result["case_id"] / "case.json").is_file()


def test_fft_wrapper_is_restored_after_failure(tmp_path, monkeypatch):
    original = np.fft.rfft
    work = tmp_path / "jobs"
    work.mkdir()

    def fail(*args, **kwargs):
        raise ValueError("injected")

    monkeypatch.setattr(evaluation, "_run_job", fail)
    with pytest.raises(ValueError, match="injected"):
        evaluation._case(tmp_path, work, "fixed", 50, 0)
    assert np.fft.rfft is original
