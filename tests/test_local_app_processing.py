"""Processing/cleanup integration tests independent of the browser assets."""

import base64
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import wave
import zipfile

import numpy as np
import pytest

from hum_dsp import local_app
from hum_dsp.audio import read_pcm16_wav, write_pcm16_wav
from hum_dsp.config import ProcessorConfig


@pytest.fixture
def job_server(tmp_path):
    work_root = tmp_path / "jobs"
    work_root.mkdir()
    return SimpleNamespace(work_root=work_root)


def _source_bytes(tmp_path):
    time = np.arange(12000) / 48000.0
    samples = 0.08 * np.sin(2 * np.pi * 233 * time) + 0.025 * np.sin(2 * np.pi * 50 * time)
    path = tmp_path / "source.wav"
    write_pcm16_wav(path, samples)
    return path.read_bytes()


@pytest.mark.parametrize("mode", ("fixed", "tracked", "guarded"))
def test_jobs_use_existing_engine_and_exact_pcm16_bypass(tmp_path, job_server, mode):
    source = _source_bytes(tmp_path)
    config = ProcessorConfig(max_attenuation_db=0)
    result = json.loads(local_app._run_job(job_server, source, mode, config, "user_file"))
    assert result["schema_version"] == 1
    assert result["mode"] == mode
    assert result["source_kind"] == "user_file"
    assert result["duration_seconds"] == 0.25
    assert base64.b64decode(result["input_wav_base64"]) == source
    output = base64.b64decode(result["output_wav_base64"])
    assert output == source
    assert result["report"]["output"]["sha256"] == hashlib.sha256(output).hexdigest()
    assert result["report"]["config"] == config.to_dict()
    assert result["report"]["processor_applied"] is False
    assert len(result["waveforms"]["input"]) == 480
    assert result["waveforms"]["input"] == result["waveforms"]["output"]
    assert max(pair[1] for pair in result["waveforms"]["input"]) < 0.11
    assert min(pair[0] for pair in result["waveforms"]["input"]) > -0.11
    assert list(job_server.work_root.iterdir()) == []
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(result["download_zip_base64"]))) as archive:
        expected = {"processed.wav", "report.json"}
        if mode != "fixed":
            expected.add("tracking.json")
        if mode == "guarded":
            expected.add("guard.json")
        assert set(archive.namelist()) == expected
        assert archive.read("processed.wav") == source
        assert json.loads(archive.read("report.json")) == result["report"]
        assert "jobs" not in archive.read("report.json").decode("utf-8")
    assert result["report"]["local_tool"]["temporary_files_removed_before_response"] is True
    assert result["report"]["local_tool"]["microphone_capture"] is False
    assert result["report"]["local_tool"]["remote_upload"] is False
    assert result["report"]["local_tool"]["autoplay"] is False


@pytest.mark.parametrize("mode, runner_name", (("fixed", "process_wav"), ("tracked", "process_tracked_wav"), ("guarded", "process_guarded_wav")))
def test_processor_fault_removes_all_job_files(tmp_path, job_server, monkeypatch, mode, runner_name):
    def fail_after_partial_write(source, destination, config):
        destination.mkdir()
        (destination / "partial.wav").write_bytes(b"not a result")
        raise RuntimeError("injected internal DSP failure")

    monkeypatch.setattr(local_app, runner_name, fail_after_partial_write)
    with pytest.raises(RuntimeError, match="injected"):
        local_app._run_job(job_server, _source_bytes(tmp_path), mode, ProcessorConfig(), "user_file")
    assert list(job_server.work_root.iterdir()) == []


def test_invalid_wav_refuses_without_leftovers(job_server):
    with pytest.raises((ValueError, wave.Error, EOFError)):
        local_app._run_job(job_server, b"not a wav", "fixed", ProcessorConfig(), "user_file")
    assert list(job_server.work_root.iterdir()) == []


def test_serialization_fault_refuses_and_cleans_job(tmp_path, job_server, monkeypatch):
    original = local_app.process_wav

    def corrupt_report(*args):
        report = original(*args)
        report["bad_numeric"] = float("nan")
        return report

    monkeypatch.setattr(local_app, "process_wav", corrupt_report)
    with pytest.raises(ValueError):
        local_app._run_job(job_server, _source_bytes(tmp_path), "fixed", ProcessorConfig(), "user_file")
    assert list(job_server.work_root.iterdir()) == []


@pytest.mark.parametrize("frequency", (50, 60))
def test_own_example_is_bounded_repeatable_pcm16(frequency):
    config = ProcessorConfig(nominal_hum_hz=frequency)
    payload = local_app._example_wav(config)
    assert payload == local_app._example_wav(config)
    with wave.open(io.BytesIO(payload), "rb") as wav:
        assert wav.getnframes() == 48000 * 6
        assert wav.getframerate() == 48000
        assert wav.getsampwidth() == 2
        assert wav.getnchannels() == 1
        samples = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(float) / 32768
    assert np.max(np.abs(samples)) < 0.18


def test_waveform_does_not_rescale_either_file():
    source = np.array([-0.1, 0.2, 0.05])
    processed = source / 2
    assert local_app._waveform(source) == [[-0.1, -0.1], [0.2, 0.2], [0.05, 0.05]]
    assert local_app._waveform(processed) == [[-0.05, -0.05], [0.1, 0.1], [0.025, 0.025]]
