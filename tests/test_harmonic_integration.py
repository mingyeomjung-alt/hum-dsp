"""H1 file, CLI and local bridge contracts; no sockets or audio device I/O."""

import base64
import hashlib
import io
import json
from types import SimpleNamespace
from urllib.parse import urlsplit
import wave
import zipfile

import numpy as np
import pytest

from hum_dsp import harmonic_processing, local_app
from hum_dsp.audio import read_pcm16_wav, write_pcm16_wav
from hum_dsp.cli import main
from hum_dsp.config import ProcessorConfig
from hum_dsp.harmonics import HarmonicConfig


RATE = 48000


def _write_codes(path, values, *, rate=RATE, channels=1, width=2):
    codes = np.asarray(values, dtype="<i2")
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(channels)
        stream.setsampwidth(width)
        stream.setframerate(rate)
        stream.writeframes(codes.tobytes() if width == 2 else bytes(codes.size * width))


def _short_source(tmp_path):
    source = tmp_path / "원본 입력.wav"
    time = np.arange(2400, dtype=np.float64) / RATE
    write_pcm16_wav(source, 0.08 * np.sin(2 * np.pi * 233 * time))
    return source


@pytest.fixture
def job_server(tmp_path):
    root = tmp_path / "temporary_jobs"
    root.mkdir()
    return SimpleNamespace(work_root=root)


def test_file_zero_depth_preserves_every_pcm16_code_and_source(tmp_path):
    values = np.arange(-32768, 32768, dtype=np.int32).astype("<i2")
    source = tmp_path / "all codes.wav"
    _write_codes(source, values)
    before = source.read_bytes()
    destination = tmp_path / "zero depth"
    settings = HarmonicConfig(orders=(1, 3, 6), max_attenuation_db=0)
    result = harmonic_processing.process_harmonic_wav(source, destination, settings)
    assert (destination / "processed.wav").read_bytes() == before
    assert source.read_bytes() == before
    assert result["processor_applied"] is False
    assert result["config"] == settings.to_dict()
    assert result["algorithm"]["processing_bypassed"] is True
    assert result["algorithm"]["shared_original_input_fft_count"] == 0
    assert result["pcm16_quantization_max_abs_error"] == 0
    assert json.loads((destination / "harmonics.json").read_text(encoding="utf-8"))["events"] == []
    assert set(path.name for path in destination.iterdir()) == {"processed.wav", "report.json", "harmonics.json"}
    assert result == json.loads((destination / "report.json").read_text(encoding="utf-8"))
    json.dumps(result, allow_nan=False)


def test_local_zero_depth_is_exact_pcm16_and_removes_temporary_files(tmp_path, job_server):
    source = _short_source(tmp_path)
    original = source.read_bytes()
    settings = HarmonicConfig(nominal_hum_hz=60, orders=(2, 3), max_attenuation_db=0)
    result = json.loads(local_app._run_job(job_server, original, "harmonics", settings, "user_file"))
    assert base64.b64decode(result["input_wav_base64"]) == original
    assert base64.b64decode(result["output_wav_base64"]) == original
    assert result["mode"] == "harmonics"
    assert result["report"]["config"] == settings.to_dict()
    assert result["report"]["processor_applied"] is False
    assert result["waveforms"]["input"] == result["waveforms"]["output"]
    assert list(job_server.work_root.iterdir()) == []
    assert source.read_bytes() == original
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(result["download_zip_base64"]))) as archive:
        assert set(archive.namelist()) == {"processed.wav", "report.json", "harmonics.json"}
        assert archive.read("processed.wav") == original
        assert json.loads(archive.read("report.json")) == result["report"]
        assert json.loads(archive.read("harmonics.json"))["events"] == []
        assert "temporary_jobs" not in archive.read("report.json").decode("utf-8")
    metadata = result["report"]["local_tool"]
    assert metadata["temporary_files_removed_before_response"] is True
    assert metadata["microphone_capture"] is False
    assert metadata["remote_upload"] is False
    assert metadata["autoplay"] is False


def test_nonzero_multitone_has_identical_file_cli_and_local_wav(tmp_path, job_server, capsys):
    time = np.arange(round(3.4 * RATE), dtype=np.float64) / RATE
    samples = 0.04 * np.sin(2 * np.pi * 233 * time + 0.23)
    for order in (1, 2, 3):
        samples += 0.02 * np.sin(2 * np.pi * 50 * order * time + 0.37 * order)
    source = tmp_path / "owned multitone.wav"
    write_pcm16_wav(source, samples)
    original = source.read_bytes()
    settings = HarmonicConfig()
    destination = tmp_path / "direct"
    report = harmonic_processing.process_harmonic_wav(source, destination, settings)
    direct_bytes = (destination / "processed.wav").read_bytes()
    assert direct_bytes != original
    assert report["processor_applied"] is True
    assert report["input"]["frames"] == report["output"]["frames"] == samples.size
    assert report["algorithm"]["samples_processed"] == samples.size
    assert report["reference_quality_metrics"] is None
    assert report["device_io"] is False
    assert report["output_gain_normalization_applied"] is False
    assert report["posthoc_alignment_applied"] is False
    assert report["input"]["sha256"] == hashlib.sha256(original).hexdigest()
    assert report["output"]["sha256"] == hashlib.sha256(direct_bytes).hexdigest()
    trace = json.loads((destination / "harmonics.json").read_text(encoding="utf-8"))
    assert report["harmonic_trace"]["event_count"] == len(trace["events"])
    assert [event["sample_index"] for event in trace["events"]] == [0, *range(96000, samples.size + 1, 4800)]
    for event in trace["events"]:
        assert [band["order"] for band in event["bands"]] == [1, 2, 3]
    for order in (1, 2, 3):
        assert any(band["order"] == order and band["target_mix"] == 1 for event in trace["events"] for band in event["bands"])
    assert report["algorithm"]["shared_original_input_fft_count"] == len(trace["events"]) - 1
    json.dumps((report, trace), allow_nan=False)

    cli_destination = tmp_path / "cli"
    assert main(["process-harmonics", str(source), "--output-dir", str(cli_destination), "--hum-hz", "50"]) == 0
    cli_payload = json.loads(capsys.readouterr().out)
    assert cli_payload["status"] == "harmonic_filter_output_created"
    assert cli_payload["device_io"] is False
    assert (cli_destination / "processed.wav").read_bytes() == direct_bytes
    assert json.loads((cli_destination / "harmonics.json").read_text(encoding="utf-8")) == trace

    local = json.loads(local_app._run_job(job_server, original, "harmonics", settings, "user_file"))
    assert base64.b64decode(local["output_wav_base64"]) == direct_bytes
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(local["download_zip_base64"]))) as archive:
        assert archive.read("processed.wav") == direct_bytes
        assert json.loads(archive.read("harmonics.json")) == trace
    assert list(job_server.work_root.iterdir()) == []
    assert source.read_bytes() == original


@pytest.mark.parametrize("orders", ((), (0,), (7,), (2, 1), (1, 1), (1.0,), (True,), "1,2,3", None))
def test_invalid_selected_orders_refused(orders):
    with pytest.raises(ValueError):
        HarmonicConfig(orders=orders)


@pytest.mark.parametrize("orders", (["0"], ["7"], ["2", "1"], ["1", "1"], ["1.5"], []))
def test_cli_invalid_orders_refuses_before_directory(tmp_path, capsys, orders):
    source = _short_source(tmp_path)
    before = source.read_bytes()
    target = tmp_path / "absent"
    with pytest.raises(SystemExit) as exc:
        main(["process-harmonics", str(source), "--output-dir", str(target), "--hum-hz", "50", "--orders", *orders])
    assert exc.value.code == 2
    assert not target.exists()
    assert source.read_bytes() == before
    capsys.readouterr()


def test_cli_explicit_nonconsecutive_orders_and_depth_are_preserved(tmp_path, capsys):
    source = _short_source(tmp_path)
    target = tmp_path / "selection"
    assert main(["process-harmonics", str(source), "--output-dir", str(target), "--hum-hz", "60", "--orders", "2", "6", "--attenuation-db", "0", "--block-size", "127"]) == 0
    result = json.loads((target / "report.json").read_text(encoding="utf-8"))
    assert result["config"]["orders"] == [2, 6]
    assert result["config"]["nominal_hum_hz"] == 60
    assert result["config"]["max_attenuation_db"] == 0
    assert result["config"]["block_size_frames"] == 127
    assert (target / "processed.wav").read_bytes() == source.read_bytes()
    capsys.readouterr()


@pytest.mark.parametrize("args", ([], ["--hum-hz", "55"], ["--hum-hz", "50", "--attenuation-db", "nan"]))
def test_cli_invalid_or_missing_frequency_depth_refuses(tmp_path, capsys, args):
    source = _short_source(tmp_path)
    target = tmp_path / "absent"
    with pytest.raises(SystemExit) as exc:
        main(["process-harmonics", str(source), "--output-dir", str(target), *args])
    assert exc.value.code == 2
    assert not target.exists()
    capsys.readouterr()


@pytest.mark.parametrize("kind", ("file", "empty_directory", "nonempty_directory"))
def test_existing_destination_is_never_replaced(tmp_path, kind):
    source = _short_source(tmp_path)
    target = tmp_path / "existing"
    if kind == "file":
        target.write_bytes(b"preserve")
    else:
        target.mkdir()
        if kind == "nonempty_directory":
            (target / "harmonics.json").write_bytes(b"preserve")
    before = source.read_bytes()
    with pytest.raises(FileExistsError):
        harmonic_processing.process_harmonic_wav(source, target)
    assert source.read_bytes() == before
    if kind == "file":
        assert target.read_bytes() == b"preserve"
    elif kind == "nonempty_directory":
        assert (target / "harmonics.json").read_bytes() == b"preserve"
        assert len(list(target.iterdir())) == 1
    else:
        assert list(target.iterdir()) == []


@pytest.mark.parametrize("settings", ({"rate": 8000}, {"rate": 44100}, {"channels": 2}, {"width": 1}, {"width": 3}))
def test_unsupported_wav_refused_before_any_output(tmp_path, settings):
    source = tmp_path / "unsupported.wav"
    _write_codes(source, np.zeros(32), **settings)
    before = source.read_bytes()
    target = tmp_path / "absent"
    with pytest.raises(ValueError):
        harmonic_processing.process_harmonic_wav(source, target)
    assert source.read_bytes() == before
    assert not target.exists()


@pytest.mark.parametrize("payload", (b"", b"not a WAV", b"RIFF\x24\x00\x00\x00WAVE"))
def test_bad_container_refused_and_local_temporary_files_cleaned(tmp_path, job_server, payload):
    source = tmp_path / "bad.wav"
    source.write_bytes(payload)
    target = tmp_path / "absent"
    with pytest.raises((ValueError, EOFError, wave.Error)):
        harmonic_processing.process_harmonic_wav(source, target)
    with pytest.raises((ValueError, EOFError, wave.Error)):
        local_app._run_job(job_server, payload, "harmonics", HarmonicConfig(), "user_file")
    assert not target.exists()
    assert source.read_bytes() == payload
    assert list(job_server.work_root.iterdir()) == []


@pytest.mark.parametrize("value", (1.0, 1.01, -1.01, float("nan"), float("inf"), -float("inf")))
def test_overshoot_or_nonfinite_render_refused_before_directory(tmp_path, monkeypatch, value):
    source = _short_source(tmp_path)
    before = source.read_bytes()
    target = tmp_path / "absent"
    monkeypatch.setattr(harmonic_processing, "render_harmonic", lambda samples, config: (np.full(samples.shape, value), {}, []))
    with pytest.raises(ValueError):
        harmonic_processing.process_harmonic_wav(source, target)
    assert not target.exists()
    assert source.read_bytes() == before


@pytest.mark.parametrize("where", ("algorithm", "trace"))
def test_nonfinite_json_refused_before_directory(tmp_path, monkeypatch, where):
    source = _short_source(tmp_path)
    target = tmp_path / "absent"
    def corrupt(samples, config):
        return samples.copy(), {"bad": float("nan")} if where == "algorithm" else {}, [{"bad": float("inf")}] if where == "trace" else []
    monkeypatch.setattr(harmonic_processing, "render_harmonic", corrupt)
    with pytest.raises(ValueError):
        harmonic_processing.process_harmonic_wav(source, target)
    assert not target.exists()


def test_harmonic_internal_failure_cleans_partially_created_local_job(tmp_path, job_server, monkeypatch):
    source = _short_source(tmp_path)
    def fail_after_partial(source_path, target_path, config):
        target_path.mkdir()
        (target_path / "partial.wav").write_bytes(b"partial")
        raise RuntimeError("injected harmonic failure")
    monkeypatch.setattr(harmonic_processing, "process_harmonic_wav", fail_after_partial)
    with pytest.raises(RuntimeError, match="injected"):
        local_app._run_job(job_server, source.read_bytes(), "harmonics", HarmonicConfig(), "user_file")
    assert list(job_server.work_root.iterdir()) == []


def _query(query):
    # The parser itself is deterministic and needs no server or socket.
    return local_app._LocalHandler._query(None, urlsplit("/api/process?" + query))


def test_local_harmonic_query_accepts_explicit_sorted_orders_and_zero_depth():
    mode, config = _query("mode=harmonics&hum_hz=60&attenuation_db=0&orders=2,3")
    assert mode == "harmonics"
    assert type(config) is HarmonicConfig
    assert config.nominal_hum_hz == 60
    assert config.max_attenuation_db == 0
    assert config.orders == (2, 3)


@pytest.mark.parametrize("query", (
    "mode=harmonics&hum_hz=50&attenuation_db=12",
    "mode=harmonics&hum_hz=50&attenuation_db=12&orders=",
    "mode=harmonics&hum_hz=50&attenuation_db=12&orders=1,1",
    "mode=harmonics&hum_hz=50&attenuation_db=12&orders=3,2",
    "mode=harmonics&hum_hz=50&attenuation_db=12&orders=0",
    "mode=harmonics&hum_hz=50&attenuation_db=12&orders=7",
    "mode=harmonics&hum_hz=50&attenuation_db=12&orders=1.0",
    "mode=harmonics&hum_hz=50&attenuation_db=12&orders=1,%202",
    "mode=harmonics&hum_hz=50&attenuation_db=12&orders=1,&orders=2",
    "mode=harmonics&hum_hz=55&attenuation_db=12&orders=1,2,3",
    "mode=harmonics&hum_hz=50&attenuation_db=nan&orders=1,2,3",
    "mode=harmonics&hum_hz=50&attenuation_db=12&orders=1,2,3&extra=1",
))
def test_local_harmonic_query_refuses_invalid_or_ambiguous_contract(query):
    with pytest.raises(local_app._RequestFailure) as exc:
        _query(query)
    assert exc.value.status == 400


@pytest.mark.parametrize("mode", ("fixed", "tracked", "guarded"))
def test_legacy_local_mode_does_not_silently_accept_harmonic_orders(mode):
    mode_result, config = _query(f"mode={mode}&hum_hz=50&attenuation_db=0")
    assert mode_result == mode
    assert type(config) is ProcessorConfig
    with pytest.raises(local_app._RequestFailure) as exc:
        _query(f"mode={mode}&hum_hz=50&attenuation_db=0&orders=1,2,3")
    assert exc.value.status == 400
