"""H2 file, command and local-bridge contracts without sockets or device I/O."""

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

from hum_dsp import band_processing, local_app
from hum_dsp.audio import read_pcm16_wav, write_pcm16_wav
from hum_dsp.band_control import BandControlConfig
from hum_dsp.cli import main
from hum_dsp.config import ProcessorConfig
from hum_dsp.harmonics import HarmonicConfig


RATE = 48000


@pytest.fixture
def job_server(tmp_path):
    directory = tmp_path / "temporary_jobs"
    directory.mkdir()
    return SimpleNamespace(work_root=directory)


def _source(tmp_path):
    source = tmp_path / "원본 입력.wav"
    time = np.arange(2400) / RATE
    write_pcm16_wav(source, 0.08 * np.sin(2 * np.pi * 233 * time))
    return source


def _query(value):
    return local_app._LocalHandler._query(None, urlsplit("/api/process?" + value))


def test_actual_compound_file_cli_and_local_outputs_and_traces_match(tmp_path, job_server, capsys):
    time = np.arange(round(3.4 * RATE)) / RATE
    mixture = 0.05 * np.sin(2 * np.pi * 419 * time + 0.13)
    for order in (1, 2, 3):
        mixture += 0.025 * np.sin(2 * np.pi * order * 50 * time + 0.37 * order)
    source = tmp_path / "owned compound.wav"
    write_pcm16_wav(source, mixture)
    original = source.read_bytes()
    config = BandControlConfig(band_depths_db=(6, 3, 0))
    direct = tmp_path / "direct"
    report = band_processing.process_band_wav(source, direct, config)
    processed = (direct / "processed.wav").read_bytes()
    trace = json.loads((direct / "bands.json").read_text(encoding="utf-8"))
    assert processed != original
    assert report["input"]["frames"] == report["output"]["frames"] == mixture.size
    assert report["algorithm"]["streaming_api"] is False
    assert report["algorithm"]["processing_mode"].startswith("offline_two_pass")
    assert report["reference_quality_metrics"] is None
    assert report["device_io"] is False
    assert report["output_gain_normalization_applied"] is False
    assert report["posthoc_alignment_applied"] is False
    assert report["input"]["sha256"] == hashlib.sha256(original).hexdigest()
    assert report["output"]["sha256"] == hashlib.sha256(processed).hexdigest()
    assert set(path.name for path in direct.iterdir()) == {"processed.wav", "report.json", "bands.json"}
    assert [stage["order"] for stage in trace["stages"]] == [1, 2]
    assert [stage["attenuation_db"] for stage in trace["stages"]] == [6, 3]
    assert [event["sample_index"] for event in trace["analysis_trace"]] == [0, *range(96000, mixture.size + 1, 4800)]
    for stage in trace["stages"]:
        assert any(event["bands"][0]["target_mix"] == 1 for event in stage["trace"])

    command = tmp_path / "command"
    assert main(["process-bands", str(source), "--output-dir", str(command), "--hum-hz", "50",
                 "--band-depths-db", "6", "3", "0"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "manual_per_band_output_created"
    assert payload["device_io"] is False
    assert payload["reference_quality_metrics"] is None
    assert (command / "processed.wav").read_bytes() == processed
    assert json.loads((command / "bands.json").read_text(encoding="utf-8")) == trace

    local = json.loads(local_app._run_job(job_server, original, "bands", config, "user_file"))
    assert local["mode"] == "bands"
    assert base64.b64decode(local["input_wav_base64"]) == original
    assert base64.b64decode(local["output_wav_base64"]) == processed
    assert local["report"]["config"] == config.to_dict()
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(local["download_zip_base64"]))) as archive:
        assert set(archive.namelist()) == {"processed.wav", "report.json", "bands.json"}
        assert archive.read("processed.wav") == processed
        assert json.loads(archive.read("bands.json")) == trace
        assert json.loads(archive.read("report.json")) == local["report"]
        assert "temporary_jobs" not in archive.read("report.json").decode("utf-8")
    assert list(job_server.work_root.iterdir()) == []
    assert source.read_bytes() == original


def test_all_zero_preserves_every_pcm16_code_in_file_and_local_zip(tmp_path, job_server):
    source = tmp_path / "all pcm16 codes.wav"
    codes = np.arange(-32768, 32768, dtype=np.int32).astype("<i2")
    with wave.open(str(source), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(RATE)
        stream.writeframes(codes.tobytes())
    original = source.read_bytes()
    config = BandControlConfig(band_depths_db=(0, 0, 0))
    target = tmp_path / "zero"
    report = band_processing.process_band_wav(source, target, config)
    assert (target / "processed.wav").read_bytes() == original
    assert source.read_bytes() == original
    assert report["processor_applied"] is False
    assert report["pcm16_quantization_max_abs_error"] == 0
    assert report["algorithm"]["processing_bypassed"] is True
    assert report["algorithm"]["shared_original_input_fft_count"] == 0
    trace = json.loads((target / "bands.json").read_text(encoding="utf-8"))
    assert trace["analysis_config"] is None
    assert trace["analysis_trace"] == trace["stages"] == []
    local = json.loads(local_app._run_job(job_server, original, "bands", config, "user_file"))
    assert base64.b64decode(local["output_wav_base64"]) == original
    assert local["waveforms"]["input"] == local["waveforms"]["output"]
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(local["download_zip_base64"]))) as archive:
        assert set(archive.namelist()) == {"processed.wav", "report.json", "bands.json"}
        assert archive.read("processed.wav") == original
        assert json.loads(archive.read("bands.json")) == trace
    metadata = local["report"]["local_tool"]
    assert metadata["temporary_files_removed_before_response"] is True
    assert metadata["microphone_capture"] is metadata["remote_upload"] is metadata["autoplay"] is False
    assert list(job_server.work_root.iterdir()) == []


@pytest.mark.parametrize("frequency", [50, 60])
def test_owned_example_source_is_identical_for_all_depth_choices(frequency):
    configurations = [(0, 0, 0), (6, 3, 0), (12, 12, 12), (0, 12, 0)]
    examples = [local_app._example_wav(BandControlConfig(nominal_hum_hz=frequency, band_depths_db=depths))
                for depths in configurations]
    assert all(payload == examples[0] for payload in examples)
    with wave.open(io.BytesIO(examples[0]), "rb") as source:
        assert source.getparams()[:4] == (1, 2, 48000, 288000)
        actual_codes = np.frombuffer(source.readframes(source.getnframes()), dtype="<i2")
    time = np.arange(RATE * 6) / RATE
    expected = (0.07 * np.sin(2 * np.pi * 233 * time)
                + 0.045 * np.sin(2 * np.pi * 419 * time + 0.2)
                + 0.025 * np.sin(2 * np.pi * 811 * time + 0.4))
    expected += sum(np.sqrt(2) * 0.012 * np.sin(2 * np.pi * order * frequency * time + 0.37 * order)
                    for order in (1, 2, 3))
    np.testing.assert_array_equal(actual_codes, np.rint(expected * 32768).astype("<i2"))


@pytest.mark.parametrize("value", (1.0, 1.01, -1.01, float("nan"), float("inf"), -float("inf")))
def test_nonfinite_or_overshoot_render_refused_before_directory(tmp_path, monkeypatch, value):
    source = _source(tmp_path)
    before = source.read_bytes()
    target = tmp_path / "absent"
    monkeypatch.setattr(band_processing, "render_band_control", lambda samples, config: (np.full(samples.shape, value), {}, {}))
    with pytest.raises(ValueError):
        band_processing.process_band_wav(source, target)
    assert not target.exists()
    assert source.read_bytes() == before


@pytest.mark.parametrize("shape", [(2401,), (2400, 1), (0,)])
def test_changed_sample_count_or_shape_refused_before_directory(tmp_path, monkeypatch, shape):
    source = _source(tmp_path)
    target = tmp_path / "absent"
    monkeypatch.setattr(band_processing, "render_band_control", lambda samples, config: (np.zeros(shape), {}, {}))
    with pytest.raises(ValueError):
        band_processing.process_band_wav(source, target)
    assert not target.exists()


@pytest.mark.parametrize("location", ["algorithm", "trace"])
def test_nonfinite_json_refused_before_directory(tmp_path, monkeypatch, location):
    source = _source(tmp_path)
    target = tmp_path / "absent"
    def malformed(samples, config):
        return (samples.copy(), {"bad": float("nan")} if location == "algorithm" else {},
                {"bad": float("inf")} if location == "trace" else {})
    monkeypatch.setattr(band_processing, "render_band_control", malformed)
    with pytest.raises(ValueError):
        band_processing.process_band_wav(source, target)
    assert not target.exists()


def test_partial_internal_failure_cleans_local_job(tmp_path, job_server, monkeypatch):
    source = _source(tmp_path)
    def fail_after_partial(source_path, target_path, config):
        target_path.mkdir()
        (target_path / "partial.wav").write_bytes(b"partial")
        raise RuntimeError("injected H2 failure")
    monkeypatch.setattr(band_processing, "process_band_wav", fail_after_partial)
    with pytest.raises(RuntimeError, match="injected H2"):
        local_app._run_job(job_server, source.read_bytes(), "bands", BandControlConfig(), "user_file")
    assert list(job_server.work_root.iterdir()) == []


@pytest.mark.parametrize("payload", (b"", b"not a wav", b"RIFF\x24\x00\x00\x00WAVE"))
def test_bad_container_cleans_local_job(job_server, payload):
    with pytest.raises((ValueError, EOFError, wave.Error)):
        local_app._run_job(job_server, payload, "bands", BandControlConfig(), "user_file")
    assert list(job_server.work_root.iterdir()) == []


def test_three_explicit_depths_and_all_zero_query_are_supported():
    mode, config = _query("mode=bands&hum_hz=60&band_depths_db=6,3,0")
    assert mode == "bands"
    assert type(config) is BandControlConfig
    assert config.nominal_hum_hz == 60
    assert config.band_depths_db == (6, 3, 0)
    assert config.orders == (1, 2)
    mode, config = _query("mode=bands&hum_hz=50&band_depths_db=0,0,0")
    assert mode == "bands"
    assert config.band_depths_db == (0, 0, 0)
    assert config.orders == ()


@pytest.mark.parametrize("query", [
    "mode=bands&hum_hz=50",
    "mode=bands&band_depths_db=6,3,0",
    "hum_hz=50&band_depths_db=6,3,0",
    "mode=bands&hum_hz=50&band_depths_db=6,3,0&attenuation_db=12",
    "mode=bands&hum_hz=50&band_depths_db=6,3,0&orders=1,2",
    "mode=bands&hum_hz=50&band_depths_db=6,3,0&extra=1",
    "mode=bands&hum_hz=50&band_depths_db=6,3,0&band_depths_db=6,3,0",
    "mode=bands&hum_hz=50&hum_hz=60&band_depths_db=6,3,0",
    "mode=bands&mode=bands&hum_hz=50&band_depths_db=6,3,0",
    "mode=bands&hum_hz=50&band_depths_db=",
    "mode=bands&hum_hz=50&band_depths_db=6,3",
    "mode=bands&hum_hz=50&band_depths_db=6,3,0,0",
    "mode=bands&hum_hz=50&band_depths_db=6,,0",
    "mode=bands&hum_hz=50&band_depths_db=6,3,",
    "mode=bands&hum_hz=50&band_depths_db=6,%203,0",
    "mode=bands&hum_hz=50&band_depths_db=6,3%20,0",
    "mode=bands&hum_hz=50&band_depths_db=%206,3,0",
    "mode=bands&hum_hz=50&band_depths_db=6,3,0%20",
    "mode=bands&hum_hz=50&band_depths_db=nan,3,0",
    "mode=bands&hum_hz=50&band_depths_db=6,inf,0",
    "mode=bands&hum_hz=50&band_depths_db=6,3,-inf",
    "mode=bands&hum_hz=50&band_depths_db=-0.1,3,0",
    "mode=bands&hum_hz=50&band_depths_db=6,12.1,0",
    "mode=bands&hum_hz=50&band_depths_db=True,3,0",
    "mode=bands&hum_hz=50&band_depths_db=6,word,0",
    "mode=bands&hum_hz=55&band_depths_db=6,3,0",
    "mode=bands&hum_hz=50.0&band_depths_db=6,3,0",
])
def test_malformed_band_query_is_rejected(query):
    with pytest.raises(local_app._RequestFailure) as caught:
        _query(query)
    assert caught.value.status == 400


@pytest.mark.parametrize("mode", ["fixed", "tracked", "guarded"])
def test_old_three_field_queries_remain_strict(mode):
    received_mode, config = _query(f"mode={mode}&hum_hz=50&attenuation_db=6")
    assert received_mode == mode
    assert type(config) is ProcessorConfig
    assert config.max_attenuation_db == 6
    with pytest.raises(local_app._RequestFailure):
        _query(f"mode={mode}&hum_hz=50&attenuation_db=6&band_depths_db=6,3,0")
    with pytest.raises(local_app._RequestFailure):
        _query(f"mode={mode}&hum_hz=50&band_depths_db=6,3,0")


def test_h1_four_field_query_retains_its_own_contract():
    mode, config = _query("mode=harmonics&hum_hz=60&attenuation_db=3&orders=2,3")
    assert mode == "harmonics"
    assert type(config) is HarmonicConfig
    assert config.orders == (2, 3)
    assert config.max_attenuation_db == 3
    with pytest.raises(local_app._RequestFailure):
        _query("mode=harmonics&hum_hz=60&band_depths_db=6,3,0")


@pytest.mark.parametrize("depths", [["6", "3"], ["6", "3", "0", "0"],
                                    ["nan", "3", "0"], ["6", "inf", "0"],
                                    ["6", "3", "-1"], ["6", "13", "0"]])
def test_cli_invalid_depths_refused_before_output(tmp_path, capsys, depths):
    source = _source(tmp_path)
    original = source.read_bytes()
    target = tmp_path / "absent"
    with pytest.raises(SystemExit) as caught:
        main(["process-bands", str(source), "--output-dir", str(target), "--hum-hz", "50",
              "--band-depths-db", *depths])
    assert caught.value.code == 2
    assert not target.exists()
    assert source.read_bytes() == original
    capsys.readouterr()


def test_existing_destination_is_never_replaced(tmp_path):
    source = _source(tmp_path)
    target = tmp_path / "existing"
    target.mkdir()
    marker = target / "bands.json"
    marker.write_bytes(b"preserve")
    with pytest.raises(FileExistsError):
        band_processing.process_band_wav(source, target)
    assert marker.read_bytes() == b"preserve"
    assert list(target.iterdir()) == [marker]
