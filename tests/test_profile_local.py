"""Opt-in local profile contract, frozen-kernel parity and truthful packaging."""

import base64
import hashlib
import http.client
import io
import json
from pathlib import Path
import threading
from types import SimpleNamespace
from urllib.parse import parse_qs, urlencode, urlsplit
import wave
import zipfile

import numpy as np
import pytest

from hum_dsp import local_app, profile_processing
from hum_dsp.audio import read_pcm16_wav, write_pcm16_wav
from hum_dsp.profile_guard import ProfileGuardConfig, render_profile_guard, replay_profile_guard


FIELDS = {"mode": "profile", "hum_hz": "60", "attenuation_db": "6", "orders": "all",
          "profile_start_seconds": "0", "profile_end_seconds": "2", "confirm_noise_only": "true"}


def _fields(**changes):
    result = dict(FIELDS, **changes)
    return {k: [v] for k, v in result.items()}


def _config(**changes):
    return profile_processing.config_from_profile_query(_fields(**changes))


def _wav(samples, *, rate=48000, channels=1, width=2):
    stream = io.BytesIO()
    with wave.open(stream, "wb") as wav:
        wav.setframerate(rate)
        wav.setnchannels(channels)
        wav.setsampwidth(width)
        payload = np.rint(samples * 32768).astype("<i2").tobytes() if width == 2 else bytes(len(samples) * width * channels)
        wav.writeframes(payload)
    return stream.getvalue()


def test_explicit_query_has_native_scope_and_exact_sample_edges():
    selected = _config(profile_start_seconds="0.000125", profile_end_seconds="2.000125", orders="1,6,21,32")
    assert selected.profile_start_sample == 6
    assert selected.profile_end_sample == 96006
    assert selected.orders == (1, 6, 21, 32)
    assert selected.max_attenuation_db == 6
    assert selected.sample_rate_hz == 48000
    assert _config().orders == tuple(range(1, 33))


@pytest.mark.parametrize("key", FIELDS)
def test_every_profile_field_is_required(key):
    fields = _fields()
    del fields[key]
    with pytest.raises(ValueError):
        profile_processing.config_from_profile_query(fields)


@pytest.mark.parametrize("changes", [
    {"confirm_noise_only": "false"}, {"confirm_noise_only": "True"}, {"confirm_noise_only": "1"},
    {"profile_start_seconds": ""}, {"profile_end_seconds": ""},
    {"profile_start_seconds": "-1"}, {"profile_start_seconds": "NaN"},
    {"profile_start_seconds": "Infinity"}, {"profile_end_seconds": "1.999"},
    {"profile_start_seconds": "2", "profile_end_seconds": "0"},
    {"profile_start_seconds": "0.00001"}, {"profile_end_seconds": "60.00001"},
    {"profile_start_seconds": "1e-5"}, {"profile_start_seconds": " 0"},
    {"profile_start_seconds": "1.999999999999999999999999999999"},
    {"profile_end_seconds": "2.000000000000000000000000000001"},
    {"orders": ""}, {"orders": "0"}, {"orders": "33"}, {"orders": "1,1"},
    {"orders": "2,1"}, {"orders": "1, 2"}, {"orders": "01"}, {"orders": "ALL"},
    {"attenuation_db": "nan"}, {"attenuation_db": "inf"},
    {"attenuation_db": "12.01"}, {"attenuation_db": "-1"}, {"hum_hz": "55"},
])
def test_malformed_or_implicit_profile_settings_are_refused(changes):
    with pytest.raises((ValueError, OverflowError)):
        _config(**changes)


def test_duplicate_unknown_and_excess_query_values_refused():
    for query in (urlencode(FIELDS) + "&orders=1", urlencode(FIELDS) + "&unknown=1",
                  urlencode({**FIELDS, "mode": "fixed"})):
        with pytest.raises(local_app._RequestFailure) as caught:
            local_app._LocalHandler._query(None, urlsplit("/api/process?" + query))
        assert caught.value.status == 400
    mode, config = local_app._LocalHandler._query(None, urlsplit("/api/process?" + urlencode(FIELDS)))
    assert mode == "profile" and config == _config()


@pytest.mark.parametrize("nominal", ("50", "60"))
def test_owned_example_has_fixed_input_independent_of_controls(nominal):
    first = _config(hum_hz=nominal)
    second = _config(hum_hz=nominal, orders="1", attenuation_db="0", profile_start_seconds="1", profile_end_seconds="3")
    payload = profile_processing.profile_example_wav(first)
    assert payload == profile_processing.profile_example_wav(second)
    with wave.open(io.BytesIO(payload), "rb") as wav:
        assert (wav.getframerate(), wav.getnchannels(), wav.getsampwidth(), wav.getnframes()) == (48000, 1, 2, 384000)
        samples = np.frombuffer(wav.readframes(384000), dtype="<i2").astype(float) / 32768
    time = np.arange(384000) / 48000
    expected = sum(np.sqrt(2) * .012 * np.sin(2 * np.pi * order * float(nominal) * time + .37 * order)
                   for order in (1, 6, 21, 24))
    np.testing.assert_allclose(samples[:96000], expected[:96000], atol=1 / 65536)
    assert np.sqrt(np.mean((samples[96000:] - expected[96000:]) ** 2)) > .04
    assert np.max(np.abs(samples)) < .18


def test_file_wrapper_output_matches_unchanged_frozen_kernel_and_replay(tmp_path):
    source = tmp_path / "source.wav"
    source.write_bytes(profile_processing.profile_example_wav(_config(attenuation_db="12")))
    before = source.read_bytes()
    config = _config(attenuation_db="12")
    samples = read_pcm16_wav(source)
    expected, _, expected_trace = render_profile_guard(samples, config)
    report = profile_processing.process_profile_wav(source, tmp_path / "result", config)
    trace = json.loads((tmp_path / "result/profile_trace.json").read_text(encoding="utf-8"))
    assert trace == expected_trace
    np.testing.assert_array_equal(replay_profile_guard(samples, config, trace), expected)
    expected_file = tmp_path / "expected.wav"
    write_pcm16_wav(expected_file, expected)
    assert (tmp_path / "result/processed.wav").read_bytes() == expected_file.read_bytes()
    assert source.read_bytes() == before
    assert report["input"]["sha256"] == hashlib.sha256(before).hexdigest()
    summary = report["profile_guard"]
    assert summary["approved_orders"] == [1, 6, 21, 24]
    assert 7 in summary["unsupported_orders"]
    assert summary["rejected_orders"] == []
    assert summary["same_input_interval"] is True
    assert summary["noise_only_interval_user_asserted"] is True
    assert summary["noise_only_interval_machine_verified"] is False
    assert summary["offline_whole_file_future_lookahead"] is True
    for key in ("real_time", "true_noise_reduction_measured", "wanted_content_preservation_verified",
                "default_product_engine_changed", "commercial_release_approved"):
        assert summary[key] is False
    assert report["output_gain_normalization_applied"] is False
    assert report["reference_quality_metrics"] is None


def test_empty_profile_packaging_is_exact_identity_not_improvement(tmp_path):
    samples = np.zeros(144000)
    samples[96000:] = .05 * np.sin(2 * np.pi * 1260 * np.arange(48000) / 48000)
    payload = _wav(samples)
    server = SimpleNamespace(work_root=tmp_path)
    result = json.loads(local_app._run_job(server, payload, "profile", _config(), "user_file"))
    assert base64.b64decode(result["output_wav_base64"]) == payload
    report = result["report"]
    assert report["processor_applied"] is False
    assert report["profile_guard"]["processing_bypassed"] is True
    assert report["profile_guard"]["approved_orders"] == []
    assert "not noise-reduction success" in report["quality_notice"]
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(result["download_zip_base64"]))) as archive:
        assert set(archive.namelist()) == {"processed.wav", "report.json", "profile_trace.json"}
        assert json.loads(archive.read("report.json")) == report
        assert json.loads(archive.read("profile_trace.json"))["approved_orders"] == []
    assert result["waveforms"]["input"] == result["waveforms"]["output"]
    assert report["local_tool"]["autoplay"] is False
    assert report["local_tool"]["microphone_capture"] is False
    assert report["local_tool"]["remote_upload"] is False
    assert list(tmp_path.iterdir()) == []


def test_missing_or_outside_interval_and_existing_directory_leave_no_output(tmp_path):
    source = tmp_path / "input.wav"
    source.write_bytes(_wav(np.zeros(96000)))
    for config in (ProfileGuardConfig(), _config(profile_end_seconds="3")):
        with pytest.raises(ValueError):
            profile_processing.process_profile_wav(source, tmp_path / "result", config)
        assert not (tmp_path / "result").exists()
    previous = tmp_path / "previous"
    previous.mkdir()
    sentinel = previous / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        profile_processing.process_profile_wav(source, previous, _config())
    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_input_mutation_during_render_is_detected_before_writing(tmp_path, monkeypatch):
    source = tmp_path / "input.wav"
    source.write_bytes(_wav(np.zeros(96000)))
    original = profile_processing.render_profile_guard
    def mutate(samples, config):
        result = original(samples, config)
        source.write_bytes(source.read_bytes() + b"mutation")
        return result
    monkeypatch.setattr(profile_processing, "render_profile_guard", mutate)
    with pytest.raises(ValueError, match="input changed"):
        profile_processing.process_profile_wav(source, tmp_path / "result", _config(attenuation_db="0"))
    assert not (tmp_path / "result").exists()


def test_growing_noise_veto_is_disclosed_without_claiming_removal(tmp_path):
    time = np.arange(144000) / 48000
    samples = np.where(time < 2, .02, .1) * np.sin(2 * np.pi * 60 * time)
    source = tmp_path / "growing.wav"
    source.write_bytes(_wav(samples))
    report = profile_processing.process_profile_wav(source, tmp_path / "result", _config(orders="1"))
    summary = report["profile_guard"]
    assert summary["approved_orders"] == []
    assert summary["rejected_orders"] == [1]
    assert summary["unsupported_orders"] == []
    assert summary["bands"][0]["reason"] == "whole_file_amplitude_exceeds_profile"
    assert summary["bands"][0]["maximum_amplitude_ratio"] > 2
    assert summary["processing_bypassed"] is True
    assert report["processor_applied"] is False
    assert (tmp_path / "result/processed.wav").read_bytes() == source.read_bytes()


@pytest.fixture
def server(tmp_path):
    running = local_app.make_server(work_root=tmp_path / "local")
    thread = threading.Thread(target=lambda: running.serve_forever(poll_interval=.02), daemon=True)
    thread.start()
    try:
        yield running
    finally:
        running.shutdown()
        running.server_close()
        thread.join(timeout=5)
        assert not thread.is_alive()


def _post(server, fields=None, *, example=False, body=None, token=None, origin=None):
    connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=60)
    headers = {"Content-Type": "audio/wav", "X-Hum-Token": server.session_token if token is None else token}
    if origin:
        headers["Origin"] = origin
    try:
        connection.request("POST", ("/api/example?" if example else "/api/process?") + urlencode(fields or FIELDS),
                           body=b"" if example else body, headers=headers)
        response = connection.getresponse()
        return response.status, json.loads(response.read())
    finally:
        connection.close()


@pytest.mark.parametrize("changes", [{"confirm_noise_only": "false"}, {"profile_end_seconds": "1"},
                                     {"orders": "2,1"}, {"profile_start_seconds": ""}])
def test_http_refuses_profile_configuration_before_processing(server, changes):
    status, response = _post(server, dict(FIELDS, **changes), example=True)
    assert status == 400 and set(response) == {"error"}
    assert list(server.work_root.iterdir()) == []


@pytest.mark.parametrize("scope", [{"rate": 44100}, {"channels": 2}, {"width": 3}])
def test_profile_http_does_not_silently_expand_format_scope(server, scope):
    status, response = _post(server, body=_wav(np.zeros(144000), **scope))
    assert status == 400 and set(response) == {"error"}
    assert list(server.work_root.iterdir()) == []


def test_http_error_recovers_and_example_keeps_no_files_or_playback(server):
    status, _ = _post(server, body=_wav(np.zeros(48000)))
    assert status == 400
    status, result = _post(server, dict(FIELDS, attenuation_db="0"), example=True)
    assert status == 200
    assert result["mode"] == "profile"
    assert result["duration_seconds"] == 8
    assert result["source_kind"] == "owned_synthetic_example"
    assert result["report"]["profile_guard"]["bypass_reason"] == "zero_depth"
    assert result["input_wav_base64"] == result["output_wav_base64"]
    assert list(server.work_root.iterdir()) == []


def test_nonzero_http_profile_output_and_zip_match_frozen_kernel(server):
    config = _config(orders="1,6,21,24", attenuation_db="12")
    payload = profile_processing.profile_example_wav(config)
    status, result = _post(server, dict(FIELDS, orders="1,6,21,24", attenuation_db="12"), body=payload)
    assert status == 200
    with wave.open(io.BytesIO(payload), "rb") as wav:
        source = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(float) / 32768
    floating, _, trace = render_profile_guard(source, config)
    expected = _wav(floating)
    assert result["report"]["processor_applied"] is True
    assert result["report"]["profile_guard"]["approved_orders"] == [1, 6, 21, 24]
    assert base64.b64decode(result["output_wav_base64"]) == expected
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(result["download_zip_base64"]))) as archive:
        assert archive.testzip() is None
        assert archive.read("processed.wav") == expected
        assert json.loads(archive.read("profile_trace.json")) == trace
        assert json.loads(archive.read("report.json")) == result["report"]
    assert list(server.work_root.iterdir()) == []


@pytest.mark.parametrize("option", [{"token": "invalid"}, {"origin": "https://foreign.example"}])
def test_profile_mode_does_not_bypass_existing_session_and_origin_guards(server, option):
    status, response = _post(server, example=True, **option)
    assert status == 403 and set(response) == {"error"}
    assert list(server.work_root.iterdir()) == []
