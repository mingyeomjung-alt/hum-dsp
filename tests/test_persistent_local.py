"""Persistent-hum loopback transport contracts; no playback or devices."""

from contextlib import contextmanager
import http.client
import io
import json
from pathlib import Path
import threading
from urllib.parse import urlsplit
import wave
import zipfile

import numpy as np
import pytest

from hum_dsp import local_app


def _wav(*, seconds=4, hum_hz=60, rate=48000, channels=2):
    time = np.arange(int(seconds * rate), dtype=np.float64) / rate
    hum = sum(.025 * np.sqrt(2) * np.sin(2 * np.pi * hum_hz * order * time)
              for order in (4, 6, 21, 24))
    values = np.column_stack((.05 * np.sin(2 * np.pi * 2300 * time) + hum,
                              .04 * np.sin(2 * np.pi * 3100 * time + .2) + .7 * hum))[:, :channels]
    stream = io.BytesIO()
    with wave.open(stream, "wb") as output:
        output.setnchannels(channels)
        output.setsampwidth(2)
        output.setframerate(rate)
        output.writeframes(np.rint(values * 32768).astype("<i2").tobytes())
    return stream.getvalue()


@contextmanager
def _running(work_root):
    server = local_app.make_server(port=0, work_root=work_root)
    thread = threading.Thread(target=lambda: server.serve_forever(poll_interval=.01), daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        assert not thread.is_alive()


@pytest.fixture
def server(tmp_path):
    with _running(tmp_path / "persistent local") as running:
        yield running


def _request(server, method="GET", path="/api/health", *, body=None, headers=None):
    target = urlsplit(server.origin)
    connection = http.client.HTTPConnection(target.hostname, target.port, timeout=30)
    try:
        fields = {"Host": target.netloc}
        if headers:
            fields.update(headers)
        connection.request(method, path, body=body, headers=fields)
        response = connection.getresponse()
        return response.status, dict(response.getheaders()), response.read()
    finally:
        connection.close()


def _headers(server):
    return {"Content-Type": "audio/wav", "X-Hum-Token": server.session_token}


def _json(payload):
    value = json.loads(payload)
    json.dumps(value, allow_nan=False)
    return value


def test_capability_is_explicit_and_example_publishes_native_files_and_trace(server):
    status, _, payload = _request(server, path="/api/capabilities")
    caps = _json(payload)
    assert status == 200 and caps["persistent"] == {
        "available": True, "explicit_opt_in": True, "maximum_seconds": 60,
        "maximum_container_bytes": 128 * 1024 * 1024,
        "whole_file_memory_processing": True, "quality_approval": False,
    }
    status, _, payload = _request(
        server, "POST", "/api/persistent/example?hum_hz=60&attenuation_db=12",
        body=b"", headers=_headers(server))
    result = _json(payload)
    assert status == 200
    assert result["mode"] == "persistent" and result["source_kind"] == "owned_synthetic_example"
    assert result["duration_seconds"] == 8
    report = result["report"]
    assert report["config"]["nominal_hum_hz"] == 60
    assert report["config"]["max_attenuation_db"] == 12
    assert len(report["algorithm"]["accepted_orders"]) >= 4
    assert report["quality_approval"] is False
    assert len(result["waveforms"]["input"]) <= 480
    assert len(result["waveforms"]["output"]) <= 480
    for key in ("input_wav_url", "output_wav_url", "report_json_url"):
        status, headers, body = _request(server, path=result[key])
        assert status == 200 and int(headers["Content-Length"]) == len(body) and body
    status, _, archive_bytes = _request(server, path=result["download_zip_url"])
    assert status == 200
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        assert set(archive.namelist()) == {"processed.wav", "report.json", "persistent_trace.json"}
        assert json.loads(archive.read("report.json"))["input"]["sha256"] == report["input"]["sha256"]
        assert json.loads(archive.read("persistent_trace.json"))["schema_version"] == 1


def test_upload_and_zero_depth_preserve_raw_audio_payload(server):
    source = _wav()
    status, _, payload = _request(
        server, "POST", "/api/persistent/process?hum_hz=60&attenuation_db=0",
        body=source, headers=_headers(server))
    result = _json(payload)
    assert status == 200 and result["report"]["processing_bypassed"] is True
    assert result["report"]["precision_io"]["raw_payload_identity"] is True
    assert _request(server, path=result["input_wav_url"])[2] == source
    output = _request(server, path=result["output_wav_url"])[2]
    with wave.open(io.BytesIO(source), "rb") as left, wave.open(io.BytesIO(output), "rb") as right:
        assert left.readframes(left.getnframes()) == right.readframes(right.getnframes())


@pytest.mark.parametrize("query", [
    "", "hum_hz=60", "attenuation_db=12", "hum_hz=55&attenuation_db=12",
    "hum_hz=60&attenuation_db=01", "hum_hz=60&attenuation_db=12.0",
    "hum_hz=60&attenuation_db=1e1", "hum_hz=60&attenuation_db=19",
    "hum_hz=60&attenuation_db=12&extra=1",
    "hum_hz=60&hum_hz=50&attenuation_db=12",
])
def test_invalid_settings_fail_before_replacing_previous_result(server, query):
    status, _, payload = _request(
        server, "POST", "/api/persistent/example?hum_hz=60&attenuation_db=0",
        body=b"", headers=_headers(server))
    assert status == 200
    previous = server.long_jobs.job_id
    status, _, payload = _request(
        server, "POST", f"/api/persistent/example?{query}", body=b"", headers=_headers(server))
    assert status == 400 and "error" in _json(payload)
    assert server.long_jobs.job_id == previous


def test_persistent_route_requires_token_and_does_not_create_work(server):
    # Keep the rejected request body below the transport's documented bounded
    # drain limit.  The authorization check runs before WAV parsing, so a full
    # multi-megabyte fixture adds no coverage and can make Windows reset an
    # HTTP/1.0 connection while the client is still sending the rejected body.
    status, _, payload = _request(
        server, "POST", "/api/persistent/process?hum_hz=60&attenuation_db=12",
        body=b"RIFF", headers={"Content-Type": "audio/wav"})
    assert status == 403 and "error" in _json(payload)
    assert server.long_jobs.job_id is None
