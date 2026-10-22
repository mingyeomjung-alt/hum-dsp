"""Local compressed adapter request, delivery, expiry and failure recovery."""

from contextlib import contextmanager
import hashlib
import http.client
import io
import json
import threading
from urllib.parse import urlsplit
import zipfile

import pytest

from hum_dsp import compressed_decode, compressed_processing, local_app
from hum_dsp.compressed_evaluation import _write_owned_compressed


@contextmanager
def _running(root):
    with local_app.make_server(work_root=root) as server:
        thread = threading.Thread(target=lambda: server.serve_forever(poll_interval=.01), daemon=True)
        thread.start()
        try:
            yield server
        finally:
            server.shutdown()
            thread.join(timeout=5)
            assert not thread.is_alive()
    assert not list(root.iterdir())


@pytest.fixture
def server(tmp_path):
    with _running(tmp_path / "jobs") as instance:
        yield instance


def _request(server, path, *, body=None, headers=None, method=None):
    parsed = urlsplit(server.origin)
    connection = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=40)
    settings = {"X-Hum-Token": server.session_token, "Content-Type": "application/octet-stream"}
    settings.update(headers or {})
    try:
        connection.request(method or ("POST" if body is not None else "GET"), path, body=body, headers=settings)
        response = connection.getresponse()
        return response.status, dict(response.getheaders()), response.read()
    finally:
        connection.close()


URL = "/api/compressed/process?hum_hz=50&band_depths_db=0,0,0"


@pytest.mark.parametrize("codec", ["mp3", "m4a"])
def test_actual_codec_upload_delivers_native_decoded_WAV_and_zip(server, tmp_path, codec):
    source = tmp_path / ("owned." + codec)
    _write_owned_compressed(source, codec.upper(), 44100, .18, 1)
    original = source.read_bytes()
    status, _, payload = _request(server, URL, body=original, headers={"X-Hum-Source-Format": codec})
    assert status == 200, payload.decode(errors="replace")
    result = json.loads(payload)
    assert result["mode"] == "compressed" and result["source_kind"] == "user_file"
    report = result["report"]
    assert report["compressed_source"]["source_sha256"] == hashlib.sha256(original).hexdigest()
    assert report["input"]["sample_format"] == report["output"]["sample_format"] == "IEEE_FLOAT32_little_endian"
    assert report["input"]["sample_rate_hz"] == report["output"]["sample_rate_hz"] == 44100
    assert report["input"]["frames"] == report["output"]["frames"]
    assert report["local_tool"]["comparison_input"] == "decoded_Float32_WAV_not_compressed_original"
    assert report["local_tool"]["autoplay"] is False
    assert report["local_tool"]["remote_upload"] is False
    assert not any("base64" in name for name in result)
    assets = {}
    for field, name in (("input_wav_url", "input.wav"), ("output_wav_url", "processed.wav"),
                        ("report_json_url", "report.json"), ("download_zip_url", "result.zip")):
        status, _, contents = _request(server, result[field])
        assert status == 200
        assets[name] = contents
    for key, name in (("input", "input.wav"), ("output", "processed.wav")):
        assert hashlib.sha256(assets[name]).hexdigest() == report[key]["sha256"]
    with zipfile.ZipFile(io.BytesIO(assets["result.zip"])) as archive:
        assert archive.testzip() is None
        assert set(archive.namelist()) == {"processed.wav", "report.json", "long_trace.jsonl"}
        assert archive.read("processed.wav") == assets["processed.wav"]
        assert archive.read("report.json") == assets["report.json"]
    status, response_headers, partial = _request(server, result["input_wav_url"], headers={"Range": "bytes=0-43"})
    assert status == 206 and partial == assets["input.wav"][:44]
    assert "Content-Range" in response_headers
    assert _request(server, result["output_wav_url"], method="HEAD")[0] == 200
    assert source.read_bytes() == original


@pytest.mark.parametrize("headers", [
    {}, {"X-Hum-Source-Format": "wav"}, {"X-Hum-Source-Format": "MP3"},
    {"X-Hum-Source-Format": "../mp3"},
    {"X-Hum-Source-Format": "mp3", "Content-Type": "audio/wav"},
])
def test_invalid_format_or_mime_is_rejected_before_job(server, headers):
    status, _, payload = _request(server, URL, body=b"owned", headers=headers)
    assert status == 400 and isinstance(json.loads(payload)["error"], str)
    assert server.long_jobs.job_id is None
    assert not list(server.work_root.iterdir())


@pytest.mark.parametrize("headers,expected", [
    ({"X-Hum-Token": "wrong", "X-Hum-Source-Format": "mp3"}, 403),
    ({"X-Hum-Source-Format": "mp3", "Origin": "https://example.com"}, 403),
    ({"X-Hum-Source-Format": "mp3", "Content-Length": "33554433"}, 413),
])
def test_auth_origin_and_compressed_size_cap(server, headers, expected):
    assert _request(server, URL, body=b"x", headers=headers)[0] == expected
    assert not list(server.work_root.iterdir())


def test_optional_backend_missing_does_not_break_WAV_routes(server, monkeypatch):
    monkeypatch.setattr(compressed_decode, "decoder_status", lambda: {"available": False})
    status, _, payload = _request(server, URL, body=b"x", headers={"X-Hum-Source-Format": "mp3"})
    assert status == 503 and "PyAV" in json.loads(payload)["error"]
    assert _request(server, "/")[0] == 200
    assert not list(server.work_root.iterdir())


def test_invalid_codec_cleanup_releases_processing_lock(server):
    status, _, _ = _request(server, URL, body=b"not MP3", headers={"X-Hum-Source-Format": "mp3"})
    assert status == 400
    assert not list(server.work_root.iterdir())
    assert server.processing_lock.acquire(blocking=False)
    server.processing_lock.release()


def test_runtime_failure_cleanup(server, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("controlled timeout")
    monkeypatch.setattr(compressed_processing, "process_compressed_file", fail)
    status, _, _ = _request(server, URL, body=b"owned", headers={"X-Hum-Source-Format": "mp3"})
    assert status == 503
    assert not list(server.work_root.iterdir())


def test_shared_lock_rejects_parallel_compressed_job(server):
    server.processing_lock.acquire()
    try:
        assert _request(server, URL, body=b"x", headers={"X-Hum-Source-Format": "mp3"})[0] == 503
    finally:
        server.processing_lock.release()


def test_owned_compressed_example_and_shared_expiry(server):
    status, _, payload = _request(server, "/api/compressed/example?hum_hz=50&band_depths_db=6,3,0", body=b"")
    assert status == 200, payload.decode(errors="replace")
    result = json.loads(payload)
    assert result["mode"] == "compressed"
    assert result["report"]["compressed_source"]["source_codec"] == "mp3"
    assert result["report"]["input"]["channels"] == 2
    assert result["source_kind"] == "owned_synthetic_example"
    previous = result["output_wav_url"]
    status, _, payload = _request(server, "/api/precision/example?hum_hz=50&band_depths_db=0,0,0", body=b"")
    assert status == 200
    assert "compressed" in json.loads(payload)["report"]["local_tool"]["temporary_result_replacement_modes"]
    assert _request(server, previous)[0] == 404


@pytest.mark.parametrize("body,headers", [(b"x", {}), (b"", {"X-Hum-Source-Format": "mp3"})])
def test_example_refuses_body_and_format_override(server, body, headers):
    assert _request(server, "/api/compressed/example?hum_hz=50&band_depths_db=0,0,0", body=body, headers=headers)[0] == 400
