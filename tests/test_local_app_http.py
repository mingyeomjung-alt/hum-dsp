"""Independent local-browser HTTP contracts; no playback, devices or network peers."""

import base64
import http.client
import io
import json
import threading
import wave
import zipfile
from contextlib import contextmanager
from urllib.parse import urlsplit

import numpy as np
import pytest

from hum_dsp import local_app


RATE = 48_000
PARAMETERS = "mode=guarded&hum_hz=50&attenuation_db=0"


def _wav(*, rate=RATE, channels=1, width=2, frames=480):
    stream = io.BytesIO()
    codes = np.rint(0.1 * 32768 * np.sin(2 * np.pi * 50 * np.arange(frames) / rate)).astype("<i2")
    with wave.open(stream, "wb") as output:
        output.setnchannels(channels)
        output.setsampwidth(width)
        output.setframerate(rate)
        output.writeframes(codes.tobytes() if width == 2 else bytes(frames * width * channels))
    return stream.getvalue()


def _codes(payload):
    with wave.open(io.BytesIO(payload), "rb") as stream:
        assert (stream.getframerate(), stream.getnchannels(), stream.getsampwidth()) == (RATE, 1, 2)
        return np.frombuffer(stream.readframes(stream.getnframes()), dtype="<i2").copy()


@contextmanager
def _running_server(work_root):
    server = local_app.make_server(port=0, work_root=work_root)
    thread = threading.Thread(target=lambda: server.serve_forever(poll_interval=0.02), daemon=True)
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
    with _running_server(tmp_path / "request work") as running:
        yield running


def _request(server, method="GET", path="/api/health", *, body=None, headers=None, raw_headers=None):
    target = urlsplit(server.origin)
    connection = http.client.HTTPConnection(target.hostname, target.port, timeout=10)
    try:
        if raw_headers is not None:
            connection.putrequest(method, path, skip_host=True, skip_accept_encoding=True)
            for name, value in raw_headers:
                connection.putheader(name, value)
            connection.endheaders(body)
        else:
            selected = {"Host": target.netloc}
            if headers:
                selected.update(headers)
            connection.request(method, path, body=body, headers=selected)
        response = connection.getresponse()
        return response.status, dict(response.getheaders()), response.read()
    finally:
        connection.close()


def _post_headers(server, **extra):
    return {"Content-Type": "audio/wav", "X-Hum-Token": server.session_token, **extra}


def _json(payload):
    record = json.loads(payload)
    json.dumps(record, allow_nan=False)
    return record


def _work_empty(server):
    assert not server.work_root.exists() or not list(server.work_root.iterdir())


def test_server_binds_only_loopback_and_has_a_per_instance_secret(tmp_path):
    with _running_server(tmp_path / "first") as first, _running_server(tmp_path / "second") as second:
        assert first.server_address[0] == second.server_address[0] == "127.0.0.1"
        assert first.origin == f"http://127.0.0.1:{first.server_address[1]}"
        assert isinstance(first.session_token, str) and len(first.session_token) >= 32
        assert first.session_token != second.session_token
        assert first.work_root.resolve().is_relative_to(tmp_path.resolve())
        assert second.work_root.resolve().is_relative_to(tmp_path.resolve())


def test_health_is_ready_without_a_processing_or_playback_side_effect(server):
    status, headers, payload = _request(server)
    assert status == 200
    assert "application/json" in headers["Content-Type"]
    assert _json(payload)["status"] == "ready"
    _work_empty(server)


@pytest.mark.parametrize("path,content_type", [("/", "text/html"), ("/app.js", "javascript"), ("/style.css", "text/css")])
def test_only_packaged_static_assets_are_served_with_private_security_headers(server, path, content_type):
    status, headers, payload = _request(server, path=path)
    assert status == 200
    assert content_type in headers["Content-Type"]
    assert payload
    assert "no-store" in headers["Cache-Control"]
    assert headers["X-Content-Type-Options"] == "nosniff"
    policy = headers["Content-Security-Policy"]
    assert "default-src 'self'" in policy or "default-src 'none'" in policy
    assert "object-src 'none'" in policy or "default-src 'none'" in policy
    assert "frame-ancestors 'none'" in policy
    assert not any(key.lower().startswith("access-control-allow") for key in headers)
    if path == "/":
        text = payload.decode("utf-8")
        assert server.session_token in text
        assert "__SESSION_TOKEN__" not in text
        assert "autoplay" not in text.lower()
    _work_empty(server)


@pytest.mark.parametrize("path", ["/missing", "/README.md", "/local_app.py", "/../config.py", "/%2e%2e/config.py", "/%2F..%2Fconfig.py", "/api/unknown"])
def test_static_and_api_paths_do_not_expose_arbitrary_files(server, path):
    status, _, payload = _request(server, path=path)
    assert status == 404
    assert server.session_token.encode() not in payload
    assert str(server.work_root).encode() not in payload
    _work_empty(server)


@pytest.mark.parametrize("host", ["localhost", "localhost:8765", "127.0.0.2", "evil.example", "127.0.0.1.evil.example"])
def test_untrusted_hosts_are_refused_even_for_health(server, host):
    status, _, _ = _request(server, headers={"Host": host})
    assert status == 403
    _work_empty(server)


@pytest.mark.parametrize("hosts", [[], [("Host", "127.0.0.1:1"), ("Host", "evil.example")]])
def test_missing_and_duplicate_host_headers_are_refused(server, hosts):
    status, _, _ = _request(server, raw_headers=hosts)
    assert status == 403
    _work_empty(server)


@pytest.mark.parametrize("token", [None, "", "wrong-token"])
def test_post_requires_the_session_token(server, token):
    headers = {"Content-Type": "audio/wav"}
    if token is not None:
        headers["X-Hum-Token"] = token
    status, _, payload = _request(server, "POST", f"/api/process?{PARAMETERS}", body=_wav(), headers=headers)
    assert status == 403
    assert server.session_token.encode() not in payload
    _work_empty(server)


@pytest.mark.parametrize("origin", ["null", "https://evil.example", "http://localhost:8765", "http://127.0.0.1:1"])
def test_token_does_not_override_a_foreign_origin(server, origin):
    status, _, _ = _request(server, "POST", f"/api/process?{PARAMETERS}", body=_wav(), headers=_post_headers(server, Origin=origin))
    assert status == 403
    _work_empty(server)


def test_cross_site_fetch_is_refused_even_when_origin_and_token_are_valid(server):
    status, _, _ = _request(server, "POST", f"/api/process?{PARAMETERS}", body=_wav(), headers=_post_headers(server, Origin=server.origin, **{"Sec-Fetch-Site": "cross-site"}))
    assert status == 403
    _work_empty(server)


@pytest.mark.parametrize("query", [
    "mode=other&hum_hz=50&attenuation_db=0",
    "mode=guarded&hum_hz=55&attenuation_db=0",
    "mode=guarded&hum_hz=50&attenuation_db=nan",
    "mode=guarded&hum_hz=50&attenuation_db=inf",
    "mode=guarded&hum_hz=50&attenuation_db=-1",
    "mode=guarded&hum_hz=50&attenuation_db=12.01",
    "mode=guarded&hum_hz=50&attenuation_db=0&unexpected=1",
    "mode=fixed&mode=guarded&hum_hz=50&attenuation_db=0",
    "mode=guarded&hum_hz=50&hum_hz=60&attenuation_db=0",
    "mode=guarded&hum_hz=50&attenuation_db=0&attenuation_db=1",
])
def test_invalid_or_ambiguous_settings_are_refused_before_any_output(server, query):
    status, _, payload = _request(server, "POST", f"/api/process?{query}", body=_wav(), headers=_post_headers(server))
    assert status == 400
    _json(payload)
    _work_empty(server)


@pytest.mark.parametrize("content_type", ["text/plain", "application/json", "multipart/form-data"])
def test_upload_is_raw_wav_not_an_unbounded_multipart_or_json_body(server, content_type):
    status, _, _ = _request(server, "POST", f"/api/process?{PARAMETERS}", body=_wav(), headers=_post_headers(server, **{"Content-Type": content_type}))
    assert status == 400
    _work_empty(server)


@pytest.mark.parametrize("fields,expected", [
    ([], 400),
    ([("Content-Length", "-1")], 400),
    ([("Content-Length", "1.5")], 400),
    ([("Content-Length", "abc")], 400),
    ([("Content-Length", "0"), ("Content-Length", "0")], 400),
    ([("Transfer-Encoding", "chunked")], 400),
    ([("Content-Length", "1"), ("Transfer-Encoding", "chunked")], 400),
    ([("Content-Length", str(32 * 1024 * 1024 + 1))], 413),
])
def test_body_length_is_checked_before_reading_untrusted_bytes(server, fields, expected):
    host = urlsplit(server.origin).netloc
    headers = [("Host", host), ("X-Hum-Token", server.session_token), ("Content-Type", "audio/wav"), *fields]
    status, _, _ = _request(server, "POST", f"/api/process?{PARAMETERS}", raw_headers=headers)
    assert status == expected
    _work_empty(server)


@pytest.mark.parametrize("payload", [b"", b"not a WAV", b"RIFF\x24\x00\x00\x00WAVE", _wav(rate=8000), _wav(channels=2), _wav(width=1)], ids=["empty", "not_wave", "truncated", "8khz", "stereo", "pcm8"])
def test_malformed_or_unsupported_wav_never_leaves_a_temporary_upload(server, payload):
    status, _, data = _request(server, "POST", f"/api/process?{PARAMETERS}", body=payload, headers=_post_headers(server))
    assert status == 400
    _json(data)
    _work_empty(server)


def test_an_in_flight_job_returns_busy_without_waiting_or_overwriting_work(server):
    assert server.processing_lock.acquire(blocking=False)
    try:
        status, headers, payload = _request(server, "POST", f"/api/process?{PARAMETERS}", body=_wav(), headers=_post_headers(server))
    finally:
        server.processing_lock.release()
    assert status == 503
    _json(payload)
    assert "no-store" in headers["Cache-Control"]
    _work_empty(server)


@pytest.mark.parametrize("mode", ["fixed", "tracked", "guarded"])
def test_valid_zero_depth_request_accepts_no_origin_and_cleans_its_work(server, mode):
    original = _wav()
    status, headers, payload = _request(server, "POST", f"/api/process?mode={mode}&hum_hz=50&attenuation_db=0", body=original, headers=_post_headers(server))
    assert status == 200
    assert "application/json" in headers["Content-Type"]
    result = _json(payload)
    assert result["schema_version"] == 1
    assert result["mode"] == mode
    assert result["source_kind"] == "user_file"
    assert result["duration_seconds"] == 0.01
    downloaded_input = base64.b64decode(result["input_wav_base64"], validate=True)
    downloaded_output = base64.b64decode(result["output_wav_base64"], validate=True)
    assert downloaded_input == original
    np.testing.assert_array_equal(_codes(downloaded_output), _codes(original))
    report = result["report"]
    assert report["config"]["nominal_hum_hz"] == 50
    assert report["config"]["max_attenuation_db"] == 0
    local = report["local_tool"]
    assert local["mode"] == mode
    for key in ("microphone_capture", "autoplay", "remote_upload"):
        assert local[key] is False
    assert local["temporary_files_removed_before_response"] is True
    assert local["waveform_scale"] == "common_pcm_full_scale_not_independently_normalized"
    input_wave = np.asarray(result["waveforms"]["input"])
    output_wave = np.asarray(result["waveforms"]["output"])
    assert 1 <= len(input_wave) <= 480
    assert input_wave.shape == output_wave.shape == (len(input_wave), 2)
    assert np.isfinite(input_wave).all() and np.isfinite(output_wave).all()
    assert np.all(input_wave[:, 0] <= input_wave[:, 1])
    assert np.max(np.abs(input_wave)) <= 1
    # Common PCM scale must not exaggerate the low-amplitude input to full scale.
    assert np.max(np.abs(input_wave)) < 0.11
    np.testing.assert_array_equal(input_wave, output_wave)
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(result["download_zip_base64"], validate=True))) as archive:
        expected = {"processed.wav", "report.json"}
        if mode in ("tracked", "guarded"):
            expected.add("tracking.json")
        if mode == "guarded":
            expected.add("guard.json")
        assert set(archive.namelist()) == expected
        assert archive.read("processed.wav") == downloaded_output
        assert _json(archive.read("report.json")) == report
        for name in archive.namelist():
            assert not name.startswith(("/", "\\")) and ".." not in name
        for name in expected - {"processed.wav", "report.json"}:
            trace = _json(archive.read(name))
            assert isinstance(trace["events"], list)
    assert server.session_token.encode() not in payload
    _work_empty(server)


def test_valid_same_origin_octet_stream_upload_is_accepted(server):
    status, _, payload = _request(server, "POST", f"/api/process?{PARAMETERS}", body=_wav(), headers=_post_headers(server, Origin=server.origin, **{"Content-Type": "application/octet-stream"}))
    assert status == 200
    assert _json(payload)
    _work_empty(server)


def test_example_route_uses_its_own_synthetic_signal_not_an_uploaded_body(server):
    status, _, payload = _request(server, "POST", f"/api/example?{PARAMETERS}", body=b"", headers=_post_headers(server))
    assert status == 200
    result = _json(payload)
    assert result["source_kind"] == "owned_synthetic_example"
    assert result["duration_seconds"] == 6
    source = base64.b64decode(result["input_wav_base64"], validate=True)
    output = base64.b64decode(result["output_wav_base64"], validate=True)
    np.testing.assert_array_equal(_codes(source), _codes(output))
    _work_empty(server)


def test_unknown_post_route_does_not_fall_back_to_processing(server):
    status, _, payload = _request(server, "POST", f"/api/missing?{PARAMETERS}", body=b"", headers=_post_headers(server))
    assert status == 404
    assert server.session_token.encode() not in payload
    _work_empty(server)


@pytest.mark.parametrize("method", ["PUT", "DELETE", "OPTIONS", "PATCH", "TRACE"])
def test_unsupported_methods_cannot_skip_host_validation(server, method):
    status, _, _ = _request(server, method, headers={"Host": "evil.example"})
    assert status == 403
    _work_empty(server)


@pytest.mark.parametrize("header", ["X-Hum-Token", "Origin", "Sec-Fetch-Site"])
def test_duplicated_security_headers_are_not_ambiguous(server, header):
    host = urlsplit(server.origin).netloc
    security = {"X-Hum-Token": server.session_token, "Origin": server.origin, "Sec-Fetch-Site": "same-origin"}
    fields = [("Host", host), ("Content-Length", "0"), ("X-Hum-Token", server.session_token)]
    if header == "X-Hum-Token":
        fields.append((header, security[header]))
    else:
        fields.extend([(header, security[header]), (header, security[header])])
    status, _, _ = _request(server, "POST", f"/api/example?{PARAMETERS}", raw_headers=fields)
    assert status == 403
    _work_empty(server)


def test_internal_engine_failure_removes_job_files_hides_exception_and_releases_lock(server, monkeypatch):
    def broken_engine(source, destination, config):
        destination.mkdir()
        (destination / "partial.wav").write_bytes(b"not a successful output")
        raise RuntimeError(f"private path {source}; secret {server.session_token}")

    monkeypatch.setattr(local_app, "process_wav", broken_engine)
    status, headers, payload = _request(server, "POST", "/api/process?mode=fixed&hum_hz=50&attenuation_db=0", body=_wav(), headers=_post_headers(server))
    assert status == 500
    assert server.session_token.encode() not in payload
    assert str(server.work_root).encode() not in payload
    assert "no-store" in headers["Cache-Control"]
    assert not any(key.lower().startswith("access-control-allow") for key in headers)
    _json(payload)
    _work_empty(server)
    assert server.processing_lock.acquire(blocking=False)
    server.processing_lock.release()
    # A failure must not permanently poison the shared local tool.
    status, _, data = _request(server, "POST", f"/api/process?{PARAMETERS}", body=_wav(), headers=_post_headers(server))
    assert status == 200
    assert _json(data)["mode"] == "guarded"
    _work_empty(server)


def test_nonzero_fixed_request_actually_changes_pcm_without_normalization(server):
    original = _wav(frames=4800)
    status, _, payload = _request(server, "POST", "/api/process?mode=fixed&hum_hz=50&attenuation_db=6", body=original, headers=_post_headers(server))
    assert status == 200
    result = _json(payload)
    processed = base64.b64decode(result["output_wav_base64"], validate=True)
    assert not np.array_equal(_codes(processed), _codes(original))
    assert result["report"]["config"]["max_attenuation_db"] == 6
    assert result["report"]["output_gain_normalization_applied"] is False
    _work_empty(server)
