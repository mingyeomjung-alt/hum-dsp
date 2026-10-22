"""Loopback capability and graceful-stop policy, without launching a browser."""

from contextlib import contextmanager
import http.client
import json
import threading
from urllib.parse import urlsplit

import pytest

from hum_dsp import local_app


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
    with _running(tmp_path / "work") as instance:
        yield instance


def _request(server, path, *, body=None, headers=None, method=None):
    parsed = urlsplit(server.origin)
    connection = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=10)
    values = {"X-Hum-Token": server.session_token, "Content-Type": "application/json"}
    values.update(headers or {})
    try:
        connection.request(method or ("POST" if body is not None else "GET"), path, body=body, headers=values)
        result = connection.getresponse()
        return result.status, dict(result.getheaders()), result.read()
    finally:
        connection.close()


def test_capabilities_disclose_normal_python_runtime_without_paths_or_tokens(server):
    status, headers, payload = _request(server, "/api/capabilities")
    assert status == 200
    result = json.loads(payload)
    assert result["version"] == "0.0.15"
    assert result["desktop"] == {"available": False, "portable_wav_only": False}
    assert result["compressed"]["available"] is True
    assert result["persistent"] == {
        "available": True, "explicit_opt_in": True, "maximum_seconds": 60,
        "maximum_container_bytes": 128 * 1024 * 1024,
        "whole_file_memory_processing": True, "quality_approval": False,
    }
    assert result["remote_upload"] is False and result["device_capture_or_autoplay"] is False
    assert server.session_token.encode() not in payload
    assert str(server.work_root).encode() not in payload
    assert headers["Cache-Control"] == "no-store"


def test_portable_backend_is_WAV_only_even_if_PyAV_is_installed(server):
    server.portable_wav_only = True
    server.desktop_shutdown_event = threading.Event()
    status, _, payload = _request(server, "/api/capabilities")
    assert status == 200
    result = json.loads(payload)
    assert result["desktop"] == {"available": True, "portable_wav_only": True}
    assert result["compressed"]["available"] is False
    assert result["compressed"]["reason"] == "portable_wav_only"
    status, _, payload = _request(server, "/api/compressed/example?hum_hz=50&band_depths_db=0,0,0", body=b"")
    assert status == 415 and "WAV" in json.loads(payload)["error"]
    assert not list(server.work_root.iterdir())


@pytest.mark.parametrize("path,headers,expected", [
    ("/api/capabilities?extra=1", {}, 400),
    ("/api/capabilities", {"Origin": "https://example.com"}, 403),
    ("/api/capabilities", {"Host": "example.com"}, 403),
])
def test_capability_read_preserves_existing_origin_and_host_rules(server, path, headers, expected):
    assert _request(server, path, headers=headers)[0] == expected


def test_stop_not_added_to_ordinary_python_server(server):
    assert _request(server, "/api/desktop/stop", body=b"")[0] == 404


@pytest.mark.parametrize("path,body,headers,expected", [
    ("/api/desktop/stop?x=1", b"", {}, 400),
    ("/api/desktop/stop", b"", {"X-Hum-Token": "wrong"}, 403),
    ("/api/desktop/stop", b"", {"Origin": "https://example.com"}, 403),
    ("/api/desktop/stop", b"{}", {}, 400),
    ("/api/desktop/stop", b"", {"Content-Type": "text/plain"}, 400),
    ("/api/desktop/stop", b"", {"Content-Encoding": "gzip"}, 400),
    ("/api/desktop/stop", b"", {"Transfer-Encoding": "chunked"}, 400),
])
def test_stop_is_authenticated_empty_same_origin_and_never_sets_event_on_rejection(server, path, body, headers, expected):
    server.desktop_shutdown_event = threading.Event()
    assert _request(server, path, body=body, headers=headers)[0] == expected
    assert not server.desktop_shutdown_event.is_set()


def test_processing_blocks_stop_without_releasing_someone_elses_lock(server):
    server.desktop_shutdown_event = threading.Event()
    server.processing_lock.acquire()
    try:
        assert _request(server, "/api/desktop/stop", body=b"")[0] == 503
        assert not server.desktop_shutdown_event.is_set()
        assert server.processing_lock.locked()
    finally:
        server.processing_lock.release()


def test_stop_response_says_cleanup_pending_and_new_jobs_fail_closed(server):
    server.desktop_shutdown_event = threading.Event()
    status, _, payload = _request(server, "/api/desktop/stop", body=b"")
    assert status == 200
    assert json.loads(payload) == {"status": "stopping", "temporary_cleanup_pending": True}
    assert server.desktop_shutdown_event.is_set()
    for path in ("/api/example?mode=fixed&hum_hz=50&attenuation_db=0",
                 "/api/precision/example?hum_hz=50&band_depths_db=0,0,0",
                 "/api/compressed/example?hum_hz=50&band_depths_db=0,0,0"):
        assert _request(server, path, body=b"")[0] == 503
    assert not list(server.work_root.iterdir())
    assert server.processing_lock.acquire(blocking=False)
    server.processing_lock.release()
