"""Small rejected POST bodies must not reset the loopback error response."""

from contextlib import contextmanager
import http.client
import io
import json
import socket
import threading
import time
from email.message import Message
from urllib.parse import urlsplit

import pytest

from hum_dsp import local_app


@contextmanager
def _server(work_root):
    server = local_app.make_server(port=0, work_root=work_root)
    thread = threading.Thread(target=lambda: server.serve_forever(poll_interval=0.01), daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        assert not thread.is_alive()


@pytest.mark.parametrize("late_body_delay", [0.01, 0.05])
def test_rejected_small_body_arriving_after_headers_returns_complete_json(tmp_path, monkeypatch, late_body_delay):
    entered_rejection = threading.Event()
    original = local_app._LocalHandler._error

    def observed_rejection(self, status, message):
        entered_rejection.set()
        return original(self, status, message)

    def forbidden_job(*args, **kwargs):
        pytest.fail("an unauthorized discarded body must never be processed")

    monkeypatch.setattr(local_app._LocalHandler, "_error", observed_rejection)
    monkeypatch.setattr(local_app, "_run_job", forbidden_job)
    with _server(tmp_path / "work") as server:
        target = urlsplit(server.origin)
        payload = b"not an audio upload" * 40
        with socket.create_connection((target.hostname, target.port), timeout=2) as connection:
            connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            connection.sendall((
                "POST /api/process?mode=guarded&hum_hz=50&attenuation_db=0 HTTP/1.1\r\n"
                f"Host: {target.netloc}\r\n"
                "Content-Type: audio/wav\r\nX-Hum-Token: wrong-token\r\n"
                f"Content-Length: {len(payload)}\r\n\r\n"
            ).encode("ascii"))
            assert entered_rejection.wait(timeout=1), "server must reject before consuming an unauthorized body"
            time.sleep(late_body_delay)
            connection.sendall(payload)
            response = http.client.HTTPResponse(connection)
            response.begin()
            data = response.read()
        assert response.status == 403
        assert json.loads(data)["error"]
        assert response.getheader("Content-Type").startswith("application/json")
        assert response.getheader("Connection") == "close"
        assert "frame-ancestors 'none'" in response.getheader("Content-Security-Policy")
        assert server.session_token.encode() not in data
        assert not list(server.work_root.iterdir())
        assert server.processing_lock.acquire(blocking=False)
        server.processing_lock.release()


def test_rejected_small_body_with_no_remaining_bytes_has_bounded_wait(tmp_path):
    with _server(tmp_path / "work") as server:
        target = urlsplit(server.origin)
        with socket.create_connection((target.hostname, target.port), timeout=2) as connection:
            start = time.monotonic()
            connection.sendall((
                "POST /api/process HTTP/1.1\r\n"
                f"Host: {target.netloc}\r\n"
                "X-Hum-Token: wrong-token\r\nContent-Length: 10\r\n\r\n"
            ).encode("ascii"))
            response = http.client.HTTPResponse(connection)
            response.begin()
            assert response.status == 403
            assert json.loads(response.read())["error"]
            assert time.monotonic() - start < 1.0
        assert not list(server.work_root.iterdir())


class _Connection:
    def __init__(self):
        self.timeout = 10.0
        self.timeouts = []

    def gettimeout(self):
        return self.timeout

    def settimeout(self, value):
        self.timeouts.append(value)
        self.timeout = value


class _Body(io.BytesIO):
    def __init__(self, value):
        super().__init__(value)
        self.read_sizes = []

    def read1(self, count):
        self.read_sizes.append(count)
        return super().read1(count)


def _handler(fields, *, body=b"unread", consumed=0, command="POST"):
    handler = object.__new__(local_app._LocalHandler)
    handler.headers = Message()
    for name, value in fields:
        handler.headers[name] = value
    handler.command = command
    handler._body_bytes_consumed = consumed
    handler.connection = _Connection()
    handler.rfile = _Body(body)
    return handler


@pytest.mark.parametrize("fields", [
    [], [("Content-Length", "-1")], [("Content-Length", "1.5")],
    [("Content-Length", "abc")], [("Content-Length", "0"), ("Content-Length", "0")],
    [("Content-Length", "65537")], [("Content-Length", "999999999999999999999999999999")],
    [("Content-Length", "4"), ("Transfer-Encoding", "chunked")],
    [("Content-Length", "4"), ("Transfer-Encoding", "")],
])
def test_ambiguous_or_large_rejection_bodies_are_not_read(fields):
    handler = _handler(fields)
    handler._discard_small_rejected_body()
    assert handler.rfile.read_sizes == []
    assert handler._body_bytes_consumed == 0


@pytest.mark.parametrize("command,consumed", [("GET", 0), ("POST", 5)])
def test_non_post_and_already_consumed_bodies_are_never_read_again(command, consumed):
    handler = _handler([("Content-Length", "5")], consumed=consumed, command=command)
    handler._discard_small_rejected_body()
    assert handler.rfile.read_sizes == []
    assert handler._body_bytes_consumed == consumed


def test_partial_consumption_discards_only_declared_remaining_bytes():
    handler = _handler([("Content-Length", "5")], body=b"abcEXTRA", consumed=2)
    handler._discard_small_rejected_body()
    assert handler._body_bytes_consumed == 5
    assert handler.rfile.read_sizes == [3]
    assert handler.rfile.read() == b"EXTRA"
    assert handler.connection.timeout == 10.0


def test_maximum_discard_is_64kib_without_processing_or_trailing_read():
    handler = _handler([("Content-Length", "65536")], body=b"x" * 65536 + b"EXTRA")
    handler._discard_small_rejected_body()
    assert handler._body_bytes_consumed == 65536
    assert handler.rfile.read_sizes == [8192] * 8
    assert handler.rfile.read() == b"EXTRA"


def test_slow_discard_uses_one_total_deadline_not_a_new_deadline_per_chunk(monkeypatch):
    handler = _handler([("Content-Length", "10")], body=b"x" * 10)
    clock = iter((0.0, 0.09, 0.18, 0.27))
    monkeypatch.setattr(local_app, "monotonic", lambda: next(clock))
    original = handler.rfile.read1
    handler.rfile.read1 = lambda count: original(1)
    handler._discard_small_rejected_body()
    assert local_app._REJECTED_BODY_DRAIN_SECONDS <= 0.2
    assert handler._body_bytes_consumed == 2
    assert handler.connection.timeouts == pytest.approx([0.11, 0.02, 10.0])


def test_discard_transport_failure_does_not_replace_original_rejection():
    handler = _handler([("Content-Length", "5")])

    def disconnected(count):
        raise ConnectionResetError("peer closed")

    def restore_after_close(value):
        if value == 10.0:
            raise OSError("socket closed before restoring timeout")

    handler.rfile.read1 = disconnected
    handler.connection.settimeout = restore_after_close
    handler._discard_small_rejected_body()
    assert handler._body_bytes_consumed == 0
