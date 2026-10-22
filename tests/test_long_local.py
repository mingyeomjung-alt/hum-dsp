"""Long-file loopback transport contracts, without playback or real devices."""

from contextlib import contextmanager
import http.client
import io
import json
from pathlib import Path
import re
import socket
import threading
from types import SimpleNamespace
from urllib.parse import urlsplit
import wave
import zipfile

import numpy as np
import pytest

from hum_dsp import local_app, long_local, long_processing


PARAMETERS = "hum_hz=50&band_depths_db=0,0,0"
PROCESS_PATH = f"/api/long/process?{PARAMETERS}"


def _wav(*, rate=48000, channels=2, width=2, frames=480):
    stream = io.BytesIO()
    phase = np.arange(frames, dtype=np.float64) / rate
    samples = np.column_stack((
        0.08 * np.sin(2 * np.pi * 233 * phase),
        -0.05 * np.sin(2 * np.pi * 311 * phase + 0.3),
    ))[:, :channels]
    with wave.open(stream, "wb") as output:
        output.setnchannels(channels)
        output.setsampwidth(width)
        output.setframerate(rate)
        output.writeframes(np.rint(samples * 32768).astype("<i2").tobytes()
                           if width == 2 else bytes(frames * channels * width))
    return stream.getvalue()


@contextmanager
def _running_server(work_root):
    server = local_app.make_server(port=0, work_root=work_root)
    thread = threading.Thread(target=lambda: server.serve_forever(poll_interval=0.01), daemon=True)
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
    with _running_server(tmp_path / "local work") as running:
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
            fields = {"Host": target.netloc}
            if headers:
                fields.update(headers)
            connection.request(method, path, body=body, headers=fields)
        response = connection.getresponse()
        return response.status, dict(response.getheaders()), response.read()
    finally:
        connection.close()


def _headers(server, **extra):
    return {"Content-Type": "audio/wav", "X-Hum-Token": server.session_token, **extra}


def _json(payload):
    result = json.loads(payload)
    json.dumps(result, allow_nan=False)
    return result


def _empty(server):
    assert not list(server.work_root.iterdir())
    assert server.long_jobs.job_id is None
    assert server.long_jobs.files == {}
    assert server.long_jobs.temporary is None


def _post(server, payload=None, **extra):
    return _request(server, "POST", PROCESS_PATH,
                    body=_wav() if payload is None else payload, headers=_headers(server, **extra))


def _published(server):
    status, _, payload = _post(server)
    assert status == 200
    return _json(payload)


@pytest.mark.parametrize("token", [None, "", "not-this-session"])
def test_long_post_requires_session_token_before_creating_work(server, token):
    headers = {"Content-Type": "audio/wav"}
    if token is not None:
        headers["X-Hum-Token"] = token
    status, _, payload = _request(server, "POST", PROCESS_PATH, body=_wav(), headers=headers)
    assert status == 403
    assert server.session_token.encode() not in payload
    _empty(server)


@pytest.mark.parametrize("fields", [
    {"Host": "evil.example"},
    {"Origin": "null"},
    {"Origin": "https://evil.example"},
    {"Sec-Fetch-Site": "cross-site"},
])
def test_long_post_token_does_not_bypass_local_header_checks(server, fields):
    status, _, payload = _post(server, **fields)
    assert status == 403
    _json(payload)
    _empty(server)


@pytest.mark.parametrize("name", ["X-Hum-Token", "Origin", "Host", "Sec-Fetch-Site", "Content-Type"])
def test_long_post_ambiguous_single_headers_are_refused(server, name):
    values = {"Host": server.expected_host, "Origin": server.origin,
              "X-Hum-Token": server.session_token, "Sec-Fetch-Site": "same-origin",
              "Content-Type": "audio/wav"}
    payload = _wav()
    headers = [("Host", server.expected_host), ("X-Hum-Token", server.session_token),
               ("Content-Type", "audio/wav"), ("Content-Length", str(len(payload)))]
    if name not in {field for field, _ in headers}:
        headers.append((name, values[name]))
    headers.append((name, values[name]))
    status, _, _ = _request(server, "POST", PROCESS_PATH, body=payload, raw_headers=headers)
    assert status == 403
    _empty(server)


@pytest.mark.parametrize("query", [
    "", "hum_hz=50", "band_depths_db=0,0,0", "hum_hz=55&band_depths_db=0,0,0",
    "hum_hz=50&band_depths_db=0,0", "hum_hz=50&band_depths_db=0,0,0,0",
    "hum_hz=50&band_depths_db=0,,0", "hum_hz=50&band_depths_db=0,%200,0",
    "hum_hz=50&band_depths_db=0,nan,0", "hum_hz=50&band_depths_db=0,inf,0",
    "hum_hz=50&band_depths_db=0,-1,0", "hum_hz=50&band_depths_db=0,12.01,0",
    "hum_hz=50&band_depths_db=0,0,0&mode=long",
    "hum_hz=50&hum_hz=60&band_depths_db=0,0,0",
    "hum_hz=50&band_depths_db=0,0,0&band_depths_db=0,0,0",
    "hum_hz=50&band_depths_db=0,0,0&unexpected=1",
])
def test_long_settings_are_unambiguous_and_checked_before_work(server, query):
    status, _, payload = _request(server, "POST", f"/api/long/process?{query}",
                                 body=_wav(), headers=_headers(server))
    assert status == 400
    _json(payload)
    _empty(server)


@pytest.mark.parametrize("fields,expected", [
    ([], 400),
    ([("Content-Length", "-1")], 400),
    ([("Content-Length", "1.5")], 400),
    ([("Content-Length", "abc")], 400),
    ([("Content-Length", "0"), ("Content-Length", "0")], 400),
    ([("Transfer-Encoding", "chunked")], 400),
    ([("Content-Length", "1"), ("Transfer-Encoding", "chunked")], 400),
    ([("Content-Length", str(long_local.MAX_UPLOAD_BYTES + 1))], 413),
    ([("Content-Length", "9" * 200)], 413),
    ([("Content-Length", "1"), ("Content-Encoding", "gzip")], 415),
])
def test_long_fixed_body_length_and_size_are_checked_before_reading(server, fields, expected):
    raw = [("Host", server.expected_host), ("X-Hum-Token", server.session_token),
           ("Content-Type", "audio/wav"), *fields]
    status, _, payload = _request(server, "POST", PROCESS_PATH, raw_headers=raw)
    assert status == expected
    _json(payload)
    _empty(server)


@pytest.mark.parametrize("kind", ["application/json", "multipart/form-data", "text/plain"])
def test_long_upload_refuses_non_raw_wav_content_types(server, kind):
    status, _, _ = _post(server, **{"Content-Type": kind})
    assert status == 400
    _empty(server)


def test_busy_long_request_does_not_replace_current_result(server):
    published = _published(server)
    old_id = server.long_jobs.job_id
    assert server.processing_lock.acquire(blocking=False)
    try:
        status, _, payload = _post(server)
    finally:
        server.processing_lock.release()
    assert status == 503
    _json(payload)
    assert server.long_jobs.job_id == old_id
    assert _request(server, path=published["output_wav_url"])[0] == 200


def test_short_zero_depth_long_request_returns_capability_urls_not_base64(server):
    source = _wav()
    status, headers, payload = _post(server, source, Origin=server.origin,
                                   **{"Content-Type": "application/octet-stream"})
    assert status == 200
    assert "application/json" in headers["Content-Type"]
    assert "no-store" in headers["Cache-Control"]
    result = _json(payload)
    assert result["schema_version"] == 1
    assert result["mode"] == "long"
    assert result["source_kind"] == "user_file"
    assert result["duration_seconds"] == 0.01
    assert not any("base64" in key for key in result)
    assert server.session_token.encode() not in payload
    assert str(server.work_root).encode() not in payload
    for key, filename in (("input_wav_url", "input.wav"), ("output_wav_url", "processed.wav"),
                          ("download_zip_url", "result.zip"), ("report_json_url", "report.json")):
        assert re.fullmatch(r"/api/long/assets/[a-f0-9]{48}/" + re.escape(filename), result[key])
    status, _, original = _request(server, path=result["input_wav_url"])
    assert status == 200 and original == source
    status, _, processed = _request(server, path=result["output_wav_url"])
    assert status == 200
    with wave.open(io.BytesIO(source), "rb") as first, wave.open(io.BytesIO(processed), "rb") as second:
        assert first.getparams() == second.getparams()
        assert first.readframes(first.getnframes()) == second.readframes(second.getnframes())
    report = result["report"]
    assert report["config"]["band_depths_db"] == [0, 0, 0]
    assert report["processing_bypassed"] is True
    assert report["output_gain_normalization_applied"] is False
    assert report["local_tool"]["temporary_files_removed_before_response"] is False
    assert report["local_tool"]["temporary_result_retained_until_next_long_job_or_server_close"] is True
    for flag in ("microphone_capture", "autoplay", "remote_upload"):
        assert report["local_tool"][flag] is False
    for kind in ("input", "output"):
        bins = np.asarray(result["waveforms"][kind])
        assert 1 <= len(bins) <= 480
        assert bins.shape == (len(bins), 2)
        assert np.isfinite(bins).all()
        assert np.max(np.abs(bins)) < 0.081
    np.testing.assert_array_equal(result["waveforms"]["input"], result["waveforms"]["output"])
    assert len(list(server.work_root.iterdir())) == 1


def test_long_report_and_download_zip_match_the_actual_served_files(server):
    result = _published(server)
    status, report_headers, report_bytes = _request(server, path=result["report_json_url"])
    assert status == 200 and _json(report_bytes) == result["report"]
    assert "application/json" in report_headers["Content-Type"]
    status, zip_headers, archive_bytes = _request(server, path=result["download_zip_url"])
    assert status == 200
    assert zip_headers["Content-Type"] == "application/zip"
    assert "attachment" in zip_headers["Content-Disposition"]
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        assert set(archive.namelist()) == {"processed.wav", "report.json", "long_trace.jsonl"}
        assert archive.read("report.json") == report_bytes
        assert archive.read("processed.wav") == _request(server, path=result["output_wav_url"])[2]
        assert archive.read("long_trace.jsonl") == b""
        assert archive.testzip() is None


@pytest.mark.parametrize("method", ["GET", "HEAD"])
def test_long_asset_get_head_security_and_exact_lengths(server, method):
    result = _published(server)
    source = _request(server, path=result["input_wav_url"])[2]
    status, headers, payload = _request(server, method, result["input_wav_url"])
    assert status == 200
    assert int(headers["Content-Length"]) == len(source)
    assert payload == (b"" if method == "HEAD" else source)
    assert headers["Content-Type"] == "audio/wav"
    assert headers["Accept-Ranges"] == "bytes"
    assert headers["Cache-Control"] == "no-store"
    assert headers["X-Content-Type-Options"] == "nosniff"
    assert headers["Referrer-Policy"] == "no-referrer"
    assert headers["X-Frame-Options"] == "DENY"
    assert "frame-ancestors 'none'" in headers["Content-Security-Policy"]
    assert not any(key.lower().startswith("access-control-allow") for key in headers)


@pytest.mark.parametrize("range_value,expected_start,expected_end", [
    ("bytes=0-9", 0, 9), ("bytes=10-", 10, None), ("bytes=-17", -17, None),
    ("bytes=0-999999", 0, None), ("bytes=-999999", 0, None),
])
@pytest.mark.parametrize("method", ["GET", "HEAD"])
def test_long_asset_supports_single_ranges_without_loading_the_full_file(server, range_value, expected_start, expected_end, method):
    result = _published(server)
    full = _request(server, path=result["input_wav_url"])[2]
    start = expected_start if expected_start >= 0 else len(full) + expected_start
    end = len(full) - 1 if expected_end is None else expected_end
    status, headers, payload = _request(server, method, result["input_wav_url"], headers={"Range": range_value})
    assert status == 206
    assert headers["Content-Range"] == f"bytes {start}-{end}/{len(full)}"
    assert int(headers["Content-Length"]) == end - start + 1
    assert payload == (b"" if method == "HEAD" else full[start:end + 1])


@pytest.mark.parametrize("range_value", [
    "bytes=999999-", "bytes=10-9", "bytes=-0", "bytes=-", "bytes=0-1,4-5",
    "items=0-1", "bytes=+1-2", "bytes=0-" + "9" * 21,
])
def test_long_unsatisfied_or_ambiguous_ranges_return_416(server, range_value):
    result = _published(server)
    size = len(_request(server, path=result["input_wav_url"])[2])
    status, headers, payload = _request(server, path=result["input_wav_url"], headers={"Range": range_value})
    assert status == 416
    assert headers["Content-Range"] == f"bytes */{size}"
    assert headers["Content-Length"] == "0"
    assert "no-store" in headers["Cache-Control"]
    assert "frame-ancestors 'none'" in headers["Content-Security-Policy"]
    assert payload == b""


def test_duplicate_range_header_is_not_silently_resolved(server):
    result = _published(server)
    fields = [("Host", server.expected_host), ("Range", "bytes=0-1"), ("Range", "bytes=2-3")]
    status, _, _ = _request(server, path=result["input_wav_url"], raw_headers=fields)
    assert status == 403


@pytest.mark.parametrize("change", ["foreign_host", "foreign_origin", "cross_site"])
def test_asset_capability_does_not_bypass_local_origin_checks(server, change):
    result = _published(server)
    fields = {"foreign_host": {"Host": "evil.example"},
              "foreign_origin": {"Origin": "https://evil.example"},
              "cross_site": {"Sec-Fetch-Site": "cross-site"}}[change]
    status, _, _ = _request(server, path=result["input_wav_url"], headers=fields)
    assert status == 403


@pytest.mark.parametrize("suffix", [
    "/missing.wav", "/../input.wav", "/%2e%2e/input.wav", "/input%2ewav",
    "/input.wav/extra", "/input.wav?token=not-needed", "/report.json?x=1",
])
def test_long_asset_route_never_resolves_arbitrary_or_query_paths(server, suffix):
    result = _published(server)
    prefix = result["input_wav_url"].rsplit("/", 1)[0]
    status, _, _ = _request(server, path=prefix + suffix)
    assert status == 404


@pytest.mark.parametrize("path", [
    "/api/long/assets/not-a-job/input.wav", "/api/long/assets/" + "A" * 48 + "/input.wav",
    "/api/long/missing", "/api/long/process/", "/api/long/assets/" + "a" * 48 + "/report.json",
])
def test_unknown_long_post_routes_do_not_fall_back_to_processing(server, path):
    status, _, _ = _request(server, "POST", path + "?" + PARAMETERS, body=_wav(), headers=_headers(server))
    assert status == 404
    _empty(server)


def test_second_successful_long_job_expires_old_capabilities_and_keeps_only_one_folder(server):
    first = _published(server)
    old_folder = Path(server.long_jobs.temporary.name)
    second = _published(server)
    assert first["input_wav_url"] != second["input_wav_url"]
    assert not old_folder.exists()
    assert len(list(server.work_root.iterdir())) == 1
    for key in ("input_wav_url", "output_wav_url", "download_zip_url", "report_json_url"):
        assert _request(server, path=first[key])[0] == 404
        assert _request(server, path=second[key])[0] == 200


def test_rejected_settings_leave_the_existing_completed_job_available(server):
    result = _published(server)
    status, _, _ = _request(server, "POST", "/api/long/process?hum_hz=55&band_depths_db=0,0,0",
                           body=_wav(), headers=_headers(server))
    assert status == 400
    assert _request(server, path=result["output_wav_url"])[0] == 200


@pytest.mark.parametrize("payload", [b"", b"not a WAV", b"RIFF\x24\x00\x00\x00WAVE",
                                     _wav(rate=32000), _wav(width=1)],
                         ids=["empty", "not_wave", "truncated", "unsupported_32000_hz", "pcm8"])
def test_bad_long_wav_cleans_work_and_the_next_request_recovers(server, payload):
    status, _, response = _post(server, payload)
    assert status == 400
    _json(response)
    _empty(server)
    assert server.processing_lock.acquire(blocking=False)
    server.processing_lock.release()
    assert _post(server)[0] == 200


def test_failed_authorized_replacement_expires_old_job_and_cleans_partial_upload(server):
    result = _published(server)
    status, _, _ = _post(server, b"malformed WAV")
    assert status == 400
    _empty(server)
    assert _request(server, path=result["input_wav_url"])[0] == 404


def test_incomplete_wire_upload_cleans_its_partial_file_and_recovers(server):
    target = urlsplit(server.origin)
    connection = http.client.HTTPConnection(target.hostname, target.port, timeout=10)
    try:
        connection.putrequest("POST", PROCESS_PATH)
        connection.putheader("X-Hum-Token", server.session_token)
        connection.putheader("Content-Type", "audio/wav")
        connection.putheader("Content-Length", "100")
        connection.endheaders(b"only three words")
        connection.sock.shutdown(socket.SHUT_WR)
        response = connection.getresponse()
        assert response.status == 400
        _json(response.read())
    finally:
        connection.close()
    _empty(server)
    assert _post(server)[0] == 200


def test_internal_long_processor_failure_does_not_expose_details_or_poison_lock(server, monkeypatch):
    original = long_processing.process_long_wav

    def broken(source, destination, config):
        destination.mkdir()
        (destination / "partial.wav").write_bytes(b"partial")
        raise RuntimeError(f"private path {source}; token {server.session_token}")

    monkeypatch.setattr(long_processing, "process_long_wav", broken)
    status, _, payload = _post(server)
    assert status == 500
    assert server.session_token.encode() not in payload
    assert str(server.work_root).encode() not in payload
    _json(payload)
    _empty(server)
    monkeypatch.setattr(long_processing, "process_long_wav", original)
    assert _post(server)[0] == 200


def test_long_example_requires_empty_body_and_uses_the_owned_generator(server, monkeypatch):
    status, _, _ = _request(server, "POST", "/api/long/example?" + PARAMETERS,
                           body=b"not allowed", headers=_headers(server))
    assert status == 400
    _empty(server)
    calls = []

    def small_owned_example(path, config):
        calls.append((path, config.nominal_hum_hz, config.band_depths_db))
        path.write_bytes(_wav(frames=960))

    monkeypatch.setattr(long_local, "write_owned_long_example", small_owned_example)
    status, _, payload = _request(server, "POST", "/api/long/example?" + PARAMETERS,
                                 body=b"", headers=_headers(server))
    assert status == 200
    result = _json(payload)
    assert result["source_kind"] == "owned_synthetic_example"
    assert result["duration_seconds"] == 0.02
    assert len(calls) == 1
    assert calls[0][1:] == (50, (0, 0, 0))


def test_server_close_removes_the_retained_long_job(tmp_path):
    work = tmp_path / "close work"
    with _running_server(work) as running:
        _published(running)
        folder = Path(running.long_jobs.temporary.name)
        assert folder.exists()
    assert not folder.exists()
    assert not list(work.iterdir())
    assert running.long_jobs.job_id is None


class _Connection:
    def __init__(self):
        self.timeouts = []

    def settimeout(self, value):
        self.timeouts.append(value)


class _BoundedReader:
    def __init__(self, payload, *, maximum=long_local.COPY_BYTES, fail=None):
        self.source = io.BytesIO(payload)
        self.maximum = maximum
        self.sizes = []
        self.fail = fail

    def read1(self, size):
        assert 0 < size <= self.maximum
        self.sizes.append(size)
        if self.fail is not None:
            raise self.fail
        return self.source.read(size)


def test_upload_copies_only_bounded_chunks_and_never_reads_beyond_declared_length(tmp_path):
    payload = b"a" * (long_local.COPY_BYTES * 2 + 13)
    reader = _BoundedReader(payload + b"not part of this body")
    connection = _Connection()
    handler = SimpleNamespace(rfile=reader, connection=connection, _body_bytes_consumed=0)
    destination = tmp_path / "upload.wav"
    long_local._copy_upload(handler, destination, len(payload))
    assert destination.read_bytes() == payload
    assert reader.sizes == [long_local.COPY_BYTES, long_local.COPY_BYTES, 13]
    assert handler._body_bytes_consumed == len(payload)
    assert reader.source.read() == b"not part of this body"
    assert connection.timeouts[-1] == 10
    assert all(0 < value <= 10 for value in connection.timeouts)


@pytest.mark.parametrize("failure", [None, OSError("read failed"), TimeoutError("slow peer")])
def test_upload_eof_io_and_timeout_are_bounded_request_failures(tmp_path, failure):
    reader = _BoundedReader(b"abc", fail=failure)
    connection = _Connection()
    handler = SimpleNamespace(rfile=reader, connection=connection, _body_bytes_consumed=0)
    with pytest.raises(local_app._RequestFailure) as error:
        long_local._copy_upload(handler, tmp_path / "partial.wav", 10)
    assert error.value.status == 400
    assert connection.timeouts[-1] == 10
    assert handler._body_bytes_consumed == (3 if failure is None else 0)


def test_upload_total_deadline_is_not_reset_per_chunk(tmp_path, monkeypatch):
    times = iter([0.0, long_local.UPLOAD_SECONDS + 1])
    monkeypatch.setattr(long_local, "monotonic", lambda: next(times))
    reader = _BoundedReader(b"abc")
    connection = _Connection()
    handler = SimpleNamespace(rfile=reader, connection=connection, _body_bytes_consumed=0)
    with pytest.raises(local_app._RequestFailure) as error:
        long_local._copy_upload(handler, tmp_path / "expired.wav", 3)
    assert error.value.status == 400
    assert reader.sizes == []
    assert handler._body_bytes_consumed == 0
    assert connection.timeouts == [10]


def test_store_does_not_publish_a_foreign_or_incomplete_directory(tmp_path):
    work = tmp_path / "store"
    work.mkdir()
    store = long_local.LongJobStore(work)
    try:
        folder = store.begin()
        with pytest.raises(ValueError, match="does not belong"):
            store.publish(tmp_path)
        with pytest.raises(ValueError, match="incomplete"):
            store.publish(folder)
        with pytest.raises(FileNotFoundError):
            with store.open_asset("a" * 48, "input.wav"):
                pytest.fail("unpublished asset must not open")
    finally:
        store.clear()
    assert not list(work.iterdir())


@pytest.mark.parametrize("value,expected", [
    (None, (0, 99, False)), ("bytes=0-9", (0, 9, True)),
    ("bytes=10-", (10, 99, True)), ("bytes=-17", (83, 99, True)),
    ("bytes=-200", (0, 99, True)), ("bytes=90-200", (90, 99, True)),
    ("bytes=0-0", (0, 0, True)),
])
def test_byte_range_parser_matches_independent_expected_boundaries(value, expected):
    assert long_local._byte_range(value, 100) == expected
