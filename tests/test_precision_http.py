"""Native-precision browser transport, independently encoded and decoded WAVs.

These tests use local HTTP only. They do not play audio, capture a microphone,
publish files or establish sound-quality improvement.
"""

from contextlib import contextmanager
import http.client
import io
import json
import re
import struct
import threading
from urllib.parse import urlsplit
import zipfile

import numpy as np
import pytest

from hum_dsp import local_app


FORMATS = {"pcm16": (1, 16, 2, "signed_PCM16_little_endian"),
           "pcm24": (1, 24, 3, "signed_PCM24_little_endian"),
           "float32": (3, 32, 4, "IEEE_FLOAT32_little_endian")}
PARAMETERS = "hum_hz=50&band_depths_db=0,0,0"


def _chunk(name, payload):
    return name + struct.pack("<I", len(payload)) + payload + b"\0" * (len(payload) % 2)


def _wav(*, kind="pcm24", rate=48000, channels=2, frames=480, values=None, fact=True):
    tag, bits, width, _ = FORMATS[kind]
    if values is None:
        t = np.arange(frames, dtype=np.float64) / rate
        values = np.column_stack([
            (0.015 + index * 0.002) * np.sin(2 * np.pi * 50 * t)
            + (0.035 + index * 0.012) * np.sin(2 * np.pi * (233 + 78 * index) * t)
            for index in range(channels)
        ])
    else:
        values = np.asarray(values).reshape(-1, channels)
        frames = len(values)
    if kind == "float32":
        payload = values.astype("<f4").tobytes()
    elif kind == "pcm16":
        payload = np.rint(values * 32768).astype("<i2").tobytes()
    else:
        codes = np.rint(values * 8388608).astype(np.int32).reshape(-1)
        payload = np.column_stack((codes & 255, (codes >> 8) & 255, (codes >> 16) & 255)).astype(np.uint8).tobytes()
    fmt = struct.pack("<HHIIHH", tag, channels, rate, rate * channels * width, channels * width, bits)
    chunks = _chunk(b"fmt ", fmt)
    if kind == "float32" and fact:
        chunks += _chunk(b"fact", struct.pack("<I", frames))
    chunks += _chunk(b"data", payload)
    return b"RIFF" + struct.pack("<I", 4 + len(chunks)) + b"WAVE" + chunks


def _read(payload):
    """Independent small-fixture RIFF parser; no product encoder or reader."""
    assert payload[:4] == b"RIFF" and payload[8:12] == b"WAVE"
    assert len(payload) == struct.unpack_from("<I", payload, 4)[0] + 8
    position, fmt, data = 12, None, None
    while position < len(payload):
        tag, size = struct.unpack_from("<4sI", payload, position)
        start = position + 8
        if tag == b"fmt ":
            fmt = struct.unpack_from("<HHIIHH", payload, start)
        elif tag == b"data":
            data = payload[start:start + size]
        position = start + size + size % 2
    assert fmt is not None and data is not None
    tag, channels, rate, byte_rate, alignment, bits = fmt
    assert alignment == channels * (bits // 8) and byte_rate == rate * alignment
    assert len(data) % alignment == 0
    if tag == 3:
        values = np.frombuffer(data, dtype="<f4").astype(np.float64)
    elif bits == 24:
        triplets = np.frombuffer(data, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        codes = triplets[:, 0] + (triplets[:, 1] << 8) + (triplets[:, 2] << 16)
        values = ((codes ^ 0x800000) - 0x800000).astype(np.float64) / 8388608
    else:
        values = np.frombuffer(data, dtype="<i2").astype(np.float64) / 32768
    return (tag, bits, rate, channels, len(data) // alignment), data, values.reshape(-1, channels)


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
    with _running_server(tmp_path / "local precision work") as instance:
        yield instance


def _request(server, path, *, body=None, method=None, headers=None):
    target = urlsplit(server.origin)
    connection = http.client.HTTPConnection(target.hostname, target.port, timeout=60)
    fields = {"Host": target.netloc}
    if body is not None:
        fields.update({"Content-Type": "audio/wav", "X-Hum-Token": server.session_token})
    fields.update(headers or {})
    try:
        connection.request(method or ("POST" if body is not None else "GET"), path,
                           body=body, headers=fields)
        response = connection.getresponse()
        return response.status, dict(response.getheaders()), response.read()
    finally:
        connection.close()


def _post(server, source, depths="0,0,0", mode="precision"):
    return _request(server, f"/api/{mode}/process?hum_hz=50&band_depths_db={depths}", body=source)


def _empty(server):
    assert not list(server.work_root.iterdir())
    assert server.long_jobs.job_id is None
    assert server.long_jobs.temporary is None


@pytest.mark.parametrize("kind", ["pcm24", "float32"])
@pytest.mark.parametrize("rate", [44100, 48000])
@pytest.mark.parametrize("channels", [1, 2])
@pytest.mark.parametrize("depths", ["0,0,0", "6,3,0"], ids=["bypass", "active"])
def test_precision_real_post_keeps_native_format_and_serves_exact_report_zip(server, kind, rate, channels, depths):
    frames = rate * 3 if depths != "0,0,0" else 481
    source = _wav(kind=kind, rate=rate, channels=channels, frames=frames)
    status, headers, payload = _post(server, source, depths)
    assert status == 200, payload.decode(errors="replace")
    assert headers["Content-Type"].startswith("application/json")
    assert headers["Cache-Control"] == "no-store"
    result = json.loads(payload)
    json.dumps(result, allow_nan=False)
    assert result["mode"] == "precision" and result["source_kind"] == "user_file"
    assert result["duration_seconds"] == pytest.approx(frames / rate)
    assert not any("base64" in key for key in result)
    assert len(payload) < 512 * 1024
    assert server.session_token.encode() not in payload
    assert str(server.work_root).encode() not in payload
    for key, name in (("input_wav_url", "input.wav"), ("output_wav_url", "processed.wav"),
                      ("report_json_url", "report.json"), ("download_zip_url", "result.zip")):
        assert re.fullmatch(r"/api/long/assets/[a-f0-9]{48}/" + re.escape(name), result[key])
    assert _request(server, result["input_wav_url"])[2] == source
    status, _, processed = _request(server, result["output_wav_url"])
    assert status == 200
    original_meta, original_data, _ = _read(source)
    output_meta, output_data, output_values = _read(processed)
    assert output_meta == original_meta == (FORMATS[kind][0], FORMATS[kind][1], rate, channels, frames)
    bypass = depths == "0,0,0"
    assert (original_data == output_data) is bypass
    assert np.isfinite(output_values).all()
    assert np.min(output_values) >= -1 and np.max(output_values) < 1
    report = result["report"]
    for side in ("input", "output"):
        assert report[side]["sample_format"] == FORMATS[kind][3]
        assert report[side]["bits_per_sample"] == FORMATS[kind][1]
        assert report[side]["sample_rate_hz"] == rate
        assert report[side]["channels"] == channels
        assert report[side]["frames"] == frames
    assert report["processing_bypassed"] is bypass
    assert report["processor_applied"] is (not bypass)
    assert report["precision_io"]["PCM16_intermediate_used"] is False
    assert report["precision_io"]["dither_applied"] is False
    assert report["precision_io"]["raw_payload_identity"] is bypass
    assert report["output_gain_normalization_applied"] is False
    assert report["device_io"] is False and report["reference_quality_metrics"] is None
    assert report["local_tool"]["mode"] == "precision"
    for field in ("autoplay", "remote_upload", "microphone_capture"):
        assert report["local_tool"][field] is False
    assert report["local_tool"]["temporary_files_removed_before_response"] is False
    assert report["stereo_policy"]["same_trace_and_coefficients_all_channels"] is True
    assert report["long_file"]["sample_rate_conversion_applied"] is (rate == 44100 and not bypass)
    if rate == 44100 and not bypass:
        assert report["conversion"]["to_engine"]["source_rate_hz"] == 44100
        assert report["conversion"]["to_engine"]["target_rate_hz"] == 48000
    else:
        assert report["conversion"]["to_engine"]["conversion_applied"] is False
    status, _, report_bytes = _request(server, result["report_json_url"])
    assert status == 200 and json.loads(report_bytes) == report
    status, download_headers, archive_bytes = _request(server, result["download_zip_url"])
    assert status == 200 and download_headers["Content-Type"] == "application/zip"
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        assert set(archive.namelist()) == {"processed.wav", "report.json", "long_trace.jsonl"}
        assert archive.read("processed.wav") == processed
        assert archive.read("report.json") == report_bytes
        assert bool(archive.read("long_trace.jsonl")) is (not bypass)
        assert archive.testzip() is None
    for side in ("input", "output"):
        bins = np.asarray(result["waveforms"][side])
        assert bins.shape == (480, 2) and np.isfinite(bins).all()
    if bypass:
        np.testing.assert_array_equal(result["waveforms"]["input"], result["waveforms"]["output"])


def test_float_bypass_preserves_negative_zero_subnormal_and_payload_without_fact(server):
    values = np.array([0.0, -0.0, np.nextafter(np.float32(0), np.float32(1)), -0.25, 0.5], dtype="<f4")
    source = _wav(kind="float32", channels=1, values=values, fact=False)
    status, _, payload = _post(server, source)
    assert status == 200, payload.decode(errors="replace")
    result = json.loads(payload)
    processed = _request(server, result["output_wav_url"])[2]
    assert _read(source)[1] == _read(processed)[1] == values.tobytes()


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, 1.0, 1.25, -1.25])
@pytest.mark.parametrize("depths", ["0,0,0", "6,3,0"])
def test_float_invalid_normalized_values_are_refused_even_on_bypass_and_recovers(server, value, depths):
    source = _wav(kind="float32", channels=1, values=[0.1, value, -0.1])
    status, _, payload = _post(server, source, depths)
    assert status == 400
    assert isinstance(json.loads(payload)["error"], str)
    assert server.session_token.encode() not in payload
    _empty(server)
    assert _post(server, _wav(kind="float32", channels=1))[0] == 200


@pytest.mark.parametrize("token", ["", "wrong-token"])
def test_precision_post_checks_session_before_work(server, token):
    status, _, _ = _request(server, "/api/precision/process?" + PARAMETERS,
                           body=_wav(), headers={"X-Hum-Token": token})
    assert status == 403
    _empty(server)


@pytest.mark.parametrize("query", [PARAMETERS + "&mode=precision", "hum_hz=50&band_depths_db=0,0",
                                    "hum_hz=55&band_depths_db=0,0,0"])
def test_precision_post_does_not_silently_ignore_extra_or_bad_settings(server, query):
    status, _, _ = _request(server, "/api/precision/process?" + query, body=_wav())
    assert status == 400
    _empty(server)


def test_precision_native_assets_have_head_range_and_security_headers(server):
    status, _, payload = _post(server, _wav(kind="float32"))
    assert status == 200
    result = json.loads(payload)
    full = _request(server, result["output_wav_url"])[2]
    status, headers, payload = _request(server, result["output_wav_url"], method="HEAD")
    assert status == 200 and not payload
    assert int(headers["Content-Length"]) == len(full)
    assert headers["Accept-Ranges"] == "bytes"
    status, headers, payload = _request(server, result["output_wav_url"], headers={"Range": "bytes=12-27"})
    assert status == 206 and payload == full[12:28]
    assert headers["Content-Range"] == f"bytes 12-27/{len(full)}"
    assert headers["Cache-Control"] == "no-store"
    assert headers["Referrer-Policy"] == "no-referrer"
    assert headers["X-Content-Type-Options"] == "nosniff"
    assert "frame-ancestors 'none'" in headers["Content-Security-Policy"]


def test_long_and_precision_jobs_replace_the_same_single_temporary_result(server):
    first = json.loads(_post(server, _wav(kind="pcm24"))[2])
    old_folder = server.long_jobs.temporary.name
    status, _, payload = _post(server, _wav(kind="pcm16"), mode="long")
    assert status == 200
    second = json.loads(payload)
    assert first["input_wav_url"] != second["input_wav_url"]
    assert _request(server, first["input_wav_url"])[0] == 404
    assert server.long_jobs.temporary.name != old_folder
    third = json.loads(_post(server, _wav(kind="float32"))[2])
    assert _request(server, second["input_wav_url"])[0] == 404
    assert _request(server, third["output_wav_url"])[0] == 200
    assert len(list(server.work_root.iterdir())) == 1


def test_precision_example_is_owned_six_second_stereo_pcm24_and_empty_body_only(server):
    path = "/api/precision/example?" + PARAMETERS
    status, _, _ = _request(server, path, body=b"not an example upload")
    assert status == 400
    _empty(server)
    status, _, payload = _request(server, path, body=b"")
    assert status == 200, payload.decode(errors="replace")
    result = json.loads(payload)
    assert result["source_kind"] == "owned_synthetic_example"
    assert result["duration_seconds"] == 6
    original = _request(server, result["input_wav_url"])[2]
    assert _read(original)[0] == (1, 24, 48000, 2, 288000)
    assert result["report"]["input"]["sample_format"] == "signed_PCM24_little_endian"


def test_precision_ui_is_explicit_native_format_mode_without_changing_existing_defaults(server):
    status, _, page = _request(server, "/")
    assert status == 200
    html = page.decode("utf-8")
    assert "v0.0.14" in html
    assert '<option value="fixed">고정 필터 · 일정한 잡음</option>' in html
    # Count processing modes, not the unrelated optional listening-rating selects.
    mode_select = html.split('<select id="mode"', 1)[1].split('</select>', 1)[0]
    assert mode_select.count('<option value="') == 10  # Nine existing modes plus opt-in profile.
    assert '<option value="precision">고해상도 WAV · 24비트·Float32</option>' in html
    assert "최대 10분과 128MiB 중 먼저" in html
    assert "유한한 −1 이상·+1 미만" in html
    assert "NaN·무한대·+1" in html
    assert "디더링·자동 음량 보정은 하지 않습니다" in html
    assert "원본 WAV의 부가 메타데이터 보존은 보장하지 않습니다" in html
    assert "새 긴 WAV·고해상도 WAV·MP3/M4A 작업 시작" in html
    assert "PCM24·Float32 재생은 브라우저 지원에 따라 다릅니다" in html
    assert "미리듣기용으로 몰래 PCM16 변환하지 않습니다" in html
    assert 'autoplay' not in html.lower()
    status, _, payload = _request(server, "/app.js")
    assert status == 200
    script = payload.decode("utf-8")
    assert "usesStoredFiles(mode) ? 128 : 32" in script
    assert "isStoredFile ? 900000 : 120000" in script
    assert "6초·48kHz·스테레오 PCM24이며 Float32·44.1kHz 시험은 아닙니다" in script
    assert "formatSamples(report.input)" in script and "formatSamples(report.output)" in script
    assert "signed_PCM24_little_endian" in script and "IEEE_FLOAT32_little_endian" in script
    assert "window.location.origin" in script and "!expectedPath.test" in script
    assert "완료 후에도 자동 재생하지 않습니다" in script
