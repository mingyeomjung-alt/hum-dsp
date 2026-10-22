"""Real 44.1 kHz long-mode loopback transport, without playback or devices."""

from contextlib import contextmanager
import http.client
import io
import json
import threading
from urllib.parse import urlsplit
import wave
import zipfile

import numpy as np
import pytest

from hum_dsp import local_app, long_local


def _wav(*, channels=1, rate=44100, frames=132300, width=2):
    phase = np.arange(frames, dtype=np.float64) / rate
    samples = np.column_stack([
        (0.012 + index * 0.002) * np.sin(2 * np.pi * 50 * phase)
        + (0.035 + index * 0.015) * np.sin(2 * np.pi * (233 + 78 * index) * phase)
        for index in range(channels)
    ])
    payload = (np.rint(samples * 32768).astype("<i2").tobytes()
               if width == 2 else bytes(frames * channels * width))
    result = io.BytesIO()
    with wave.open(result, "wb") as destination:
        destination.setnchannels(channels)
        destination.setsampwidth(width)
        destination.setframerate(rate)
        destination.writeframes(payload)
    return result.getvalue()


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
    with _running_server(tmp_path / "local work") as instance:
        yield instance


def _request(server, path, *, payload=None):
    target = urlsplit(server.origin)
    connection = http.client.HTTPConnection(target.hostname, target.port, timeout=60)
    try:
        headers = {"Host": target.netloc}
        if payload is not None:
            headers.update({"Content-Type": "audio/wav", "X-Hum-Token": server.session_token})
        connection.request("POST" if payload is not None else "GET", path,
                           body=payload, headers=headers)
        response = connection.getresponse()
        return response.status, dict(response.getheaders()), response.read()
    finally:
        connection.close()


def _post(server, payload, depths="0,0,0"):
    return _request(server, f"/api/long/process?hum_hz=50&band_depths_db={depths}", payload=payload)


def _audio(payload):
    with wave.open(io.BytesIO(payload), "rb") as source:
        return (source.getframerate(), source.getnchannels(), source.getnframes(),
                source.readframes(source.getnframes()))


@pytest.mark.parametrize("channels", [1, 2])
@pytest.mark.parametrize("depths", ["0,0,0", "6,3,0"], ids=["bypass", "active_FIR"])
def test_real_44100_long_post_preserves_native_format_and_returns_downloads(server, channels, depths):
    source = _wav(channels=channels)
    status, headers, payload = _post(server, source, depths)
    assert status == 200, payload.decode(errors="replace")
    assert headers["Content-Type"].startswith("application/json")
    result = json.loads(payload)
    assert result["mode"] == "long"
    assert result["source_kind"] == "user_file"
    assert result["duration_seconds"] == 3
    assert not any("base64" in key for key in result)
    assert server.session_token.encode() not in payload
    assert str(server.work_root).encode() not in payload
    report = result["report"]
    for side in ("input", "output"):
        assert report[side]["sample_rate_hz"] == 44100
        assert report[side]["channels"] == channels
        assert report[side]["frames"] == 132300
    bypass = depths == "0,0,0"
    assert report["processing_bypassed"] is bypass
    assert report["device_io"] is False
    assert report["output_gain_normalization_applied"] is False
    assert report["reference_quality_metrics"] is None
    assert report["stereo_policy"]["same_trace_and_coefficients_all_channels"] is True
    assert report["stereo_policy"]["independent_channel_frequency_tracking"] is False
    assert report["local_tool"]["autoplay"] is False
    assert report["local_tool"]["microphone_capture"] is False
    assert report["local_tool"]["remote_upload"] is False
    assert report["local_tool"]["temporary_result_retained_until_next_long_job_or_server_close"] is True
    status, _, original = _request(server, result["input_wav_url"])
    assert status == 200 and original == source
    status, _, processed = _request(server, result["output_wav_url"])
    assert status == 200
    decoded = _audio(processed)
    assert decoded[:3] == (44100, channels, 132300)
    assert (decoded[3] == _audio(source)[3]) is bypass
    assert report["processor_applied"] is (not bypass)
    status, report_headers, report_bytes = _request(server, result["report_json_url"])
    assert status == 200 and json.loads(report_bytes) == report
    assert report_headers["Content-Type"].startswith("application/json")
    status, _, archive_bytes = _request(server, result["download_zip_url"])
    assert status == 200
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        assert set(archive.namelist()) == {"processed.wav", "report.json", "long_trace.jsonl"}
        assert archive.read("processed.wav") == processed
        assert archive.read("report.json") == report_bytes
        events = archive.read("long_trace.jsonl").decode("utf-8").splitlines()
        assert (len(events) == 0) is bypass
        for event in events:
            json.dumps(json.loads(event), allow_nan=False)
        assert archive.testzip() is None
    for side in ("input", "output"):
        bins = np.asarray(result["waveforms"][side])
        assert bins.shape == (480, 2)
        assert np.isfinite(bins).all()
    if bypass:
        np.testing.assert_array_equal(result["waveforms"]["input"], result["waveforms"]["output"])


def test_real_44100_long_bypass_does_not_inherit_the_old_60_second_limit(server):
    frames = 61 * 44100 + 17
    source = _wav(channels=2, frames=frames)
    status, _, payload = _post(server, source)
    assert status == 200, payload.decode(errors="replace")
    result = json.loads(payload)
    assert result["duration_seconds"] == pytest.approx(frames / 44100)
    assert result["report"]["processing_bypassed"] is True
    assert result["report"]["input"]["frames"] == frames
    status, _, processed = _request(server, result["output_wav_url"])
    assert status == 200
    assert _audio(processed) == _audio(source)


@pytest.mark.parametrize("source", [
    _wav(rate=32000, frames=32),
    _wav(channels=3, frames=44),
    _wav(width=1, frames=44),
    _wav(width=3, frames=44),
    _wav(frames=44)[:-1],
    b"RIFF\x24\0\0\0WAVE",
], ids=["unsupported_rate", "three_channels", "PCM8", "PCM24", "truncated", "bad_header"])
def test_44100_capability_does_not_relax_other_format_checks_and_recovers(server, source):
    status, _, payload = _post(server, source)
    assert status == 400
    assert isinstance(json.loads(payload)["error"], str)
    assert not list(server.work_root.iterdir())
    assert server.long_jobs.job_id is None
    assert server.processing_lock.acquire(blocking=False)
    server.processing_lock.release()
    assert _post(server, _wav(frames=44))[0] == 200


def test_long_example_remains_48000_and_is_not_misrepresented_as_conversion(server, monkeypatch):
    calls = []

    def small_example(path, config):
        calls.append(config.nominal_hum_hz)
        path.write_bytes(_wav(rate=48000, channels=2, frames=480))

    monkeypatch.setattr(long_local, "write_owned_long_example", small_example)
    status, _, payload = _request(server, "/api/long/example?hum_hz=50&band_depths_db=0,0,0", payload=b"")
    assert status == 200
    result = json.loads(payload)
    assert calls == [50]
    assert result["source_kind"] == "owned_synthetic_example"
    assert result["report"]["input"]["sample_rate_hz"] == 48000
    assert result["report"]["input"]["channels"] == 2
    assert result["report"]["output"]["sample_rate_hz"] == 48000


def test_long_ui_discloses_conversion_without_changing_old_mode_defaults(server):
    status, _, html = _request(server, "/")
    assert status == 200
    page = html.decode("utf-8")
    assert "v0.0.15" in page
    assert '<option value="long">Long WAV · up to 10 minutes (44.1/48 kHz)</option>' in page
    assert '<option value="fixed">Fixed filter · steady hum</option>' in page
    assert "fixed FIR" in page and "preceding and following samples" in page
    assert "boundaries, high frequencies and peaks" in page and "all bands at 0 dB" in page
    assert "forced termination or power loss" in page and "Secure erasure is not guaranteed" in page
    status, _, javascript = _request(server, "/app.js")
    assert status == 200
    script = javascript.decode("utf-8")
    assert "The Long WAV example is a 65-second, 48 kHz stereo" in script
    assert "To test 44.1 kHz resampling, select a matching WAV file" in script
    assert "playback does not start automatically when processing finishes" in script
    assert "isStoredFile ? 900000 : 120000" in script
