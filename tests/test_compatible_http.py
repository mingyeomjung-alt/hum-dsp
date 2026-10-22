"""Explicit compatibility-mode HTTP contracts using owned in-memory WAV only."""

import base64
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

from hum_dsp import local_app


PARAMETERS = "mode=compatible&hum_hz=50&band_depths_db=0,0,0"


def _wav(*, rate=44100, channels=2, width=2, frames=None):
    count = rate // 100 if frames is None else frames
    time = np.arange(count) / rate
    first = np.rint(0.09 * 32768 * np.sin(2 * np.pi * 311 * time)).astype("<i2")
    # Opposite polarity must not disappear into a signed downmix waveform.
    codes = np.column_stack([first if index % 2 == 0 else -first
                             for index in range(channels)])
    stream = io.BytesIO()
    with wave.open(stream, "wb") as wav:
        wav.setframerate(rate)
        wav.setnchannels(channels)
        wav.setsampwidth(width)
        wav.writeframes(codes.tobytes() if width == 2 else bytes(count * channels * width))
    return stream.getvalue()


def _decode(payload):
    with wave.open(io.BytesIO(payload), "rb") as wav:
        metadata = (wav.getframerate(), wav.getnchannels(), wav.getnframes(), wav.getsampwidth())
        codes = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").copy()
    return metadata, codes


@contextmanager
def _running_server(work_root):
    running = local_app.make_server(port=0, work_root=work_root)
    thread = threading.Thread(target=lambda: running.serve_forever(poll_interval=0.02), daemon=True)
    thread.start()
    try:
        yield running
    finally:
        running.shutdown()
        running.server_close()
        thread.join(timeout=5)
        assert not thread.is_alive()


@pytest.fixture
def server(tmp_path):
    with _running_server(tmp_path / "compatible request work") as running:
        yield running


def _post(server, body, query=PARAMETERS, *, token=None, route="process"):
    target = urlsplit(server.origin)
    connection = http.client.HTTPConnection(target.hostname, target.port, timeout=10)
    try:
        connection.request("POST", f"/api/{route}?{query}", body=body,
                           headers={"Host": target.netloc, "Content-Type": "audio/wav",
                                    "X-Hum-Token": server.session_token if token is None else token})
        response = connection.getresponse()
        payload = response.read()
        result = json.loads(payload)
        json.dumps(result, allow_nan=False)
        assert "no-store" in response.getheader("Cache-Control")
        assert not any(name.lower().startswith("access-control-allow") for name, _ in response.getheaders())
        assert server.session_token.encode() not in payload
        return response.status, result
    finally:
        connection.close()


def _work_empty(server):
    assert not server.work_root.exists() or not list(server.work_root.iterdir())


@pytest.mark.parametrize("rate", (44100, 48000))
@pytest.mark.parametrize("channels", (1, 2))
def test_compatible_http_zero_depth_preserves_native_pcm_rate_channels_and_time(server, rate, channels):
    source = _wav(rate=rate, channels=channels)
    status, result = _post(server, source)
    assert status == 200
    assert result["mode"] == "compatible"
    assert result["duration_seconds"] == 0.01
    assert base64.b64decode(result["input_wav_base64"], validate=True) == source
    output = base64.b64decode(result["output_wav_base64"], validate=True)
    input_metadata, input_codes = _decode(source)
    output_metadata, output_codes = _decode(output)
    assert output_metadata == input_metadata == (rate, channels, rate // 100, 2)
    np.testing.assert_array_equal(output_codes, input_codes)
    report = result["report"]
    for side in ("input", "output"):
        assert report[side]["sample_rate_hz"] == rate
        assert report[side]["channels"] == channels
        assert report[side]["frames"] == rate // 100
    assert report["processing_bypassed"] is True
    assert report["algorithm"]["shared_original_input_fft_count"] == 0
    assert report["conversion"]["to_engine"]["conversion_applied"] is False
    assert report["reference_quality_metrics"] is None
    for key in ("microphone_capture", "autoplay", "remote_upload"):
        assert report["local_tool"][key] is False
    assert report["local_tool"]["temporary_files_removed_before_response"] is True
    assert report["local_tool"]["waveform_channel_policy"] == "per_time_bin_extrema_across_channels_no_signed_downmix"
    extrema = np.asarray(result["waveforms"]["input"])
    assert extrema.shape == (min(rate // 100, 480), 2)
    assert np.max(np.abs(extrema)) > 0.08
    if channels == 2:
        np.testing.assert_array_equal(extrema[:, 0], -extrema[:, 1])
    assert result["waveforms"]["input"] == result["waveforms"]["output"]
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(result["download_zip_base64"], validate=True))) as archive:
        assert set(archive.namelist()) == {"processed.wav", "report.json", "compatible.json"}
        assert archive.read("processed.wav") == output
        assert json.loads(archive.read("report.json")) == report
    _work_empty(server)


@pytest.mark.parametrize("mode", ("fixed", "tracked", "guarded", "harmonics", "bands"))
@pytest.mark.parametrize("rate,channels", ((44100, 1), (48000, 2)))
def test_compatible_support_does_not_expand_legacy_modes(server, mode, rate, channels):
    if mode == "bands":
        query = f"mode={mode}&hum_hz=50&band_depths_db=0,0,0"
    else:
        query = f"mode={mode}&hum_hz=50&attenuation_db=0"
        if mode == "harmonics":
            query += "&orders=1,2,3"
    status, result = _post(server, _wav(rate=rate, channels=channels), query)
    assert status == 400
    assert isinstance(result["error"], str)
    _work_empty(server)


def test_compatible_wrong_token_cannot_reach_any_processing(server, monkeypatch):
    calls = []

    def forbidden(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("unauthorized request reached processing")

    monkeypatch.setattr(local_app, "_run_job", forbidden)
    status, result = _post(server, _wav(), token="wrong-token")
    assert status == 403
    assert "error" in result
    assert calls == []
    _work_empty(server)


@pytest.mark.parametrize("source", (b"", b"not a WAV", b"RIFF\x24\x00\x00\x00WAVE",
                                     _wav(rate=8000), _wav(channels=3), _wav(width=3),
                                     _wav() + b"trailing bytes"),
                         ids=("empty", "not-wav", "truncated", "8khz", "3channels", "pcm24", "trailing"))
def test_compatible_malformed_input_is_refused_cleaned_and_does_not_poison_next_request(server, source):
    status, result = _post(server, source)
    assert status == 400
    assert isinstance(result["error"], str)
    _work_empty(server)
    status, result = _post(server, _wav())
    assert status == 200
    assert result["mode"] == "compatible"
    _work_empty(server)


def test_compatible_engine_error_cleans_partial_files_hides_paths_and_recovers(server, monkeypatch):
    from hum_dsp import compatible_processing

    def broken(source, destination, config):
        destination.mkdir()
        (destination / "partial.wav").write_bytes(b"partial output")
        raise RuntimeError(f"private path {source}; secret {server.session_token}")

    with monkeypatch.context() as patch:
        patch.setattr(compatible_processing, "process_compatible_wav", broken)
        status, result = _post(server, _wav())
    assert status == 500
    assert str(server.work_root) not in json.dumps(result)
    _work_empty(server)
    status, result = _post(server, _wav())
    assert status == 200
    assert result["mode"] == "compatible"
    _work_empty(server)


def test_compatible_example_is_owned_native_stereo_and_zero_depth_identity(server):
    status, result = _post(server, b"", route="example")
    assert status == 200
    assert result["source_kind"] == "owned_synthetic_example"
    assert result["duration_seconds"] == 6
    source = base64.b64decode(result["input_wav_base64"], validate=True)
    output = base64.b64decode(result["output_wav_base64"], validate=True)
    assert _decode(source)[0] == _decode(output)[0] == (48000, 2, 288000, 2)
    np.testing.assert_array_equal(_decode(source)[1], _decode(output)[1])
    _work_empty(server)
