"""H3 local/CLI contracts, including stereo envelopes without mono cancellation."""

import base64
import io
import json
from types import SimpleNamespace
from urllib.parse import urlsplit
import wave
import zipfile

import numpy as np
import pytest

from hum_dsp import local_app
from hum_dsp.band_control import BandControlConfig
from hum_dsp.cli import main


def _write(path, values, rate):
    values = np.asarray(values, dtype="<i2")
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(values.shape[1])
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(values.tobytes())


def _handler():
    return object.__new__(local_app._LocalHandler)


@pytest.mark.parametrize("depths", ["6,3,0", "0,0,0", "12,12,12", "1.25,0,7"])
def test_compatible_query_has_exact_explicit_settings(depths):
    mode, config = _handler()._query(urlsplit("/?mode=compatible&hum_hz=50&band_depths_db=" + depths))
    assert mode == "compatible"
    assert config.band_depths_db == tuple(map(float, depths.split(",")))


@pytest.mark.parametrize("query", [
    "mode=compatible&hum_hz=50", "mode=compatible&hum_hz=50&attenuation_db=3",
    "mode=compatible&hum_hz=50&band_depths_db=6,3,0&orders=1,2",
    "mode=compatible&hum_hz=50&band_depths_db=NaN,3,0",
    "mode=compatible&hum_hz=50&band_depths_db=6,3,13",
    "mode=compatible&hum_hz=50&band_depths_db=6,3",
    "mode=compatible&hum_hz=50&band_depths_db=6,3,0&band_depths_db=0,0,0",
    "mode=compatible&hum_hz=44100&band_depths_db=6,3,0",
])
def test_bad_compatible_query_is_not_silently_reinterpreted(query):
    with pytest.raises(local_app._RequestFailure) as exc:
        _handler()._query(urlsplit("/?" + query))
    assert exc.value.status == 400


def test_antiphase_display_uses_extrema_not_L_plus_R():
    values = np.column_stack((np.linspace(-0.3, 0.3, 1200), -np.linspace(-0.3, 0.3, 1200)))
    bins = np.asarray(local_app._compatible_waveform(values))
    assert bins.shape == (480, 2)
    assert bins[:, 0].min() == -0.3
    assert bins[:, 1].max() == 0.3
    assert np.max(np.abs(bins)) > 0.29
    assert np.sum(values, axis=1).max() == 0


@pytest.mark.parametrize("rate,channels", [(44100, 1), (44100, 2), (48000, 1), (48000, 2)])
def test_zero_native_frame_duration_channels_ZIP_and_cleanup(tmp_path, rate, channels):
    values = np.arange(-1000, 1001, dtype=np.int16)[:, None]
    values = np.repeat(values, channels, axis=1)
    source = tmp_path / "source.wav"
    _write(source, values, rate)
    root = tmp_path / "jobs"
    root.mkdir()
    config = BandControlConfig(band_depths_db=(0, 0, 0))
    reply = json.loads(local_app._run_job(SimpleNamespace(work_root=root), source.read_bytes(), "compatible", config, "user_file"))
    assert reply["duration_seconds"] == values.shape[0] / rate
    assert reply["report"]["input"]["sample_rate_hz"] == rate
    assert reply["report"]["output"]["channels"] == channels
    assert base64.b64decode(reply["output_wav_base64"]) == source.read_bytes()
    assert reply["waveforms"]["input"] == reply["waveforms"]["output"]
    assert list(root.iterdir()) == []
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(reply["download_zip_base64"]))) as archive:
        assert set(archive.namelist()) == {"processed.wav", "report.json", "compatible.json"}
        assert archive.read("processed.wav") == source.read_bytes()
    assert reply["report"]["local_tool"]["waveform_channel_policy"] == "per_time_bin_extrema_across_channels_no_signed_downmix"


def test_44k_stereo_actual_direct_CLI_and_local_output_match(tmp_path, capsys):
    from hum_dsp.compatible_processing import process_compatible_wav
    rate = 44100
    t = np.arange(round(rate * 3.4)) / rate
    values = 0.045 * np.sin(2 * np.pi * 233 * t)
    values += sum(0.02 * np.sin(2 * np.pi * 50 * n * t + 0.37 * n) for n in (1, 2, 3))
    codes = np.rint(values * 32768).astype(np.int16)
    source = tmp_path / "native stereo.wav"
    _write(source, np.column_stack((codes, -codes)), rate)
    original = source.read_bytes()
    direct = tmp_path / "direct"
    config = BandControlConfig()
    result = process_compatible_wav(source, direct, config)
    output = (direct / "processed.wav").read_bytes()
    trace = json.loads((direct / "compatible.json").read_text(encoding="utf-8"))
    assert output != original
    assert result["reference_quality_metrics"] is None
    assert result["input"]["frames"] == result["output"]["frames"] == codes.size
    cli = tmp_path / "cli"
    assert main(["process-compatible", str(source), "--hum-hz", "50", "--band-depths-db", "6", "3", "0", "--output-dir", str(cli)]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "compatible_WAV_output_created"
    assert (cli / "processed.wav").read_bytes() == output
    assert json.loads((cli / "compatible.json").read_text(encoding="utf-8")) == trace
    root = tmp_path / "jobs"
    root.mkdir()
    reply = json.loads(local_app._run_job(SimpleNamespace(work_root=root), original, "compatible", config, "user_file"))
    assert base64.b64decode(reply["output_wav_base64"]) == output
    assert reply["report"]["input"]["channels"] == 2
    assert list(root.iterdir()) == []
    assert source.read_bytes() == original


def test_compatible_example_is_stereo_and_source_independent_of_depth():
    a = local_app._compatible_example_wav(BandControlConfig())
    b = local_app._compatible_example_wav(BandControlConfig(band_depths_db=(0, 0, 0)))
    assert a == b
    with wave.open(io.BytesIO(a), "rb") as wav:
        assert (wav.getframerate(), wav.getnchannels(), wav.getsampwidth(), wav.getnframes()) == (48000, 2, 2, 288000)
