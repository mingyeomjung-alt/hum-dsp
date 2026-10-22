"""Explicit opt-in local file transport for persistent harmonic suppression.

The unchanged offline candidate keeps its complete, at most 60-second signal
and FFT workspace in memory. HTTP upload/download is bounded and file-backed;
that does not make the processing engine a bounded stream or device engine.
"""

import hmac
import json
import math
import re
from urllib.parse import parse_qs
import wave
import zipfile

import numpy as np

from .long_local import _copy_upload, _length
from .long_processing import _Statistics
from .persistent_comb import CombConfig
from .persistent_processing import process_persistent_file
from .precision_wav import PrecisionWavReader, inspect_precision_wav
from .sdk import HumDSPError


def _settings(parsed):
    """Two unambiguous settings; no exponent, whitespace or duplicate keys."""
    from .local_app import _RequestFailure
    try:
        fields = parse_qs(parsed.query, keep_blank_values=True,
                          strict_parsing=True, max_num_fields=2)
        if (set(fields) != {"hum_hz", "attenuation_db"}
                or any(len(values) != 1 for values in fields.values())
                or fields["hum_hz"][0] not in ("50", "60")):
            raise ValueError("settings keys or frequency")
        depth = fields["attenuation_db"][0]
        # Plain canonical decimal notation: 0, 12, 0.5, 12.25, not 01/1e1/1.0.
        if (len(depth) > 24
                or re.fullmatch(r"(?:0|[1-9][0-9]*)(?:\.[0-9]*[1-9])?", depth) is None):
            raise ValueError("canonical attenuation")
        return CombConfig(nominal_hum_hz=float(fields["hum_hz"][0]),
                          max_attenuation_db=float(depth))
    except (ValueError, OverflowError) as exc:
        raise _RequestFailure(400, "Persistent mode requires hum_hz 50 or 60 and a plain finite attenuation_db from 0 to 18, with no duplicate or extra settings.") from exc


def write_owned_persistent_example(path, config):
    """Eight-second native stereo example, fixed across attenuation settings.

    Four persistent injected family tones are distinct from both wanted tones.
    This is a mathematical demonstration, not a recording or natural-hum test.
    """
    rate, frames = 48000, 48000 * 8
    orders = (4, 6, 21, 24)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(2)
        output.setsampwidth(2)
        output.setframerate(rate)
        for start in range(0, frames, 4096):
            t = np.arange(start, min(start + 4096, frames), dtype=np.float64) / rate
            hum = sum(.025 * math.sqrt(2) * np.sin(
                2 * math.pi * config.nominal_hum_hz * order * t + .17 * order
            ) for order in orders)
            left = (.07 * np.sin(2 * math.pi * 2300 * t)
                    + .045 * np.sin(2 * math.pi * 3100 * t + .2) + hum)
            right = (.045 * np.sin(2 * math.pi * 2300 * t + .4)
                     + .06 * np.sin(2 * math.pi * 3100 * t + .7) + .7 * hum)
            output.writeframesraw(np.rint(np.column_stack((left, right)) * 32768).astype("<i2").tobytes())


def _waveform(path, expected_sha):
    """Native full-scale extrema, bounded IO and no stereo signed downmix."""
    metadata = inspect_precision_wav(path)
    statistics = _Statistics(metadata["frames"], metadata["channels"])
    with PrecisionWavReader(path, metadata) as reader:
        for _payload, values in reader:
            statistics.add(values)
        if reader.sha256 != expected_sha:
            raise ValueError("audio changed before local publication")
    return statistics.waveform()


def process_request(handler, parsed):
    from .desktop_control import ensure_accepting_jobs
    from .local_app import _RequestFailure
    routes = {"/api/persistent/process": False, "/api/persistent/example": True}
    if parsed.path not in routes:
        raise _RequestFailure(404, "This persistent-mode request address is not supported.")
    example = routes[parsed.path]
    token = handler._one_header("X-Hum-Token", required=True)
    if not hmac.compare_digest(token.encode("utf-8"), handler.server.session_token.encode("ascii")):
        raise _RequestFailure(403, "Reopen this local screen and try again.")
    length = _length(handler)
    config = _settings(parsed)
    if (example and length != 0) or (not example and length <= 0):
        raise _RequestFailure(400, "Send no file for the example, or select a nonempty WAV for file processing.")
    if not example and handler._one_header("Content-Type", required=True) not in ("audio/wav", "application/octet-stream"):
        raise _RequestFailure(400, "Select a raw native PCM16/PCM24/Float32 WAV file.")
    if not handler.server.processing_lock.acquire(blocking=False):
        raise _RequestFailure(503, "Another file is being processed. Wait for it to finish.")
    store, folder, success = handler.server.long_jobs, None, False
    try:
        ensure_accepting_jobs(handler.server)
        # Rejected settings, authorization, length and busy requests never
        # replace an existing completed result. An accepted job does replace it.
        folder = store.begin()
        source = folder / "input.wav"
        if example:
            write_owned_persistent_example(source, config)
        else:
            _copy_upload(handler, source, length)
        result = process_persistent_file(source, output_dir=folder / "result",
                                         hum_hz=config.nominal_hum_hz,
                                         attenuation_db=config.max_attenuation_db)
        report = dict(result.report)
        report["waveforms"] = {
            "input": _waveform(source, report["input"]["sha256"]),
            "output": _waveform(result.output_wav, report["output"]["sha256"]),
        }
        source_kind = "owned_synthetic_example" if example else "user_file"
        report["local_tool"] = {
            "schema_version": 1, "mode": "persistent", "source_kind": source_kind,
            "transport": "loopback_streamed_files_no_base64",
            "microphone_capture": False, "autoplay": False, "remote_upload": False,
            "user_initiated_browser_playback_available": True,
            "temporary_result_retained_until_next_long_job_or_server_close": True,
            "temporary_result_replacement_modes": ["long", "precision", "compressed", "persistent"],
            "temporary_files_removed_before_response": False,
            "waveform_channel_policy": "per_time_bin_extrema_across_channels_no_signed_downmix",
            "waveform_scale": "common_pcm_full_scale_not_independently_normalized",
            "asset_urls": "unguessable_local_session_capabilities_not_remote_authentication",
            "offline_whole_file_memory_processing": True,
            "quality_approval": False,
            "comparison_notice": "Explicit opt-in candidate. Wanted tones in the same harmonic family can also be reduced; compare before keeping the result. Full-file audio and FFT workspace are retained in memory. No perceptual quality approval or real-time device claim.",
        }
        report_bytes = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8")
        result.report_path.write_bytes(report_bytes)
        with zipfile.ZipFile(folder / "result.zip", "x", compression=zipfile.ZIP_DEFLATED) as archive:
            for name in ("processed.wav", "report.json", "persistent_trace.json"):
                archive.write(folder / "result" / name, arcname=name)
        urls = store.publish(folder)
        response = {
            "schema_version": 1, "mode": "persistent", "source_kind": source_kind,
            "duration_seconds": report["input"]["frames"] / report["input"]["sample_rate_hz"],
            "report": report, "waveforms": report["waveforms"],
            "input_wav_url": urls["input.wav"], "output_wav_url": urls["processed.wav"],
            "download_zip_url": urls["result.zip"], "report_json_url": urls["report.json"],
        }
        payload = json.dumps(response, ensure_ascii=False, allow_nan=False).encode("utf-8")
        success = True
        handler._send(200, payload, "application/json; charset=utf-8")
    except _RequestFailure:
        raise
    except HumDSPError as exc:
        status = 500 if exc.code == "dependency_unavailable" else 400
        raise _RequestFailure(status, "Persistent processing was refused. Use a finite normalized native 44.1/48 kHz mono/stereo PCM16/PCM24/Float32 WAV, at most 60 seconds and 128 MiB. No automatic normalization, resampling or clipping is performed.") from exc
    except (ValueError, wave.Error, EOFError) as exc:
        raise _RequestFailure(400, "Persistent processing could not safely complete. Check the native WAV format, normalized sample range and 60-second/128-MiB limits. The original file is unchanged.") from exc
    except Exception as exc:
        raise _RequestFailure(500, "Persistent processing failed. This job's temporary result is removed; original files are unchanged.") from exc
    finally:
        try:
            if folder is not None and not success:
                store.clear()
        finally:
            handler.server.processing_lock.release()
