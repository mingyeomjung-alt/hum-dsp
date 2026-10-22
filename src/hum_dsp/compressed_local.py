"""Opt-in local compressed-file import; shared bounded stored-file delivery."""

from fractions import Fraction
import hmac
import json
import math

import numpy as np

from . import long_local


def write_owned_compressed_example(path, config):
    """Encode a six-second owned mathematical MP3, never a microphone input."""
    import av
    rate = 48000
    with path.open("xb") as owned, av.open(owned, "w", format="mp3") as container:
        stream = container.add_stream("libmp3lame", rate=rate)
        stream.layout = "stereo"
        stream.bit_rate = 128000
        for start in range(0, rate * 6, 4096):
            t = np.arange(start, min(start + 4096, rate * 6), dtype=np.float64) / rate
            hum = sum(math.sqrt(2) * .012 * np.sin(2 * math.pi * config.nominal_hum_hz * n * t + .37 * n)
                      for n in (1, 2, 3))
            values = np.stack((.06 * np.sin(2 * math.pi * 233 * t) + hum,
                               .04 * np.sin(2 * math.pi * 311 * t + .3) + .6 * hum)).astype(np.float32)
            frame = av.AudioFrame.from_ndarray(values, format="fltp", layout="stereo")
            frame.sample_rate, frame.pts, frame.time_base = rate, start, Fraction(1, rate)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)


def process_request(handler, parsed):
    from .local_app import _RequestFailure
    from .compressed_decode import decoder_status
    from .compressed_processing import process_compressed_file, MAX_SOURCE_BYTES

    if parsed.path not in ("/api/compressed/process", "/api/compressed/example"):
        raise _RequestFailure(404, "이 압축 파일 요청 주소는 지원하지 않습니다.")
    token = handler._one_header("X-Hum-Token", required=True)
    if not hmac.compare_digest(token.encode("utf-8"), handler.server.session_token.encode("ascii")):
        raise _RequestFailure(403, "로컬 화면을 새로 열고 다시 실행해 주세요.")
    try:
        length = long_local._length(handler)
    except _RequestFailure as exc:
        if exc.status == 413:
            raise _RequestFailure(413, "MP3·M4A 입력은 최대 32 MiB까지 지원합니다.") from exc
        raise
    if length > MAX_SOURCE_BYTES:
        raise _RequestFailure(413, "MP3·M4A 입력은 최대 32 MiB까지 지원합니다.")
    config = long_local._settings(parsed)
    example = parsed.path.endswith("/example")
    source_format = handler._one_header("X-Hum-Source-Format")
    if example:
        if length != 0 or source_format is not None:
            raise _RequestFailure(400, "압축 예제 요청에는 파일이나 파일 형식을 보내지 마세요.")
        source_format = "mp3"
    elif (length <= 0 or source_format not in ("mp3", "m4a") or
          handler._one_header("Content-Type", required=True) != "application/octet-stream"):
        raise _RequestFailure(400, "32 MiB 이하의 MP3 또는 AAC M4A 파일 하나를 선택하세요.")
    if not decoder_status()["available"]:
        raise _RequestFailure(503, "선택 기능인 PyAV 19 디코더가 없습니다. 압축 파일용 추가 구성 요소를 설치하거나 WAV 모드를 사용하세요. 자동 설치는 하지 않습니다.")
    if not handler.server.processing_lock.acquire(blocking=False):
        raise _RequestFailure(503, "다른 파일을 처리하고 있습니다. 완료 후 다시 실행해 주세요.")
    store, success = handler.server.long_jobs, False
    try:
        from .desktop_control import ensure_accepting_jobs
        ensure_accepting_jobs(handler.server)
        folder = store.begin()
        source = folder / ("source." + source_format)
        if example:
            write_owned_compressed_example(source, config)
        else:
            long_local._copy_upload(handler, source, length)
        report = process_compressed_file(source, folder / "result", config, source_format=source_format)
        report = dict(report)
        report["local_tool"] = {
            "schema_version": 1, "mode": "compressed", "transport": "loopback_streamed_files_no_base64",
            "microphone_capture": False, "autoplay": False, "remote_upload": False,
            "temporary_result_replacement_modes": ["long", "precision", "compressed", "persistent"],
            "temporary_result_retained_until_next_stored_job_or_server_close": True,
            "temporary_files_removed_before_response": False,
            "comparison_input": "decoded_Float32_WAV_not_compressed_original",
            "waveform_channel_policy": "per_time_bin_extrema_across_channels_no_signed_downmix",
            "asset_urls": "unguessable_local_session_capabilities_not_remote_authentication",
        }
        (folder / "result/report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        with long_local.zipfile.ZipFile(folder / "result.zip", "x", compression=long_local.zipfile.ZIP_DEFLATED) as archive:
            for name in ("processed.wav", "report.json", "long_trace.jsonl"):
                archive.write(folder / "result" / name, arcname=name)
        urls = store.publish(folder, input_wav=folder / "result/decoded_input.wav")
        response = {"schema_version": 1, "mode": "compressed",
                    "source_kind": "owned_synthetic_example" if example else "user_file",
                    "duration_seconds": report["input"]["frames"] / report["input"]["sample_rate_hz"],
                    "report": report, "waveforms": report["waveforms"],
                    "input_wav_url": urls["input.wav"], "output_wav_url": urls["processed.wav"],
                    "download_zip_url": urls["result.zip"], "report_json_url": urls["report.json"]}
        payload = json.dumps(response, ensure_ascii=False, allow_nan=False).encode("utf-8")
        success = True
        handler._send(200, payload, "application/json; charset=utf-8")
    except _RequestFailure:
        raise
    except RuntimeError as exc:
        raise _RequestFailure(503, "압축 디코더를 실행하지 못했거나 제한 시간을 넘었습니다. WAV로 변환해 다시 시도하세요. 이번 임시 결과는 정리합니다.") from exc
    except (ValueError, EOFError) as exc:
        raise _RequestFailure(400, "지원하지 않거나 손상된 압축 파일·출력 범위입니다. MP3 또는 단일 AAC M4A, 44.1/48 kHz·모노/스테레오만 받습니다. 디코딩 후 최대 10분·128 MiB, 유한한 -1 이상 1 미만 값이어야 합니다. 자동 다운믹스·클리핑·정규화는 하지 않습니다.") from exc
    except Exception as exc:
        raise _RequestFailure(500, "압축 파일 처리에 실패했습니다. 이번 임시 결과를 정리하며 원본은 변경하지 않습니다.") from exc
    finally:
        try:
            if not success:
                store.clear()
        finally:
            handler.server.processing_lock.release()
