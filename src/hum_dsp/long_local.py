"""Bounded loopback transport for long WAV jobs; no base64 audio responses.

One temporary result is retained for user-initiated playback/download until the
next accepted long request or server close. Random asset capabilities are local
session-only, not account authentication or a remotely deployable API.
"""

from contextlib import contextmanager
import hmac
import json
import math
from pathlib import Path
import re
import secrets
import socket
import tempfile
import threading
from time import monotonic
from urllib.parse import parse_qs
import wave
import zipfile

import numpy as np

from .band_control import BandControlConfig


MAX_UPLOAD_BYTES = 128 * 1024 * 1024
COPY_BYTES = 65536
UPLOAD_SECONDS = 60.0
ASSET_TRANSFER_SECONDS = 30.0
ASSET_NAMES = {"input.wav": "audio/wav", "processed.wav": "audio/wav",
               "report.json": "application/json; charset=utf-8", "result.zip": "application/zip"}
ASSET_ROUTE = re.compile(r"/api/long/assets/([a-f0-9]{48})/(input\.wav|processed\.wav|report\.json|result\.zip)")


class LongJobStore:
    def __init__(self, work_root):
        self.work_root = Path(work_root).resolve()
        self.lock = threading.RLock()
        self.temporary = None
        self.job_id = None
        self.files = {}

    def clear(self):
        with self.lock:
            self.files = {}
            self.job_id = None
            if self.temporary is not None:
                self.temporary.cleanup()
                self.temporary = None

    def begin(self):
        with self.lock:
            self.clear()
            self.temporary = tempfile.TemporaryDirectory(prefix="long_job_", dir=self.work_root)
            return Path(self.temporary.name)

    def publish(self, folder, *, input_wav=None):
        with self.lock:
            if self.temporary is None or folder.resolve() != Path(self.temporary.name).resolve():
                raise ValueError("long job directory does not belong to this session")
            selected_input = folder / "input.wav" if input_wav is None else input_wav
            if selected_input.resolve().parent not in (folder.resolve(), (folder / "result").resolve()):
                raise ValueError("input WAV must belong to this owned job")
            paths = {"input.wav": selected_input, "processed.wav": folder / "result/processed.wav",
                     "report.json": folder / "result/report.json", "result.zip": folder / "result.zip"}
            if any(not path.is_file() or path.is_symlink() for path in paths.values()):
                raise ValueError("long job assets are incomplete")
            if sum(path.stat().st_size for path in folder.rglob("*") if path.is_file()) > 512 * 1024 * 1024:
                raise ValueError("long job temporary files exceed the 512 MiB session limit")
            self.job_id = secrets.token_hex(24)
            self.files = paths
            return {name: f"/api/long/assets/{self.job_id}/{name}" for name in paths}

    @contextmanager
    def open_asset(self, job_id, name):
        # Serialize replacement with active readers. Transfers have one bounded
        # deadline, so a slow asset cannot indefinitely pin an old temporary job.
        with self.lock:
            if self.job_id != job_id or name not in self.files:
                raise FileNotFoundError("long job no longer exists")
            with self.files[name].open("rb") as source:
                yield source, self.files[name].stat().st_size


def _settings(parsed):
    from .local_app import _RequestFailure
    try:
        fields = parse_qs(parsed.query, keep_blank_values=True, strict_parsing=True, max_num_fields=2)
        if set(fields) != {"hum_hz", "band_depths_db"} or any(len(values) != 1 for values in fields.values()):
            raise ValueError("settings keys")
        if fields["hum_hz"][0] not in ("50", "60"):
            raise ValueError("nominal frequency")
        tokens = fields["band_depths_db"][0].split(",")
        if len(tokens) != 3 or any(not value or value.strip() != value for value in tokens):
            raise ValueError("band depths")
        return BandControlConfig(nominal_hum_hz=float(fields["hum_hz"][0]),
                                 band_depths_db=tuple(map(float, tokens)), block_size_frames=4096)
    except (ValueError, OverflowError) as exc:
        raise _RequestFailure(400, "긴 WAV는 50/60 Hz와 세 대역의 0~12 dB 설정이 필요합니다.") from exc


def _length(handler):
    from .local_app import _RequestFailure
    if handler.headers.get_all("Transfer-Encoding", []):
        raise _RequestFailure(400, "고정 길이 WAV 요청만 지원합니다.")
    if handler._one_header("Content-Encoding") is not None:
        raise _RequestFailure(415, "압축된 요청 본문은 지원하지 않습니다.")
    values = handler.headers.get_all("Content-Length", [])
    if len(values) != 1 or re.fullmatch(r"[0-9]+", values[0]) is None:
        raise _RequestFailure(400, "파일 크기 정보가 올바르지 않습니다.")
    digits = values[0].lstrip("0") or "0"
    if len(digits) > 9 or int(digits) > MAX_UPLOAD_BYTES:
        raise _RequestFailure(413, "긴 WAV는 최대 128 MiB까지 지원합니다.")
    return int(digits)


def _copy_upload(handler, destination, length):
    from .local_app import _RequestFailure
    remaining = length
    deadline = monotonic() + UPLOAD_SECONDS
    try:
        with destination.open("xb") as target:
            while remaining:
                timeout = deadline - monotonic()
                if timeout <= 0:
                    raise TimeoutError("upload deadline")
                handler.connection.settimeout(min(timeout, 10.0))
                chunk = handler.rfile.read1(min(COPY_BYTES, remaining))
                if not chunk:
                    raise EOFError("incomplete upload")
                handler._body_bytes_consumed += len(chunk)
                remaining -= len(chunk)
                target.write(chunk)
    except (OSError, TimeoutError, EOFError) as exc:
        raise _RequestFailure(400, "긴 WAV 전송이 완료되지 않았습니다. 파일을 다시 선택해 주세요.") from exc
    finally:
        try:
            handler.connection.settimeout(10.0)
        except OSError:
            pass  # A peer abort must not obscure the original upload failure.


def write_owned_long_example(path, config):
    """Fixed 65-second stereo source generated in blocks, with no playback."""
    with wave.open(str(path), "wb") as output:
        output.setnchannels(2)
        output.setsampwidth(2)
        output.setframerate(48000)
        for start in range(0, 48000 * 65, 4096):
            t = np.arange(start, min(start + 4096, 48000 * 65), dtype=np.float64) / 48000
            hum = sum(math.sqrt(2) * 0.012 * np.sin(2 * math.pi * config.nominal_hum_hz * order * t + 0.37 * order) for order in (1, 2, 3))
            left = 0.06 * np.sin(2 * math.pi * 233 * t) + hum
            right = 0.04 * np.sin(2 * math.pi * 311 * t + 0.3) + 0.6 * hum
            output.writeframesraw(np.rint(np.column_stack((left, right)) * 32768).astype("<i2").tobytes())


def write_owned_precision_example(path, config):
    """Six-second PCM24 stereo fixture, independent of requested band depths."""
    from .precision_wav import PrecisionWavWriter, encode_precision_block
    metadata = {"sample_rate_hz": 48000, "channels": 2, "frames": 48000 * 6,
                "sample_format": "signed_PCM24_little_endian", "bits_per_sample": 24,
                "sample_width_bytes": 3}
    with PrecisionWavWriter(path, metadata) as output:
        for start in range(0, metadata["frames"], 4096):
            t = np.arange(start, min(start + 4096, metadata["frames"]), dtype=np.float64) / 48000
            hum = sum(math.sqrt(2) * .012 * np.sin(2 * math.pi * config.nominal_hum_hz * order * t + .37 * order) for order in (1, 2, 3))
            samples = np.column_stack((.06 * np.sin(2 * math.pi * 233 * t) + hum,
                                       .04 * np.sin(2 * math.pi * 311 * t + .3) + .6 * hum))
            payload, _ = encode_precision_block(samples, metadata["sample_format"])
            output.write_payload(payload)


def process_request(handler, parsed):
    from .local_app import _RequestFailure
    routes = {"/api/long/process": ("long", False), "/api/long/example": ("long", True),
              "/api/precision/process": ("precision", False), "/api/precision/example": ("precision", True)}
    if parsed.path not in routes:
        raise _RequestFailure(404, "이 긴 파일 요청 주소는 지원하지 않습니다.")
    mode, example = routes[parsed.path]
    token = handler._one_header("X-Hum-Token", required=True)
    if not hmac.compare_digest(token.encode("utf-8"), handler.server.session_token.encode("ascii")):
        raise _RequestFailure(403, "로컬 화면을 새로 열고 다시 실행해 주세요.")
    length = _length(handler)
    config = _settings(parsed)
    if (example and length != 0) or (not example and length <= 0):
        raise _RequestFailure(400, "예제에는 파일을 보내지 말고, 파일 처리에는 비어 있지 않은 WAV를 선택하세요.")
    if not example and handler._one_header("Content-Type", required=True) not in ("audio/wav", "application/octet-stream"):
        raise _RequestFailure(400, "지원되는 44.1/48 kHz WAV 파일을 선택해 주세요.")
    locked = handler.server.processing_lock.acquire(blocking=False)
    if not locked:
        raise _RequestFailure(503, "다른 파일을 처리하고 있습니다. 완료 후 다시 실행해 주세요.")
    store = handler.server.long_jobs
    success = False
    try:
        from .desktop_control import ensure_accepting_jobs
        ensure_accepting_jobs(handler.server)
        folder = store.begin()
        source = folder / "input.wav"
        if example:
            (write_owned_precision_example if mode == "precision" else write_owned_long_example)(source, config)
        else:
            _copy_upload(handler, source, length)
        if mode == "precision":
            from .precision_processing import process_precision_wav
            report = process_precision_wav(source, folder / "result", config)
        else:
            from .long_compatible_processing import process_long_compatible_wav
            report = process_long_compatible_wav(source, folder / "result", config)
        report = dict(report)
        report["local_tool"] = {
            "schema_version": 1, "mode": mode, "transport": "loopback_streamed_files_no_base64",
            "microphone_capture": False, "autoplay": False, "remote_upload": False,
            "temporary_result_retained_until_next_long_job_or_server_close": True,
            "temporary_result_replacement_modes": ["long", "precision", "compressed", "persistent"],
            "temporary_files_removed_before_response": False,
            "waveform_channel_policy": "per_time_bin_extrema_across_channels_no_signed_downmix",
            "asset_urls": "unguessable_local_session_capabilities_not_remote_authentication",
        }
        report_bytes = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8")
        (folder / "result/report.json").write_bytes(report_bytes)
        with zipfile.ZipFile(folder / "result.zip", "x", compression=zipfile.ZIP_DEFLATED) as archive:
            for name in ("processed.wav", "report.json", "long_trace.jsonl"):
                archive.write(folder / "result" / name, arcname=name)
        urls = store.publish(folder)
        response = {"schema_version": 1, "mode": mode,
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
    except (ValueError, wave.Error, EOFError) as exc:
        if mode == "precision":
            raise _RequestFailure(400, "고해상도 WAV는 44.1/48 kHz·모노/스테레오·PCM16/24 또는 Float32를 지원합니다. 최대 10분과 128 MiB 중 먼저 도달하는 제한이 적용됩니다. Float32는 유한한 -1 이상 1 미만 값만 받으며, 입력과 같은 형식으로 저장합니다. 자동 정규화·클리핑은 하지 않습니다.") from exc
        raise _RequestFailure(400, "긴 WAV는 44.1/48 kHz·모노/스테레오·PCM16, 최대 10분·128 MiB를 지원합니다. 44.1 kHz 활성 처리는 명시된 FIR 변환을 포함합니다. 입력과 출력 피크를 확인하세요. 자동 정규화·클리핑은 하지 않습니다.") from exc
    except Exception as exc:
        raise _RequestFailure(500, "긴 파일 처리에 실패했습니다. 이번 임시 결과는 정리하며 기존 원본은 변경하지 않습니다.") from exc
    finally:
        try:
            if not success:
                store.clear()
        finally:
            handler.server.processing_lock.release()


def _byte_range(value, size):
    if value is None:
        return 0, size - 1, False
    match = re.fullmatch(r"bytes=([0-9]{0,20})-([0-9]{0,20})", value)
    if match is None or not any(match.groups()):
        raise ValueError("invalid single range")
    first, last = match.groups()
    if not first:
        suffix = int(last)
        if suffix <= 0:
            raise ValueError("empty suffix")
        return max(0, size - suffix), size - 1, True
    start = int(first)
    end = size - 1 if not last else min(int(last), size - 1)
    if start >= size or end < start:
        raise ValueError("unsatisfied range")
    return start, end, True


def serve_asset(handler, parsed, *, head_only=False):
    from .local_app import _CSP, _RequestFailure
    match = ASSET_ROUTE.fullmatch(parsed.path)
    if parsed.query or match is None:
        raise _RequestFailure(404, "이 결과 파일은 제공하지 않습니다.")
    job_id, name = match.groups()
    try:
        with handler.server.long_jobs.open_asset(job_id, name) as (stream, size):
            try:
                start, end, partial = _byte_range(handler._one_header("Range"), size)
            except ValueError:
                handler.send_response(416)
                handler.send_header("Content-Range", f"bytes */{size}")
                handler.send_header("Content-Length", "0")
                handler.send_header("Content-Security-Policy", _CSP)
                handler.send_header("X-Content-Type-Options", "nosniff")
                handler.send_header("Cache-Control", "no-store")
                handler.send_header("Referrer-Policy", "no-referrer")
                handler.send_header("X-Frame-Options", "DENY")
                handler.send_header("Connection", "close")
                handler.end_headers()
                handler.close_connection = True
                return
            handler.send_response(206 if partial else 200)
            for key, value in (("Content-Type", ASSET_NAMES[name]), ("Content-Length", str(end - start + 1)),
                               ("Accept-Ranges", "bytes"), ("Content-Security-Policy", _CSP),
                               ("X-Content-Type-Options", "nosniff"), ("Cache-Control", "no-store"),
                               ("Referrer-Policy", "no-referrer"), ("X-Frame-Options", "DENY"), ("Connection", "close")):
                handler.send_header(key, value)
            if partial:
                handler.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            if name == "result.zip":
                handler.send_header("Content-Disposition", 'attachment; filename="hum-dsp-long-result.zip"')
            handler.end_headers()
            handler.close_connection = True
            if head_only:
                return
            stream.seek(start)
            remaining = end - start + 1
            deadline = monotonic() + ASSET_TRANSFER_SECONDS
            try:
                while remaining:
                    timeout = deadline - monotonic()
                    if timeout <= 0:
                        break
                    handler.connection.settimeout(min(timeout, 10.0))
                    chunk = stream.read(min(COPY_BYTES, remaining))
                    if not chunk:
                        break
                    handler.wfile.write(chunk)
                    remaining -= len(chunk)
            except (OSError, socket.timeout):
                pass
    except FileNotFoundError as exc:
        raise _RequestFailure(404, "이 긴 파일 결과는 만료됐습니다. 다시 처리해 주세요.") from exc
