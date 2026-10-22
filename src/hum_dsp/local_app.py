"""Loopback-only, temporary-file UI bridge to the unchanged offline DSP engines.

This is a local desktop helper, not a remotely deployable service. No audio
device is opened and no network connection is initiated by this module.
"""

import base64
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import math
from pathlib import Path
import re
import secrets
import socket
import tempfile
import threading
from time import monotonic
from urllib.parse import parse_qs, urlsplit
import wave
import zipfile

import numpy as np

from .audio import MAX_FILE_BYTES, SAMPLE_RATE_HZ, read_pcm16_wav
from .config import ProcessorConfig
from .guarded_processing import process_guarded_wav
from .processing import process_wav
from .tracked_processing import process_tracked_wav


_WEB_ROOT = Path(__file__).resolve().parent / "web"
_BODY_TIMEOUT_SECONDS = 10.0
_REJECTED_BODY_MAX_BYTES = 65536
_REJECTED_BODY_DRAIN_SECONDS = 0.2
_MAX_WAVEFORM_BINS = 480
_CSP = (
    "default-src 'none'; script-src 'self'; style-src 'self'; "
    "connect-src 'self'; media-src 'self' blob:; img-src 'self' data:; "
    "base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
)


class _RequestFailure(Exception):
    def __init__(self, status, message):
        self.status = status
        self.message = message


def _waveform(samples):
    """Common full-scale min/max bins, without per-file gain normalization."""
    count = min(_MAX_WAVEFORM_BINS, int(samples.size))
    boundaries = np.linspace(0, samples.size, count + 1, dtype=np.int64)
    return [
        [float(np.min(samples[start:stop])), float(np.max(samples[start:stop]))]
        for start, stop in zip(boundaries[:-1], boundaries[1:])
    ]


def _compatible_waveform(samples):
    """Per-time-bin extrema across channels, never a signed mono downmix."""
    count = min(_MAX_WAVEFORM_BINS, int(samples.shape[0]))
    boundaries = np.linspace(0, samples.shape[0], count + 1, dtype=np.int64)
    return [[float(np.min(samples[start:stop, :])), float(np.max(samples[start:stop, :]))]
            for start, stop in zip(boundaries[:-1], boundaries[1:])]


def _example_wav(config):
    """Owned six-second mathematical example, not a recording or data fixture."""
    time = np.arange(SAMPLE_RATE_HZ * 6, dtype=np.float64) / SAMPLE_RATE_HZ
    clean = (
        0.07 * np.sin(2 * math.pi * 233.0 * time)
        + 0.045 * np.sin(2 * math.pi * 419.0 * time + 0.2)
        + 0.025 * np.sin(2 * math.pi * 811.0 * time + 0.4)
    )
    if hasattr(config, "band_depths_db"):
        # Fixed source regardless of manual depth: changing controls must not
        # quietly remove noise from the comparison input.
        noise = sum(math.sqrt(2) * 0.012 * np.sin(2 * math.pi * order * config.nominal_hum_hz * time + 0.37 * order) for order in (1, 2, 3))
    elif hasattr(config, "orders"):
        noise = sum(math.sqrt(2) * 0.012 * np.sin(2 * math.pi * order * config.nominal_hum_hz * time + 0.37 * order) for order in config.orders)
    else:
        noise = math.sqrt(2) * 0.02 * np.sin(2 * math.pi * config.nominal_hum_hz * time + 0.37)
    codes = np.rint((clean + noise) * 32768.0).astype("<i2")
    result = io.BytesIO()
    with wave.open(result, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE_HZ)
        wav.writeframes(codes.tobytes())
    return result.getvalue()


def _compatible_example_wav(config):
    """Fixed 48 kHz stereo input independent of chosen attenuation values."""
    time = np.arange(SAMPLE_RATE_HZ * 6, dtype=np.float64) / SAMPLE_RATE_HZ
    noise = sum(math.sqrt(2) * 0.012 * np.sin(2 * math.pi * order * config.nominal_hum_hz * time + 0.37 * order) for order in (1, 2, 3))
    left = 0.07 * np.sin(2 * math.pi * 233 * time) + 0.045 * np.sin(2 * math.pi * 419 * time + 0.2) + noise
    right = 0.05 * np.sin(2 * math.pi * 311 * time + 0.3) + 0.025 * np.sin(2 * math.pi * 557 * time + 0.4) + 0.6 * noise
    codes = np.rint(np.column_stack((left, right)) * 32768).astype("<i2")
    result = io.BytesIO()
    with wave.open(result, "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE_HZ)
        wav.writeframes(codes.tobytes())
    return result.getvalue()


def _run_job(server, source_bytes, mode, config, source_kind):
    """Finish packaging and clean temporary disk files before returning data."""
    runners = {"fixed": process_wav, "tracked": process_tracked_wav, "guarded": process_guarded_wav}
    if mode == "harmonics":
        from .harmonic_processing import process_harmonic_wav
        runners["harmonics"] = process_harmonic_wav
    elif mode == "bands":
        from .band_processing import process_band_wav
        runners["bands"] = process_band_wav
    elif mode == "compatible":
        from .compatible_processing import process_compatible_wav
        runners["compatible"] = process_compatible_wav
    elif mode == "profile":
        from .profile_processing import process_profile_wav
        runners["profile"] = process_profile_wav
    with tempfile.TemporaryDirectory(prefix="job_", dir=server.work_root) as temporary:
        directory = Path(temporary)
        source = directory / "input.wav"
        source.write_bytes(source_bytes)
        # The bounded reader and the unchanged engine both enforce the contract.
        if mode == "compatible":
            from .compatible_processing import read_compatible_wav
            input_audio, input_info = read_compatible_wav(source)
            input_rate = input_info["sample_rate_hz"]
            waveform = _compatible_waveform
        else:
            input_audio = read_pcm16_wav(source)
            input_rate = SAMPLE_RATE_HZ
            waveform = _waveform
        destination = directory / "result"
        report = runners[mode](source, destination, config)
        output_bytes = (destination / "processed.wav").read_bytes()
        if mode == "compatible":
            output_audio, output_info = read_compatible_wav(destination / "processed.wav")
            if output_info["sample_rate_hz"] != input_rate:
                raise ValueError("processor changed the output sample rate")
        else:
            output_audio = read_pcm16_wav(destination / "processed.wav")
        if output_audio.shape != input_audio.shape:
            raise ValueError("processor changed the number of samples")
        report = dict(report)
        report["local_tool"] = {
            "schema_version": 1,
            "mode": mode,
            "source_kind": source_kind,
            "transport": "loopback_http_only",
            "microphone_capture": False,
            "autoplay": False,
            "remote_upload": False,
            "user_initiated_browser_playback_available": True,
            "engine_metadata_scope": "Engine device_io and no-network limitations describe only the offline DSP kernel. This wrapper uses local loopback HTTP and offers browser playback only when the user presses Play; it never starts playback or captures a microphone itself.",
            "temporary_files_only": True,
            "temporary_files_removed_before_response": True,
            "waveform_scale": "common_pcm_full_scale_not_independently_normalized",
            "comparison_notice": "Playback and file statistics do not establish perceptual improvement. Desired content at the selected frequency is also attenuated.",
        }
        if mode == "compatible":
            report["local_tool"]["waveform_channel_policy"] = "per_time_bin_extrema_across_channels_no_signed_downmix"
        report_bytes = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8")
        archive = io.BytesIO()
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zipped:
            zipped.writestr("processed.wav", output_bytes)
            zipped.writestr("report.json", report_bytes)
            for filename in ("tracking.json", "guard.json", "harmonics.json", "bands.json", "compatible.json", "profile_trace.json"):
                path = destination / filename
                if path.exists():
                    zipped.writestr(filename, path.read_bytes())
        response = {
            "schema_version": 1,
            "mode": mode,
            "source_kind": source_kind,
            "duration_seconds": int(input_audio.shape[0]) / input_rate,
            "input_wav_base64": base64.b64encode(source_bytes).decode("ascii"),
            "output_wav_base64": base64.b64encode(output_bytes).decode("ascii"),
            "download_zip_base64": base64.b64encode(archive.getvalue()).decode("ascii"),
            "report": report,
            "waveforms": {"input": waveform(input_audio), "output": waveform(output_audio)},
        }
        # Serialize before cleanup so numerical/serialization faults fail closed.
        result = json.dumps(response, ensure_ascii=False, allow_nan=False).encode("utf-8")
    return result


class _LocalHandler(BaseHTTPRequestHandler):
    server_version = "LocalHumDSP"
    sys_version = ""
    protocol_version = "HTTP/1.0"

    def setup(self):
        super().setup()
        self.connection.settimeout(_BODY_TIMEOUT_SECONDS)
        self._body_bytes_consumed = 0

    def log_message(self, format, *args):
        # Avoid filenames, user paths, tokens or query contents in console logs.
        return

    def parse_request(self):
        if not super().parse_request():
            return False
        # Even unsupported HTTP methods must not bypass the loopback checks.
        try:
            self._check_local()
        except _RequestFailure as exc:
            self._error(exc.status, exc.message)
            return False
        return True

    def _send(self, status, body, content_type):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Content-Security-Policy", _CSP)
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError, socket.timeout):
            pass

    def _error(self, status, message):
        self._discard_small_rejected_body()
        self._send(status, json.dumps({"error": message}, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

    def _discard_small_rejected_body(self):
        """Bounded transport cleanup, never authorization or audio processing.

        HTTP/1.0 closes after each response. Closing with a small unread POST
        body can reset the connection on Windows while the client reads our
        JSON rejection. Only one unambiguous, fixed-length body up to 64 KiB is
        discarded, within one total 200 ms deadline. Large, chunked, ambiguous
        or malformed lengths are never read here. Bytes already read by the
        authorized upload path are not read a second time. Nothing is stored.
        """
        headers = getattr(self, "headers", None)
        if getattr(self, "command", None) != "POST" or headers is None:
            return
        if headers.get_all("Transfer-Encoding", []):
            return
        lengths = headers.get_all("Content-Length", [])
        if len(lengths) != 1 or re.fullmatch(r"[0-9]+", lengths[0]) is None:
            return
        digits = lengths[0].lstrip("0") or "0"
        if len(digits) > len(str(_REJECTED_BODY_MAX_BYTES)):
            return
        length = int(digits)
        consumed = self._body_bytes_consumed
        if (length > _REJECTED_BODY_MAX_BYTES or type(consumed) is not int
                or not 0 <= consumed <= length):
            return
        remaining = length - consumed
        if not remaining:
            return
        deadline = monotonic() + _REJECTED_BODY_DRAIN_SECONDS
        previous_timeout = self.connection.gettimeout()
        try:
            while remaining:
                timeout = deadline - monotonic()
                if timeout <= 0:
                    break
                self.connection.settimeout(timeout)
                chunk = self.rfile.read1(min(8192, remaining))
                if not chunk:
                    break
                self._body_bytes_consumed += len(chunk)
                remaining -= len(chunk)
        except (socket.timeout, TimeoutError, OSError):
            # Incomplete/slow discarded input must not hide the original error.
            pass
        finally:
            try:
                self.connection.settimeout(previous_timeout)
            except OSError:
                pass  # The peer may already have closed this rejected request.

    def send_error(self, code, message=None, explain=None):
        # Base-class parse/unsupported-method errors must not echo request paths
        # or fall back to a different HTML/CSP response contract.
        self._error(code, "요청을 처리할 수 없습니다. 로컬 화면에서 다시 실행해 주세요.")

    def _one_header(self, name, *, required=False):
        values = self.headers.get_all(name, [])
        if len(values) > 1 or (required and len(values) != 1):
            raise _RequestFailure(403, "요청 헤더를 확인하고 로컬 화면에서 다시 실행해 주세요.")
        return values[0] if values else None

    def _check_local(self):
        if self._one_header("Host", required=True) != self.server.expected_host:
            raise _RequestFailure(403, "출력된 127.0.0.1 로컬 주소로 접속해 주세요.")
        if (self._one_header("Sec-Fetch-Site") or "").lower() == "cross-site":
            raise _RequestFailure(403, "다른 사이트에서 보낸 요청은 허용하지 않습니다.")
        origin = self._one_header("Origin")
        if origin is not None and origin != self.server.origin:
            raise _RequestFailure(403, "동일한 로컬 화면에서 다시 실행해 주세요.")
        try:
            parsed = urlsplit(self.path)
        except ValueError as exc:
            raise _RequestFailure(400, "요청 주소가 올바르지 않습니다.") from exc
        if parsed.scheme or parsed.netloc or parsed.fragment or not parsed.path.startswith("/"):
            raise _RequestFailure(400, "요청 주소가 올바르지 않습니다.")
        return parsed

    def _query(self, parsed):
        try:
            fields = parse_qs(parsed.query, keep_blank_values=True, strict_parsing=True, max_num_fields=7)
            if not {"mode", "hum_hz"}.issubset(fields) or any(len(value) != 1 for value in fields.values()):
                raise ValueError("query keys")
            mode = fields["mode"][0]
            if mode == "profile":
                from .profile_processing import config_from_profile_query
                try:
                    return mode, config_from_profile_query(fields)
                except (ValueError, OverflowError) as exc:
                    raise _RequestFailure(400, "프로필 모드는 같은 파일의 2초 이상 잡음 전용 구간, 명시적 확인, 50/60 Hz, 0~12 dB, 정렬된 1~32차 대역 설정이 필요합니다. 구간은 정확한 샘플 경계여야 합니다.") from exc
            if mode in ("bands", "compatible"):
                from .band_control import BandControlConfig
                if set(fields) != {"mode", "hum_hz", "band_depths_db"} or fields["hum_hz"][0] not in ("50", "60"):
                    raise ValueError("per-band query keys")
                tokens = fields["band_depths_db"][0].split(",")
                if len(tokens) != 3 or any(not value or value.strip() != value for value in tokens):
                    raise ValueError("three per-band depths required")
                return mode, BandControlConfig(nominal_hum_hz=float(fields["hum_hz"][0]), band_depths_db=tuple(map(float, tokens)))
            keys = {"mode", "hum_hz", "attenuation_db", "orders"} if mode == "harmonics" else {"mode", "hum_hz", "attenuation_db"}
            if set(fields) != keys or mode not in ("fixed", "tracked", "guarded", "harmonics") or fields["hum_hz"][0] not in ("50", "60"):
                raise ValueError("mode or frequency")
            depth = float(fields["attenuation_db"][0])
            if mode == "harmonics":
                from .harmonics import HarmonicConfig
                tokens = fields["orders"][0].split(",")
                if not 1 <= len(tokens) <= 6 or any(re.fullmatch(r"[1-6]", value) is None for value in tokens):
                    raise ValueError("orders")
                config = HarmonicConfig(nominal_hum_hz=float(fields["hum_hz"][0]), max_attenuation_db=depth, orders=tuple(map(int, tokens)))
            else:
                config = ProcessorConfig(nominal_hum_hz=float(fields["hum_hz"][0]), max_attenuation_db=depth)
            return mode, config
        except (ValueError, OverflowError) as exc:
            raise _RequestFailure(400, "모드, 50/60 Hz 주파수와 0~12 dB 감쇠 설정을 확인해 주세요. 대역별 조절은 세 개의 감쇠값이 필요합니다.") from exc

    def do_GET(self):
        try:
            parsed = self._check_local()
            if parsed.path.startswith("/api/long/assets/"):
                from .long_local import serve_asset
                serve_asset(self, parsed)
                return
            if parsed.query:
                raise _RequestFailure(400, "이 주소에는 추가 설정을 넣을 수 없습니다.")
            if parsed.path == "/api/health":
                self._send(200, b'{"status":"ready","loopback_only":true}', "application/json; charset=utf-8")
                return
            if parsed.path == "/api/capabilities":
                from .desktop_control import capabilities
                payload = json.dumps(capabilities(self.server), ensure_ascii=False, allow_nan=False).encode("utf-8")
                self._send(200, payload, "application/json; charset=utf-8")
                return
            assets = {"/": ("index.html", "text/html; charset=utf-8"), "/app.js": ("app.js", "text/javascript; charset=utf-8"), "/ab-review.js": ("ab-review.js", "text/javascript; charset=utf-8"), "/preflight.js": ("preflight.js", "text/javascript; charset=utf-8"), "/style.css": ("style.css", "text/css; charset=utf-8")}
            if parsed.path not in assets:
                raise _RequestFailure(404, "이 로컬 도구에서 제공하지 않는 주소입니다.")
            filename, content_type = assets[parsed.path]
            try:
                content = (_WEB_ROOT / filename).read_bytes()
            except OSError as exc:
                raise _RequestFailure(500, "화면 파일이 없습니다. 패키지를 다시 설치해 주세요.") from exc
            if filename == "index.html":
                content = content.replace(b"__SESSION_TOKEN__", self.server.session_token.encode("ascii"))
            self._send(200, content, content_type)
        except _RequestFailure as exc:
            self._error(exc.status, exc.message)

    def do_POST(self):
        locked = False
        try:
            parsed = self._check_local()
            from .desktop_control import ensure_accepting_jobs, request_stop
            if parsed.path == "/api/desktop/stop":
                request_stop(self, parsed)
                return
            ensure_accepting_jobs(self.server)
            if parsed.path.startswith("/api/persistent/"):
                from .persistent_local import process_request
                process_request(self, parsed)
                return
            if parsed.path.startswith("/api/compressed/"):
                if getattr(self.server, "portable_wav_only", False):
                    raise _RequestFailure(415, "이 독립 실행판은 WAV 전용입니다. MP3·M4A 디코더는 포함하지 않았습니다.")
                from .compressed_local import process_request
                process_request(self, parsed)
                return
            if parsed.path.startswith(("/api/long/", "/api/precision/")):
                from .long_local import process_request
                process_request(self, parsed)
                return
            if parsed.path not in ("/api/process", "/api/example"):
                raise _RequestFailure(404, "이 로컬 도구에서 제공하지 않는 주소입니다.")
            token = self._one_header("X-Hum-Token", required=True)
            if not hmac.compare_digest(token.encode("utf-8"), self.server.session_token.encode("ascii")):
                raise _RequestFailure(403, "로컬 화면을 새로 열고 다시 실행해 주세요.")
            if self.headers.get_all("Transfer-Encoding", []):
                raise _RequestFailure(400, "고정 길이 WAV 파일 요청만 지원합니다.")
            lengths = self.headers.get_all("Content-Length", [])
            if len(lengths) != 1:
                raise _RequestFailure(400, "파일 크기 정보는 한 번만 지정해야 합니다.")
            length_text = lengths[0]
            if re.fullmatch(r"[0-9]+", length_text) is None:
                raise _RequestFailure(400, "파일 크기 정보가 올바르지 않습니다.")
            if len(length_text.lstrip("0")) > 8:
                raise _RequestFailure(413, "WAV 파일은 최대 32 MiB까지 지원합니다.")
            length = int(length_text.lstrip("0") or "0")
            if length > MAX_FILE_BYTES:
                raise _RequestFailure(413, "WAV 파일은 최대 32 MiB까지 지원합니다.")
            mode, config = self._query(parsed)
            example = parsed.path == "/api/example"
            if example:
                if length != 0:
                    raise _RequestFailure(400, "예제 실행에는 파일을 보내지 마세요.")
            else:
                content_type = self._one_header("Content-Type", required=True)
                if content_type not in ("audio/wav", "application/octet-stream"):
                    raise _RequestFailure(400, "PCM16 WAV 파일을 선택해 주세요.")
                if length <= 0:
                    raise _RequestFailure(400, "비어 있지 않은 WAV 파일을 선택해 주세요.")
            locked = self.server.processing_lock.acquire(blocking=False)
            if not locked:
                raise _RequestFailure(503, "다른 파일을 처리하고 있습니다. 완료 후 다시 실행해 주세요.")
            ensure_accepting_jobs(self.server)
            if example:
                if mode == "profile":
                    from .profile_processing import profile_example_wav
                    body = profile_example_wav(config)
                else:
                    body = _compatible_example_wav(config) if mode == "compatible" else _example_wav(config)
            else:
                try:
                    deadline = monotonic() + _BODY_TIMEOUT_SECONDS
                    chunks = []
                    remaining = length
                    while remaining:
                        timeout = deadline - monotonic()
                        if timeout <= 0:
                            raise TimeoutError("body deadline")
                        self.connection.settimeout(timeout)
                        chunk = self.rfile.read1(min(65536, remaining))
                        if not chunk:
                            break
                        self._body_bytes_consumed += len(chunk)
                        chunks.append(chunk)
                        remaining -= len(chunk)
                    body = b"".join(chunks)
                except (socket.timeout, TimeoutError, OSError) as exc:
                    raise _RequestFailure(400, "파일 전송 시간이 초과되었습니다. 파일을 다시 선택해 주세요.") from exc
                if len(body) != length:
                    raise _RequestFailure(400, "파일 전송이 완료되지 않았습니다. 다시 실행해 주세요.")
            try:
                payload = _run_job(self.server, body, mode, config, "owned_synthetic_example" if example else "user_file")
            except (ValueError, wave.Error, EOFError) as exc:
                message = ("프로필 모드는 48 kHz·모노·PCM16 WAV, 최대 60초만 지원합니다. 지정한 2초 이상 잡음 전용 구간이 이 파일 안에 있는지 확인하세요. 자동 변환·정규화는 하지 않습니다."
                           if mode == "profile" else "처리할 수 없는 WAV 또는 출력 범위입니다. 일반 WAV 모드는 44.1/48 kHz·모노/스테레오·PCM16, 0초 초과~60초를 지원합니다. 변환으로 피크가 커질 수 있으며 자동 정규화·클리핑은 하지 않습니다."
                           if mode == "compatible" else "처리할 수 없는 WAV 또는 출력 범위입니다. 기존 모드는 48 kHz·모노·PCM16, 0초 초과~60초를 지원합니다. 다른 WAV는 일반 WAV 모드를 선택하세요. 자동 변환·정규화는 하지 않습니다.")
                raise _RequestFailure(400, message) from exc
            except Exception as exc:
                raise _RequestFailure(500, "처리 중 내부 오류가 발생했습니다. 성공 결과는 저장하지 않았습니다. 다시 시도하거나 명령줄에서 진단해 주세요.") from exc
            self._send(200, payload, "application/json; charset=utf-8")
        except _RequestFailure as exc:
            self._error(exc.status, exc.message)
        finally:
            if locked:
                self.server.processing_lock.release()

    def _unsupported(self):
        try:
            self._check_local()
            raise _RequestFailure(405, "GET 또는 POST 요청만 지원합니다.")
        except _RequestFailure as exc:
            self._error(exc.status, exc.message)

    def do_HEAD(self):
        try:
            parsed = self._check_local()
            if parsed.path.startswith("/api/long/assets/"):
                from .long_local import serve_asset
                serve_asset(self, parsed, head_only=True)
                return
            raise _RequestFailure(405, "이 주소는 HEAD 요청을 지원하지 않습니다.")
        except _RequestFailure as exc:
            self._error(exc.status, exc.message)

    do_PUT = _unsupported
    do_DELETE = _unsupported
    do_OPTIONS = _unsupported
    do_PATCH = _unsupported


class _LocalServer(ThreadingHTTPServer):
    daemon_threads = True

    def server_close(self):
        try:
            if hasattr(self, "long_jobs"):
                self.long_jobs.clear()
        finally:
            super().server_close()


def make_server(port=0, *, work_root=None):
    """Create a loopback-only server; caller explicitly chooses when to serve."""
    if type(port) is not int or not 0 <= port <= 65535:
        raise ValueError("port must be an integer from 0 to 65535")
    selected_root = Path(__file__).resolve().parents[2] / "outputs" / "local_sessions" if work_root is None else Path(work_root)
    selected_root.mkdir(parents=True, exist_ok=True)
    server = _LocalServer(("127.0.0.1", port), _LocalHandler)
    actual_port = server.server_address[1]
    server.session_token = secrets.token_urlsafe(32)
    server.expected_host = f"127.0.0.1:{actual_port}"
    server.origin = f"http://{server.expected_host}"
    server.work_root = selected_root.resolve()
    server.processing_lock = threading.Lock()
    from .long_local import LongJobStore
    server.long_jobs = LongJobStore(server.work_root)
    # Set only when this updated server is created. An already-running older
    # server may serve new static files but must not advertise missing routes.
    server.persistent_processing_available = True
    return server


def serve_local(port=0, *, work_root=None):
    """Serve until Ctrl+C. No browser, microphone or audio playback is launched."""
    with make_server(port, work_root=work_root) as server:
        print(f"LocalURL: {server.origin}/", flush=True)
        print("로컬 파일 처리 도구입니다. 종료하려면 Ctrl+C를 누르세요.", flush=True)
        try:
            server.serve_forever(poll_interval=0.25)
        except KeyboardInterrupt:
            pass
