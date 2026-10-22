"""Capability disclosure and authenticated local graceful-stop control.

Stopping requests cleanup; it never reports cleanup completed before the
desktop runtime has drained requests and closed the owned session.
"""

import hmac
import json

from . import __version__


def capabilities(server):
    portable = bool(getattr(server, "portable_wav_only", False))
    if portable:
        compressed = {"available": False, "reason": "portable_wav_only", "required_version": ">=19,<20"}
    else:
        from .compressed_decode import decoder_status
        availability = decoder_status()
        compressed = {key: availability[key] for key in ("available", "reason", "required_version")}
    return {"schema_version": 1, "version": __version__,
            "desktop": {"available": getattr(server, "desktop_shutdown_event", None) is not None,
                        "portable_wav_only": portable},
            "compressed": compressed,
            "device_capture_or_autoplay": False, "remote_upload": False}


def ensure_accepting_jobs(server):
    from .local_app import _RequestFailure
    event = getattr(server, "desktop_shutdown_event", None)
    if event is not None and event.is_set():
        raise _RequestFailure(503, "프로그램을 종료하고 있습니다. 새 작업은 실행하지 않습니다.")


def request_stop(handler, parsed):
    from .local_app import _RequestFailure
    if parsed.query:
        raise _RequestFailure(400, "종료 요청에는 추가 설정을 넣을 수 없습니다.")
    event = getattr(handler.server, "desktop_shutdown_event", None)
    if event is None:
        raise _RequestFailure(404, "이 실행 방식에는 화면 종료 기능이 없습니다.")
    token = handler._one_header("X-Hum-Token", required=True)
    if not hmac.compare_digest(token.encode("utf-8"), handler.server.session_token.encode("ascii")):
        raise _RequestFailure(403, "로컬 화면을 새로 열고 다시 실행해 주세요.")
    if (handler.headers.get_all("Transfer-Encoding", [])
            or handler._one_header("Content-Encoding") is not None
            or handler._one_header("Content-Length", required=True) != "0"
            or handler._one_header("Content-Type", required=True) != "application/json"):
        raise _RequestFailure(400, "종료는 본문이 없는 로컬 JSON 요청만 지원합니다.")
    if not handler.server.processing_lock.acquire(blocking=False):
        raise _RequestFailure(503, "파일을 처리 중입니다. 완료 후 필요한 결과를 저장하고 종료해 주세요.")
    try:
        body = json.dumps({"status": "stopping", "temporary_cleanup_pending": True}).encode("ascii")
        handler._send(200, body, "application/json; charset=utf-8")
        try:
            handler.wfile.flush()
        finally:
            event.set()
    finally:
        handler.server.processing_lock.release()
