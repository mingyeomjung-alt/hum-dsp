"""Owned local desktop lifecycle; no microphone, playback or remote service.

Only a new, randomly named session directory is disposable.  An interrupted
process may leave that directory behind; this module never sweeps old sessions
or claims secure erasure.  The optional state files are non-secret QA metadata,
not an authentication or control token.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
from socketserver import ThreadingMixIn
import stat
import sys
import tempfile
import threading
import time
from urllib.parse import urlsplit
import webbrowser


_POLL_SECONDS = 0.1
_SESSION_PREFIX = "session_"
_REPARSE_POINT = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _absolute_path(value):
    selected = Path(value)
    if not selected.is_absolute():
        raise ValueError("an absolute path is required")
    # Do not resolve first: doing so would hide a symlink or junction.
    return Path(os.path.abspath(os.fspath(selected)))


def _is_link_or_reparse(info):
    return stat.S_ISLNK(info.st_mode) or bool(
        getattr(info, "st_file_attributes", 0) & _REPARSE_POINT
    )


def _check_existing_chain(path):
    """Refuse links/junctions, including any ancestor of an owned location."""
    selected = _absolute_path(path)
    for component in reversed((selected, *selected.parents)):
        try:
            info = component.lstat()
        except FileNotFoundError:
            continue
        if _is_link_or_reparse(info):
            raise ValueError("symlink and reparse-point locations are not supported")
        if not stat.S_ISDIR(info.st_mode):
            raise ValueError("a directory ancestor is not a directory")
    return selected


def _safe_existing_directory(path):
    selected = _check_existing_chain(path)
    if not selected.is_dir():
        raise ValueError("an existing directory is required")
    return selected


def _ensure_safe_directory(path):
    selected = _absolute_path(path)
    for component in reversed((selected, *selected.parents)):
        if not os.path.lexists(component):
            try:
                component.mkdir()
            except FileExistsError:
                pass
        info = component.lstat()
        if _is_link_or_reparse(info) or not stat.S_ISDIR(info.st_mode):
            raise ValueError("unsafe desktop session directory")
    return _safe_existing_directory(selected)


def _session_base(explicit_root):
    if explicit_root is not None:
        # QA chooses an already existing parent, never a directory to erase.
        return _safe_existing_directory(explicit_root)
    local_data = os.environ.get("LOCALAPPDATA")
    selected = Path(local_data) if local_data else Path.home() / "AppData" / "Local"
    return _ensure_safe_directory(selected / "Swipsion" / "HumDSP" / "sessions")


def _state_paths(value):
    if value is None:
        return None, None
    state_file = _absolute_path(value)
    _safe_existing_directory(state_file.parent)
    closed_file = state_file.with_name(state_file.name + ".closed.json")
    for path in (state_file, closed_file):
        if os.path.lexists(path):
            raise FileExistsError("desktop QA state output already exists")
    return state_file, closed_file


def _write_json_exclusive(path, payload):
    """Publish a complete metadata file atomically without replacing anything.

    Windows rename is atomic and refuses an existing destination.  POSIX needs
    a same-directory hard link for those no-replace semantics.  Unsupported
    filesystems fail instead of silently using a replace operation.  Neither
    the draft nor the published file contains a session token or audio.
    """
    selected = _absolute_path(path)
    _safe_existing_directory(selected.parent)
    draft = None
    try:
        descriptor, filename = tempfile.mkstemp(prefix=".hum-state-", suffix=".tmp", dir=selected.parent)
        draft = Path(filename)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        _safe_existing_directory(selected.parent)
        if os.name == "nt":
            os.rename(draft, selected)
        else:
            os.link(draft, selected, follow_symlinks=False)
    finally:
        if draft is not None:
            draft.unlink(missing_ok=True)


def _create_session(base):
    checked_base = _safe_existing_directory(base)
    selected = Path(tempfile.mkdtemp(prefix=_SESSION_PREFIX, dir=checked_base))
    _safe_existing_directory(selected)
    if selected.parent != checked_base:
        raise ValueError("session is outside its owned parent")
    return selected


def _remove_owned_session(session, base):
    selected, checked_base = _absolute_path(session), _safe_existing_directory(base)
    if selected.parent != checked_base or not selected.name.startswith(_SESSION_PREFIX):
        raise ValueError("refusing to remove a non-session location")
    if not os.path.lexists(selected):
        return True
    _safe_existing_directory(selected)
    # This is the exact unique child made by _create_session, never its parent.
    shutil.rmtree(selected)
    return not os.path.lexists(selected)


def _stop_marker_present(marker):
    try:
        info = marker.lstat()
    except FileNotFoundError:
        return False
    if _is_link_or_reparse(info) or not stat.S_ISREG(info.st_mode) or info.st_size != 0:
        raise ValueError("desktop stop marker must be a new empty regular file")
    return True


def _make_server(work_root):
    # In particular, no optional compressed decoder is imported by this module.
    from .local_app import make_server
    return make_server(work_root=work_root)


def _checked_origin(server):
    parsed = urlsplit(server.origin)
    if (parsed.scheme != "http" or parsed.hostname != "127.0.0.1" or
            parsed.username is not None or parsed.password is not None or
            parsed.path or parsed.query or parsed.fragment or parsed.port is None):
        raise ValueError("desktop runtime requires an IPv4 loopback HTTP origin")
    return server.origin


def _drain_and_close(server, worker):
    """Drain HTTP workers before LongJobStore cleanup, including downloads."""
    if worker is not None and worker.ident is not None:
        server.shutdown()
        worker.join()
    try:
        if isinstance(server, ThreadingMixIn):
            # The ordinary local server is daemon-threaded.  Desktop runtime
            # switches it to tracked, non-daemon request workers before serving.
            # Call the mixin's close first so server.server_close cannot clear
            # temporary files while a request still uses them.
            ThreadingMixIn.server_close(server)
    finally:
        server.server_close()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Hum DSP 로컬 데스크톱 실행기")
    parser.add_argument("--no-browser", action="store_true", help="자동으로 브라우저를 열지 않습니다(QA).")
    parser.add_argument("--session-root", type=Path, help="기존 절대 경로 아래에 새 임시 세션만 만듭니다(QA).")
    parser.add_argument("--state-file", type=Path, help="새 절대 경로에 비밀정보 없는 상태 JSON을 저장합니다(QA).")
    args = parser.parse_args(argv)
    state_file = closed_file = base = session = server = worker = None
    started_at = _utc_now()
    stop_reason = None
    browser_opened = False
    normal_shutdown = False
    temporary_results_cleaned = False
    error_code = None
    published_state = False
    worker_errors = []
    try:
        state_file, closed_file = _state_paths(args.state_file)
        base = _session_base(args.session_root)
        session = _create_session(base)
        marker = session / "stop.request"
        server = _make_server(session / "work")
        origin = _checked_origin(server)
        server.portable_wav_only = bool(getattr(sys, "frozen", False))
        server.desktop_shutdown_event = threading.Event()
        server.daemon_threads = False
        server.block_on_close = True

        def serve():
            try:
                server.serve_forever(poll_interval=_POLL_SECONDS)
            except BaseException as exc:
                worker_errors.append(exc)

        if state_file is not None:
            _write_json_exclusive(state_file, {
                "schema_version": 1, "lifecycle": "running", "pid": os.getpid(),
                "session_id": session.name, "started_at": started_at,
                "origin": origin, "session_directory": str(session), "stop_marker": str(marker),
                "browser_requested": not args.no_browser, "device_io": False,
                "remote_upload": False, "secure_erasure_claimed": False,
                "abnormal_termination_may_leave_temporary_files": True,
            })
            published_state = True
        worker = threading.Thread(target=serve, name="HumDSP-local-http", daemon=True)
        worker.start()
        print(f"LocalURL: {origin}/", flush=True)
        print("로컬 파일 처리만 수행합니다. 화면의 프로그램 종료 버튼 또는 Ctrl+C로 종료하세요.", flush=True)
        if not args.no_browser:
            try:
                browser_opened = bool(webbrowser.open(origin + "/", new=2))
            except Exception:
                browser_opened = False
            if not browser_opened:
                print("브라우저 자동 열기에 실패했습니다. 위 로컬 주소를 직접 열어 주세요.", file=sys.stderr, flush=True)
        try:
            while True:
                if server.desktop_shutdown_event.is_set():
                    stop_reason = "ui_request"
                    break
                if _stop_marker_present(marker):
                    server.desktop_shutdown_event.set()
                    stop_reason = "owned_stop_marker"
                    break
                if not worker.is_alive():
                    raise RuntimeError("local HTTP worker stopped unexpectedly")
                time.sleep(_POLL_SECONDS)
        except KeyboardInterrupt:
            server.desktop_shutdown_event.set()
            stop_reason = "keyboard_interrupt"
        normal_shutdown = not worker_errors
    except Exception as exc:
        error_code = type(exc).__name__
        print(f"프로그램을 시작하거나 종료하지 못했습니다({error_code}). 원본 파일은 변경하지 않습니다.", file=sys.stderr, flush=True)
    finally:
        if server is not None:
            try:
                if hasattr(server, "desktop_shutdown_event"):
                    server.desktop_shutdown_event.set()
                _drain_and_close(server, worker)
                if worker_errors:
                    error_code = error_code or type(worker_errors[0]).__name__
                    normal_shutdown = False
            except Exception as exc:
                error_code = error_code or type(exc).__name__
                normal_shutdown = False
        if session is not None and base is not None:
            try:
                temporary_results_cleaned = _remove_owned_session(session, base)
            except Exception as exc:
                error_code = error_code or type(exc).__name__
        if published_state and closed_file is not None:
            try:
                _write_json_exclusive(closed_file, {
                    "schema_version": 1, "lifecycle": "closed" if error_code is None else "failed",
                    "pid": os.getpid(), "session_id": session.name, "started_at": started_at,
                    "closed_at": _utc_now(), "stop_reason": stop_reason,
                    "normal_shutdown": normal_shutdown and error_code is None,
                    "temporary_results_cleaned": temporary_results_cleaned,
                    "browser_opened": browser_opened, "error_code": error_code,
                    "device_io": False, "remote_upload": False, "secure_erasure_claimed": False,
                })
            except Exception as exc:
                error_code = error_code or type(exc).__name__
                normal_shutdown = False
        if error_code is None and normal_shutdown and temporary_results_cleaned:
            print("프로그램을 정상 종료하고 이번 세션의 임시 파일을 정리했습니다. 보안 삭제를 의미하지 않습니다.", flush=True)
        elif session is not None:
            print("정상 종료 확인이 완료되지 않았습니다. 강제 종료 시 임시 파일이 남을 수 있습니다.", file=sys.stderr, flush=True)
    return 0 if error_code is None and normal_shutdown and temporary_results_cleaned else 1


if __name__ == "__main__":
    raise SystemExit(main())
