"""Desktop lifecycle tests use fake HTTP servers, never browser/audio devices."""

import json
import os
from pathlib import Path
import stat
import threading
from types import SimpleNamespace

import pytest

from hum_dsp import desktop_runtime as runtime


class FakeServer:
    origin = "http://127.0.0.1:54321"

    def __init__(self, work_root, action=None):
        self.work_root = Path(work_root)
        self.work_root.mkdir()
        (self.work_root / "owned-result.wav").write_bytes(b"owned fixture, not recorded audio")
        self.closed = False
        self.shutdown_called = False
        self.action = action
        self.stop = threading.Event()

    def serve_forever(self, *, poll_interval):
        assert self.daemon_threads is False
        assert self.block_on_close is True
        if self.action is not None:
            self.action(self)
        else:
            self.desktop_shutdown_event.set()
        self.stop.wait(timeout=5)

    def shutdown(self):
        self.shutdown_called = True
        self.stop.set()

    def server_close(self):
        self.closed = True
        assert self.work_root.exists()


@pytest.fixture
def fake_server(monkeypatch):
    servers = []

    def make(work_root):
        server = FakeServer(work_root)
        servers.append(server)
        return server

    monkeypatch.setattr(runtime, "_make_server", make)
    monkeypatch.setattr(runtime.webbrowser, "open", lambda *a, **kw: pytest.fail("must not open a browser"))
    return servers


def _arguments(tmp_path, *additional):
    return ["--no-browser", "--session-root", str(tmp_path), "--state-file", str(tmp_path / "run.json"), *additional]


def test_no_browser_lifecycle_cleans_only_new_child_and_keeps_metadata(tmp_path, fake_server, capsys):
    original = tmp_path / "original.wav"
    original.write_bytes(b"preserve user original")
    other = tmp_path / "session_unrelated"
    other.mkdir()
    (other / "old.txt").write_text("not ours", encoding="utf-8")
    assert runtime.main(_arguments(tmp_path)) == 0
    server, = fake_server
    assert server.closed and server.shutdown_called
    assert not server.work_root.parent.exists()
    assert original.read_bytes() == b"preserve user original"
    assert (other / "old.txt").read_text(encoding="utf-8") == "not ours"
    state = json.loads((tmp_path / "run.json").read_text(encoding="utf-8"))
    closed = json.loads((tmp_path / "run.json.closed.json").read_text(encoding="utf-8"))
    assert state["lifecycle"] == "running"
    assert state["origin"] == server.origin
    assert state["browser_requested"] is False
    assert not any("token" in key for key in state)
    assert state["device_io"] is state["remote_upload"] is False
    assert closed["normal_shutdown"] is True
    assert closed["temporary_results_cleaned"] is True
    assert closed["stop_reason"] == "ui_request"
    assert closed["secure_erasure_claimed"] is False
    output = capsys.readouterr()
    assert f"LocalURL: {server.origin}/" in output.out
    assert "정상 종료" in output.out


def test_owned_marker_stops_without_needing_server_token(tmp_path, monkeypatch):
    def make(work_root):
        return FakeServer(work_root, action=lambda server: (server.work_root.parent / "stop.request").touch())

    monkeypatch.setattr(runtime, "_make_server", make)
    assert runtime.main(_arguments(tmp_path)) == 0
    closed = json.loads((tmp_path / "run.json.closed.json").read_text(encoding="utf-8"))
    assert closed["stop_reason"] == "owned_stop_marker"


@pytest.mark.parametrize("existing", ["run.json", "run.json.closed.json"])
def test_existing_qa_state_is_never_replaced_and_server_never_starts(tmp_path, fake_server, existing):
    target = tmp_path / existing
    target.write_bytes(b"keep me")
    assert runtime.main(_arguments(tmp_path)) == 1
    assert target.read_bytes() == b"keep me"
    assert fake_server == []
    assert not list(tmp_path.glob("session_*"))


def test_atomic_publish_never_overwrites_even_when_target_appears_late(tmp_path):
    target = tmp_path / "state.json"
    target.write_bytes(b"earlier content")
    with pytest.raises(FileExistsError):
        runtime._write_json_exclusive(target, {"new": True})
    assert target.read_bytes() == b"earlier content"
    assert not list(tmp_path.glob(".hum-state-*"))


def test_atomic_publish_writes_complete_utf8_and_no_draft_remains(tmp_path):
    target = tmp_path / "state.json"
    runtime._write_json_exclusive(target, {"message": "정상", "device_io": False})
    assert json.loads(target.read_text(encoding="utf-8")) == {"message": "정상", "device_io": False}
    assert not list(tmp_path.glob(".hum-state-*"))


def test_publish_filesystem_without_link_support_fails_without_partial_output(tmp_path, monkeypatch):
    def refuse(*args, **kwargs):
        raise OSError("hard links unavailable")

    monkeypatch.setattr(runtime.os, "rename" if os.name == "nt" else "link", refuse)
    with pytest.raises(OSError):
        runtime._write_json_exclusive(tmp_path / "state.json", {"ready": True})
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("relative", ["relative", "../outside"])
def test_relative_session_roots_are_refused(relative, tmp_path, fake_server):
    assert runtime.main(["--no-browser", "--session-root", relative]) == 1
    assert fake_server == []


def test_explicit_session_parent_must_already_exist(tmp_path, fake_server):
    missing = tmp_path / "not-created"
    assert runtime.main(["--no-browser", "--session-root", str(missing)]) == 1
    assert not missing.exists()
    assert fake_server == []


def test_default_appdata_creates_only_product_session_parent(tmp_path, fake_server, monkeypatch):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    assert runtime.main(["--no-browser"]) == 0
    base = tmp_path / "Swipsion" / "HumDSP" / "sessions"
    assert base.is_dir() and list(base.iterdir()) == []
    assert len(fake_server) == 1


@pytest.mark.parametrize("frozen", [False, True])
def test_portable_build_capability_flag_is_set_before_serving(tmp_path, fake_server, monkeypatch, frozen):
    monkeypatch.setattr(runtime.sys, "frozen", frozen, raising=False)
    assert runtime.main(_arguments(tmp_path)) == 0
    assert fake_server[0].portable_wav_only is frozen


def test_default_fallback_home_does_not_mutate_home_environment(tmp_path, fake_server, monkeypatch):
    monkeypatch.delenv("LOCALAPPDATA", raising=False)
    monkeypatch.setattr(runtime.Path, "home", classmethod(lambda cls: tmp_path))
    previous_home = os.environ.get("HOME")
    assert runtime.main(["--no-browser"]) == 0
    assert os.environ.get("HOME") == previous_home
    assert (tmp_path / "AppData" / "Local" / "Swipsion" / "HumDSP" / "sessions").is_dir()


@pytest.mark.parametrize("location", ["session_parent", "state_parent"])
def test_reparse_ancestor_is_refused_without_creating_or_deleting_files(tmp_path, fake_server, monkeypatch, location):
    real_lstat = runtime.Path.lstat

    def patched(path):
        info = real_lstat(path)
        if path == tmp_path:
            return SimpleNamespace(st_mode=info.st_mode, st_file_attributes=runtime._REPARSE_POINT)
        return info

    monkeypatch.setattr(runtime.Path, "lstat", patched)
    args = (["--no-browser", "--session-root", str(tmp_path)] if location == "session_parent"
            else ["--no-browser", "--state-file", str(tmp_path / "run.json")])
    assert runtime.main(args) == 1
    assert fake_server == []
    assert list(tmp_path.iterdir()) == []


def test_symlink_classification_and_reparse_classification_are_explicit():
    assert runtime._is_link_or_reparse(SimpleNamespace(st_mode=stat.S_IFLNK))
    assert runtime._is_link_or_reparse(SimpleNamespace(st_mode=stat.S_IFDIR, st_file_attributes=0x400))
    assert not runtime._is_link_or_reparse(SimpleNamespace(st_mode=stat.S_IFDIR, st_file_attributes=0))


@pytest.mark.parametrize("target_kind", ["base", "other_child", "relative"])
def test_cleanup_refuses_broad_or_unowned_paths(tmp_path, target_kind):
    keep = tmp_path / "keep.txt"
    keep.write_text("preserve", encoding="utf-8")
    other = tmp_path / "other"
    other.mkdir()
    target = {"base": tmp_path, "other_child": other, "relative": Path("session_relative")}[target_kind]
    with pytest.raises(ValueError):
        runtime._remove_owned_session(target, tmp_path)
    assert keep.read_text(encoding="utf-8") == "preserve"
    assert other.exists()


def test_cleanup_does_not_follow_replaced_session_junction(tmp_path, monkeypatch):
    session = runtime._create_session(tmp_path)
    keep = session / "keep.txt"
    keep.write_bytes(b"must not recurse")
    real_lstat = runtime.Path.lstat

    def patched(path):
        info = real_lstat(path)
        if path == session:
            return SimpleNamespace(st_mode=info.st_mode, st_file_attributes=0x400)
        return info

    monkeypatch.setattr(runtime.Path, "lstat", patched)
    with pytest.raises(ValueError):
        runtime._remove_owned_session(session, tmp_path)
    assert keep.read_bytes() == b"must not recurse"


def test_browser_false_return_is_nonfatal_and_manual_url_is_printed(tmp_path, fake_server, monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(runtime.webbrowser, "open", lambda url, **kwargs: calls.append((url, kwargs)) or False)
    args = _arguments(tmp_path)[1:]
    assert runtime.main(args) == 0
    assert calls == [("http://127.0.0.1:54321/", {"new": 2})]
    output = capsys.readouterr()
    assert "직접 열어" in output.err
    assert "LocalURL:" in output.out
    assert json.loads((tmp_path / "run.json.closed.json").read_text(encoding="utf-8"))["browser_opened"] is False


def test_browser_exception_is_nonfatal(tmp_path, fake_server, monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("browser unavailable")

    monkeypatch.setattr(runtime.webbrowser, "open", fail)
    assert runtime.main(_arguments(tmp_path)[1:]) == 0


def test_browser_success_is_reported_without_playing_audio(tmp_path, fake_server, monkeypatch):
    monkeypatch.setattr(runtime.webbrowser, "open", lambda url, **kwargs: True)
    assert runtime.main(_arguments(tmp_path)[1:]) == 0
    assert json.loads((tmp_path / "run.json.closed.json").read_text(encoding="utf-8"))["browser_opened"] is True


def test_server_start_failure_cleans_new_session_only(tmp_path, monkeypatch):
    original = tmp_path / "original.wav"
    original.write_bytes(b"preserve")

    def fail(work_root):
        raise OSError("bind failed")

    monkeypatch.setattr(runtime, "_make_server", fail)
    assert runtime.main(_arguments(tmp_path)) == 1
    assert original.read_bytes() == b"preserve"
    assert not list(tmp_path.glob("session_*"))
    assert not (tmp_path / "run.json").exists()


def test_worker_failure_is_not_reported_as_normal_shutdown(tmp_path, monkeypatch):
    class FailingServer(FakeServer):
        def serve_forever(self, **kwargs):
            raise RuntimeError("worker failure")

    monkeypatch.setattr(runtime, "_make_server", FailingServer)
    assert runtime.main(_arguments(tmp_path)) == 1
    closed = json.loads((tmp_path / "run.json.closed.json").read_text(encoding="utf-8"))
    assert closed["normal_shutdown"] is False
    assert closed["temporary_results_cleaned"] is True
    assert closed["error_code"] == "RuntimeError"


def test_request_threads_drain_before_application_clears_files():
    events = []

    class Base:
        def server_close(self):
            events.append("listener_closed")

    class Server(runtime.ThreadingMixIn, Base):
        _threads = SimpleNamespace(join=lambda: events.append("workers_drained"))

        def server_close(self):
            events.append("application_files_cleared")

    runtime._drain_and_close(Server(), None)
    assert events == ["listener_closed", "workers_drained", "application_files_cleared"]


def test_unstarted_worker_does_not_call_shutdown_and_deadlock():
    events = []
    server = SimpleNamespace(shutdown=lambda: pytest.fail("unstarted serve_forever must not be shut down"), server_close=lambda: events.append("closed"))
    worker = threading.Thread(target=lambda: None)
    runtime._drain_and_close(server, worker)
    assert events == ["closed"]


def test_keyboard_interrupt_stops_normally_after_worker_is_drained(tmp_path, fake_server, monkeypatch):
    def make(work_root):
        return FakeServer(work_root, action=lambda server: None)

    monkeypatch.setattr(runtime, "_make_server", make)
    monkeypatch.setattr(runtime.time, "sleep", lambda seconds: (_ for _ in ()).throw(KeyboardInterrupt()))
    assert runtime.main(_arguments(tmp_path)) == 0
    assert json.loads((tmp_path / "run.json.closed.json").read_text(encoding="utf-8"))["stop_reason"] == "keyboard_interrupt"


@pytest.mark.parametrize("marker_kind", ["directory", "nonempty", "reparse"])
def test_stop_marker_must_be_empty_regular_owned_file(tmp_path, marker_kind, monkeypatch):
    marker = tmp_path / "stop.request"
    if marker_kind == "directory":
        marker.mkdir()
    else:
        marker.write_bytes(b"not empty" if marker_kind == "nonempty" else b"")
    if marker_kind == "reparse":
        real_lstat = runtime.Path.lstat
        monkeypatch.setattr(runtime.Path, "lstat", lambda path: SimpleNamespace(st_mode=stat.S_IFREG, st_size=0, st_file_attributes=0x400) if path == marker else real_lstat(path))
    with pytest.raises(ValueError):
        runtime._stop_marker_present(marker)


def test_missing_and_regular_empty_marker(tmp_path):
    marker = tmp_path / "stop.request"
    assert runtime._stop_marker_present(marker) is False
    marker.touch()
    assert runtime._stop_marker_present(marker) is True


@pytest.mark.parametrize("origin", ["https://127.0.0.1:54321", "http://localhost:54321", "http://example.com:54321", "http://127.0.0.1:54321/a", "http://user@127.0.0.1:54321", "http://127.0.0.1:54321?x=1", "http://127.0.0.1:54321#x", "http://127.0.0.1"])
def test_non_loopback_or_ambiguous_origin_is_refused(origin):
    with pytest.raises(ValueError):
        runtime._checked_origin(SimpleNamespace(origin=origin))


def test_successful_origin_validation():
    assert runtime._checked_origin(SimpleNamespace(origin="http://127.0.0.1:54321")) == "http://127.0.0.1:54321"


def test_unsafe_server_origin_is_closed_before_any_browser_or_state(tmp_path, fake_server, monkeypatch):
    def make(work_root):
        server = FakeServer(work_root)
        server.origin = "https://not-local.invalid"
        fake_server.append(server)
        return server

    monkeypatch.setattr(runtime, "_make_server", make)
    assert runtime.main(_arguments(tmp_path)) == 1
    assert fake_server[0].closed
    assert not (tmp_path / "run.json").exists()
    assert not list(tmp_path.glob("session_*"))


def test_state_publish_failure_closes_server_and_cleans_owned_child(tmp_path, fake_server, monkeypatch):
    monkeypatch.setattr(runtime, "_write_json_exclusive", lambda *args: (_ for _ in ()).throw(OSError("write refused")))
    assert runtime.main(_arguments(tmp_path)) == 1
    assert fake_server[0].closed
    assert not list(tmp_path.glob("session_*"))


def test_runtime_source_has_no_audio_device_or_codec_imports():
    import ast
    source = Path(runtime.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = {alias.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    imported.update(node.module.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module)
    assert not imported & {"av", "sounddevice", "pyaudio", "winsound", "pygame", "requests", "urllib3", "httpx"}
