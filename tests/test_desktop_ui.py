"""Static contracts for capability-aware private desktop controls.

No sockets, devices, audio playback, or publication are used in these checks.
The desktop lifecycle and HTTP authorization are covered separately.
"""

from html.parser import HTMLParser
from pathlib import Path
import re

import pytest


ASSETS = Path(__file__).resolve().parents[1] / "src" / "hum_dsp" / "web"
HTML = (ASSETS / "index.html").read_text(encoding="utf-8")
SCRIPT = (ASSETS / "app.js").read_text(encoding="utf-8")
CSS = (ASSETS / "style.css").read_text(encoding="utf-8")


class _Elements(HTMLParser):
    def __init__(self):
        super().__init__()
        self.elements = []

    def handle_starttag(self, tag, attrs):
        self.elements.append((tag, dict(attrs)))


ELEMENTS = _Elements()
ELEMENTS.feed(HTML)


def _with_id(name):
    found = [(tag, attrs) for tag, attrs in ELEMENTS.elements if attrs.get("id") == name]
    assert len(found) == 1
    return found[0]


def _function(name, next_marker):
    return SCRIPT.split(f"function {name}(", 1)[1].split(next_marker, 1)[0]


def test_version_and_existing_nine_modes_preserve_fixed_default():
    assert "v0.0.14" in HTML
    modes = [(attrs["value"], attrs) for tag, attrs in ELEMENTS.elements if tag == "option"
             and attrs.get("value") in {"fixed", "tracked", "guarded", "harmonics", "bands",
                                        "compatible", "long", "precision", "compressed"}]
    assert [value for value, _ in modes] == ["fixed", "tracked", "guarded", "harmonics",
                                          "bands", "compatible", "long", "precision", "compressed"]
    assert all("selected" not in attrs for _, attrs in modes)
    assert "disabled" in modes[-1][1]
    assert all("disabled" not in attrs for _, attrs in modes[:-1])
    assert _with_id("audio-file")[1]["accept"] == ".wav,audio/wav,audio/x-wav"


def test_all_ids_are_unique():
    ids = [attrs["id"] for _, attrs in ELEMENTS.elements if "id" in attrs]
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize("name", ["desktop-controls", "desktop-stop-confirm", "desktop-stop-status"])
def test_desktop_controls_are_hidden_until_capability_confirmation(name):
    assert "hidden" in _with_id(name)[1]


def test_capability_fetch_uses_same_origin_get_without_token_or_user_input():
    capability = _function("loadCapabilities", "$('desktop-stop').addEventListener")
    assert "fetch('/api/capabilities', {" in capability
    assert "method: 'GET', credentials: 'same-origin', cache: 'no-store'" in capability
    assert "setTimeout(() => abort.abort(), 10000)" in capability
    assert "'X-Hum-Token'" not in capability and "?" not in capability.split("fetch(", 1)[1].split("});", 1)[0]
    assert "void loadCapabilities();" in SCRIPT


@pytest.mark.parametrize("contract", [
    "value.schema_version !== 1", "value.version !== '0.0.14'",
    "typeof value.desktop?.available !== 'boolean'",
    "typeof value.desktop?.portable_wav_only !== 'boolean'",
    "typeof value.compressed?.available !== 'boolean'",
    "value.compressed.required_version !== '>=19,<20'",
    "value.desktop.portable_wav_only && value.compressed.available",
])
def test_capability_response_is_validated_before_enabling_features(contract):
    assert contract in _function("loadCapabilities", "$('desktop-stop').addEventListener")


def test_capability_failure_is_closed_only_for_optional_features_not_wav():
    capability = _function("loadCapabilities", "$('desktop-stop').addEventListener")
    catch = capability.split("} catch {", 1)[1].split("} finally", 1)[0]
    assert "compressedAvailable = false" in catch
    assert "desktopAvailable = false" in catch
    assert "WAV는 계속 사용할 수 있습니다" in catch
    assert "settings-fields" not in catch
    assert "querySelector('option[value=\"compressed\"]').disabled = !compressedAvailable" in capability
    assert "이 배포판은 WAV 전용입니다. MP3·M4A 기능은 포함하지 않았습니다." in capability
    assert "현재 환경에 지원되는 MP3·M4A 디코더가 없습니다" in capability
    assert "pip install" not in SCRIPT and "location.reload" not in SCRIPT


def test_compressed_run_and_upload_validation_check_capability_again():
    validate = _function("validateFile", "function showError")
    assert "if (!compressedAvailable) return" in validate
    run = _function("run", "$('processor-form').addEventListener")
    assert "if (mode === 'compressed' && !compressedAvailable)" in run
    assert run.index("!compressedAvailable") < run.index("busy = true")


def test_stop_is_an_explicit_two_step_cancelable_local_action():
    tag, stop = _with_id("desktop-stop")
    assert tag == "button" and stop["type"] == "button"
    assert stop["aria-expanded"] == "false" and stop["aria-controls"] == "desktop-stop-confirm"
    assert "프로그램 종료" in HTML and "종료 확인" in HTML and "계속 사용" in HTML
    assert "필요한 결과를 먼저 저장하세요" in HTML
    assert "원본 파일과 직접 저장한 결과는 삭제하지 않습니다" in HTML
    assert "$('desktop-stop-cancel').focus()" in SCRIPT
    assert "event.key === 'Escape'" in SCRIPT
    assert "window.confirm" not in SCRIPT and "window.close" not in SCRIPT


def test_stop_request_has_header_token_empty_body_no_query_or_beacon():
    request = SCRIPT.split("fetch('/api/desktop/stop', {", 1)[1].split("});", 1)[0]
    assert "method: 'POST'" in request
    assert "'Content-Type': 'application/json', 'X-Hum-Token': token" in request
    assert "body: new Uint8Array(0)" in request
    assert "credentials: 'same-origin', cache: 'no-store', signal: abort.signal" in request
    assert "?" not in request and "URLSearchParams" not in request
    assert "sendBeacon" not in SCRIPT
    unload = SCRIPT.split("window.addEventListener('beforeunload'", 1)[1].split("updateSettingsControls();", 1)[0]
    assert "fetch(" not in unload and "desktop/stop" not in unload


def test_busy_processing_disables_stop_and_blocks_both_stop_handlers():
    assert "const interactionLocked = () => busy || stopping || stopRequested" in SCRIPT
    assert "$('desktop-stop').disabled = interactionLocked()" in SCRIPT
    assert SCRIPT.count("if (interactionLocked() || !desktopAvailable) return") == 2
    run = _function("run", "$('processor-form').addEventListener")
    assert "busy = true;\n    dismissStopConfirmation();\n    updateDesktopControls();" in run
    assert "busy = false;\n      updateDesktopControls();" in run


def test_stop_pauses_audio_locks_controls_and_preserves_cleanup_uncertainty():
    action = SCRIPT.split("$('desktop-stop-confirm-button').addEventListener", 1)[1].split("const usesStoredFiles", 1)[0]
    assert "for (const player of audio) { player.pause(); player.controls = false; }" in action
    assert "$('settings-fields').disabled = true" in action
    assert "setAttribute('aria-disabled', 'true')" in action
    assert "if (stopping || stopRequested) event.preventDefault()" in action
    assert "result.status !== 'stopping' || result.temporary_cleanup_pending !== true" in action
    assert "stopRequested = true" in action
    assert "종료를 요청했습니다. 정상 종료 후 임시 사본이 정리됩니다. 이 창을 닫아도 됩니다." in action
    assert "임시 사본의 정리 완료를 이 화면에서 확인하지는 않습니다" in action
    assert "정상 종료 여부와 임시 사본의 정리는 확인되지 않았습니다" in action
    assert "정리가 완료되었습니다" not in action


def test_failed_stop_keeps_acknowledgement_uncertain_and_restores_controls():
    action = SCRIPT.split("$('desktop-stop-confirm-button').addEventListener", 1)[1].split("const usesStoredFiles", 1)[0]
    catch = action.split("} catch {", 1)[1].split("} finally", 1)[0]
    assert "종료 응답을 확인하지 못했습니다" in catch
    assert "$('settings-fields').disabled = false" in catch
    assert "player.controls = true" in catch
    assert "removeAttribute('aria-disabled')" in catch
    assert "stopRequested = true" not in catch


def test_no_capture_playback_installation_external_script_or_network_auto_close():
    assert "autoplay" not in HTML.lower() and ".play(" not in SCRIPT
    for name in ("input-audio", "output-audio"):
        assert _with_id(name)[1]["preload"] == "none"
    for forbidden in ("getUserMedia", "AudioContext", "decodeAudioData", "sendBeacon", "window.close("):
        assert forbidden not in SCRIPT
    for tag, attrs in ELEMENTS.elements:
        if tag == "script":
            assert attrs.get("src", "").startswith("/")
    assert "外部" not in HTML


def test_capability_and_shutdown_ui_preserve_existing_accessible_design():
    assert _with_id("mode")[1]["aria-describedby"] == "mode-help capability-notice"
    assert _with_id("capability-notice")[1]["role"] == "status"
    assert _with_id("desktop-stop-status")[1]["aria-live"] == "polite"
    assert _with_id("desktop-stop-confirm")[1]["aria-labelledby"] == "desktop-stop-title"
    assert "[hidden]{display:none!important}" in CSS
    assert ".topbar-actions" in CSS and ".desktop-stop-confirm" in CSS
    assert "a[aria-disabled=true]" in CSS and "pointer-events:none" in CSS
    assert "button:focus-visible" in CSS and "prefers-reduced-motion" in CSS

