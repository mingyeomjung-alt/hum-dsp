"""Static contracts for the optional local compressed-audio browser mode.

These checks do not open sockets, decode audio, play sound, or publish assets.
HTTP/decode behavior is verified separately by the integration suite.
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
    return [(tag, attrs) for tag, attrs in ELEMENTS.elements if attrs.get("id") == name]


def test_ninth_mode_preserves_previous_eight_modes_and_fixed_default():
    modes = re.search(r'<select id="mode" name="mode"[^>]*>(.*?)</select>', HTML).group(1)
    options = re.findall(r'<option value="([^"]+)"[^>]*>([^<]+)</option>', modes)
    assert [name for name, _ in options] == [
        "fixed", "tracked", "guarded", "harmonics", "bands", "compatible", "long",
        "precision", "compressed", "profile",
    ]
    assert options[8] == ("compressed", "MP3·M4A → WAV")
    assert options[9][1].startswith("실험")
    assert " selected" not in modes
    assert options[0][1] == "고정 필터 · 일정한 잡음"


def test_version_and_default_file_contract_remain_wav():
    assert "v0.0.14" in HTML
    assert _with_id("audio-file")[0][1]["accept"] == ".wav,audio/wav,audio/x-wav"
    assert 'id="file-type-label">WAV 파일 선택' in HTML
    assert 'id="compressed-notice" class="compressed-notice" hidden' in HTML


def test_all_html_ids_are_unique_and_dynamic_caption_targets_exist():
    ids = [attrs["id"] for _, attrs in ELEMENTS.elements if "id" in attrs]
    assert len(ids) == len(set(ids))
    for name in ["file-type-label", "compressed-notice", "input-track-title",
                 "input-track-description", "output-track-title", "input-peak-label",
                 "listen-note"]:
        assert len(_with_id(name)) == 1


@pytest.mark.parametrize("disclosure", [
    "MP3·M4A 파일은 최대 32MiB", "디코딩된 오디오는 44.1/48kHz·모노/스테레오",
    "M4A는 AAC만 지원", "ALAC·여러 트랙·앨범 아트", "Float32 WAV",
    "16비트로 낮추거나 MP3·M4A로 다시 인코딩하지 않습니다",
    "A는 압축 파일 자체가 아닌 디코딩 원본 WAV",
    "압축 파일 바이트 보존이나 압축 전 원음 복원을 뜻하지 않으며",
    "자동 설치하거나 외부 변환 서버로 보내지 않습니다",
    "코덱에 따른 피크 초과나 처리 후 범위 초과는 거절",
    "Float32 WAV 재생은 브라우저 지원에 따라 다릅니다",
])
def test_compressed_scope_is_disclosed_without_audio_improvement_claim(disclosure):
    notice = re.search(r'<div id="compressed-notice".*?</div>', HTML).group(0)
    assert disclosure in notice
    assert "음질 개선은 보장하지 않습니다" in notice


def test_long_precision_and_compressed_share_expiring_result_copy():
    assert "mode === 'long' || mode === 'precision' || mode === 'compressed'" in SCRIPT
    assert "mode === 'compressed' ? 32 : (usesStoredFiles(mode) ? 128 : 32)" in SCRIPT
    assert "새 긴 WAV·고해상도 WAV·MP3/M4A 작업 시작" in HTML
    assert "새 긴 WAV·고해상도 WAV·MP3/M4A 작업이나 처리기 정상 종료" in SCRIPT
    assert "강제 종료·전원 중단 시 임시 사본이 남을 수" in HTML
    assert "보안 삭제를 보장하지 않습니다" in HTML


def test_file_selection_updates_extensions_labels_and_keeps_wav_default():
    assert "$('audio-file').accept = mode === 'compressed' ? '.mp3,.m4a,audio/mpeg,audio/mp4,audio/x-m4a' : '.wav,audio/wav,audio/x-wav'" in SCRIPT
    assert "$('file-type-label').textContent = mode === 'compressed' ? 'MP3·M4A 파일 선택' : 'WAV 파일 선택'" in SCRIPT
    assert "$('compressed-notice').hidden = mode !== 'compressed'" in SCRIPT
    assert "if (mode === 'compressed')" in SCRIPT
    assert r"/\.(mp3|m4a)$/i.test(file.name)" in SCRIPT
    assert "file.size > 0 && file.size <= fileLimitMiB(mode) * 1024 * 1024" in SCRIPT
    assert "if (id === 'mode' && selectedFile)" in SCRIPT
    assert "validateFile(selectedFile, $('mode').value)" in SCRIPT


def test_new_mode_uses_shared_band_controls_without_changing_6_3_0_defaults():
    assert "mode === 'bands' || mode === 'compatible' || usesStoredFiles(mode)" in SCRIPT
    for n, value in [(1, 6), (2, 3), (3, 0)]:
        attrs = _with_id(f"band-depth-{n}")[0][1]
        assert (attrs["min"], attrs["max"], attrs["step"], attrs["value"]) == (
            "0", "12", "1", str(value))
    assert "parameters.set('band_depths_db', bandDepths.join(','))" in SCRIPT
    assert "모두 0 dB이면 디코딩 원본 WAV의 오디오 데이터 값을 유지" in SCRIPT
    assert "압축 파일 보존·압축 전 원음 복원이 아니며" in SCRIPT


def test_source_header_is_extension_derived_for_uploads_not_example():
    assert "const isCompressed = mode === 'compressed'" in SCRIPT
    assert "'Content-Type': isCompressed ? 'application/octet-stream' : 'audio/wav'" in SCRIPT
    assert "'X-Hum-Token': token" in SCRIPT
    assert "if (isCompressed && !isExample) requestHeaders['X-Hum-Source-Format'] = /\\.mp3$/i.test(selectedFile.name) ? 'mp3' : 'm4a'" in SCRIPT
    assert "`/api/${isStoredFile ? mode + '/' : ''}${isExample ? 'example' : 'process'}?${parameters}`" in SCRIPT
    assert "body: isExample ? new Uint8Array(0) : selectedFile" in SCRIPT
    assert "credentials: 'same-origin', cache: 'no-store'" in SCRIPT


def test_stored_asset_urls_are_same_origin_validated_and_no_inline_audio():
    assert "value.startsWith('/api/long/assets/')" in SCRIPT
    assert "url.origin !== window.location.origin" in SCRIPT
    assert "url.search || url.hash || !expectedPath.test" in SCRIPT
    assert "localAssetURL(result.input_wav_url, 'input.wav')" in SCRIPT
    assert "localAssetURL(result.output_wav_url, 'processed.wav')" in SCRIPT
    assert "localAssetURL(result.download_zip_url, 'result.zip')" in SCRIPT


def test_compressed_result_validates_float32_native_rate_channels_and_frames():
    assert "report.input.sample_format !== 'IEEE_FLOAT32_little_endian'" in SCRIPT
    assert "report.output.sample_format !== 'IEEE_FLOAT32_little_endian'" in SCRIPT
    assert "![44100, 48000].includes(inputRate) || outputRate !== inputRate" in SCRIPT
    assert "outputChannels !== inputChannels || report.output.frames !== report.input.frames" in SCRIPT
    assert "result.duration_seconds > 600" in SCRIPT
    assert "formatSamples(report.input)" in SCRIPT and "formatSamples(report.output)" in SCRIPT


def test_compressed_players_and_statistics_are_clearly_decoded_not_compressed_source():
    assert "$('input-track-title').textContent = isCompressed ? '디코딩 원본' : '원본'" in SCRIPT
    assert "압축 파일에서 디코딩한 Float32 WAV" in SCRIPT
    assert "$('output-track-title').textContent = isCompressed ? '처리 WAV' : '처리본'" in SCRIPT
    assert "$('input-peak-label').textContent = isCompressed ? '디코딩 원본 피크' : '원본 피크'" in SCRIPT
    assert "디코딩 원본 WAV 수동 재생" in SCRIPT
    assert "A는 압축 파일 자체가 아닌 디코딩 원본 Float32 WAV이며 B도 Float32 WAV" in SCRIPT


def test_builtin_mp3_example_and_optional_local_decoder_are_disclosed():
    assert "6초·48kHz·스테레오 합성음을 로컬에서 MP3로 인코딩·디코딩한 예제" in SCRIPT
    assert "PyAV 19.x가 없으면 실행되지 않으며 자동 설치하지 않습니다" in SCRIPT
    assert "PyAV 19.x 디코더가 로컬 환경에 있어야 합니다" in HTML
    assert "최대 10분과 128MiB 중 먼저 닿는 한도" in HTML
    assert "디코딩 후 최대 10분 또는 128MiB 중 먼저 닿는 한도" in SCRIPT
    assert "response.ok" in SCRIPT
    assert "typeof result.error === 'string' ? result.error" in SCRIPT
    assert "pip install" not in SCRIPT and "cdn." not in SCRIPT


def test_existing_bounded_wait_offline_and_expiry_behavior_is_preserved():
    assert "isStoredFile ? 900000 : 120000" in SCRIPT
    assert "로컬 처리기에 연결하지 못했습니다" in SCRIPT
    assert "서버 처리는 아직 진행 중일 수 있습니다" in SCRIPT
    assert "완료 후에도 자동 재생하지 않습니다" in SCRIPT
    assert "설정이 바뀌었습니다. 새 설정으로 다시 처리해 주세요" in SCRIPT


def test_no_auto_play_microphone_external_script_or_browser_decoder_added():
    assert "autoplay" not in HTML.lower()
    assert len(_with_id("input-audio")) == len(_with_id("output-audio")) == 1
    for name in ["input-audio", "output-audio"]:
        assert _with_id(name)[0][1]["preload"] == "none"
    assert ".play(" not in SCRIPT
    assert "getUserMedia" not in SCRIPT and "AudioContext" not in SCRIPT
    assert "decodeAudioData" not in SCRIPT
    for tag, attrs in ELEMENTS.elements:
        if tag == "script":
            assert attrs.get("src", "").startswith("/")
    assert "외부 서버로 파일을 보내지 않습니다" in HTML


def test_new_notice_matches_existing_design_and_accessible_disclosures():
    assert ".long-notice,.precision-notice,.compressed-notice" in CSS
    assert ".precision-disclosure,.compressed-disclosure" in CSS
    assert ".file-label b{font-weight:inherit}" in CSS
    assert '<details class="compressed-disclosure"><summary>디코딩·0 dB·보관 안내</summary>' in HTML
    assert "summary:focus-visible" in CSS
    assert "[hidden]{display:none!important}" in CSS
