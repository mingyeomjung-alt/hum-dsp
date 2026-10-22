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
    assert options[9][1].startswith("Experimental")
    assert re.search(r'<option\b[^>]*\sselected(?:\s|=|>)', modes) is None
    assert options[0][1] == "Fixed filter · steady hum"


def test_version_and_default_file_contract_remain_wav():
    assert "v0.0.15" in HTML
    assert _with_id("audio-file")[0][1]["accept"] == ".wav,audio/wav,audio/x-wav"
    assert 'id="file-type-label">Choose a WAV file' in HTML
    assert 'id="compressed-notice" class="compressed-notice" hidden' in HTML


def test_all_html_ids_are_unique_and_dynamic_caption_targets_exist():
    ids = [attrs["id"] for _, attrs in ELEMENTS.elements if "id" in attrs]
    assert len(ids) == len(set(ids))
    for name in ["file-type-label", "compressed-notice", "input-track-title",
                 "input-track-description", "output-track-title", "input-peak-label",
                 "listen-note"]:
        assert len(_with_id(name)) == 1


@pytest.mark.parametrize("disclosure", [
    "MP3/M4A files must be at most 32 MiB", "Decoded audio must be 44.1/48 kHz mono or stereo",
    "M4A supports AAC only", "ALAC, multiple tracks, album artwork", "Float32 WAV",
    "not reduced to 16-bit or re-encoded as MP3/M4A",
    "A is the decoded input WAV, not the compressed file itself",
    "does not preserve the compressed file bytes or restore the pre-compression signal",
    "nothing is installed automatically or sent to an external conversion service",
    "Codec overshoots and post-processing range violations are rejected",
    "Float32 WAV playback depends on browser support",
])
def test_compressed_scope_is_disclosed_without_audio_improvement_claim(disclosure):
    notice = re.search(r'<div id="compressed-notice".*?</div>', HTML).group(0)
    assert disclosure in notice
    assert "improved audio quality are not guaranteed" in notice


def test_long_precision_and_compressed_share_expiring_result_copy():
    assert "mode === 'long' || mode === 'precision' || mode === 'compressed' || mode === 'persistent'" in SCRIPT
    assert "mode === 'compressed' ? 32 : (usesStoredFiles(mode) ? 128 : 32)" in SCRIPT
    assert "a new Long WAV, High-resolution WAV or MP3/M4A job starts" in HTML
    assert "starting another stored-file job (Long WAV, High-resolution WAV, MP3/M4A or Persistent hum), or before a normal processor shutdown" in SCRIPT
    assert "Temporary copies may remain after forced termination or power loss" in HTML
    assert "Secure erasure is not guaranteed" in HTML


def test_file_selection_updates_extensions_labels_and_keeps_wav_default():
    assert "$('audio-file').accept = mode === 'compressed' ? '.mp3,.m4a,audio/mpeg,audio/mp4,audio/x-m4a' : '.wav,audio/wav,audio/x-wav'" in SCRIPT
    assert "$('file-type-label').textContent = mode === 'compressed' ? 'Choose an MP3/M4A file' : 'Choose a WAV file'" in SCRIPT
    assert "$('compressed-notice').hidden = mode !== 'compressed'" in SCRIPT
    assert "if (mode === 'compressed')" in SCRIPT
    assert r"/\.(mp3|m4a)$/i.test(file.name)" in SCRIPT
    assert "file.size > 0 && file.size <= fileLimitMiB(mode) * 1024 * 1024" in SCRIPT
    # Retain a selected file for explicit mode guidance rather than discarding
    # it merely because the current mode has a lower size/format limit.
    assert "updateFilePreflight();" in SCRIPT
    assert "HumPreflight.assess(fileMetadata,$('mode').value,{compressedAvailable,persistentAvailable})" in SCRIPT
    assert "if (id === 'mode' && selectedFile)" not in SCRIPT
    assert "validateFile(selectedFile,$('mode').value)" in SCRIPT


def test_new_mode_uses_shared_band_controls_without_changing_6_3_0_defaults():
    assert "mode === 'bands' || mode === 'compatible' || (usesStoredFiles(mode) && mode !== 'persistent')" in SCRIPT
    for n, value in [(1, 6), (2, 3), (3, 0)]:
        attrs = _with_id(f"band-depth-{n}")[0][1]
        assert (attrs["min"], attrs["max"], attrs["step"], attrs["value"]) == (
            "0", "12", "1", str(value))
    assert "parameters.set('band_depths_db', bandDepths.join(','))" in SCRIPT
    assert "With all bands at 0 dB, decoded input WAV sample values are preserved" in SCRIPT
    assert "does not preserve the compressed file or restore the pre-compression signal" in SCRIPT


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
    assert "result.duration_seconds > (isPersistent ? 60 : 600)" in SCRIPT
    assert "formatSamples(report.input)" in SCRIPT and "formatSamples(report.output)" in SCRIPT


def test_compressed_players_and_statistics_are_clearly_decoded_not_compressed_source():
    assert "$('input-track-title').textContent = isCompressed ? 'Decoded input' : 'Input'" in SCRIPT
    assert "Float32 WAV decoded from the compressed file" in SCRIPT
    assert "$('output-track-title').textContent = isCompressed ? 'Processed WAV' : 'Output'" in SCRIPT
    assert "$('input-peak-label').textContent = isCompressed ? 'Decoded input peak' : 'Input peak'" in SCRIPT
    assert "Play decoded input WAV manually" in SCRIPT
    assert "A is the decoded input Float32 WAV, not the compressed file; B is also Float32 WAV" in SCRIPT


def test_builtin_mp3_example_and_optional_local_decoder_are_disclosed():
    assert "A 6-second, 48 kHz stereo synthetic signal is locally encoded to MP3 and decoded for this example" in SCRIPT
    assert "It requires PyAV 19.x, which is not installed automatically" in SCRIPT
    assert "The local environment must include a PyAV 19.x decoder" in HTML
    assert "up to 10 minutes or 128 MiB, whichever comes first" in HTML
    assert "decoded audio is limited to 10 minutes or 128 MiB, whichever comes first" in SCRIPT
    assert "response.ok" in SCRIPT
    assert "typeof result.error === 'string' ? result.error" in SCRIPT
    assert "pip install" not in SCRIPT and "cdn." not in SCRIPT


def test_existing_bounded_wait_offline_and_expiry_behavior_is_preserved():
    assert "isStoredFile ? 900000 : 120000" in SCRIPT
    assert "Could not connect to the local processor" in SCRIPT
    assert "Server processing may still be running" in SCRIPT
    assert "playback does not start automatically when processing finishes" in SCRIPT
    assert "Settings changed. Process again with the new settings" in SCRIPT


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
    assert "Files are not sent to external servers" in HTML


def test_new_notice_matches_existing_design_and_accessible_disclosures():
    assert ".long-notice,.precision-notice,.compressed-notice" in CSS
    assert ".precision-disclosure,.compressed-disclosure" in CSS
    assert ".file-label b{font-weight:inherit}" in CSS
    assert '<details class="compressed-disclosure"><summary>Decoding, 0 dB and temporary storage</summary>' in HTML
    assert "summary:focus-visible" in CSS
    assert "[hidden]{display:none!important}" in CSS
