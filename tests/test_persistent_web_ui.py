"""Static and executable preflight contracts for explicit persistent mode."""

from html.parser import HTMLParser
import json
from pathlib import Path
import re
import shutil
import subprocess

import pytest


ASSETS = Path(__file__).resolve().parents[1] / "src" / "hum_dsp" / "web"
HTML = (ASSETS / "index.html").read_text(encoding="utf-8")
APP = (ASSETS / "app.js").read_text(encoding="utf-8")
PREFLIGHT = ASSETS / "preflight.js"
NODE = shutil.which("node")


class Elements(HTMLParser):
    def __init__(self):
        super().__init__()
        self.items = []

    def handle_starttag(self, tag, attrs):
        self.items.append((tag, dict(attrs)))


PARSED = Elements()
PARSED.feed(HTML)


def _one(element_id):
    found = [(tag, attrs) for tag, attrs in PARSED.items if attrs.get("id") == element_id]
    assert len(found) == 1
    return found[0]


def test_persistent_mode_is_disabled_until_capability_and_has_separate_controls():
    option = _one("persistent-option")[1]
    assert option["value"] == "persistent" and "disabled" in option
    depth = _one("persistent-depth")[1]
    assert depth == {"id": "persistent-depth", "type": "range", "min": "0", "max": "18",
                     "step": "1", "value": "12", "aria-describedby": "persistent-depth-help"}
    assert "value=\"fixed\"" in HTML.split("<select id=\"mode\"", 1)[1].split("</select>", 1)[0]
    select_markup = HTML.split("<select id=\"mode\"", 1)[1].split("</select>", 1)[0]
    assert re.search(r'<option\b[^>]*\sselected(?:\s|=|>)', select_markup) is None
    assert "persistentAvailable = value.persistent?.available === true" in APP
    assert "mode === 'persistent' && !persistentAvailable" in APP


@pytest.mark.parametrize("text", [
    "at least four stable harmonic lines", "Wanted sound in the same 50/60 Hz harmonic family",
    "not streaming, microphone processing or real-time cancellation", "There is no automatic normalization",
    "abstention, not successful removal", "persistent_trace.json",
])
def test_visible_limit_disclosures(text):
    assert text in HTML


HARNESS = r'''const fs=require('fs'),vm=require('vm');
global.window={};
vm.runInThisContext(fs.readFileSync(process.argv[1],'utf8'));
const info=JSON.parse(process.argv[2]), mode=process.argv[3], caps=JSON.parse(process.argv[4]);
process.stdout.write(JSON.stringify(window.HumPreflight.assess(info,mode,caps)));'''


def _assess(info, available):
    if NODE is None:
        pytest.skip("Node.js is not installed")
    result = subprocess.run([NODE, "-e", HARNESS, str(PREFLIGHT), json.dumps(info), "persistent",
                             json.dumps({"persistentAvailable": available})],
                            capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


@pytest.mark.parametrize("fmt", ["PCM16", "PCM24", "Float32"])
@pytest.mark.parametrize("rate", [44100, 48000])
@pytest.mark.parametrize("channels", [1, 2])
def test_persistent_preflight_accepts_native_contract(fmt, rate, channels):
    result = _assess({"status":"metadata", "kind":"wav", "message":"header", "sample_rate_hz":rate,
                      "channels":channels, "sample_format":fmt, "duration_seconds":60,
                      "container_bytes":128*1024*1024, "needs_server_layout_check":False}, True)
    assert result["allowed"] is True
    assert result["recommended_mode"] != "persistent"


def test_persistent_preflight_fails_closed_without_capability_and_rejects_over_60_seconds():
    base = {"status":"metadata", "kind":"wav", "message":"header", "sample_rate_hz":48000,
            "channels":2, "sample_format":"PCM24", "duration_seconds":60,
            "container_bytes":1024, "needs_server_layout_check":False}
    assert _assess(base, False)["allowed"] is False
    base["duration_seconds"] = 60.001
    assert _assess(base, True)["allowed"] is False
