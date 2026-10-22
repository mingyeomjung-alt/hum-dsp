"""Profile-mode browser contract and executable, device-free DOM checks.

The Node harness runs the shipped JavaScript with a small test DOM and mocked
HTTP responses. It makes no network requests and never decodes or plays audio.
Real HTTP, engine and browser rendering are checked separately.
"""

from html.parser import HTMLParser
import json
from pathlib import Path
import re
import shutil
import subprocess

import pytest


ASSETS = Path(__file__).resolve().parents[1] / "src" / "hum_dsp" / "web"
HTML = (ASSETS / "index.html").read_text(encoding="utf-8")
SCRIPT = (ASSETS / "app.js").read_text(encoding="utf-8")
CSS = (ASSETS / "style.css").read_text(encoding="utf-8")
NODE = shutil.which("node")


class _Elements(HTMLParser):
    def __init__(self):
        super().__init__()
        self.elements = []

    def handle_starttag(self, tag, attrs):
        self.elements.append((tag, dict(attrs)))


PARSED = _Elements()
PARSED.feed(HTML)


def _element(name):
    found = [(tag, attrs) for tag, attrs in PARSED.elements if attrs.get("id") == name]
    assert len(found) == 1
    return found[0]


def test_profile_and_persistent_candidates_are_opt_in_after_all_previous_modes():
    mode_markup = re.search(r'<select id="mode" name="mode"[^>]*>(.*?)</select>', HTML).group(1)
    assert re.findall(r'<option\b[^>]*\bvalue="([^"]+)"', mode_markup) == [
        "fixed", "tracked", "guarded", "harmonics", "bands", "compatible", "long",
        "precision", "compressed", "profile", "persistent",
    ]
    assert re.search(r'<option\b[^>]*\sselected(?:\s|=|>)', mode_markup) is None
    assert "v0.0.15" in HTML and "value.version !== '0.0.15'" in SCRIPT
    assert "Experimental · noise-profile band guard" in HTML
    assert "hidden" in _element("profile-options")[1]
    assert "hidden" in _element("profile-result")[1]


def test_profile_interval_and_confirmation_start_empty_and_unasserted():
    for name in ["profile-start", "profile-end"]:
        attrs = _element(name)[1]
        assert attrs["inputmode"] == "decimal"
        assert "value" not in attrs
        assert attrs["aria-describedby"] == "profile-interval-help"
    assert "checked" not in _element("profile-confirm")[1]
    assert _element("profile-orders")[1]["value"] == "all"
    assert _element("profile-depth")[1]["value"] == "6"
    assert _element("attenuation")[1]["value"] == "12"


@pytest.mark.parametrize("notice", [
    "at least 2 seconds within the same input file", "the beginning of a file is not automatically treated as noise",
    "A mistaken declaration can reduce wanted sound", "amplitude exceeds twice the reference anywhere in the file",
    "Growing noise may also remain", "This offline analysis reads future portions",
    "Equal-amplitude overlapping wanted sound", "48 kHz mono PCM16 WAV up to 60 seconds",
    "does not mean noise removal succeeded", "profile_trace.json",
])
def test_experimental_assumptions_and_withholding_limitations_are_visible(notice):
    assert notice in HTML


def test_all_ids_labels_and_new_accessible_targets_exist():
    ids = [attrs["id"] for _, attrs in PARSED.elements if "id" in attrs]
    assert len(ids) == len(set(ids))
    for _, attrs in PARSED.elements:
        if "for" in attrs:
            for name in attrs["for"].split():
                assert name in ids
        for name in attrs.get("aria-describedby", "").split():
            assert name in ids
    assert ".profile-band-list{max-height:" in CSS
    assert ".profile-interval{display:grid;grid-template-columns:1fr 1fr" in CSS


def test_no_auto_audio_devices_storage_or_external_telemetry():
    assert "autoplay" not in HTML.lower()
    assert ".play(" not in SCRIPT
    for prohibited in ["getUserMedia", "AudioContext", "localStorage", "sessionStorage", "sendBeacon", "innerHTML"]:
        assert prohibited not in SCRIPT
    assert "link.click()" in SCRIPT  # Only the explicit manual-review export button.
    assert "personal_listening_self_report" in SCRIPT
    assert "automatic_quality_pass: false" in SCRIPT
    assert "commercial_approval: false" in SCRIPT
    assert "verification_status: 'unverified_self_report'" in SCRIPT
    for name in ["input-audio", "output-audio"]:
        assert _element(name)[1]["preload"] == "none"


_HARNESS = r"""
const fs = require('fs');
const vm = require('vm');
const root = process.argv[1];
const scenario = JSON.parse(process.argv[2]);
const html = fs.readFileSync(root + '/index.html', 'utf8');
const source = fs.readFileSync(root + '/app.js', 'utf8');
const elements = new Map(); const requests = []; const blobs = new Map(); const clicked = [];
let serial = 0;
class Element {
  constructor(attrs={}) {
    this.attrs = {...attrs}; this.value = attrs.value || ''; this.checked = 'checked' in attrs;
    this.hidden = 'hidden' in attrs; this.disabled = 'disabled' in attrs; this.listeners = {};
    this.children = []; this.textContent = ''; this.className = ''; this.files = [];
    this.classList = {add(){}, remove(){}, toggle(){}}; this.controls = true;
  }
  addEventListener(name, fn) { (this.listeners[name] ||= []).push(fn); }
  emit(name, event={}) { for (const fn of this.listeners[name] || []) fn({target:this, preventDefault(){}, ...event}); }
  setAttribute(name,value) { this.attrs[name] = String(value); }
  removeAttribute(name) { delete this.attrs[name]; if(name==='src') this.src=''; if(name==='href') this.href=''; }
  replaceChildren(...items) { this.children = items; }
  append(...items) { this.children.push(...items); }
  pause(){} load(){} focus(){}
  click(){ clicked.push({href:this.href, download:this.download}); this.emit('click'); }
  querySelector() { return elements.get('compressed-option'); }
  getBoundingClientRect(){ return {width:600}; }
  getContext(){ return {scale(){}, beginPath(){}, moveTo(){}, lineTo(){}, stroke(){}}; }
}
for(const match of html.matchAll(/<\w+\s[^>]*id="([^"]+)"[^>]*>/g)) {
 const attrs={}; for(const a of match[0].matchAll(/([\w-]+)(?:="([^"]*)")?/g)) attrs[a[1]]=a[2] ?? '';
 elements.set(match[1], new Element(attrs));
}
elements.set('compressed-option', new Element({disabled:''}));
elements.get('mode').value='fixed'; elements.get('hum-hz').value='50';
const $ = id => elements.get(id);
const document = {getElementById:$, querySelector:()=>({content:'test-token'}), createElement:()=>new Element()};
const window = {location:{origin:'http://127.0.0.1:1234'}, devicePixelRatio:1, addEventListener(){}};
class TestURL extends URL {}
TestURL.createObjectURL = blob => { const name='blob:test-'+(++serial); blobs.set(name,blob); return name; };
TestURL.revokeObjectURL = name => blobs.delete(name);
const report={config:{nominal_hum_hz:60, max_attenuation_db:6,orders:[1,6,21,24],profile_start_sample:0,profile_end_sample:96000},
 input:{frames:384000,sample_rate_hz:48000,peak_abs:.1,sha256:'input-test'},
 output:{frames:384000,sample_rate_hz:48000,peak_abs:.08,sha256:'output-test'},
 timing:{file_processing_seconds:.3},profile_guard:{experimental:true,same_input_interval:true,real_time:false,
 noise_only_interval_user_asserted:true,input_wav_sha256:'input-test',profile_start_seconds:0,profile_end_seconds:2,
 approved_orders:[1,6,24],rejected_orders:[21],unsupported_orders:[],processing_bypassed:false,
 offline_whole_file_future_lookahead:true,noise_only_interval_machine_verified:false,
 bands:[{order:1,approved:true,reason:'profile_supported_amplitude_consistent'},
        {order:21,approved:false,amplitude_guard_rejected:true,reason:'whole_file_amplitude_exceeds_profile'}]}};
const fetch = async (url,options={}) => {
 requests.push({url,options});
 if(url==='/api/capabilities') return {ok:true,json:async()=>({schema_version:1,version:'0.0.15',desktop:{available:false,portable_wav_only:true},compressed:{available:false,required_version:'>=19,<20'}})};
 if(scenario.response?.startsWith('server_error_')) return {ok:false,status:Number(scenario.response.split('_').at(-1)),json:async()=>({error:'입력 파일이나 설정을 확인하세요.'})};
 if(scenario.response==='english_server_error') return {ok:false,status:400,json:async()=>({error:'Example English server diagnostic.'})};
 if(scenario.response==='bad_profile') report.profile_guard.noise_only_interval_machine_verified=true;
 if(scenario.response==='zero_depth') Object.assign(report.profile_guard,{bands:[],approved_orders:[],rejected_orders:[],unsupported_orders:[],processing_bypassed:true,bypass_reason:'zero_depth'});
 const mode = new URL(url,window.location.origin).searchParams.get('mode');
 return {ok:true,json:async()=>({mode,report,waveforms:{input:[[-.1,.1]],output:[[-.08,.08]]},input_wav_base64:'UklGRg==',output_wav_base64:'UklGRg==',download_zip_base64:'UEs='})};
};
vm.runInNewContext(source,{document,window,URL:TestURL,URLSearchParams,Blob,AbortController,
 setTimeout,clearTimeout,fetch,atob,Uint8Array,Date,console});
async function settle(){ for(let i=0;i<8;i++) await new Promise(setImmediate); }
async function set(id,value,kind='input'){ if(typeof value==='boolean') $(id).checked=value; else $(id).value=value; $(id).emit(kind); await settle(); }
function flatten(el){ return [el.textContent,...el.children.map(flatten)].join(' '); }
(async()=>{
 await settle();
 for(const action of scenario.actions || []) {
   if(action.set) await set(action.set, action.value);
   if(action.click){ $(action.click).emit('click'); await settle(); }
   if(action.file){ $('audio-file').files=[{name:action.file,size:300}]; $('audio-file').emit('change'); await settle(); }
   if(action.submit){ $('processor-form').emit('submit'); await settle(); }
 }
 let exported=null;
 if(clicked.length){ const blob=blobs.get(clicked.at(-1).href); if(blob) exported=JSON.parse(await blob.text()); }
 const state={}; for(const id of ['profile-start','profile-end','profile-confirm','profile-options','attenuation-options','profile-result','profile-result-state','profile-result-interval','error','results','review-export','review-confirm','review-status','process-button','status','result-settings']) {
  const el=$(id); state[id]={value:el.value,checked:el.checked,hidden:el.hidden,disabled:el.disabled,text:el.textContent};
 }
 console.log(JSON.stringify({state,requests:requests.map(r=>({url:r.url,method:r.options.method,bodyLength:r.options.body?.length,headers:r.options.headers})),
   resultText:flatten($('profile-band-list')),href:$('download-wav').href || null,exported,clicked}));
})().catch(error=>{console.error(error);process.exitCode=1});
"""


def _run(actions=(), response=None):
    if NODE is None:
        pytest.skip("Node.js is not installed; static and HTTP checks remain separate")
    proc = subprocess.run([NODE, "-e", _HARNESS, str(ASSETS), json.dumps({"actions": actions, "response": response})],
                          capture_output=True, text=True, encoding="utf-8", timeout=20)
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


def _profile_ready(start="0", end="2", orders="all", *, example_interval=False):
    actions = [{"set": "mode", "value": "profile"}, {"set": "hum-hz", "value": "60"}]
    if example_interval:
        actions += [{"click": "profile-example-interval"}]
    else:
        actions += [{"set": "profile-start", "value": start}, {"set": "profile-end", "value": end}]
    actions += [{"set": "profile-orders", "value": orders}, {"set": "profile-confirm", "value": True}]
    return actions


def _process_requests(result):
    return [r for r in result["requests"] if "/example?" in r["url"] or "/process?" in r["url"]]


def test_runtime_default_retains_fixed_mode_and_blank_profile():
    result = _run()
    assert result["state"]["profile-options"]["hidden"]
    assert result["state"]["profile-start"]["value"] == result["state"]["profile-end"]["value"] == ""
    assert not result["state"]["profile-confirm"]["checked"]
    assert not result["state"]["attenuation-options"]["hidden"]
    assert _process_requests(result) == []


def test_user_interface_is_english_without_changing_report_or_review_schema():
    assert '<html lang="en">' in HTML
    for source in (HTML, SCRIPT, (ASSETS / "ab-review.js").read_text(encoding="utf-8")):
        assert re.search(r"[\u1100-\u11ff\u3130-\u318f\uac00-\ud7a3]", source) is None
    assert "unverified_self_report" in SCRIPT
    assert "noise_only_interval_machine_verified !== false" in SCRIPT
    assert "automatic_quality_pass: false" in SCRIPT


@pytest.mark.parametrize("status,instruction", [
    (400, "Check the file format and settings"),
    (403, "Reload this local page"),
    (413, "exceeds the current mode limits"),
    (503, "may be busy or an optional decoder may be unavailable"),
    (500, "could not complete the request"),
])
def test_untranslated_backend_diagnostics_have_english_recovery_guidance(status, instruction):
    result = _run(_profile_ready() + [{"click": "example-button"}], response=f"server_error_{status}")
    error = result["state"]["error"]
    assert not error["hidden"]
    assert instruction in error["text"]
    assert re.search(r"[\uac00-\ud7a3]", error["text"]) is None
    if status != 403:
        assert "Experimental" in error["text"] and "noise" in error["text"]
    assert result["href"] is None
    assert result["state"]["results"]["hidden"]


def test_existing_english_backend_diagnostic_is_preserved():
    result = _run(_profile_ready() + [{"click": "example-button"}], response="english_server_error")
    assert result["state"]["error"]["text"] == "Example English server diagnostic."


@pytest.mark.parametrize("start,end,orders", [
    ("", "", "all"), ("0", "1.999", "all"), ("2", "0", "all"), ("0", "61", "all"),
    ("NaN", "2", "all"), ("-1", "2", "all"), ("0.00001", "3", "all"),
    ("1e-3", "3", "all"), ("0", "2", ""), ("0", "2", "2,1"),
    ("0", "2", "1,1"), ("0", "2", "33"), ("0", "2", "1, 6"),
])
def test_runtime_invalid_profile_configuration_never_reaches_process(start, end, orders):
    result = _run(_profile_ready(start, end, orders) + [{"click": "example-button"}])
    assert _process_requests(result) == []
    assert not result["state"]["error"]["hidden"]
    assert result["href"] is None


def test_runtime_confirmation_is_required_for_known_example_too():
    result = _run([{"set": "mode", "value": "profile"}, {"click": "profile-example-interval"}, {"click": "example-button"}])
    assert result["state"]["profile-start"]["value"] == "0"
    assert result["state"]["profile-end"]["value"] == "2"
    assert not result["state"]["profile-confirm"]["checked"]
    assert _process_requests(result) == []


@pytest.mark.parametrize("orders", ["all", "1,6,21,24"])
def test_runtime_explicit_profile_uses_bounded_route_and_shows_approvals_and_vetoes(orders):
    result = _run(_profile_ready(orders=orders) + [{"click": "example-button"}])
    requests = _process_requests(result)
    assert len(requests) == 1
    assert requests[0]["url"].startswith("/api/example?mode=profile&hum_hz=60&")
    assert "profile_start_seconds=0" in requests[0]["url"] and "profile_end_seconds=2" in requests[0]["url"]
    assert "confirm_noise_only=true" in requests[0]["url"] and "attenuation_db=6" in requests[0]["url"]
    assert requests[0]["headers"]["X-Hum-Token"] == "test-token"
    assert requests[0]["bodyLength"] == 0
    assert not result["state"]["results"]["hidden"]
    assert not result["state"]["profile-result"]["hidden"]
    assert "Withheld for the whole file because amplitude increased" in result["resultText"]
    assert "no quality guarantee" in result["resultText"]
    assert result["href"].startswith("blob:test-")
    assert result["state"]["profile-start"]["value"] == ""  # Example assertion does not transfer.
    assert not result["state"]["profile-confirm"]["checked"]


def test_runtime_zero_depth_is_labeled_unchanged_not_denoising_success():
    result = _run(_profile_ready() + [{"set": "profile-depth", "value": "0"}, {"click": "example-button"}], response="zero_depth")
    assert "Unchanged" in result["state"]["profile-result-state"]["text"]
    assert "0 dB" in result["resultText"] and "input unchanged" in result["resultText"]
    assert result["href"] is not None


def test_runtime_claim_of_machine_verified_noise_interval_is_rejected():
    result = _run(_profile_ready() + [{"click": "example-button"}], response="bad_profile")
    assert result["state"]["results"]["hidden"]
    assert not result["state"]["error"]["hidden"]
    assert result["href"] is None


@pytest.mark.parametrize("control,value", [
    ("profile-start", "1"), ("profile-end", "3"), ("profile-orders", "1,6"),
    ("profile-depth", "3"), ("profile-confirm", False), ("hum-hz", "50"), ("mode", "fixed"),
])
def test_runtime_every_profile_setting_invalidates_stale_output(control, value):
    result = _run(_profile_ready() + [{"click": "example-button"}, {"set": control, "value": value}])
    assert len(_process_requests(result)) == 1
    assert result["state"]["results"]["hidden"]
    assert result["state"]["profile-result"]["hidden"]
    assert result["href"] is None
    assert result["state"]["review-export"]["disabled"]


def test_runtime_file_change_clears_noise_declaration_and_interval():
    result = _run(_profile_ready() + [{"file": "my-file.wav"}])
    assert result["state"]["profile-start"]["value"] == result["state"]["profile-end"]["value"] == ""
    assert not result["state"]["profile-confirm"]["checked"]


def test_runtime_example_interval_helper_cannot_assert_an_uploaded_file():
    actions = [{"file": "my-file.wav"}] + _profile_ready(example_interval=True) + [{"submit": True}]
    result = _run(actions)
    assert _process_requests(result) == []
    assert "synthetic example" in result["state"]["error"]["text"]
    assert not result["state"]["profile-confirm"]["checked"]


def test_runtime_user_file_declaration_cannot_transfer_to_synthetic_example():
    # In the synthetic fixture, 4~6 seconds contains wanted audio, unlike the
    # selected user's file assertion. The source-kind boundary must stop HTTP.
    actions = [{"file": "my-file.wav"}] + _profile_ready(start="4", end="6") + [{"click": "example-button"}]
    result = _run(actions)
    assert _process_requests(result) == []
    assert "Your current confirmation is for your own file" in result["state"]["error"]["text"]
    assert "0–2 s example interval" in result["state"]["error"]["text"]
    assert not result["state"]["profile-confirm"]["checked"]
    assert result["href"] is None


def test_runtime_source_mismatch_requires_example_helper_and_fresh_confirmation_to_recover():
    actions = [{"file": "my-file.wav"}] + _profile_ready(start="4", end="6") + [
        {"click": "example-button"}, {"click": "example-button"},
        {"click": "profile-example-interval"}, {"set": "profile-confirm", "value": True},
        {"click": "example-button"},
    ]
    result = _run(actions)
    requests = _process_requests(result)
    assert len(requests) == 1
    assert requests[0]["url"].startswith("/api/example?mode=profile")
    assert "profile_start_seconds=0" in requests[0]["url"] and "profile_end_seconds=2" in requests[0]["url"]
    assert not result["state"]["results"]["hidden"]
    assert result["state"]["error"]["hidden"]
    assert not result["state"]["profile-confirm"]["checked"]


def test_runtime_upload_with_explicit_own_interval_does_not_clear_user_input_after_processing():
    result = _run([{"file": "my-file.wav"}] + _profile_ready(start="0.5", end="2.5") + [{"submit": True}])
    assert len(_process_requests(result)) == 1
    assert _process_requests(result)[0]["url"].startswith("/api/process?mode=profile")
    assert result["state"]["profile-start"]["value"] == "0.5"
    assert not result["state"]["results"]["hidden"]


def test_runtime_personal_review_never_blocks_wav_and_requires_attestation_after_edits():
    base = _profile_ready() + [{"click": "example-button"}]
    initial = _run(base)
    assert initial["href"] is not None and initial["state"]["review-export"]["disabled"]
    reviewed = base + [{"set": "review-hum", "value": "less"}, {"set": "review-wanted", "value": "uncertain"},
                       {"set": "review-artifacts", "value": "not_noticed"}, {"set": "review-confirm", "value": True}]
    ready = _run(reviewed)
    assert not ready["state"]["review-export"]["disabled"]
    edited = _run(reviewed + [{"set": "review-notes", "value": "new note"}])
    assert not edited["state"]["review-confirm"]["checked"] and edited["state"]["review-export"]["disabled"]
    assert edited["href"] is not None


def test_runtime_review_export_is_local_unverified_and_binds_original_report():
    actions = _profile_ready() + [{"click": "example-button"}, {"set": "review-hum", "value": "less"},
        {"set": "review-wanted", "value": "uncertain"}, {"set": "review-artifacts", "value": "not_noticed"},
        {"set": "review-notes", "value": "test-only simulated review"}, {"set": "review-confirm", "value": True},
        {"click": "review-export"}]
    result = _run(actions)
    assert len(_process_requests(result)) == 1  # No review submission endpoint.
    record = result["exported"]
    assert record["verification_status"] == "unverified_self_report"
    assert record["automatic_quality_pass"] is record["commercial_approval"] is False
    assert record["report"]["input"]["sha256"] == "input-test"
    assert record["observations"]["notes"] == "test-only simulated review"
    assert result["clicked"][0]["download"] == "hum-dsp-personal-listening.json"


def test_shipped_script_has_valid_node_syntax():
    if NODE is None:
        pytest.skip("Node.js not installed")
    proc = subprocess.run([NODE, "--check", str(ASSETS / "app.js")], capture_output=True, text=True, encoding="utf-8", timeout=20)
    assert proc.returncode == 0, proc.stderr
