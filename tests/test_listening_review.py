"""Offline manual-review generation, provenance binding and defensive paths."""

import hashlib
import json
import re
import shutil
import subprocess
import wave

import pytest

from hum_dsp.listening_review import build_listening_review


@pytest.fixture
def review_manifest(tmp_path):
    audio = tmp_path / "case" / "wanted.wav"
    audio.parent.mkdir()
    with wave.open(str(audio), "wb") as stream:
        stream.setparams((1, 2, 48000, 0, "NONE", "not compressed"))
        stream.writeframes(b"\x00\x00" * 64)
    asset = {"label": "Full", "file_name": "case/wanted.wav", "sha256": hashlib.sha256(audio.read_bytes()).hexdigest()}
    manifest = {"schema_version": 1, "title": "A private study", "cases": [{
        "case_id": "piano_50", "title": "Piano + recorded hum", "kind": "constructed_mixture",
        "context": "Fixed settings, no automatic quality decision.",
        "reference": {**asset, "label": "Input"}, "variants": [asset],
    }]}
    path = tmp_path / "listening_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path, manifest


def save(path, manifest):
    path.write_text(json.dumps(manifest), encoding="utf-8")


def embedded(path):
    body = path.read_text(encoding="utf-8")
    return body, json.loads(re.search(r'<script id="review-data" type="application/json">(.*?)</script>', body, re.S).group(1))


def test_builds_offline_review_and_pins_exact_manifest_and_audio(review_manifest):
    path, manifest = review_manifest
    original = path.read_bytes()
    output = build_listening_review(path)
    body, payload = embedded(output)
    assert output == path.with_name("review.html")
    assert payload["manifest_sha256"] == hashlib.sha256(original).hexdigest()
    assert payload["cases"][0]["variants"] == manifest["cases"][0]["variants"]
    assert path.read_bytes() == original
    assert 'audio.preload=\'none\'' in body
    assert "audio.volume=.5" in body
    assert "other.pause()" in body
    assert "autoplay" not in body.lower()
    assert "localStorage" not in body and "fetch(" not in body
    assert "connect-src 'none'" in body
    assert '<button id="export" disabled>' in body
    assert "No human reviews completed." in body
    assert "automatic_quality_approval:false" in body
    assert "listener_self_reported_listened:true" in body
    assert "human_listening_basis:'unverified_self_report'" in body
    assert "human_listening_verified:false" in body
    assert "asset_hash_validation:'at_HTML_generation_only_not_rechecked_during_browser_playback'" in body
    assert "responses.has(item.case_id)" in body
    assert "new Blob(" in body and "URL.revokeObjectURL" in body
    assert "check.checked=false;if(key==='evaluated_variant')" in body
    assert "['residual_hum','wanted_loss','distortion'].forEach" in body
    assert "notes.addEventListener('input',()=>{check.checked=false;update();})" in body


def test_manifest_strings_cannot_escape_json_script(review_manifest):
    path, manifest = review_manifest
    attack = '</script><img src=x onerror="alert(1)">'
    manifest["title"] = attack
    manifest["cases"][0]["context"] = attack
    manifest["cases"][0]["variants"][0]["label"] = attack
    save(path, manifest)
    body, payload = embedded(build_listening_review(path))
    assert attack not in body
    assert payload["title"] == attack
    assert "textContent=value" in body


@pytest.mark.parametrize("name", ["../wanted.wav", "/wanted.wav", "C:/wanted.wav", "case\\wanted.wav", "case//wanted.wav", "case/./wanted.wav", "case/../wanted.wav", "case/%2e%2e.wav", "//host/a.wav", "case/wanted.wav?x", "case/wanted.mp3"])
def test_rejects_noncanonical_or_unsafe_paths(review_manifest, name):
    path, manifest = review_manifest
    manifest["cases"][0]["variants"][0]["file_name"] = name
    save(path, manifest)
    with pytest.raises(ValueError, match="safe relative"):
        build_listening_review(path)
    assert not path.with_name("review.html").exists()


@pytest.mark.parametrize("kind", ["recorded_hum_only", "natural_hum_only", "constructed_mixture", "desired_only"])
def test_supported_evidence_kinds_are_explicit(review_manifest, kind):
    path, manifest = review_manifest
    manifest["cases"][0]["kind"] = kind
    save(path, manifest)
    _, payload = embedded(build_listening_review(path))
    assert payload["cases"][0]["kind"] == kind


@pytest.mark.parametrize("change", ["hash", "missing", "not_wav", "short", "duplicate_case", "duplicate_label", "kind", "cases", "variants", "schema", "id"])
def test_invalid_manifests_fail_before_output_creation(review_manifest, change):
    path, manifest = review_manifest
    case = manifest["cases"][0]
    audio = path.parent / case["variants"][0]["file_name"]
    if change == "hash":
        case["variants"][0]["sha256"] = "0" * 64
    elif change == "missing":
        audio.unlink()
    elif change == "not_wav":
        audio.write_bytes(b"x" * 128)
    elif change == "short":
        audio.write_bytes(b"RIFF")
    elif change == "duplicate_case":
        manifest["cases"].append(case)
    elif change == "duplicate_label":
        case["variants"].append(case["variants"][0])
    elif change == "kind":
        case["kind"] = "clean_ground_truth"
    elif change == "cases":
        manifest["cases"] = []
    elif change == "variants":
        case["variants"] = []
    elif change == "schema":
        manifest["schema_version"] = True
    else:
        case["case_id"] = "../bad"
    save(path, manifest)
    with pytest.raises(ValueError):
        build_listening_review(path)
    assert not path.with_name("review.html").exists()


def test_output_never_overwrites_and_stays_beside_manifest(review_manifest, tmp_path):
    path, _ = review_manifest
    output = build_listening_review(path, path.with_name("custom.html"))
    before = output.read_bytes()
    with pytest.raises(FileExistsError):
        build_listening_review(path, output)
    assert output.read_bytes() == before
    other = tmp_path / "other"
    other.mkdir()
    with pytest.raises(ValueError, match="beside"):
        build_listening_review(path, other / "review.html")


def test_rejects_duplicate_json_keys(review_manifest):
    path, _ = review_manifest
    path.write_text('{"schema_version":1,"schema_version":1,"cases":[]}', encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate manifest"):
        build_listening_review(path)


def test_rejects_symlink_audio_if_platform_can_create_links(review_manifest):
    path, manifest = review_manifest
    audio = path.parent / "case/wanted.wav"
    target = path.parent / "real.wav"
    target.write_bytes(audio.read_bytes())
    audio.unlink()
    try:
        audio.symlink_to(target)
    except OSError:
        pytest.skip("Host does not permit creating symlinks")
    with pytest.raises(ValueError, match="Symbolic"):
        build_listening_review(path)


def test_embedded_javascript_syntax_and_synthetic_export_payload(review_manifest):
    """Only serialize synthetic data; never impersonate a listening reviewer."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js parser unavailable")
    path, _ = review_manifest
    body, payload = embedded(build_listening_review(path))
    script = re.search(r"</script><script>(.*?)</script>", body, re.S).group(1)
    parsed = subprocess.run([node, "-e", "new Function(JSON.parse(process.argv[1]));", json.dumps(script)], capture_output=True, text=True, timeout=10)
    assert parsed.returncode == 0, parsed.stderr
    serializer = re.search(r"function exportPayload\(.*?/\* end exportPayload \*/", script, re.S).group(0)
    synthetic = {"cases": [{"case_id": "synthetic_a", "variants": [{"label": "A", "sha256": "0" * 64}]}, {"case_id": "synthetic_b", "variants": []}], "manifest_sha256": "1" * 64}
    code = serializer + ";const study=JSON.parse(process.argv[1]);const records=new Map([['synthetic_a',{case_id:'synthetic_a',test_only:true}]]);process.stdout.write(JSON.stringify(exportPayload(study,records,'TEST_ONLY_NOT_HUMAN_REVIEW')));"
    result = subprocess.run([node, "-e", code, json.dumps(synthetic)], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    exported = json.loads(result.stdout)
    assert exported["human_review_status"] == "partial"
    assert exported["human_listening_verified"] is False
    assert exported["human_listening_basis"] == "unverified_self_report"
    assert exported["automatic_quality_approval"] is False
    assert exported["manifest_sha256"] == "1" * 64
    assert exported["completed_case_count"] == 1
    assert [record["case_id"] for record in exported["records"]] == ["synthetic_a"]
    assert exported["records"][0]["test_only"] is True
    assert exported["records"][0]["variants"] == synthetic["cases"][0]["variants"]


def test_synthetic_dom_invalidates_attestation_when_ratings_or_variant_change(review_manifest):
    """An in-memory fake DOM checks form state; no audio or review is produced."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js unavailable")
    path, manifest = review_manifest
    manifest["cases"][0]["variants"].append({**manifest["cases"][0]["variants"][0], "label": "Gentle"})
    save(path, manifest)
    body, payload = embedded(build_listening_review(path))
    script = re.search(r"</script><script>(.*?)</script>", body, re.S).group(1)
    fake_dom = r'''
const nodes=new Map();
class Element{constructor(tag){this.tag=tag;this.children=[];this.listeners={};this.value='';this.checked=false;this.disabled=false;this.textContent='';}set id(value){this._id=value;nodes.set(value,this);}get id(){return this._id;}append(...children){this.children.push(...children);}addEventListener(name,fn){(this.listeners[name]??=[]).push(fn);}setAttribute(){}pause(){this.paused=true;}emit(name){(this.listeners[name]??[]).forEach(fn=>fn());}}
globalThis.Option=class{constructor(label,value){this.label=label;this.value=value;}};
globalThis.document={getElementById:(id)=>{if(!nodes.has(id)){const node=new Element('div');node.id=id;}return nodes.get(id);},createElement:(tag)=>new Element(tag),body:new Element('body')};
document.getElementById('review-data').textContent=process.argv[1];
'''
    checks = r'''
const caseId=study.cases[0].case_id, control=(name)=>nodes.get(caseId+'-'+name), confirm=control('confirm');
function fillSyntheticForm(){for(const name of ['evaluated_variant','residual_hum','wanted_loss','distortion','preference']){control(name).value=name==='evaluated_variant'||name==='preference'?'Full':'unsure';control(name).emit('change');}confirm.checked=true;confirm.emit('change');}
fillSyntheticForm();if(responses.size!==1)throw Error('Synthetic completed state not recorded');
control('evaluated_variant').value='Gentle';control('evaluated_variant').emit('change');
if(confirm.checked||responses.size||!nodes.get('export').disabled)throw Error('Variant change retained attestation');
for(const name of ['residual_hum','wanted_loss','distortion'])if(control(name).value!=='')throw Error('Variant-specific rating was not cleared');
fillSyntheticForm();control('residual_hum').value='slight';control('residual_hum').emit('change');
if(confirm.checked||responses.size)throw Error('Rating edit retained attestation');
fillSyntheticForm();control('notes').value='Synthetic unit test only';control('notes').emit('input');
if(confirm.checked||responses.size)throw Error('Notes edit retained attestation');
process.stdout.write('SYNTHETIC_FORM_STATE_PASS_NO_LISTENING_OR_EXPORT');
'''
    result = subprocess.run([node, "-e", fake_dom + script + checks, json.dumps(payload)], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert result.stdout == "SYNTHETIC_FORM_STATE_PASS_NO_LISTENING_OR_EXPORT"
