"""Build a private, offline, explicitly manual listening worksheet.

This evaluation helper does not run DSP, make a quality decision, play sound,
or change shipped product assets. Its manifest pins every local WAV by hash.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path, PurePosixPath


_SAFE_SEGMENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,100}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_KINDS = {
    "recorded_hum_only": "Recorded hum · no independent clean reference",
    "natural_hum_only": "Recorded hum · no independent clean reference",
    "constructed_mixture": "Constructed mixture · separately recorded hum component",
    "desired_only": "Wanted audio only · preservation control",
}


def _reject_links(path: Path) -> None:
    for part in (path, *path.parents):
        if part.is_symlink() or getattr(part, "is_junction", lambda: False)():
            raise ValueError("Symbolic links and junctions are not accepted")


def _text(value: object, name: str, maximum: int = 1000) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise ValueError(f"Invalid {name}")
    return value


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate manifest key")
        result[key] = value
    return result


def _asset(value: object, root: Path) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ValueError("Invalid audio asset")
    label = _text(value.get("label"), "audio label", 100)
    name = _text(value.get("file_name"), "audio file name", 1000)
    parts = PurePosixPath(name).parts
    if not parts or "/".join(parts) != name or any(
        not _SAFE_SEGMENT.fullmatch(part) or part in (".", "..") for part in parts
    ) or not name.lower().endswith(".wav"):
        raise ValueError("Audio file names must be safe relative POSIX WAV paths")
    digest = value.get("sha256")
    if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
        raise ValueError("Invalid audio SHA-256")
    path = root.joinpath(*parts)
    _reject_links(path)
    if not path.is_file() or not 44 <= path.stat().st_size <= 128 * 1024 * 1024:
        raise ValueError("Missing or oversized WAV asset")
    calculated = hashlib.sha256()
    with path.open("rb") as source:
        header = source.read(12)
        if header[:4] != b"RIFF" or header[8:12] != b"WAVE":
            raise ValueError("Audio asset is not a RIFF WAV file")
        calculated.update(header)
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            calculated.update(chunk)
    if calculated.hexdigest() != digest:
        raise ValueError("WAV asset SHA-256 mismatch")
    return {"label": label, "file_name": name, "sha256": digest}


def build_listening_review(manifest_path: str | Path, output_path: str | Path | None = None) -> Path:
    """Verify a schema-1 manifest and create ``review.html`` without overwrite.

    Each case has ``case_id``, ``title``, ``kind``, ``context``, ``variants``
    and an optional ``reference``. Audio objects contain ``label``, relative
    POSIX ``file_name`` and lowercase ``sha256``. The output must be beside
    the manifest so that relative audio links retain the same meaning.
    """
    manifest_path = Path(manifest_path).absolute()
    _reject_links(manifest_path)
    if not manifest_path.is_file() or manifest_path.stat().st_size > 1024 * 1024:
        raise ValueError("Missing or oversized review manifest")
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw, object_pairs_hook=_unique_object)
    if not isinstance(manifest, dict) or type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1:
        raise ValueError("Expected review manifest schema_version 1")
    cases = manifest.get("cases")
    if not isinstance(cases, list) or not 1 <= len(cases) <= 100:
        raise ValueError("Expected 1 to 100 review cases")
    validated, case_ids = [], set()
    for item in cases:
        if not isinstance(item, dict):
            raise ValueError("Invalid review case")
        case_id = _text(item.get("case_id"), "case ID", 100)
        if not _SAFE_SEGMENT.fullmatch(case_id) or case_id in case_ids:
            raise ValueError("Unsafe or duplicate case ID")
        case_ids.add(case_id)
        kind = item.get("kind")
        if not isinstance(kind, str) or kind not in _KINDS:
            raise ValueError("Unknown review case kind")
        variants = item.get("variants")
        if not isinstance(variants, list) or not 1 <= len(variants) <= 6:
            raise ValueError("Expected 1 to 6 variants per case")
        assets = [_asset(asset, manifest_path.parent) for asset in variants]
        if len({asset["label"] for asset in assets}) != len(assets):
            raise ValueError("Duplicate variant labels")
        case = {"case_id": case_id, "title": _text(item.get("title"), "case title", 200),
                "kind": kind, "kind_label": _KINDS[kind],
                "context": _text(item.get("context"), "case context", 3000), "variants": assets}
        if item.get("reference") is not None:
            case["reference"] = _asset(item["reference"], manifest_path.parent)
        validated.append(case)
    output = Path(output_path).absolute() if output_path is not None else manifest_path.with_name("review.html")
    _reject_links(output)
    if output.parent.resolve() != manifest_path.parent.resolve() or output.suffix.lower() != ".html":
        raise ValueError("Review HTML must be beside its manifest")
    payload = {"schema_version": 1, "manifest_sha256": hashlib.sha256(raw).hexdigest(),
               "title": _text(manifest.get("title", "Hum DSP listening review"), "title", 200),
               "description": _text(manifest.get("description", "A manual worksheet, not an automatic quality approval."), "description", 3000),
               "cases": validated}
    embedded = json.dumps(payload, ensure_ascii=True, separators=(",", ":")).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    with output.open("x", encoding="utf-8", newline="\n") as destination:
        destination.write(_HTML.replace("__REVIEW_PAYLOAD__", embedded))
    return output


_HTML = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; media-src 'self' file:; connect-src 'none'; img-src 'none'; base-uri 'none'; form-action 'none'">
<title>Hum DSP · Manual listening review</title>
<style>
:root{color-scheme:dark;--bg:#111715;--surface:#19221e;--ink:#f1f4ed;--muted:#a4b2a9;--line:#334239;--accent:#c4ef80}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.55 system-ui,sans-serif}main{max-width:1140px;margin:auto;padding:52px 28px 76px}header{padding-bottom:28px;border-bottom:1px solid var(--line)}.eyebrow{color:var(--accent);font-size:12px;letter-spacing:.17em;text-transform:uppercase}h1{font-size:clamp(30px,4vw,48px);line-height:1.13;letter-spacing:-.04em;margin:14px 0}p{margin:10px 0;color:var(--muted)}.notice{background:#222c22;border:1px solid #43543d;border-radius:12px;padding:16px 20px;margin-top:22px}.notice strong{color:var(--ink)}.toolbar{display:flex;align-items:center;justify-content:space-between;gap:16px;margin:30px 0 24px}.status{color:var(--muted);font-size:14px}button{font:inherit;font-weight:650;color:#14200e;background:var(--accent);border:0;border-radius:9px;padding:12px 18px;cursor:pointer}button:disabled{opacity:.45;cursor:not-allowed}button:focus-visible,select:focus-visible,textarea:focus-visible,input:focus-visible{outline:3px solid var(--accent);outline-offset:3px}.case{border:1px solid var(--line);border-radius:16px;background:var(--surface);padding:26px;margin:22px 0}.case-top{display:flex;gap:20px;justify-content:space-between}.number{font:13px/1.5 ui-monospace,monospace;color:var(--accent)}h2{font-size:23px;margin:5px 0 8px;letter-spacing:-.02em}.tag{display:inline-block;font-size:12px;background:#2e3b31;border:1px solid #465448;border-radius:6px;padding:5px 9px;color:#dae6d5}.players{display:grid;grid-template-columns:repeat(auto-fit,minmax(235px,1fr));gap:12px;margin:20px 0}.player{padding:16px;border:1px solid var(--line);border-radius:10px;background:#141c18}.player b{display:block;margin-bottom:12px;font-size:14px}audio{width:100%;height:38px}.file{font-size:10px;color:var(--muted);word-break:break-all;margin-top:10px}.instruction{font-size:13px}.ratings{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:15px;margin-top:24px}label{display:block;font-size:13px;color:var(--muted)}select,textarea{font:inherit;color:var(--ink);background:#111915;border:1px solid #465449;border-radius:8px;padding:10px;width:100%;margin-top:7px}textarea{min-height:82px;resize:vertical}.full{grid-column:1/-1}.confirm{display:flex;align-items:flex-start;gap:10px;margin:16px 0 0;color:var(--ink);font-size:14px}.confirm input{margin-top:5px;accent-color:var(--accent)}footer{margin-top:35px;border-top:1px solid var(--line);padding-top:20px;font-size:12px;color:var(--muted)}.case-status{font-size:12px;color:var(--muted);margin:14px 0 0}.hash{font-family:ui-monospace,monospace;word-break:break-all}.live{font-size:13px;color:var(--accent);margin-left:8px}@media(max-width:620px){main{padding:30px 16px 50px}.case{padding:18px}.ratings{grid-template-columns:1fr}.toolbar{align-items:flex-start;flex-direction:column}.case-top{flex-direction:column;gap:2px}}
</style></head><body><main>
<header><div class="eyebrow">Private evaluation · manual listening</div><h1 id="title"></h1><p id="description"></p>
<div class="notice"><strong>You decide when to listen.</strong> Nothing plays automatically. Start at a comfortable device volume; each player starts at 50%. No gain matching, normalization, or automatic quality approval is applied.<br><span class="instruction">These are narrow-band 50/60 Hz and selected-harmonic controls, not a broadband denoiser. Wanted audio in the same bands may also be reduced.</span></div></header>
<div class="toolbar"><div><div id="progress" class="status" role="status" aria-live="polite">No human reviews completed.</div><div class="instruction">Only fully completed cases are included in the export.</div></div><div><button id="export" disabled>Export completed reviews</button><span id="export-status" class="live" role="status"></span></div></div>
<section id="cases" aria-label="Listening cases"></section>
<footer>This worksheet stays in this browser tab. No uploads, analytics, cookies, or saved browser storage. Closing or reloading loses unexported responses. Listening confirmations are unverified self-reports, not proof that audio was heard or a DSP pass/fail result. Asset hashes were checked when this HTML was generated; files are not rechecked during browser playback.<p>Manifest SHA-256: <span id="manifest-hash" class="hash"></span></p></footer>
</main><script id="review-data" type="application/json">__REVIEW_PAYLOAD__</script><script>
'use strict';
const study=JSON.parse(document.getElementById('review-data').textContent), responses=new Map(), players=[];
document.getElementById('title').textContent=study.title;document.getElementById('description').textContent=study.description;document.getElementById('manifest-hash').textContent=study.manifest_sha256;
const text=(tag,value,cls)=>{const el=document.createElement(tag);el.textContent=value;if(cls)el.className=cls;return el;};
function player(asset,reference){const box=text('div','','player');box.append(text('b',(reference?'Reference / input · ':'')+asset.label));const audio=document.createElement('audio');audio.controls=true;audio.preload='none';audio.src=asset.file_name;audio.volume=.5;audio.addEventListener('play',()=>players.forEach(other=>{if(other!==audio)other.pause();}));players.push(audio);box.append(audio,text('div',asset.file_name+' · SHA-256 '+asset.sha256,'file'));return box;}
function choose(label,options,id){const wrapper=text('label',label), select=document.createElement('select');select.id=id;select.append(new Option('Choose a response',''));options.forEach(([value,name])=>select.append(new Option(name,value)));wrapper.htmlFor=id;wrapper.append(select);return [wrapper,select];}
const levels=[['none','None noticed'],['slight','Slight'],['moderate','Moderate'],['strong','Strong'],['unsure','Unsure / not assessable']];
function exportPayload(study,responses,exportedAt){const completed=study.cases.filter(item=>responses.has(item.case_id));return {schema_version:1,source:'manual_local_listening_review',exported_at:exportedAt,manifest_sha256:study.manifest_sha256,human_review_status:responses.size===study.cases.length?'completed':'partial',human_listening_basis:'unverified_self_report',human_listening_verified:false,asset_hash_validation:'at_HTML_generation_only_not_rechecked_during_browser_playback',automatic_quality_approval:false,not_a_blinded_experiment:true,completed_case_count:completed.length,total_case_count:study.cases.length,records:completed.map(item=>({...responses.get(item.case_id),reference:item.reference??null,variants:item.variants}))};}/* end exportPayload */
study.cases.forEach((item,index)=>{const section=text('article','','case'),top=text('div','','case-top'),heading=document.createElement('div');heading.append(text('div',String(index+1).padStart(2,'0')+' / '+String(study.cases.length).padStart(2,'0'),'number'),text('h2',item.title));top.append(heading);section.append(top,text('span',item.kind_label,'tag'),text('p',item.context));
const explanation=item.kind==='constructed_mixture'?'This is a controlled constructed mixture, not a naturally noisy/clean field pair. Separately recorded hum was added to wanted audio.':item.kind==='desired_only'?'No hum was intentionally added. Listen for unwanted damage to the original content.':'There is no independent clean recording of this scene. Wanted-content preservation and true remaining-noise levels cannot be established from this clip alone.';
section.append(text('p',explanation,'instruction'));const audioGrid=text('div','','players');if(item.reference)audioGrid.append(player(item.reference,true));item.variants.forEach(asset=>audioGrid.append(player(asset,false)));section.append(audioGrid,text('p','Listen manually, then rate the single variant selected below. These are named comparisons, not a blinded listening experiment.','instruction'));
const grid=text('div','','ratings'),controls={};const fields=[['evaluated_variant','Variant being rated',item.variants.map(v=>[v.label,v.label])],['residual_hum','Residual hum in the selected variant',levels],['wanted_loss','Wanted-content loss in the selected variant',levels],['distortion','New distortion / artifacts in the selected variant',levels],['preference','Overall preference for this case',[...item.variants.map(v=>[v.label,v.label]),['no_preference','No preference'],['unsure','Unsure / not assessable']]]];
fields.forEach(([key,label,options])=>{const [wrapper,control]=choose(label,options,item.case_id+'-'+key);controls[key]=control;grid.append(wrapper);});const notesLabel=text('label','Optional notes (do not include personal or sensitive information)','full'),notes=document.createElement('textarea');notes.maxLength=4000;notes.id=item.case_id+'-notes';notesLabel.htmlFor=notes.id;notesLabel.append(notes);grid.append(notesLabel);section.append(grid);
const confirmation=text('label','','confirm'),check=document.createElement('input');check.type='checkbox';check.id=item.case_id+'-confirm';confirmation.htmlFor=check.id;confirmation.append(check,text('span','I personally listened and completed these responses. An “unsure” answer is acceptable.'));section.append(confirmation);const state=text('div','Not completed · no human quality judgment recorded.','case-status');state.setAttribute('role','status');section.append(state);
function update(){const values=Object.fromEntries(Object.entries(controls).map(([key,control])=>[key,control.value]));if(check.checked&&Object.values(values).every(Boolean)){responses.set(item.case_id,{case_id:item.case_id,kind:item.kind,...values,notes:notes.value,listener_self_reported_listened:true});state.textContent='Completed by self-report · included in the next export.';}else{responses.delete(item.case_id);state.textContent='Not completed · excluded from the export.';}document.getElementById('progress').textContent=responses.size+' of '+study.cases.length+' cases completed by self-report.';document.getElementById('export').disabled=responses.size===0;document.getElementById('export-status').textContent='';}
Object.entries(controls).forEach(([key,control])=>control.addEventListener('change',()=>{check.checked=false;if(key==='evaluated_variant'){['residual_hum','wanted_loss','distortion'].forEach(field=>{controls[field].value='';});}update();}));check.addEventListener('change',update);notes.addEventListener('input',()=>{check.checked=false;update();});document.getElementById('cases').append(section);});
document.getElementById('export').addEventListener('click',()=>{if(!responses.size)return;const output=exportPayload(study,responses,new Date().toISOString());const url=URL.createObjectURL(new Blob([JSON.stringify(output,null,2)+'\n'],{type:'application/json'})),link=document.createElement('a');link.href=url;link.download='hum-dsp-manual-review-'+new Date().toISOString().slice(0,10)+'.json';document.body.append(link);link.click();link.remove();setTimeout(()=>URL.revokeObjectURL(url),30000);document.getElementById('export-status').textContent='JSON export requested.';});
</script></body></html>'''
