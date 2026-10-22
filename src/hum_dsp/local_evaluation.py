"""Scoped M5 integration evidence, not another noise-reduction benchmark.

This repository-only evaluator compares the local UI processing bridge with
actual offline CLI subprocesses. It neither serves HTTP nor opens an audio
device. Its owned mathematical inputs are not independent real-world evidence.
"""

import base64
import hashlib
import html
import io
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch
import zipfile

import numpy as np

from .audio import SAMPLE_RATE_HZ, read_pcm16_wav
from .config import ProcessorConfig
from .local_app import _example_wav, _run_job


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SOURCE_ROOT = Path(__file__).resolve().parent
_COMMANDS = {"fixed": "process", "tracked": "process-tracked", "guarded": "process-guarded"}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path, record):
    Path(path).write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _preflight():
    baseline = _PROJECT_ROOT / "outputs" / "m4_20261021_v2" / "benchmark.json"
    milestones_path = _PROJECT_ROOT / "milestones.json"
    scope_path = _SOURCE_ROOT / "m5_scope.json"
    record = _json(baseline)
    scope = _json(scope_path)
    if record.get("m4_development_targets_met") is not True or scope.get("engine_changes_allowed") is not False:
        raise ValueError("M4 evidence and frozen M5 scope are required before evaluation")
    expected = {}
    for field in ("preserved_m3_source_sha256", "source_sha256"):
        mapping = record[field]
        if not isinstance(mapping, dict) or not mapping:
            raise ValueError("frozen source hash evidence is missing")
        for name, digest in mapping.items():
            if Path(name).name != name or len(digest) != 64:
                raise ValueError("invalid frozen source evidence")
            if name in expected and expected[name] != digest:
                raise ValueError("conflicting frozen source hashes")
            expected[name] = digest
    m2 = next(item for item in _json(milestones_path)["milestones"] if item["id"] == "M2")
    expected["processing.py"] = m2["evidence"]["source_sha256"]["src/hum_dsp/processing.py"]
    current = {name: _sha(_SOURCE_ROOT / name) for name in expected}
    if current != expected:
        raise ValueError("frozen M1-M4 source hashes differ; evaluation refused")
    return {
        "m4_benchmark_sha256": _sha(baseline),
        "m5_scope_sha256": _sha(scope_path),
        "frozen_engine_source_sha256": expected,
        "local_source_sha256": {name: _sha(_SOURCE_ROOT / name) for name in ("local_app.py", "local_evaluation.py")},
        "scope": scope,
    }


def _run_cli(source, destination, mode, config):
    """Use the real CLI entry point, not a Python call to its selected engine."""
    command = [
        sys.executable, "-c", "from hum_dsp.cli import main; raise SystemExit(main())",
        _COMMANDS[mode], str(Path(source).resolve()), "--output-dir", str(Path(destination).resolve()),
        "--hum-hz", str(int(config.nominal_hum_hz)),
        "--attenuation-db", str(config.max_attenuation_db),
    ]
    environment = dict(os.environ)
    source_path = str(_PROJECT_ROOT / "src")
    environment["PYTHONPATH"] = source_path + (os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else "")
    completed = subprocess.run(
        command, cwd=_PROJECT_ROOT, env=environment, capture_output=True,
        text=True, encoding="utf-8", errors="replace", timeout=120,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        check=False,
    )
    # stdout contains local absolute paths; do not copy it into shared reports.
    return {
        "exit_code": completed.returncode,
        "entry_point": "sys.executable -c hum_dsp.cli.main",
        "command": _COMMANDS[mode],
        "stdout_sha256": hashlib.sha256(completed.stdout.encode("utf-8")).hexdigest(),
        "stderr_sha256": hashlib.sha256(completed.stderr.encode("utf-8")).hexdigest(),
        "raw_console_output_stored": False,
    }


def _tracking_description(mode, report):
    if mode == "guarded":
        return report["algorithm"]["raw_tracker"]
    if mode == "tracked":
        return report["algorithm"]
    return None


def _acquisition(mode, report, trace):
    description = _tracking_description(mode, report)
    if description is None or report["config"]["max_attenuation_db"] == 0:
        return {"required": False, "met": True, "first_acquired_sample": None}
    first = description.get("first_acquired_sample")
    trace_acquired = any(event.get("estimated_frequency_hz") is not None and event.get("acquired") is True for event in trace)
    met = type(first) is int and first >= 96000 and description.get("acquired") is True and trace_acquired
    mix = report["algorithm"].get("mix") if mode == "guarded" else None
    if mode == "guarded":
        met = met and isinstance(mix, (int, float)) and 0 < mix <= 1
    return {"required": True, "met": bool(met), "first_acquired_sample": first, "final_guard_mix": mix}


def _case(target, work_root, mode, frequency, depth):
    case_id = f"{mode}_{frequency}hz_{int(depth)}db"
    directory = target / case_id
    directory.mkdir()
    config = ProcessorConfig(nominal_hum_hz=frequency, max_attenuation_db=depth)
    input_bytes = _example_wav(config)
    source = directory / "input.wav"
    source.write_bytes(input_bytes)
    input_hash = _sha(source)
    fft_calls = 0
    original_fft = np.fft.rfft

    def counted_fft(*args, **kwargs):
        nonlocal fft_calls
        fft_calls += 1
        return original_fft(*args, **kwargs)

    # This records invocation count only; all numerical work delegates unchanged.
    with patch.object(np.fft, "rfft", counted_fft):
        response = json.loads(_run_job(SimpleNamespace(work_root=work_root), input_bytes, mode, config, "owned_synthetic_example"))
    ui_bytes = base64.b64decode(response["output_wav_base64"], validate=True)
    (directory / "ui_processed.wav").write_bytes(ui_bytes)
    _write_json(directory / "ui_report.json", response["report"])
    archive_bytes = base64.b64decode(response["download_zip_base64"], validate=True)
    (directory / "ui_result.zip").write_bytes(archive_bytes)
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        trace = json.loads(archive.read("tracking.json"))["events"] if mode != "fixed" else []
        zip_output_match = archive.read("processed.wav") == ui_bytes
        zip_report_match = json.loads(archive.read("report.json")) == response["report"]
    cli_path = directory / "cli"
    cli = _run_cli(source, cli_path, mode, config)
    cli_available = cli["exit_code"] == 0 and (cli_path / "processed.wav").is_file() and (cli_path / "report.json").is_file()
    cli_report = _json(cli_path / "report.json") if cli_available else None
    output = read_pcm16_wav(directory / "ui_processed.wav")
    original = read_pcm16_wav(source)
    ui_report = response["report"]
    byte_equal = cli_available and (cli_path / "processed.wav").read_bytes() == ui_bytes
    config_match = cli_available and ui_report["config"] == cli_report["config"] == config.to_dict()
    counts_match = cli_available and int(output.size) == int(original.size) == ui_report["input"]["frames"] == ui_report["output"]["frames"] == cli_report["input"]["frames"] == cli_report["output"]["frames"]
    changed = not np.array_equal(original, output)
    temp_clean = not list(work_root.iterdir())
    input_unchanged = _sha(source) == input_hash and base64.b64decode(response["input_wav_base64"], validate=True) == input_bytes
    acquisition = _acquisition(mode, ui_report, trace)
    if cli_available and mode != "fixed":
        cli_trace_path = cli_path / "tracking.json"
        cli_trace = _json(cli_trace_path)["events"] if cli_trace_path.is_file() else []
    else:
        cli_trace = []
    cli_acquisition = _acquisition(mode, cli_report, cli_trace) if cli_available else {"required": depth > 0 and mode != "fixed", "met": False}
    bypass = None
    if depth == 0:
        tracking_description = _tracking_description(mode, ui_report)
        no_acquisition = tracking_description is None or (tracking_description.get("first_acquired_sample") is None and tracking_description.get("acquired") is False and tracking_description.get("trace_event_count") == 0)
        bypass = {"exact_pcm16_identity": not changed, "exact_wav_byte_identity": input_bytes == ui_bytes, "local_path_fft_call_count": fft_calls, "no_tracking_events": not trace, "no_acquisition": no_acquisition}
        mode_target = all(value is True for key, value in bypass.items() if key != "local_path_fft_call_count") and fft_calls == 0
        if cli_available and mode != "fixed":
            cli_description = _tracking_description(mode, cli_report)
            mode_target = mode_target and not cli_trace and cli_description.get("first_acquired_sample") is None and cli_description.get("acquired") is False and cli_description.get("trace_event_count") == 0
    else:
        mode_target = changed and acquisition["met"] and cli_acquisition["met"]
    checks = {
        "actual_cli_output_matches_ui_bridge": bool(byte_equal),
        "configuration_consistent": bool(config_match),
        "frame_counts_preserved": bool(counts_match),
        "input_unchanged": input_unchanged,
        "temporary_jobs_cleaned": temp_clean,
        "zip_output_matches": zip_output_match,
        "zip_report_matches": zip_report_match,
        "declared_mode_target_met": bool(mode_target),
        "ui_report_output_hash_matches": ui_report["output"]["sha256"] == hashlib.sha256(ui_bytes).hexdigest(),
    }
    result = {
        "case_id": case_id, "mode": mode, "config": config.to_dict(),
        "source_kind": "owned_synthetic_example", "duration_seconds": response["duration_seconds"],
        "frames": int(original.size), "pcm16_changed": changed,
        "local_path_fft_call_count": fft_calls,
        "acquisition": acquisition, "cli_acquisition": cli_acquisition, "zero_depth": bypass,
        "checks": checks, "all_checks_met": all(checks.values()), "cli": cli,
        "input_sha256": input_hash, "ui_output_sha256": hashlib.sha256(ui_bytes).hexdigest(),
        "artifacts": {name: {"relative_file": f"{case_id}/{name}", "sha256": _sha(directory / name)} for name in ("input.wav", "ui_processed.wav", "ui_report.json", "ui_result.zip")},
    }
    _write_json(directory / "case.json", result)
    return result


def _malformed_case(work_root):
    refused = False
    try:
        _run_job(SimpleNamespace(work_root=work_root), b"not an audio file", "guarded", ProcessorConfig(), "user_file")
    except (ValueError, EOFError):
        refused = True
    clean = not list(work_root.iterdir())
    return {"refused": refused, "temporary_jobs_cleaned": clean, "successful_payload_created": not refused, "all_checks_met": refused and clean}


def _html_report(report):
    rows = []
    for case in report["cases"]:
        identity = "unchanged bypass" if case["config"]["max_attenuation_db"] == 0 else "changed as requested"
        status = "PASS" if case["all_checks_met"] else "FAIL"
        prefix = html.escape(case["case_id"], quote=True)
        rows.append(f'<tr><td>{prefix}</td><td>{status}</td><td>{identity}</td><td>{case["local_path_fft_call_count"]}</td><td><a href="{prefix}/input.wav">Input</a> · <a href="{prefix}/ui_processed.wav">Processed WAV</a> · <a href="{prefix}/ui_report.json">Report</a> · <a href="{prefix}/ui_result.zip">ZIP</a></td></tr>')
    status = "PASS" if report["all_scoped_checks_met"] else "FAIL"
    return """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>M5 local integration evidence</title><style>body{font:16px/1.6 system-ui;max-width:1100px;margin:36px auto;padding:0 20px;color:#18352f;background:#f6f9f6}table{border-collapse:collapse;width:100%;background:white}td,th{padding:10px;border-bottom:1px solid #d9e2dc;text-align:left}a{color:#176c59}.scroll{overflow:auto}</style><h1>M5 local integration evidence: """ + status + """</h1><p>Nine owned six-second mathematical examples compare the actual offline CLI with the local UI processing bridge. Six use 12 dB and three use exact 0 dB bypass. No microphone, speaker, playback or HTTP service is started by this evaluation.</p><p>This proves integration parity and declared bypass behavior only. It does not establish real speech/music quality, independent noise-reduction performance, real-time device behavior or commercial readiness. Listening links are manual; no autoplay is embedded.</p><div class="scroll"><table><thead><tr><th>Case</th><th>Result</th><th>PCM16 output</th><th>Local FFT calls</th><th>Artifacts</th></tr></thead><tbody>""" + "".join(rows) + """</tbody></table></div><p>Malformed input refused and temporary files removed: """ + str(report["malformed_input"]["all_checks_met"]) + """. Frozen M1–M4 sources and evidence preserved: """ + str(report["frozen_evidence_preserved"]) + """.</p><p><a href="m5_local_evaluation.json">Complete JSON evidence</a></p></html>"""


def run_local_evaluation(output_directory):
    """Write new repository-local parity evidence; never overwrite prior output."""
    if not isinstance(output_directory, (str, os.PathLike)) or not str(output_directory).strip():
        raise ValueError("output directory must be a nonempty path")
    target = Path(output_directory)
    if target.exists():
        raise FileExistsError("output directory must be new; existing results are never replaced")
    provenance = _preflight()
    target.mkdir(parents=True, exist_ok=False)
    work_root = target / "temporary_jobs"
    work_root.mkdir()
    cases = [_case(target, work_root, mode, frequency, 12) for mode in _COMMANDS for frequency in (50, 60)]
    cases.extend(_case(target, work_root, mode, 50, 0) for mode in _COMMANDS)
    malformed = _malformed_case(work_root)
    after = _preflight()
    frozen_preserved = provenance == after
    report = {
        "schema_version": 1, "stage": "M5_local_tool_scoped_integration",
        "scope": "nine_owned_synthetic_examples_actual_CLI_vs_local_UI_bridge_not_independent_audio_quality",
        "case_count": len(cases), "nonzero_case_count": 6, "zero_depth_case_count": 3,
        "actual_cli_parity_count": sum(case["checks"]["actual_cli_output_matches_ui_bridge"] for case in cases),
        "zero_depth_targets_met_count": sum(case["all_checks_met"] for case in cases if case["zero_depth"] is not None),
        "cases": cases, "malformed_input": malformed,
        "frozen_evidence_preserved": frozen_preserved, "provenance": provenance,
        "all_scoped_checks_met": frozen_preserved and all(case["all_checks_met"] for case in cases) and malformed["all_checks_met"],
        "microphone_capture": False, "audio_playback": False, "network_or_HTTP_serving": False,
        "DSP_parameter_or_algorithm_changes": False,
        "FFT_instrumentation": "Only local bridge calls are counted by a temporary wrapper delegating all arguments and numerical work to the unchanged original np.fft.rfft; restored after each case. CLI invocation counts are not instrumented.",
        "limitations": [
            "The local UI bridge is tested directly here; HTTP security and browser behavior are separate checks.",
            "Only owned mathematical examples are used; no independent or perceptual quality conclusion is justified.",
            "The CLI output is generated by a real Python subprocess for each case; timing is not a real-time device benchmark.",
            "All modes still attenuate desired content at the selected frequency.",
        ],
    }
    _write_json(target / "m5_local_evaluation.json", report)
    (target / "report.html").write_text(_html_report(report), encoding="utf-8")
    return report
