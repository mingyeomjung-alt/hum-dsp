"""Paired M3 evaluation; fixture truth is used here, never by the engine."""

import hashlib
import html
import json
from pathlib import Path

import numpy as np

from .audio import read_pcm16_wav, write_pcm16_wav
from .config import ProcessorConfig
from .evaluation import measure_windows
from .fixtures import _dump_json, fixture_specs, verify_fixture_suite
from .processing import _input_sha256
from .tracked_processing import render_tracked
from .tracking_evaluation import measure_tracking


def run_tracking_benchmark(suite, baseline_directory, output_directory) -> dict:
    inputs = Path(suite).resolve(strict=True)
    baseline = Path(baseline_directory).resolve(strict=True)
    target = Path(output_directory)
    verified = verify_fixture_suite(inputs)
    module = Path(__file__).parent
    candidate = json.loads((module / "m3_candidate.json").read_text(encoding="utf-8"))
    for name, key in (("filter.py", "baseline_filter_sha256"), ("processing.py", "baseline_processing_sha256")):
        if hashlib.sha256((module / name).read_bytes()).hexdigest() != candidate[key]:
            raise ValueError("frozen M2 baseline source has changed")
    baseline_manifest = json.loads((baseline / "benchmark.json").read_text(encoding="utf-8"))
    if baseline_manifest.get("stage") != "M2" or baseline_manifest.get("case_count") != 16:
        raise ValueError("baseline must be the preserved sixteen-case M2 result")
    baseline_cases = {item["case_id"]: item for item in baseline_manifest["cases"]}
    specs = fixture_specs()
    if set(baseline_cases) != {spec.case_id for spec in specs}:
        raise ValueError("baseline case inventory differs from the frozen suite")
    target.mkdir(parents=True, exist_ok=False)
    cases = []
    for spec in specs:
        folder = inputs / spec.case_id
        source = folder / "input.wav"
        samples = read_pcm16_wav(source)
        clean = read_pcm16_wav(folder / "clean.wav")
        noise = samples - clean
        settings = ProcessorConfig(nominal_hum_hz=spec.nominal_hum_hz)
        prior_report = json.loads((baseline / spec.case_id / "report.json").read_text(encoding="utf-8"))
        prior_wav = baseline / spec.case_id / "processed.wav"
        if prior_report["config"] != settings.to_dict() or prior_report["input"]["sha256"] != _input_sha256(source):
            raise ValueError("baseline input/settings differ from candidate")
        if hashlib.sha256(prior_wav.read_bytes()).hexdigest() != baseline_cases[spec.case_id]["output"]["sha256"]:
            raise ValueError("preserved baseline WAV changed")
        baseline_audio = read_pcm16_wav(prior_wav)
        if baseline_audio.shape != samples.shape:
            raise ValueError("baseline frame count differs")
        floating, algorithm, trace = render_tracked(samples, settings)
        case_target = target / spec.case_id
        case_target.mkdir()
        write_pcm16_wav(case_target / "processed.wav", floating)
        saved = read_pcm16_wav(case_target / "processed.wav")
        _dump_json(case_target / "tracking.json", {"schema_version": 1, "sample_rate_hz": 48000, "events": trace})
        candidate_metrics = measure_windows(clean, noise, saved)
        baseline_metrics = measure_windows(clean, noise, baseline_audio)
        floating_metrics = measure_windows(clean, noise, floating)
        frequency = measure_tracking(trace, spec, frames=int(samples.size))
        noise_only = spec.kind in ("stationary_target", "moving_target")
        before_db = baseline_metrics["settled"]["reference_error_reduction_db"] if noise_only else None
        after_db = candidate_metrics["settled"]["reference_error_reduction_db"] if noise_only else None
        attenuation_met = None
        if spec.kind == "stationary_target":
            attenuation_met = after_db is not None and after_db >= 10.0
        if spec.kind == "moving_target":
            attenuation_met = after_db is not None and before_db is not None and after_db >= 6.0 and after_db > before_db
        record = {
            "case_id": spec.case_id,
            "kind": spec.kind,
            "expected_use": spec.expected_use,
            "config": settings.to_dict(),
            "algorithm": algorithm,
            "tracking": frequency,
            "baseline_noise_only_attenuation_db": before_db,
            "candidate_noise_only_attenuation_db": after_db,
            "attenuation_target_met": attenuation_met,
            "baseline_decoded_wav_measurements": baseline_metrics,
            "candidate_decoded_wav_measurements": candidate_metrics,
            "candidate_float_measurements": floating_metrics,
            "output": {"file_name": "processed.wav", "frames": int(saved.size), "sha256": hashlib.sha256((case_target / "processed.wav").read_bytes()).hexdigest()},
            "input_unchanged": _input_sha256(source) == prior_report["input"]["sha256"],
            "frame_count_preserved": saved.size == samples.size,
            "pcm16_quantization_max_abs_error": float(np.max(np.abs(saved - floating))),
            "desired_content_preservation_guaranteed": False,
        }
        _dump_json(case_target / "measurements.json", record)
        cases.append(record)
    attenuated = [case for case in cases if case["attenuation_target_met"] is not None]
    frequency_cases = [case for case in cases if case["tracking"]["frequency_truth_available"]]
    attenuation_ok = len(attenuated) == 4 and all(case["attenuation_target_met"] for case in attenuated)
    frequency_ok = len(frequency_cases) == 10 and all(case["tracking"]["frequency_target_met"] for case in frequency_cases)
    sources = ("tracking.py", "filter.py", "config.py", "tracked_processing.py", "tracking_evaluation.py", "tracking_benchmark.py", "m3_candidate.json", "protocol.json")
    report = {
        "schema_version": 1,
        "stage": "M3",
        "candidate": candidate,
        "candidate_revision": candidate["revision"],
        "repeated_development_evaluation_is_not_independent_validation": True,
        "scope": "paired_owned_synthetic_development_conditions_not_independent_real_audio_validation",
        "fixture_verification": verified,
        "baseline_benchmark_sha256": hashlib.sha256((baseline / "benchmark.json").read_bytes()).hexdigest(),
        "source_sha256": {name: hashlib.sha256((module / name).read_bytes()).hexdigest() for name in sources},
        "case_count": len(cases),
        "attenuation_target_case_count": len(attenuated),
        "frequency_target_case_count": len(frequency_cases),
        "all_attenuation_targets_met": attenuation_ok,
        "all_frequency_targets_met": frequency_ok,
        "m3_development_targets_met": attenuation_ok and frequency_ok,
        "all_input_files_unchanged": all(case["input_unchanged"] for case in cases),
        "all_frame_counts_preserved": all(case["frame_count_preserved"] for case in cases),
        "real_time_device_performance_verified": False,
        "cases": cases,
        "interpretation": [
            "Same decoded input, depth, pole-design bandwidth, zero-state start, and fixed 2s settled boundary as M2.",
            "Only target-only RMS reduction is noise attenuation. Mixture errors include wanted-signal distortion.",
            "Both estimator and held applied-center errors use current causal truth time, including every post-boundary output sample.",
            "Missing/held estimates and the initial acquisition jump are retained in traces and whole/startup metrics.",
            "A desired same-frequency tone can be tracked and suppressed. Heuristic confidence does not resolve semantics.",
            "Coefficient-transition audibility and protection controls remain M4 work; no real audio/device claim.",
        ],
        "remaining_development_milestones_if_targets_met": ["M4", "M5"],
    }
    _dump_json(target / "benchmark.json", report)
    _write_html(target / "report.html", report)
    return report


def _write_html(path: Path, report: dict) -> None:
    rows = []
    for case in report["cases"]:
        old = case["baseline_noise_only_attenuation_db"]
        new = case["candidate_noise_only_attenuation_db"]
        old_text = "not noise-only" if old is None else f"{old:.2f} dB"
        new_text = "not noise-only" if new is None else f"{new:.2f} dB"
        error = case["tracking"]["all_post_boundary_samples_applied_center_max_error_hz"]
        error_text = "no noise truth" if error is None else f"{error:.4f} Hz"
        rows.append(f"<tr><td>{html.escape(case['case_id'])}</td><td>{old_text}</td><td>{new_text}</td><td>{error_text}</td><td><a href='{case['case_id']}/processed.wav'>WAV</a> / <a href='{case['case_id']}/tracking.json'>Trace</a></td></tr>")
    page = (
        "<!doctype html><html lang='en'><meta charset='utf-8'><title>Hum DSP M3 tracking comparison</title>"
        "<style>body{font:16px system-ui;max-width:1250px;margin:40px auto;padding:0 20px;color:#16222a}table{border-collapse:collapse;width:100%;font-size:14px}td,th{border:1px solid #ccd5da;padding:10px;text-align:left}.note{padding:16px;background:#fff4da;line-height:1.6}</style>"
        "<h1>Hum DSP — M3 causal center tracking</h1>"
        "<p class='note'>Synthetic development comparison, not independent real-world performance. "
        "A manually selected band is required. Wanted tones at the same frequency are suppressed too. "
        "Only filter center changes: depth 12 dB and pole-design bandwidth 1 Hz match M2. "
        "First estimate uses two seconds of past input; no microphone, speaker, cloud or network was used.</p>"
        f"<p>Development targets met: {str(report['m3_development_targets_met']).lower()}. Cases: {report['case_count']}. "
        "Table shows measurements after the unchanged 2 s boundary. Startup and whole-record errors remain in JSON.</p>"
        "<table><thead><tr><th>Fixture</th><th>M2 fixed attenuation</th><th>M3 tracked attenuation</th><th>Max applied-center error (all settled samples)</th><th>Outputs</th></tr></thead><tbody>"
        + "".join(rows) + "</tbody></table>"
        "<p>Mixtures are not labeled as noise-only attenuation. Missing estimates, held centers, frequency-estimation error, acquisition time and source hashes are in <a href='benchmark.json'>full JSON evidence</a>.</p></html>"
    )
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(page)

