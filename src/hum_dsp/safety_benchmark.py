"""M4 guard evaluation against preserved M3 evidence and owned stress inputs.

This is a repeated synthetic development evaluation, not an independent quality
or real-device safety certification. Truth is consumed only by this harness.
"""

import hashlib
import html
import json
from pathlib import Path

import numpy as np

from .audio import PCM16_STEP, SAMPLE_RATE_HZ, read_pcm16_wav, write_pcm16_wav
from .config import ProcessorConfig
from .evaluation import measure_windows
from .fixtures import _dump_json, fixture_specs, verify_fixture_suite
from .guarded_processing import render_guarded
from .processing import _input_sha256
from .safety_evaluation import measure_guard, reconstruct_mix
from .safety_fixtures import safety_fixtures
from .tracked_processing import render_tracked


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _save_result(folder, floating, guard, tracking):
    write_pcm16_wav(folder / "processed.wav", floating)
    for name, trace in (("guard", guard), ("tracking", tracking)):
        _dump_json(folder / f"{name}.json", {"schema_version": 1, "sample_rate_hz": SAMPLE_RATE_HZ, "events": trace})
    return read_pcm16_wav(folder / "processed.wav")


def _guard_targets(evidence, tolerance):
    return (
        evidence["mix_min"] >= 0.0
        and evidence["mix_max"] <= 1.0
        and evidence["maximum_mix_sample_step"] <= evidence["theoretical_attack_step_bound"] + 1e-10
        and evidence["reconstruction_max_error"] <= tolerance
    )


def _recovery(guard, evidence, noise_end):
    """Report source-offset latency separately from stale-observation hold."""
    final = evidence["final_dry_from_sample"]
    releases = [event for event in guard if event["target_changed"] and event["target_mix"] == 0]
    release = releases[-1] if releases else None
    qualified = None if release is None else release["last_qualified_sample"]
    return {
        "source_noise_end_sample": noise_end,
        "last_qualified_sample_before_release": qualified,
        "release_start_sample": None if release is None else release["sample_index"],
        "final_dry_from_sample": final,
        "source_end_to_final_dry_seconds": None if noise_end is None or final is None else (final - noise_end) / SAMPLE_RATE_HZ,
        "last_qualified_to_release_seconds": None if qualified is None else (release["sample_index"] - qualified) / SAMPLE_RATE_HZ,
        "release_to_final_dry_seconds": None if release is None or final is None else (final - release["sample_index"]) / SAMPLE_RATE_HZ,
        "latency_scope": "includes_trailing_estimator_history_not_only_hold_plus_release",
    }


def run_safety_benchmark(suite, baseline_directory, output_directory):
    inputs = Path(suite).resolve(strict=True)
    baseline = Path(baseline_directory).resolve(strict=True)
    target = Path(output_directory)
    if target.exists():
        raise FileExistsError("benchmark output directory must be new")
    verified = verify_fixture_suite(inputs)
    module = Path(__file__).parent
    policy = json.loads((module / "m4_policy.json").read_text(encoding="utf-8"))
    limits = policy["targets"]
    if _sha(inputs / "manifest.json") != policy["frozen_fixture_manifest_sha256"]:
        raise ValueError("inputs differ from the frozen M1 suite used for the M3 baseline")
    if _sha(baseline / "benchmark.json") != policy["frozen_m3_benchmark_sha256"]:
        raise ValueError("M3 benchmark manifest differs from frozen baseline provenance")
    previous = json.loads((baseline / "benchmark.json").read_text(encoding="utf-8"))
    if previous.get("stage") != "M3" or previous.get("case_count") != 16:
        raise ValueError("a preserved sixteen-case M3 baseline is required")
    expected_sources = {"config.py", "filter.py", "m3_candidate.json", "protocol.json", "tracked_processing.py", "tracking.py", "tracking_benchmark.py", "tracking_evaluation.py"}
    if set(previous.get("source_sha256", {})) != expected_sources:
        raise ValueError("M3 source hash inventory is incomplete or unexpected")
    for name, expected in previous["source_sha256"].items():
        if Path(name).name != name or _sha(module / name) != expected:
            raise ValueError("frozen M3 source differs from baseline evidence")
    for name, key in (("filter.py", "m2_filter_sha256"), ("tracking.py", "m3_tracking_sha256")):
        if _sha(module / name) != policy[key]:
            raise ValueError("M4 policy baseline source changed")
    prior_cases = {case["case_id"]: case for case in previous["cases"]}
    specs = fixture_specs()
    if len(prior_cases) != 16 or set(prior_cases) != {spec.case_id for spec in specs}:
        raise ValueError("M3 case inventory differs from fixed M1 suite")
    # Refuse corrupt baseline evidence before creating this run's directory.
    for spec in specs:
        prior = baseline / spec.case_id / "processed.wav"
        if _sha(prior) != prior_cases[spec.case_id]["output"]["sha256"]:
            raise ValueError("preserved M3 output changed")
        if prior_cases[spec.case_id]["config"] != ProcessorConfig(nominal_hum_hz=spec.nominal_hum_hz).to_dict():
            raise ValueError("M3 settings differ from the declared paired settings")
        if read_pcm16_wav(prior).shape != read_pcm16_wav(inputs / spec.case_id / "input.wav").shape:
            raise ValueError("M3 baseline frame count differs from the frozen input")
    target.mkdir(parents=True, exist_ok=False)
    retained = []
    for spec in specs:
        folder = inputs / spec.case_id
        source_sha = _input_sha256(folder / "input.wav")
        source, clean = [read_pcm16_wav(folder / f"{name}.wav") for name in ("input", "clean")]
        raw = read_pcm16_wav(baseline / spec.case_id / "processed.wav")
        settings = ProcessorConfig(nominal_hum_hz=spec.nominal_hum_hz)
        if prior_cases[spec.case_id]["config"] != settings.to_dict() or raw.shape != source.shape:
            raise ValueError("M3 input shape or settings differ")
        floating, description, guard, tracking = render_guarded(source, settings)
        case_target = target / spec.case_id
        case_target.mkdir()
        saved = _save_result(case_target, floating, guard, tracking)
        measured = measure_windows(clean, source - clean, saved)
        evidence = measure_guard(source, raw, floating, guard)
        # M3 reference here is decoded PCM16; reconstruction retains that bound.
        envelope_met = _guard_targets(evidence, PCM16_STEP)
        noise_only = spec.kind in ("stationary_target", "moving_target")
        before = prior_cases[spec.case_id]["candidate_noise_only_attenuation_db"] if noise_only else None
        after = measured["settled"]["reference_error_reduction_db"] if noise_only else None
        regression = None if before is None or after is None else before - after
        attenuation_met = None
        if noise_only:
            key = "steady_development_stationary_attenuation_db_min" if spec.kind == "stationary_target" else "steady_development_moving_attenuation_db_min"
            attenuation_met = bool(after is not None and regression is not None and after >= limits[key]
                                   and regression <= limits["noise_only_attenuation_regression_vs_m3_db_max"])
        record = {
            "case_id": spec.case_id, "kind": spec.kind, "config": settings.to_dict(), "algorithm": description,
            "guard_measurements": evidence, "envelope_targets_met": envelope_met,
            "reconstruction_raw_reference": "preserved_M3_PCM16_output_tolerance_one_LSB",
            "transition_metrics_role": "diagnostic_quantized_M3_reference_not_an_inaudibility_claim",
            "m3_noise_only_attenuation_db": before, "m4_noise_only_attenuation_db": after,
            "attenuation_loss_vs_m3_db": regression, "attenuation_target_met": attenuation_met,
            "decoded_wav_measurements": measured,
            "float_measurements": measure_windows(clean, source - clean, floating),
            "input_unchanged": _input_sha256(folder / "input.wav") == source_sha,
            "frame_count_preserved": saved.size == source.size,
            "output_sha256": _sha(case_target / "processed.wav"),
        }
        _dump_json(case_target / "measurements.json", record)
        retained.append(record)
    stress = []
    for spec in safety_fixtures():
        folder = target / spec.case_id
        folder.mkdir()
        for name, samples in (("input", spec.input_audio), ("clean", spec.clean), ("noise", spec.noise)):
            write_pcm16_wav(folder / f"{name}.wav", samples)
        source, clean = [read_pcm16_wav(folder / f"{name}.wav") for name in ("input", "clean")]
        before_sha = _sha(folder / "input.wav")
        settings = ProcessorConfig(nominal_hum_hz=spec.nominal_hum_hz)
        floating, description, guard, tracking = render_guarded(source, settings, controls=spec.controls)
        raw, _, _ = render_tracked(source, settings)
        saved = _save_result(folder, floating, guard, tracking)
        evidence = measure_guard(source, raw, floating, guard)
        mix, _ = reconstruct_mix(guard, int(source.size))
        recovery = _recovery(guard, evidence, spec.noise_end_sample)
        envelope_met = _guard_targets(evidence, 1e-12)
        exact_dry_met = None if not spec.expect_exact_dry_everywhere else bool(np.array_equal(source, floating) and np.array_equal(source, saved) and np.all(mix == 0))
        final_dry_met = None if not spec.expected_final_dry else bool(np.all(mix[-SAMPLE_RATE_HZ:] == 0) and np.array_equal(floating[-SAMPLE_RATE_HZ:], source[-SAMPLE_RATE_HZ:]))
        soft_off_met = None
        if spec.disable_interval is not None:
            start, stop = spec.disable_interval
            soft_off_met = bool(np.all(mix[start:stop] == 0) and np.array_equal(floating[start:stop], source[start:stop]) and np.array_equal(saved[start:stop], source[start:stop]))
        recovery_met = None
        if spec.noise_end_sample is not None:
            latency = recovery["source_end_to_final_dry_seconds"]
            recovery_met = bool(latency is not None and 0 <= latency <= limits["tone_end_to_final_dry_seconds_max_including_estimator_history"] and evidence["mix_max"] > 0.99)
        activated = None if spec.expect_exact_dry_everywhere else evidence["mix_max"] > 0.99
        checks = [envelope_met, exact_dry_met, final_dry_met, soft_off_met, recovery_met, activated]
        record = {
            "case_id": spec.case_id, "notes": spec.notes, "config": settings.to_dict(), "controls": spec.controls,
            "measurement_noise_reference": "decoded_input_minus_decoded_clean",
            "reconstruction_raw_reference": "same_decoded_input_frozen_M3_float_output",
            "guard_measurements": evidence, "recovery": recovery, "algorithm": description,
            "envelope_targets_met": envelope_met, "exact_dry_target_met": exact_dry_met,
            "final_dry_target_met": final_dry_met, "soft_off_target_met": soft_off_met,
            "source_offset_recovery_target_met": recovery_met, "activation_observed_when_expected": activated,
            "declared_targets_met": all(check is None or bool(check) for check in checks),
            "decoded_wav_measurements": measure_windows(clean, source - clean, saved),
            "float_measurements": measure_windows(clean, source - clean, floating),
            "wanted_tone_preservation_guaranteed": False,
            "input_unchanged": _sha(folder / "input.wav") == before_sha,
            "frame_count_preserved": saved.size == source.size,
            "output_sha256": _sha(folder / "processed.wav"),
        }
        _dump_json(folder / "measurements.json", record)
        stress.append(record)
    attenuation_cases = [case for case in retained if case["attenuation_target_met"] is not None]
    retained_ok = len(attenuation_cases) == 4 and all(case["attenuation_target_met"] for case in attenuation_cases) and all(case["envelope_targets_met"] for case in retained)
    stress_ok = len(stress) == 10 and all(case["declared_targets_met"] for case in stress)
    all_cases = retained + stress
    suite_reverified = verify_fixture_suite(inputs) == verified
    inputs_unchanged = suite_reverified and all(case["input_unchanged"] for case in all_cases)
    frames_preserved = all(case["frame_count_preserved"] for case in all_cases)
    sources = ("guarded.py", "guarded_processing.py", "safety_fixtures.py", "safety_evaluation.py", "safety_benchmark.py", "m4_policy.json")
    report = {
        "schema_version": 1, "stage": "M4", "policy": policy,
        "scope": "owned_synthetic_development_not_real_world_quality_or_physical_safety_certification",
        "parameter_changes_after_full_evaluation": 0,
        "fixture_verification": verified, "retained_case_count": len(retained), "safety_case_count": len(stress),
        "all_frozen_fixture_files_reverified_after_run": suite_reverified,
        "retained_attenuation_and_envelope_targets_met": retained_ok, "safety_targets_met": stress_ok,
        "all_input_files_unchanged": inputs_unchanged,
        "all_frame_counts_preserved": frames_preserved,
        "m4_development_targets_met": retained_ok and stress_ok and inputs_unchanged and frames_preserved,
        "baseline_manifest_sha256": _sha(baseline / "benchmark.json"),
        "fixture_manifest_sha256": _sha(inputs / "manifest.json"),
        "preserved_m3_source_sha256": previous["source_sha256"],
        "source_sha256": {name: _sha(module / name) for name in sources},
        "retained_cases": retained, "safety_cases": stress,
        "device_io": False, "real_time_device_performance_verified": False,
        "interpretation": [
            "The M4 mix changes effective attenuation; this is not M3 center-only equivalence.",
            "Noise-only attenuation uses the unchanged post-2-second window; whole/startup errors remain.",
            "Release after a source tone ends includes the trailing 2-second observation history.",
            "Residual first differences and high-frequency energy compare the same gate event times to hard switching; neither proves inaudibility.",
            "Same-frequency desired tones are still suppressed; no semantic noise protection or automatic quality approval.",
            "Soft OFF fades back to the input, not silence, and is not an emergency mute.",
        ],
        "remaining_development_milestones_if_targets_met": ["M5"],
    }
    _dump_json(target / "benchmark.json", report)
    _write_html(target / "report.html", report)
    return report


def _write_html(path, report):
    rows = []
    for case in report["retained_cases"]:
        before, after = case["m3_noise_only_attenuation_db"], case["m4_noise_only_attenuation_db"]
        label = "not noise-only" if after is None else f"{before:.3f} → {after:.3f} dB"
        rows.append(f"<tr><td>{html.escape(case['case_id'])}</td><td>{label}</td><td>{case['envelope_targets_met']}</td><td><a href='{case['case_id']}/processed.wav'>Processed WAV</a> / <a href='{case['case_id']}/measurements.json'>JSON</a></td></tr>")
    safety_rows = []
    for case in report["safety_cases"]:
        delay = case["recovery"]["source_end_to_final_dry_seconds"]
        delay_text = "not applicable" if delay is None else f"{delay:.3f} s"
        safety_rows.append(f"<tr><td>{html.escape(case['case_id'])}</td><td>{case['declared_targets_met']}</td><td>{delay_text}</td><td><a href='{case['case_id']}/input.wav'>Input</a> / <a href='{case['case_id']}/processed.wav'>Output</a> / <a href='{case['case_id']}/measurements.json'>Evidence</a></td></tr>")
    page = ("<!doctype html><html lang='en'><meta charset='utf-8'><title>Hum DSP M4 guard</title>"
            "<style>body{font:16px system-ui;max-width:1200px;margin:40px auto;padding:0 20px;color:#17232c}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:10px;border:1px solid #ccd5da;text-align:left}.note{padding:18px;background:#fff0cf;line-height:1.6}</style>"
            "<h1>Hum DSP — M4 distortion/stability controls</h1>"
            "<p class='note'>Synthetic development only. No microphone, playback, cloud, real-audio quality approval or physical safety certification. This opt-in mode starts dry, requires two qualified observations, then uses a 100 ms attack and 250 ms soft release. Recovery also includes stale estimator history. Same-frequency wanted tones remain vulnerable.</p>"
            f"<p>Declared development targets met: {report['m4_development_targets_met']}. Retained conditions: 16. Additional guard cases: 10.</p>"
            "<h2>Attenuation trade-off under the unchanged 2 s boundary</h2><p>The guard reduces startup processing. It may attenuate less than M3. This is not an improved-noise-attenuation claim.</p>"
            "<table><tr><th>Case</th><th>M3 → M4 attenuation</th><th>Envelope checks</th><th>Evidence</th></tr>" + "".join(rows) + "</table>"
            "<h2>Guard stress cases</h2><table><tr><th>Case</th><th>Declared checks</th><th>Source-end → final dry</th><th>Files</th></tr>" + "".join(safety_rows) + "</table>"
            "<p>Hard-switch residual comparisons are objective diagnostics, not a click-free or perceptual audibility guarantee. Soft OFF is not mute. Full traces, whole/startup metrics, source hashes and limitations: <a href='benchmark.json'>benchmark.json</a>.</p></html>")
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(page)
