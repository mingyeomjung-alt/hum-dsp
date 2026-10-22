"""H1 owned-development comparison and frozen-decision distortion evidence."""

import hashlib
import html
import json
from pathlib import Path

import numpy as np

from .audio import SAMPLE_RATE_HZ, read_pcm16_wav, write_pcm16_wav
from .config import ProcessorConfig
from .evaluation import measure_windows
from .fixtures import _dump_json
from .guarded_processing import render_guarded
from .harmonic_fixtures import harmonic_fixtures
from .harmonic_metrics import measure_harmonic_components
from .harmonics import HarmonicConfig, render_harmonic, render_harmonic_replay
from .safety_evaluation import reconstruct_mix


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _preflight():
    source = Path(__file__).parent
    project = source.parents[1]
    legacy = json.loads((project / "outputs/m4_20261021_v2/benchmark.json").read_text(encoding="utf-8"))
    hashes = {**legacy["source_sha256"], **legacy["preserved_m3_source_sha256"]}
    milestones = json.loads((project / "milestones.json").read_text(encoding="utf-8"))
    m2 = next(item for item in milestones["milestones"] if item["id"] == "M2")
    hashes["processing.py"] = m2["evidence"]["source_sha256"]["src/hum_dsp/processing.py"]
    if any(Path(name).name != name or _sha(source / name) != value for name, value in hashes.items()):
        raise ValueError("frozen legacy source changed; paired H1 evaluation refused")
    policy = json.loads((source / "h1_policy.json").read_text(encoding="utf-8"))
    return policy, hashes


def _components(clean, noise, output, replay_clean, replay_noise):
    window = 2 * SAMPLE_RATE_HZ
    return {
        "whole_record": measure_harmonic_components(clean, noise, output, replay_clean, replay_noise),
        "startup": measure_harmonic_components(clean[:window], noise[:window], output[:window], replay_clean[:window], replay_noise[:window]),
        "settled": measure_harmonic_components(clean, noise, output, replay_clean, replay_noise, start_sample=window),
        "settling_samples": window,
        "scope": "float_output_known_reference_oracle_replay_not_reference_free_real_audio_quality",
    }


def _band_usage(trace, description):
    result = []
    for band in description["bands"]:
        order = band["order"]
        events = [next(item for item in event["bands"] if item["order"] == order) for event in trace]
        result.append({
            "order": order,
            "acquired": band["acquired"],
            "first_acquired_sample": band["first_acquired_sample"],
            "maximum_recorded_mix": max((item["mix"] for item in events), default=0.0),
            "final_mix": band["mix"],
            "final_filter_frequency_hz": band["filter_frequency_hz"],
            "post_acquisition_slew_hz_per_s_max": band["maximum_post_acquisition_slew_hz_per_s"],
            "qualified_observation_count": sum(item["qualified"] for item in events),
        })
    return result


def _recovery_diagnostics(trace, frames, source_offset):
    if source_offset is None:
        return None
    result = []
    for initial in trace[0]["bands"]:
        order = initial["order"]
        events = [{**next(b for b in event["bands"] if b["order"] == order), "sample_index": event["sample_index"]} for event in trace]
        mix, _ = reconstruct_mix(events, frames)
        positive = np.flatnonzero(mix > 0)
        last = int(positive[-1] + 1) if positive.size else 0
        final = None if last == frames else last
        result.append({"order": order, "source_offset_sample": source_offset, "final_dry_from_sample": final,
                       "source_offset_to_final_dry_seconds": None if final is None else (final - source_offset) / SAMPLE_RATE_HZ,
                       "latency_scope": "includes_trailing_observation_history_and_release_not_release_only"})
    return result


def run_harmonic_benchmark(output_directory):
    target = Path(output_directory)
    if target.exists():
        raise FileExistsError("output directory must be new; previous evidence is preserved")
    policy, old_hashes = _preflight()
    limits = policy["targets"]
    target.mkdir(parents=True, exist_ok=False)
    cases = []
    for spec in harmonic_fixtures():
        folder = target / spec.case_id
        folder.mkdir()
        for name, samples in (("input", spec.input_audio), ("clean", spec.clean), ("noise", spec.noise)):
            write_pcm16_wav(folder / f"{name}.wav", samples)
        input_hashes = {name: _sha(folder / f"{name}.wav") for name in ("input", "clean", "noise")}
        source, clean = [read_pcm16_wav(folder / f"{name}.wav") for name in ("input", "clean")]
        noise = source - clean
        settings = HarmonicConfig(nominal_hum_hz=spec.nominal_hum_hz)
        output, description, trace = render_harmonic(source, settings)
        mixed_replay = render_harmonic_replay(source, settings, trace)
        replay_clean = render_harmonic_replay(clean, settings, trace)
        replay_noise = render_harmonic_replay(noise, settings, trace)
        write_pcm16_wav(folder / "processed.wav", output)
        decoded = read_pcm16_wav(folder / "processed.wav")
        _dump_json(folder / "harmonics.json", {"schema_version": 1, "sample_rate_hz": SAMPLE_RATE_HZ, "events": trace})
        old_float, old_description, old_guard, old_tracking = render_guarded(source, ProcessorConfig(nominal_hum_hz=spec.nominal_hum_hz))
        write_pcm16_wav(folder / "single_band_M4.wav", old_float)
        old_decoded = read_pcm16_wav(folder / "single_band_M4.wav")
        measured = measure_windows(clean, noise, decoded)
        before = measure_windows(clean, noise, old_decoded)
        components = _components(clean, noise, output, replay_clean, replay_noise)
        settled = components["settled"]
        reconstruction_error = float(np.max(np.abs(output - mixed_replay)))
        replay_ok = reconstruction_error <= limits["replay_mixture_reconstruction_max_abs_error"]
        additive_ok = all(item["component_interpretation_valid"] for key, item in components.items() if key in ("whole_record", "startup", "settled"))
        usage = _band_usage(trace, description)
        baseline_db = before["settled"]["reference_error_reduction_db"]
        output_db = measured["settled"]["reference_error_reduction_db"]
        noise_only = spec.kind in ("stationary_noise_only", "fundamental_only", "missing_fundamental", "abrupt_end", "compound_drift")
        noise_target = None
        clean_target = None
        selective_target = None
        if spec.kind == "stationary_noise_only":
            noise_target = bool(output_db is not None and baseline_db is not None
                                and output_db >= limits["compound_stationary_noise_only_total_attenuation_db_min"]
                                and output_db - baseline_db >= limits["compound_stationary_noise_only_gain_vs_single_band_db_min"])
        if spec.kind == "stationary_mixture":
            noise_target = bool(settled["noise_attenuation_db"] is not None and settled["noise_attenuation_db"] >= limits["stationary_mixture_oracle_noise_attenuation_db_min"])
            clean_target = bool(settled["clean_relative_error_rmse"] is not None and settled["clean_relative_error_rmse"] <= limits["stationary_mixture_clean_relative_error_RMSE_max"]
                                and settled["clean_level_change_db"] is not None and abs(settled["clean_level_change_db"]) <= limits["stationary_mixture_clean_level_change_abs_db_max"])
        if spec.kind in ("clean_program", "silence"):
            clean_target = bool(np.array_equal(source, output) and np.array_equal(source, decoded))
        if spec.kind == "fundamental_only":
            selective_target = all(not band["acquired"] and band["maximum_recorded_mix"] == 0 for band in usage if band["order"] in (2, 3))
        if spec.kind == "missing_fundamental":
            selective_target = all((band["acquired"] and band["maximum_recorded_mix"] > 0.99) if band["order"] in (2, 3) else (not band["acquired"] and band["maximum_recorded_mix"] == 0) for band in usage)
        inputs_preserved = all(_sha(folder / f"{name}.wav") == value for name, value in input_hashes.items())
        frame_count_met = source.size == decoded.size
        checks = [replay_ok, additive_ok, inputs_preserved, frame_count_met, noise_target, clean_target, selective_target]
        record = {
            "case_id": spec.case_id, "kind": spec.kind, "notes": spec.notes,
            "config": settings.to_dict(), "algorithm": description, "band_usage": usage,
            "recovery_diagnostics": _recovery_diagnostics(trace, int(source.size), 192960 if spec.kind == "abrupt_end" else None),
            "input_sha256": input_hashes, "inputs_unchanged": inputs_preserved, "frame_count_preserved": frame_count_met,
            "replay_mixture_max_abs_error": reconstruction_error, "replay_reconstruction_target_met": replay_ok,
            "component_additivity_target_met": additive_ok, "component_float_measurements": components,
            "harmonic_decoded_WAV_raw_measurements": measured, "single_band_decoded_WAV_raw_measurements": before,
            "noise_only_case": noise_only,
            "single_band_noise_only_attenuation_db": baseline_db if noise_only else None,
            "harmonic_noise_only_attenuation_db": output_db if noise_only else None,
            "noise_target_met": noise_target, "clean_target_met": clean_target, "selective_target_met": selective_target,
            "known_wanted_content_limit_case": spec.kind in ("wanted_harmonic_limit", "neighboring_desired_limit"),
            "desired_content_preservation_guaranteed": False,
            "declared_development_checks_met": all(check is None or bool(check) for check in checks),
            "output_sha256": _sha(folder / "processed.wav"),
        }
        _dump_json(folder / "measurements.json", record)
        cases.append(record)
    # Fixed OFF behavior is separate from the twelve nonzero fixture conditions.
    bypass = []
    example = harmonic_fixtures()[2]
    for orders in ((1,), (2, 3), (1, 2, 3)):
        output, description, trace = render_harmonic(example.input_audio, HarmonicConfig(orders=orders, max_attenuation_db=0))
        bypass.append({"orders": list(orders), "exact_identity": bool(np.array_equal(output, example.input_audio)), "fft_count": description["shared_original_input_fft_count"], "trace_events": len(trace)})
    bypass_ok = all(item["exact_identity"] and item["fft_count"] == 0 and item["trace_events"] == 0 for item in bypass)
    _, after_hashes = _preflight()
    frozen_ok = after_hashes == old_hashes
    noise_count = sum(case["noise_target_met"] is not None for case in cases)
    clean_count = sum(case["clean_target_met"] is not None for case in cases)
    selective_count = sum(case["selective_target_met"] is not None for case in cases)
    source = Path(__file__).parent
    sources = ("harmonics.py", "harmonic_processing.py", "harmonic_fixtures.py", "harmonic_metrics.py", "harmonic_benchmark.py", "h1_policy.json")
    report = {
        "schema_version": 1, "stage": "H1", "policy": policy,
        "parameter_changes_after_full_evaluation": 0,
        "scope": "owned_synthetic_development_conditions_not_independent_real_audio_quality",
        "case_count": len(cases), "noise_target_cases": noise_count, "clean_target_cases": clean_count, "selective_target_cases": selective_count,
        "all_development_targets_met": len(cases) == 12 and noise_count == 4 and clean_count == 4 and selective_count == 2 and bypass_ok and frozen_ok and all(case["declared_development_checks_met"] for case in cases),
        "zero_depth_checks": bypass, "all_zero_depth_checks_met": bypass_ok,
        "legacy_sources_preserved": frozen_ok, "legacy_source_sha256": old_hashes,
        "source_sha256": {name: _sha(source / name) for name in sources}, "cases": cases,
        "device_io": False, "commercial_readiness_or_real_world_quality_claim": False,
        "interpretation": [
            "The same decoded source and fixed two-second boundary are used for paired single-band M4 comparison; whole/startup errors remain.",
            "Only noise-only RMS reduction is labeled directly as noise attenuation from an output file.",
            "Mixture component figures use known-reference frozen-decision FLOAT replays; mixture reconstruction and component additivity must pass first.",
            "Clean relative RMSE includes phase/timing errors and applies no gain fitting or alignment.",
            "Wanted tones inside selected bands can be attenuated; known-limit cases are disclosures, not successful preservation tests.",
            "Independent band detections do not identify a true electrical hum source or establish phase/harmonic coherence.",
            "Moving-tone and recovery figures are diagnostic, not new acceptance floors.",
        ],
    }
    _dump_json(target / "benchmark.json", report)
    _write_html(target / "report.html", report)
    return report


def _write_html(path, report):
    rows = []
    for case in report["cases"]:
        old, new = case["single_band_noise_only_attenuation_db"], case["harmonic_noise_only_attenuation_db"]
        attenuation = "not noise-only" if new is None else f"{old:.3f} → {new:.3f} dB"
        loss = case["component_float_measurements"]["settled"]["clean_relative_error_rmse"]
        damage = "no clean reference" if loss is None else f"{loss * 100:.3f}%"
        label = "KNOWN WANTED-TONE LIMIT" if case["known_wanted_content_limit_case"] else "declared checks " + str(case["declared_development_checks_met"])
        prefix = html.escape(case["case_id"], quote=True)
        rows.append(f"<tr><td>{prefix}<br><small>{label}</small></td><td>{attenuation}</td><td>{damage}</td><td><a href='{prefix}/input.wav'>Input</a> · <a href='{prefix}/processed.wav'>Output</a> · <a href='{prefix}/single_band_M4.wav'>Old single band</a> · <a href='{prefix}/measurements.json'>JSON</a></td></tr>")
    page = ("<!doctype html><html lang='en'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>H1 harmonic processing</title><style>body{font:15px/1.6 system-ui;margin:36px auto;max-width:1280px;padding:0 20px;color:#213b35}table{border-collapse:collapse;width:100%;font-size:13px}td,th{border:1px solid #d9e2d5;padding:10px;text-align:left}.notice{background:#fff1d4;padding:18px}.table{overflow:auto}small{color:#805824}</style>"
            "<h1>H1 selective harmonic processing and wanted-signal distortion</h1>"
            "<p class='notice'>Owned synthetic development evidence only. Independently observed manually selected bands are not semantic noise recognition. Wanted tones can be removed too. No sound was played, no device was accessed, and no commercial or real-world quality claim is made.</p>"
            f"<p>Declared checks met: {report['all_development_targets_met']}. Twelve nonzero conditions plus three OFF controls. The unchanged 2 s settled boundary is used; startup and full-record metrics are retained.</p>"
            "<div class='table'><table><tr><th>Condition</th><th>Single-band → harmonic noise attenuation</th><th>Wanted-signal relative RMS error (oracle float replay)</th><th>Files</th></tr>" + "".join(rows) + "</table></div>"
            "<p>Mixture component diagnostics require both original-mixture reconstruction and additive replay checks. They are not available from arbitrary real recordings without known references. Near-band and same-frequency wanted-tone damage is shown deliberately, not relabeled as noise reduction.</p><p><a href='benchmark.json'>Complete evidence, policy and hashes</a></p></html>")
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(page)
