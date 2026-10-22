"""H2c development-only coverage comparison; not an independent quality test.

Previously inspected H2b PCM16 files are retained as the actual input boundary.
Frozen mixture decisions replay known components before PCM16 export. The
recorded-noise mixture is constructed, not an independently clean field pair.
No audio playback, device use, parameter search or default-engine adoption.
"""

from pathlib import Path
import hashlib
import html
import json
import math

import numpy as np

from .audio import read_pcm16_wav, validate_audio, write_pcm16_wav
from .band_control import BandControlConfig, render_band_control, render_band_control_replay
from .harmonic_metrics import measure_harmonic_components


RATE = 48000
BOUNDARY = 96000
ACTUAL_CASES = tuple(f"{kind}_{nominal}" for kind in
                     ("recorded_hum_only", "constructed_mixture", "desired_only")
                     for nominal in (50, 60))


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")


def _policy():
    path = Path(__file__).with_name("h2c_policy.json")
    result = json.loads(path.read_text(encoding="utf-8"))
    if (type(result) is not dict or type(result.get("schema_version")) is not int
            or result["schema_version"] != 1 or result.get("stage") != "H2c"):
        raise ValueError("H2c policy schema is invalid")
    return result


def _engine_api():
    # Lazy import keeps this evaluation helper separate from product entrypoints.
    from .extended_bank import ExtendedBankConfig, render_extended_bank, replay_extended_bank
    return ExtendedBankConfig, render_extended_bank, replay_extended_bank


def _rms(array):
    return float(np.sqrt(np.mean(array * array)))


def _db(after, before):
    return float(20 * np.log10(after / before)) if after > 0 and before > 0 else None


def _windows(source, output, components=None):
    result = {"post_two_seconds_not_guaranteed_settled": True}
    for label, selected in (("whole_record", slice(None)), ("startup", slice(0, BOUNDARY)),
                            ("post_two_seconds", slice(BOUNDARY, None))):
        if selected.start is not None and source.size <= selected.start:
            result[label] = None
            continue
        before, after = source[selected], output[selected]
        observed = {"sample_count": int(before.size), "input_rms": _rms(before),
                    "output_rms": _rms(after), "output_to_input_rms_db": _db(_rms(after), _rms(before)),
                    "raw_change_rmse": _rms(after - before),
                    "interpretation": "observed_signal_change_not_true_noise_reduction_or_wanted_preservation"}
        if components is not None:
            clean, noise, replay_clean, replay_noise = (value[selected] for value in components)
            diagnostic = measure_harmonic_components(clean, noise, after, replay_clean, replay_noise)
            diagnostic["interpretation"] = "known_components_frozen_mixture_replay_before_PCM16_not_field_or_perceptual_quality"
            observed["known_component_diagnostic"] = diagnostic
        result[label] = observed
    return result


def _synthetic_cases(policy):
    synthetic = policy["owned_synthetic"]
    length = int(RATE * synthetic["seconds"])
    t = np.arange(length, dtype=np.float64) / RATE
    desired = sum(synthetic["wanted_amplitude_each"] * np.sin(2 * np.pi * frequency * t)
                  for frequency in synthetic["wanted_frequencies_hz"])
    cases = []
    for nominal in policy["candidate"]["nominal_hum_hz"]:
        noise = sum(math.sqrt(2) * synthetic["noise_rms_each"]
                    * np.sin(2 * np.pi * nominal * order * t + .37 * order)
                    for order in synthetic["noise_orders"])
        cases.append({"case_id": f"owned_extended_tones_{nominal}", "kind": "owned_synthetic",
                      "nominal_hum_hz": nominal, "input": validate_audio(desired + noise),
                      "desired": validate_audio(desired), "retained_baseline": None,
                      "context": "Owned deterministic synthetic fixture; not an independent recording."})
    wanted = validate_audio(synthetic["wanted_limit_amplitude"] * np.sin(2 * np.pi * synthetic["wanted_limit_hz"] * t))
    cases.append({"case_id": "wanted_1260_hz_overlap", "kind": "wanted_overlap_control",
                  "nominal_hum_hz": 60, "input": wanted, "desired": wanted.copy(),
                  "retained_baseline": None,
                  "context": "Intentionally wanted selected-frequency tone; suppression is damage, not noise reduction."})
    return cases


def _load_actual(root, metadata):
    if metadata.get("stage") != "H2b" or metadata.get("commercial_release_approved") is not False:
        raise ValueError("baseline must be a development-only H2b study")
    entries = {case["case_id"]: case for case in metadata.get("cases", [])}
    if any(case_id not in entries for case_id in ACTUAL_CASES):
        raise ValueError("baseline is missing one of the six fixed actual-audio cases")
    cases, assets = [], []
    for case_id in ACTUAL_CASES:
        entry = entries[case_id]
        folder = root / case_id
        input_path, baseline_path = folder / "input.wav", folder / "full.wav"
        # H2b's bypass is byte-identical to its decoded input boundary.
        if (_sha(input_path) != entry["settings"]["bypass"]["output_sha256"]
                or _sha(baseline_path) != entry["settings"]["full"]["output_sha256"]):
            raise ValueError("retained H2b input or baseline hash does not match its result")
        source = read_pcm16_wav(input_path)
        desired = None
        assets.extend((input_path, baseline_path))
        if entry["kind"] != "recorded_hum_only":
            desired_path = folder / "desired_reference.wav"
            expected_desired_hash = entries[f"desired_only_{entry['nominal_hum_hz']}"]["settings"]["bypass"]["output_sha256"]
            if _sha(desired_path) != expected_desired_hash:
                raise ValueError("retained desired reference hash does not match the H2b desired-only input")
            desired = read_pcm16_wav(desired_path)
            assets.append(desired_path)
            if desired.shape != source.shape:
                raise ValueError("retained desired reference has a different PCM boundary")
        if source.size != entry["frames"]:
            raise ValueError("retained H2b frame boundary disagrees with metadata")
        if (_sha(input_path) != entry["settings"]["bypass"]["output_sha256"]
                or _sha(baseline_path) != entry["settings"]["full"]["output_sha256"]
                or (desired is not None and _sha(desired_path) != expected_desired_hash)):
            raise ValueError("retained H2b asset changed during preflight")
        cases.append({"case_id": case_id, "kind": entry["kind"],
                      "nominal_hum_hz": entry["nominal_hum_hz"], "input": source, "desired": desired,
                      "retained_baseline": baseline_path,
                      "context": "Previously inspected development data. " + entry["context"]})
    return cases, assets


def _known(record):
    window = record["measurements"]["post_two_seconds"]
    return None if window is None else window.get("known_component_diagnostic")


def _development_decision(cases, policy, structural):
    targets = policy["development_targets"]
    indexed = {case["case_id"]: case for case in cases}
    comparisons = {}
    for case in cases:
        before, after = (_known(case["settings"][name]) for name in ("baseline", "extended"))
        improvement = None
        if before is not None and after is not None:
            a, b = before["noise_attenuation_db"], after["noise_attenuation_db"]
            if a is not None and b is not None:
                improvement = b - a
        comparisons[case["case_id"]] = {"extra_known_component_attenuation_db": improvement}
    recorded = indexed["constructed_mixture_60"]
    recorded_metric = _known(recorded["settings"]["extended"])
    gain = comparisons["constructed_mixture_60"]["extra_known_component_attenuation_db"]
    checks = {
        "structural_checks": structural,
        "recorded_60_extra_attenuation": gain is not None and gain >= targets["constructed_60_noise_attenuation_gain_db_min"],
        "recorded_60_wanted_relative_error": recorded_metric is not None and recorded_metric["clean_relative_error_rmse"] is not None
        and recorded_metric["clean_relative_error_rmse"] <= targets["constructed_60_clean_relative_error_rmse_max"],
        "recorded_60_wanted_level_change": recorded_metric is not None and recorded_metric["clean_level_change_db"] is not None
        and abs(recorded_metric["clean_level_change_db"]) <= targets["constructed_60_clean_level_change_abs_db_max"],
        "actual_desired_only_exact_identity": all(indexed[f"desired_only_{nominal}"]["settings"]["extended"]["output_exact_input"]
                                                 for nominal in (50, 60)),
    }
    for nominal in policy["candidate"]["nominal_hum_hz"]:
        improvement = comparisons[f"owned_extended_tones_{nominal}"]["extra_known_component_attenuation_db"]
        checks[f"synthetic_{nominal}_extra_attenuation"] = improvement is not None and improvement >= targets["synthetic_extended_noise_gain_db_min"]
    harm = _known(indexed["wanted_1260_hz_overlap"]["settings"]["extended"])
    return {"status": "development_targets_met_not_adopted" if all(checks.values()) else "not_adopted_development_targets_failed",
            "checks": checks, "comparisons": comparisons, "targets_from_policy": targets,
            "wanted_overlap_harm_diagnostic": harm,
            "harm_warning": "Wanted content at a selected frequency can be suppressed identically; no semantic noise separation.",
            "independent_quality_claim": False, "default_engine_changed": False,
            "commercial_release_approved": False}


def _report(result, path):
    rows = []
    for case in result["cases"]:
        for label, record in case["settings"].items():
            known = _known(record) or {}
            values = (case["case_id"], label, known.get("noise_attenuation_db"),
                      known.get("clean_relative_error_rmse"), known.get("clean_level_change_db"), record["structural_checks_met"])
            rows.append("<tr>" + "".join("<td>" + html.escape(str(value)) + "</td>" for value in values) + "</tr>")
    Path(path).write_text("""<!doctype html><html lang="en"><meta charset="utf-8"><title>H2c extended bank development study</title>
<style>body{font:16px system-ui;max-width:1200px;margin:40px auto;padding:20px}p{line-height:1.6}table{border-collapse:collapse}td,th{border:1px solid #bbb;padding:9px}</style>
<h1>Extended harmonic coverage: development comparison</h1>
<p>Previously inspected development recordings are NOT an independent evaluation. The recorded-noise mixture has constructed known components, not an independently clean field reference. A natural recording without clean reference supports observations only.</p>
<p>All component metrics use frozen original-mixture decisions and floating replay before PCM16 export. No alignment, gain fitting, normalization, threshold search or automatic best-setting selection. Startup and post-two-seconds windows are reported separately; post-two-seconds is not guaranteed settled.</p>
<p>Wanted tones in selected bands can be suppressed like noise. No listening, perceptual approval, default engine replacement or commercial release approval has occurred.</p>
<p><a href="evaluation.json">Complete measurements, hashes and policy</a></p>
<h2>Development decision: """ + html.escape(result["development_decision"]["status"]) + """</h2>
<table><tr><th>Case</th><th>Engine</th><th>Known noise attenuation dB</th><th>Wanted relative RMSE</th><th>Wanted level change dB</th><th>Structural checks</th></tr>"""
                            + "".join(rows) + "</table></html>", encoding="utf-8")


def run_extended_bank_evaluation(baseline_study_dir, output_dir):
    """Render fixed actual/synthetic cases into a NEW output directory only."""
    root, target = Path(baseline_study_dir), Path(output_dir)
    if target.exists():
        raise FileExistsError("extended bank evaluation output directory must be new")
    policy_path = Path(__file__).with_name("h2c_policy.json")
    initial_policy_hash = _sha(policy_path)
    policy = _policy()
    if _sha(policy_path) != initial_policy_hash:
        raise ValueError("policy changed during preflight")
    metadata_path = root / "evaluation.json"
    metadata_bytes = metadata_path.read_bytes()
    initial_metadata_hash = hashlib.sha256(metadata_bytes).hexdigest()
    metadata = json.loads(metadata_bytes)
    if _sha(metadata_path) != initial_metadata_hash:
        raise ValueError("baseline metadata changed during preflight")
    actual, assets = _load_actual(root, metadata)
    cases_to_run = [*actual, *_synthetic_cases(policy)]
    source_files = [Path(__file__), policy_path,
                    *(Path(__file__).with_name(name) for name in
                      ("extended_bank.py", "band_control.py", "harmonics.py", "harmonic_metrics.py", "audio.py"))]
    # Include copied provenance/catalog without following their original paths.
    assets.extend([metadata_path, *(root / name for name in ("noise_provenance.json", "source_catalog.json"))])
    identity_paths = [*assets, *source_files]
    identities_before = {str(path.resolve()): _sha(path) for path in identity_paths}
    identities_before[str(policy_path.resolve())] = initial_policy_hash
    identities_before[str(metadata_path.resolve())] = initial_metadata_hash
    ExtendedBankConfig, render_extended_bank, replay_extended_bank = _engine_api()
    # Validate both configurations before creating output artifacts.
    configs = {nominal: ExtendedBankConfig(nominal_hum_hz=nominal,
                                         orders=tuple(policy["candidate"]["orders"]),
                                         max_attenuation_db=policy["candidate"]["attenuation_db"],
                                         block_size_frames=policy["candidate"]["block_size_frames"])
               for nominal in policy["candidate"]["nominal_hum_hz"]}
    if any(_sha(path) != identities_before[str(path.resolve())] for path in identity_paths):
        raise ValueError("source or policy changed during preflight")
    target.mkdir(parents=True, exist_ok=False)
    _dump(target / "locked_policy.json", policy)
    cases = []
    for case in cases_to_run:
        folder = target / case["case_id"]
        folder.mkdir()
        write_pcm16_wav(folder / "input.wav", case["input"])
        source = read_pcm16_wav(folder / "input.wav")
        desired = case["desired"]
        if desired is not None:
            # Decode the saved PCM16 boundary for every component diagnostic.
            write_pcm16_wav(folder / "desired_reference.wav", desired)
            desired = read_pcm16_wav(folder / "desired_reference.wav")
            noise = source - desired
        baseline_config = BandControlConfig(nominal_hum_hz=case["nominal_hum_hz"], band_depths_db=tuple(policy["baseline"]["band_depths_db"]),
                                            block_size_frames=policy["candidate"]["block_size_frames"])
        records = {}
        for label in ("baseline", "extended"):
            config = baseline_config if label == "baseline" else configs[case["nominal_hum_hz"]]
            render = render_band_control if label == "baseline" else render_extended_bank
            replay = render_band_control_replay if label == "baseline" else replay_extended_bank
            output, descriptor, trace = render(source, config)
            repeated = replay(source, config, trace)
            replay_error = float(np.max(np.abs(output - repeated)))
            components = None if desired is None else (desired, noise, replay(desired, config, trace), replay(noise, config, trace))
            windows = _windows(source, output, components)
            write_pcm16_wav(folder / (label + ".wav"), output)
            _dump(folder / (label + "_trace.json"), trace)
            additive_ok = components is None or all(window["known_component_diagnostic"]["component_interpretation_valid"]
                                                    for key, window in windows.items() if isinstance(window, dict))
            baseline_match = label != "baseline" or case["retained_baseline"] is None or _sha(folder / "baseline.wav") == _sha(case["retained_baseline"])
            record = {"descriptor": descriptor, "measurements": windows,
                      "replay_mixture_max_abs_error": replay_error,
                      "component_additivity_check": additive_ok, "retained_baseline_PCM16_match": baseline_match,
                      "output_exact_input": bool(np.array_equal(source, output)),
                      "maximum_output_peak": float(np.max(np.abs(output))),
                      "output_sha256": _sha(folder / (label + ".wav")),
                      "structural_checks_met": replay_error <= policy["development_targets"]["frozen_mixture_replay_max_abs_error"] and additive_ok and baseline_match}
            records[label] = record
        cases.append({"case_id": case["case_id"], "kind": case["kind"], "context": case["context"],
                      "nominal_hum_hz": case["nominal_hum_hz"], "frames": int(source.size),
                      "input_sha256": _sha(folder / "input.wav"),
                      "desired_reference_sha256": None if desired is None else _sha(folder / "desired_reference.wav"),
                      "known_components": desired is not None, "independent_clean_field_reference": False,
                      "settings": records})
    identities_after = {str(path.resolve()): _sha(path) for path in identity_paths}
    unchanged = identities_before == identities_after
    structural = unchanged and all(record["structural_checks_met"] for case in cases for record in case["settings"].values())
    result = {"schema_version": 1, "stage": "H2c", "evaluation_only": True,
              "scope": "previously_inspected_development_audio_extended_harmonic_coverage_comparison",
              "policy": policy, "case_count": len(cases), "render_count": 2 * len(cases), "cases": cases,
              "identities_before": identities_before, "identities_after": identities_after,
              "source_files_unchanged": unchanged, "all_declared_structural_checks_met": structural,
              "development_decision": _development_decision(cases, policy, structural),
              "audio_played": False, "device_io": False, "human_review_completed": False,
              "independent_perceptual_or_field_quality_verified": False, "commercial_release_approved": False,
              "default_engine_changed": False, "remaining_product_workstreams": 2}
    _dump(target / "evaluation.json", result)
    _report(result, target / "report.html")
    return result
