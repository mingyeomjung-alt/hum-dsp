"""H2d fixed development comparison, not an independent perceptual evaluation.

Only the candidate sees the declared interval of the SAME input file. Its
whole-file amplitude veto is explicitly offline/noncausal. Separate components
are used exclusively for frozen-decision measurement, never candidate fitting.
"""

from pathlib import Path
import hashlib
import html
import json
import math

import numpy as np

from .audio import read_pcm16_wav, validate_audio, write_pcm16_wav
from .harmonic_metrics import measure_harmonic_components

RATE = 48000
FRAMES = 384000
PROFILE_END = 96000
POST_FOUR = 192000
ONSET_FRAMES = 24000
SELECTIVE_CASES = ("selective_50", "selective_60")
PROTECTED_CASES = (*SELECTIVE_CASES, "strong_overlap_60", "late_strong_overlap_60")
LIMIT_CASES = ("masked_overlap_limit", "contaminated_profile_limit")


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")


def _policy():
    path = Path(__file__).with_name("h2d_policy.json")
    result = json.loads(path.read_bytes())
    if (type(result) is not dict or type(result.get("schema_version")) is not int
            or result["schema_version"] != 1 or result.get("stage") != "H2d"):
        raise ValueError("H2d policy schema is invalid")
    measurement, candidate = result.get("measurement", {}), result.get("candidate", {})
    if (measurement.get("sample_rate_hz") != RATE or measurement.get("total_frames") != FRAMES
            or measurement.get("startup_end_sample") != PROFILE_END
            or measurement.get("post_four_seconds_start_sample") != POST_FOUR
            or measurement.get("onset_window_frames") != ONSET_FRAMES
            or candidate.get("profile_start_sample") != 0 or candidate.get("profile_end_sample") != PROFILE_END):
        raise ValueError("H2d fixed sample boundaries disagree with harness")
    return result


def _engine_api():
    from .extended_bank import ExtendedBankConfig, render_extended_bank, replay_extended_bank
    from .profile_guard import ProfileGuardConfig, render_profile_guard, replay_profile_guard
    return (ExtendedBankConfig, render_extended_bank, replay_extended_bank,
            ProfileGuardConfig, render_profile_guard, replay_profile_guard)


def _rms(array):
    return float(np.sqrt(np.mean(array * array)))


def _db(after, before):
    return float(20 * np.log10(after / before)) if after > 0 and before > 0 else None


def _windows(source, output, components, *, onset_sample=PROFILE_END):
    """Retain prefix, complete program and late-onset transitions without fitting."""
    intervals = {"whole_record": (0, source.size), "profile_startup": (0, PROFILE_END),
                 "program": (PROFILE_END, source.size),
                 "program_onset": (PROFILE_END, PROFILE_END + ONSET_FRAMES),
                 "post_four_seconds": (POST_FOUR, source.size),
                 "declared_wanted_onset": (onset_sample, onset_sample + ONSET_FRAMES)}
    result = {"post_four_seconds_is_guaranteed_settled": False}
    for name, (start, stop) in intervals.items():
        stop = min(stop, source.size)
        if start >= stop:
            result[name] = None
            continue
        selected = slice(start, stop)
        before, after = source[selected], output[selected]
        diagnostic = measure_harmonic_components(
            *(value[selected] for value in (components[0], components[1], output, components[2], components[3])))
        diagnostic["interpretation"] = "known_components_frozen_mixture_replay_before_PCM16_not_field_or_perceptual_quality"
        result[name] = {"start_sample": start, "end_sample_exclusive": stop,
                        "sample_count": stop - start, "input_rms": _rms(before),
                        "output_rms": _rms(after), "output_to_input_rms_db": _db(_rms(after), _rms(before)),
                        "raw_change_rmse": _rms(after - before),
                        "known_component_diagnostic": diagnostic}
    return result


def _owned_cases(policy):
    """These deterministic fixtures are development data with explicit labels."""
    specification = policy["owned_synthetic"]
    t = np.arange(FRAMES, dtype=np.float64) / RATE
    gate = np.arange(FRAMES) >= PROFILE_END
    desired = gate * sum(specification["wanted_amplitude_each"] * np.sin(2 * np.pi * f * t)
                         for f in specification["wanted_frequencies_hz"])

    def noise_for(nominal):
        return sum(math.sqrt(2) * specification["noise_rms_each"]
                   * np.sin(2 * np.pi * nominal * order * t + specification["noise_phase_per_order_radians"] * order)
                   for order in specification["noise_orders"])

    cases = []

    def append(case_id, wanted, noise, nominal=60, kind="owned_synthetic", onset=PROFILE_END, context=""):
        cases.append({"case_id": case_id, "kind": kind, "nominal_hum_hz": nominal,
                      "input": validate_audio(wanted + noise), "desired": validate_audio(wanted),
                      "wanted_onset_sample": onset, "context": context or
                      "Owned deterministic development fixture; no independent recording or semantic source recognition."})

    for nominal in (50, 60):
        append(f"selective_{nominal}", desired, noise_for(nominal), nominal)
    wanted_frequency = specification["wanted_limit_hz"]
    wanted_amplitude = specification["wanted_limit_amplitude"]
    overlap = wanted_amplitude * np.sin(2 * np.pi * wanted_frequency * t)
    append("strong_overlap_60", overlap * gate, noise_for(60))
    append("late_strong_overlap_60", overlap * (np.arange(FRAMES) >= POST_FOUR), noise_for(60), onset=POST_FOUR)
    append("empty_profile_60", overlap * gate, np.zeros(FRAMES), context=
           "Declared profile is silent. Later wanted selected-frequency content must not create a noise profile.")
    growing = noise_for(60) * np.where(np.arange(FRAMES) >= POST_FOUR, specification["growing_noise_multiplier"], 1.0)
    append("growing_noise_60", specification["wanted_amplitude_each"] * np.sin(2 * np.pi * specification["wanted_frequencies_hz"][0] * t) * gate, growing,
           kind="growing_noise_report_only", context=
           "True noise grows threefold at four seconds: conservative veto may leave noise. Report-only tradeoff.")
    amplitude = math.sqrt(2) * specification["noise_rms_each"]
    phase = 0.0
    masking_noise = amplitude * np.sin(2 * np.pi * wanted_frequency * t + phase)
    masking_wanted = amplitude * np.sin(2 * np.pi * wanted_frequency * t + phase + specification["masked_overlap_wanted_phase_difference_radians"]) * gate
    append("masked_overlap_limit", masking_wanted, masking_noise, kind="wanted_overlap_hard_limit", context=
           "Same-frequency equal-amplitude components differ by 120 degrees. Their resultant amplitude equals the profile amplitude; an amplitude guard cannot guarantee preservation.")
    append("contaminated_profile_limit", overlap, np.zeros(FRAMES), kind="contaminated_profile_hard_limit", onset=0, context=
           "Profile declaration is intentionally wrong: wanted 1260 Hz exists throughout. Disclose damage; this is not a preservation-pass case.")
    return cases


def _load_recorded_case(root, metadata):
    if metadata.get("stage") != "H2b" or metadata.get("commercial_release_approved") is not False:
        raise ValueError("source must be a development-only H2b study")
    indexed = {case["case_id"]: case for case in metadata.get("cases", [])}
    try:
        noise_hash = indexed["recorded_hum_only_60"]["settings"]["bypass"]["output_sha256"]
        desired_hash = indexed["desired_only_60"]["settings"]["bypass"]["output_sha256"]
        declared_provenance = metadata["noise_provenance"]
    except (KeyError, TypeError) as exc:
        raise ValueError("H2b source metadata lacks fixed source identities") from exc
    noise_path = root / "prepared_recorded_noise.wav"
    desired_path = root / "desired_only_60" / "input.wav"
    provenance_path, catalog_path = (root / name for name in ("noise_provenance.json", "source_catalog.json"))
    # H2b records hashes of the ORIGINAL input JSON, but its retained copies
    # were reserialized. Anchor those copies semantically, then freeze their
    # actual bytes; never pretend the original and copied JSON hashes match.
    provenance_bytes, catalog_bytes = provenance_path.read_bytes(), catalog_path.read_bytes()
    provenance_hash = hashlib.sha256(provenance_bytes).hexdigest()
    catalog_hash = hashlib.sha256(catalog_bytes).hexdigest()
    expected = {noise_path: noise_hash, desired_path: desired_hash,
                provenance_path: provenance_hash, catalog_path: catalog_hash}
    if any(_sha(path) != digest for path, digest in expected.items()):
        raise ValueError("retained H2b source hash does not match metadata")
    # Freeze serialized JSON BEFORE parsing and recheck during use.
    try:
        provenance, catalog = json.loads(provenance_bytes), json.loads(catalog_bytes)
    except (ValueError, UnicodeError) as exc:
        raise ValueError("retained source JSON hash/provenance link is invalid") from exc
    piano_sources = [source for source in catalog.get("sources", []) if source.get("case_id") == "piano"]
    if (provenance.get("license") != "CC0-1.0" or catalog.get("evaluation_only") is not True
            or provenance != declared_provenance
            or provenance.get("sha256") != metadata.get("recorded_noise_sha256")
            or len(piano_sources) != 1 or piano_sources[0].get("prepared", {}).get("sha256") != desired_hash):
        raise ValueError("H2b source hash/provenance or evaluation-only declaration is invalid")
    noise, piano = read_pcm16_wav(noise_path), read_pcm16_wav(desired_path)
    if noise.size != FRAMES or piano.size < FRAMES - PROFILE_END:
        raise ValueError("recorded source must provide eight seconds noise and six seconds piano without repetition")
    rms = _rms(noise)
    if rms <= 0:
        raise ValueError("recorded noise must not be silent")
    gain = .02 / rms
    scaled_noise = noise * gain
    desired = np.concatenate((np.zeros(PROFILE_END), piano[:FRAMES - PROFILE_END]))
    mixed = validate_audio(desired + scaled_noise)
    if any(_sha(path) != digest for path, digest in expected.items()):
        raise ValueError("source changed during preflight")
    return {"case_id": "recorded_profile_then_piano_60", "kind": "constructed_recorded_mixture",
            "nominal_hum_hz": 60, "input": mixed, "desired": desired,
            "wanted_onset_sample": PROFILE_END,
            "context": "Previously inspected MP3-preview hum and piano are combined for development; not an independent naturally noisy/clean field pair. Nominal 60 Hz is declared, not source truth.",
            "fixture_preparation": {"noise_RMS_before_PCM16": .02, "fixed_noise_gain": gain,
                                    "noise_source_frames": [0, FRAMES],
                                    "piano_source_frames": [0, FRAMES - PROFILE_END],
                                    "piano_prefix_zero_frames": PROFILE_END,
                                    "repetition_or_time_stretch": False,
                                    "retained_JSON_identity": "actual_copy_bytes_frozen_provenance_matches_metadata_and_piano_catalog_matches_PCM_hash",
                                    "fixture_gain_not_production_normalization": True}}, expected, provenance, catalog


def _known(case, setting, window):
    value = case["settings"][setting]["measurements"].get(window)
    return None if value is None else value.get("known_component_diagnostic")


def _development_decision(cases, policy, structural):
    indexed = {case["case_id"]: case for case in cases}
    target = policy["development_targets"]
    checks = {"structural_checks": structural,
              "empty_profile_exact_identity": indexed["empty_profile_60"]["settings"]["profile_guard"]["output_exact_input"]}
    for case_id in PROTECTED_CASES:
        for window in ("program", "post_four_seconds"):
            metrics = _known(indexed[case_id], "profile_guard", window) or {}
            error, level = metrics.get("clean_relative_error_rmse"), metrics.get("clean_level_change_db")
            checks[f"{case_id}_{window}_wanted_error"] = error is not None and error <= target["clean_relative_error_rmse_max"]
            checks[f"{case_id}_{window}_wanted_level"] = level is not None and abs(level) <= target["clean_level_change_abs_db_max"]
    for case_id in SELECTIVE_CASES:
        attenuation = (_known(indexed[case_id], "profile_guard", "post_four_seconds") or {}).get("noise_attenuation_db")
        checks[f"{case_id}_post_four_noise_attenuation"] = attenuation is not None and attenuation >= target["selective_noise_attenuation_db_min"]
    return {"status": "development_targets_met_not_adopted" if all(checks.values()) else "not_adopted_development_targets_failed",
            "checks": checks, "targets_from_policy": target,
            "report_only_cases": ["growing_noise_60", *LIMIT_CASES, "recorded_profile_then_piano_60"],
            "hard_limit_diagnostics": {case_id: {label: _known(indexed[case_id], label, "program")
                                                 for label in ("baseline", "profile_guard")} for case_id in LIMIT_CASES},
            "independent_quality_claim": False, "default_engine_changed": False,
            "commercial_release_approved": False,
            "warning": "Whole-file lookahead veto is offline and can leave increased noise. Same-frequency masked overlap and a contaminated profile can still suppress wanted sound."}


def _report(result, path):
    rows = []
    for case in result["cases"]:
        for setting in ("baseline", "profile_guard"):
            for window in ("whole_record", "profile_startup", "program", "program_onset", "declared_wanted_onset", "post_four_seconds"):
                values = _known(case, setting, window) or {}
                row = (case["case_id"], setting, window, values.get("noise_attenuation_db"),
                       values.get("clean_relative_error_rmse"), values.get("clean_level_change_db"))
                rows.append("<tr>" + "".join("<td>" + html.escape(str(value)) + "</td>" for value in row) + "</tr>")
    Path(path).write_text("""<!doctype html><html lang="en"><meta charset="utf-8"><title>H2d profile guard development study</title>
<style>body{font:16px system-ui;max-width:1400px;margin:40px auto;padding:20px}p{line-height:1.6}table{border-collapse:collapse}td,th{border:1px solid #bbb;padding:8px}</style>
<h1>Same-file noise profile and offline amplitude veto</h1>
<p>Previously inspected development recordings are NOT an independent evaluation. The recorded-hum mixture is constructed, not an independently clean field pair. The first two seconds are a user-declared noise-only interval, never an oracle component supplied to the engine.</p>
<p>The candidate uses whole-file lookahead to veto amplitude changes; it is OFFLINE / NONCAUSAL, not a live or real-time noise canceller. All components replay frozen mixture decisions before PCM16 export. No fitting, normalization, alignment or best-setting search is applied to measurement. Post-four-seconds is not guaranteed settled.</p>
<p>Masked overlap and a contaminated profile are disclosed hard limits, not preservation-pass cases. Growing noise and recorded-mixture results are report-only. No listening, perceptual approval, default replacement or commercial release approval has occurred.</p>
<p><a href="evaluation.json">Complete measurements, locked policy and source hashes</a></p>
<h2>Development decision: """ + html.escape(result["development_decision"]["status"]) + """</h2>
<table><tr><th>Case</th><th>Engine</th><th>Window</th><th>Known noise attenuation dB</th><th>Wanted relative RMSE</th><th>Wanted level change dB</th></tr>"""
                            + "".join(rows) + "</table></html>", encoding="utf-8")


def run_profile_guard_evaluation(source_study_dir_h2b, output_dir):
    """Create nine predeclared cases and eighteen outputs in a NEW directory."""
    root, target = Path(source_study_dir_h2b), Path(output_dir)
    if target.exists():
        raise FileExistsError("profile guard evaluation output directory must be new")
    policy_path, metadata_path = Path(__file__).with_name("h2d_policy.json"), root / "evaluation.json"
    policy_bytes, metadata_bytes = policy_path.read_bytes(), metadata_path.read_bytes()
    expected_policy, expected_metadata = hashlib.sha256(policy_bytes).hexdigest(), hashlib.sha256(metadata_bytes).hexdigest()
    policy, metadata = _policy(), json.loads(metadata_bytes)
    if _sha(policy_path) != expected_policy or _sha(metadata_path) != expected_metadata:
        raise ValueError("policy or metadata changed during preflight")
    recorded, assets, provenance, catalog = _load_recorded_case(root, metadata)
    cases_to_run = [*_owned_cases(policy), recorded]
    source_paths = [Path(__file__), policy_path, metadata_path,
                    *(Path(__file__).with_name(name) for name in ("profile_guard.py", "extended_bank.py", "audio.py", "harmonic_metrics.py"))]
    identities_before = {str(path.resolve()): _sha(path) for path in [*source_paths, *assets]}
    identities_before[str(policy_path.resolve())], identities_before[str(metadata_path.resolve())] = expected_policy, expected_metadata
    for path, digest in assets.items():
        identities_before[str(path.resolve())] = digest
    BaseConfig, base_render, base_replay, GuardConfig, guard_render, guard_replay = _engine_api()
    candidate = policy["candidate"]
    base_configs, guard_configs = {}, {}
    for nominal in (50, 60):
        shared = {"sample_rate_hz": RATE, "nominal_hum_hz": nominal,
                  "orders": tuple(candidate["orders"]), "max_attenuation_db": candidate["attenuation_db"],
                  "block_size_frames": candidate["block_size_frames"]}
        base_configs[nominal] = BaseConfig(**shared)
        guard_configs[nominal] = GuardConfig(**shared, profile_start_sample=0, profile_end_sample=PROFILE_END)
    if any(_sha(path) != digest for path, digest in ((Path(name), value) for name, value in identities_before.items())):
        raise ValueError("source or policy changed during preflight")
    target.mkdir(parents=True, exist_ok=False)
    _dump(target / "locked_policy.json", policy)
    _dump(target / "noise_provenance.json", provenance)
    _dump(target / "source_catalog.json", catalog)
    cases = []
    for case in cases_to_run:
        folder = target / case["case_id"]
        folder.mkdir()
        write_pcm16_wav(folder / "input.wav", case["input"])
        write_pcm16_wav(folder / "desired_reference.wav", case["desired"])
        source = read_pcm16_wav(folder / "input.wav")
        desired = read_pcm16_wav(folder / "desired_reference.wav")
        noise = source - desired
        records = {}
        for label, configs, render, replay in (("baseline", base_configs, base_render, base_replay),
                                               ("profile_guard", guard_configs, guard_render, guard_replay)):
            config = configs[case["nominal_hum_hz"]]
            output, descriptor, trace = render(source, config)
            output = validate_audio(output)
            if output.shape != source.shape:
                raise ValueError("engine changed output frame boundary")
            repeated = replay(source, config, trace)
            replay_error = float(np.max(np.abs(output - repeated)))
            components = (desired, noise, replay(desired, config, trace), replay(noise, config, trace))
            windows = _windows(source, output, components, onset_sample=case["wanted_onset_sample"])
            additive_error = max(value["known_component_diagnostic"]["decomposition_additivity_max_abs_error"]
                                 for value in windows.values() if isinstance(value, dict))
            structural = (replay_error <= policy["development_targets"]["replay_max_abs_error_max"]
                          and additive_error <= policy["development_targets"]["additive_max_abs_error_max"])
            write_pcm16_wav(folder / (label + ".wav"), output)
            _dump(folder / (label + "_trace.json"), trace)
            records[label] = {"descriptor": descriptor, "measurements": windows,
                              "replay_mixture_max_abs_error": replay_error,
                              "component_additivity_max_abs_error": additive_error,
                              "structural_checks_met": structural, "output_exact_input": bool(np.array_equal(source, output)),
                              "output_sha256": _sha(folder / (label + ".wav")), "maximum_output_peak": float(np.max(np.abs(output)))}
        cases.append({"case_id": case["case_id"], "kind": case["kind"], "context": case["context"],
                      "nominal_hum_hz": case["nominal_hum_hz"], "frames": int(source.size),
                      "input_sha256": _sha(folder / "input.wav"), "desired_reference_sha256": _sha(folder / "desired_reference.wav"),
                      "known_components": True, "independent_clean_field_reference": False,
                      "profile_interval_from_actual_input": [0, PROFILE_END],
                      "profile_noise_only_declaration": "explicit_owned_fixture_assertion_not_automatic_detection",
                      "profile_truly_noise_only_in_known_fixture": bool(np.all(desired[:PROFILE_END] == 0)),
                      "wanted_onset_sample": case["wanted_onset_sample"], "fixture_preparation": case.get("fixture_preparation"),
                      "settings": records})
    identities_after = {name: _sha(name) for name in identities_before}
    unchanged = identities_before == identities_after
    structural = unchanged and all(record["structural_checks_met"] for case in cases for record in case["settings"].values())
    result = {"schema_version": 1, "stage": "H2d", "evaluation_only": True,
              "scope": "same_file_profile_plus_whole_file_offline_veto_development_comparison",
              "policy": policy, "case_count": len(cases), "render_count": 2 * len(cases), "cases": cases,
              "identities_before": identities_before, "identities_after": identities_after,
              "source_files_unchanged": unchanged, "all_declared_structural_checks_met": structural,
              "development_decision": _development_decision(cases, policy, structural),
              "audio_played": False, "device_io": False, "human_review_completed": False,
              "independent_perceptual_or_field_quality_verified": False, "commercial_release_approved": False,
              "default_engine_changed": False, "causal_or_live_processing": False,
              "remaining_product_workstreams": 2}
    _dump(target / "evaluation.json", result)
    _report(result, target / "report.html")
    return result
