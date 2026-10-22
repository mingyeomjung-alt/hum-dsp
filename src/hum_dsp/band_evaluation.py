"""Small, declared real-source/injected-hum study. No listening or devices.

The desired reference is a prepared public source, not certified pristine audio.
Known injected components permit frozen-decision diagnostics; this is neither a
natural-hum clean/noisy pair nor an independent perceptual/generalization test.
"""

import hashlib
import html
import json
from pathlib import Path

import numpy as np

from .audio import read_pcm16_wav, validate_audio, write_pcm16_wav
from .band_control import BandControlConfig, render_band_control, render_band_control_replay
from .fixtures import _dump_json
from .harmonic_metrics import measure_harmonic_components
from .harmonics import HarmonicConfig, render_harmonic


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _metric_window(desired, noise, output, desired_replay, noise_replay, start=0):
    result = measure_harmonic_components(desired, noise, output, desired_replay, noise_replay, start_sample=start)
    result["interpretation"] = "known_prepared_source_plus_controlled_noise_frozen_decision_diagnostic_not_blind_quality"
    return result


def _windows(desired, noise, output, desired_replay, noise_replay):
    arrays = (desired, noise, output, desired_replay, noise_replay)
    count = min(96000, desired.size)
    return {
        "whole_record": _metric_window(*arrays),
        "startup": _metric_window(*(a[:count] for a in arrays)),
        "settled": None if desired.size <= 96000 else _metric_window(*arrays, start=96000),
        "settled_unavailable_reason": "source_not_longer_than_two_seconds_not_repeated" if desired.size <= 96000 else None,
        "startup_end_sample": count,
        "settled_start_sample": 96000,
    }


def _catalog(catalog_path, policy):
    path = Path(catalog_path)
    catalog = json.loads(path.read_text(encoding="utf-8"))
    if (catalog.get("schema_version") != 1 or catalog.get("evaluation_only") is not True
            or catalog.get("sample_rate_hz") != 48000 or catalog.get("fixed_gain") != 0.25
            or [s.get("case_id") for s in catalog.get("sources", [])] != policy["source_ids"]):
        raise ValueError("evaluation requires the fixed prepared three-source catalog")
    loaded = []
    for item in catalog["sources"]:
        case_id = item["case_id"]
        filename = item["prepared"]["file_name"]
        if (filename != case_id + ".wav" or item.get("license") != "CC0-1.0"
                or item["original"]["sha256"] != policy["source_sha256"][case_id]
                or item["preparation"]["gain"] != 0.25
                or item["preparation"]["peak_normalization"] is not False):
            raise ValueError("source provenance or fixed preparation settings changed")
        audio_path = path.parent / filename
        original_path = Path(item["original"]["local_path"])
        if original_path.stat().st_size > 5 * 1024 * 1024 or _sha(original_path) != policy["source_sha256"][case_id]:
            raise ValueError("original public source hash changed")
        if _sha(audio_path) != item["prepared"]["sha256"]:
            raise ValueError("prepared evaluation source hash changed")
        samples = read_pcm16_wav(audio_path)
        if int(samples.size) != item["prepared"]["frames"]:
            raise ValueError("prepared source frame count changed")
        loaded.append((item, audio_path, samples))
    return catalog, loaded


def _number(value):
    return "n/a" if value is None else f"{value:.5g}"


def _report(result, path):
    rows = []
    for case in result["cases"]:
        for label, record in case["settings"].items():
            metric = record["metrics"]["settled"] or record["metrics"]["whole_record"]
            window = "settled" if record["metrics"]["settled"] else "whole / too short for settled"
            folder = html.escape(case["case_id"], quote=True)
            safe_label = html.escape(label, quote=True)
            rows.append("<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in (
                case["case_id"], label, window, _number(metric["noise_attenuation_db"]),
                _number(metric["clean_relative_error_rmse"]), _number(metric["clean_level_change_db"])))
                + f'<td><a href="{folder}/{safe_label}.wav">WAV</a></td></tr>')
    text = """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>H2 manual control and real-source study</title><style>body{font:16px system-ui;margin:40px;color:#173b2b;background:#f6f8f3}main{max-width:1250px;margin:auto}table{border-collapse:collapse;background:white;width:100%;font-size:14px}td,th{padding:10px;border:1px solid #cbd5c5;text-align:left}.scroll{overflow-x:auto}p{line-height:1.65}a{color:#206348}</style><main>
<h1>H2: manual depth / wanted-content tradeoffs</h1>
<p>Three CC0 real source clips, prepared by declared conversion and fixed 0.25 gain, with or without controlled mathematical hum. This is NOT natural-noise, perceptual or independent final quality validation. The desired reference may already contain background noise. No audio was played.</p>
<p>Full: 12/12/12 dB. Gentle: 6/3/0 dB. Bypass: 0/0/0 dB. Settings were fixed before running this study, not selected from these outcomes. Lower depth can leave more noise. Same-frequency wanted tones remain confusable.</p>
<p>RMS damage includes phase/time change, without alignment or gain fitting. Component metrics describe floating renders/replays BEFORE final PCM16 rounding, not decompositions of the downloadable WAV. JSON separately retains decoded PCM16 raw error and quantization error. Noise reduction uses frozen decisions replayed on the known injected component. The fixed 'settled' boundary is exactly 2 seconds: qualification/attack may still be progressing, not guaranteed fully settled. Short clips are not repeated; their settled interval is unavailable. Startup and whole-file results are retained in JSON.</p>
<p><a href="evaluation.json">All measurements, trace provenance and source licenses</a> · <a href="source_catalog.json">Prepared source catalog</a></p>
<div class="scroll"><table><thead><tr><th>Source / condition</th><th>Setting</th><th>Window</th><th>Noise reduction dB</th><th>Desired relative RMS error</th><th>Desired level change dB</th><th>Manual download</th></tr></thead><tbody>"""
    text += "".join(rows) + "</tbody></table></div><p>Commercial quality approval: NO. Original H1 engine preserved. See owned 100 Hz limit/control diagnostics in JSON; switching its band off is manual exclusion, not source separation.</p></main></html>"
    path.write_text(text, encoding="utf-8")


def run_band_evaluation(catalog_path, output_directory):
    target = Path(output_directory)
    if target.exists():
        raise FileExistsError("evaluation output directory must be new")
    source_root = Path(__file__).parent
    policy = json.loads((source_root / "h2_policy.json").read_text(encoding="utf-8"))
    if policy["settings"] != {"full": [12, 12, 12], "gentle": [6, 3, 0], "bypass": [0, 0, 0]}:
        raise ValueError("declared H2 comparison settings changed")
    # H1 is preserved and the paired baseline remains the exact shipped engine.
    expected_h1 = "72e5b77ffd9873b5610dc81e44ad8681d9f399cc1305a01d8d8e1aba3d14ab9e"
    if _sha(source_root / "harmonics.py") != expected_h1:
        raise ValueError("preserved H1 source changed")
    catalog, sources = _catalog(catalog_path, policy)
    target.mkdir(parents=True, exist_ok=False)
    _dump_json(target / "source_catalog.json", catalog)
    cases = []
    for item, audio_path, desired in sources:
        for condition in policy["conditions"]:
            nominal = 60 if condition.endswith("60") else 50
            if condition.startswith("injected"):
                t = np.arange(desired.size, dtype=np.float64) / 48000.0
                injected = sum(np.sqrt(2) * 0.012 * np.sin(2 * np.pi * nominal * order * t + 0.37 * order) for order in (1, 2, 3))
            else:
                injected = np.zeros_like(desired)
            mixture = validate_audio(desired + injected)
            case_id = item["case_id"] + "_" + condition
            folder = target / case_id
            folder.mkdir()
            write_pcm16_wav(folder / "input.wav", mixture)
            write_pcm16_wav(folder / "desired_reference.wav", desired)
            mixture = read_pcm16_wav(folder / "input.wav")
            # PCM16 reference at the actual input boundary, not independently
            # quantized mathematical noise that would break exact additivity.
            noise = mixture - desired
            baseline, _, _ = render_harmonic(mixture, HarmonicConfig(nominal_hum_hz=nominal))
            write_pcm16_wav(folder / "H1_reference.wav", baseline)
            records = {}
            for label, depths in policy["settings"].items():
                config = BandControlConfig(nominal_hum_hz=nominal, band_depths_db=depths)
                output, description, trace = render_band_control(mixture, config)
                output = validate_audio(output)
                replay = render_band_control_replay(mixture, config, trace)
                replay_desired = render_band_control_replay(desired, config, trace)
                replay_noise = render_band_control_replay(noise, config, trace)
                metrics = _windows(desired, noise, output, replay_desired, replay_noise)
                write_pcm16_wav(folder / (label + ".wav"), output)
                decoded_output = read_pcm16_wav(folder / (label + ".wav"))
                _dump_json(folder / (label + "_trace.json"), trace)
                max_error = float(np.max(np.abs(output - replay)))
                additive_ok = all(m["component_interpretation_valid"] for m in (metrics["whole_record"], metrics["startup"], metrics["settled"]) if m is not None)
                bypass_ok = label != "bypass" or (np.array_equal(output, mixture) and description["shared_original_input_fft_count"] == 0)
                baseline_ok = label != "full" or ((folder / "full.wav").read_bytes() == (folder / "H1_reference.wav").read_bytes())
                records[label] = {
                    "config": config.to_dict(), "descriptor": description, "metrics": metrics,
                    "component_metrics_signal_domain": "floating_render_and_frozen_replay_before_final_PCM16_rounding",
                    "decoded_PCM16_diagnostics": {
                        "whole_raw_desired_error_RMSE": float(np.sqrt(np.mean((decoded_output - desired) ** 2))),
                        "peak_abs": float(np.max(np.abs(decoded_output))),
                        "quantization_max_abs_error": float(np.max(np.abs(decoded_output - output))),
                        "noise_attenuation_inferred_from_file_RMS": False,
                    },
                    "output_sha256": _sha(folder / (label + ".wav")),
                    "replay_mixture_max_abs_error": max_error,
                    "component_additivity_check": additive_ok,
                    "bypass_check": bypass_ok,
                    "full_H1_equivalence_check": baseline_ok,
                    "checks_met": max_error <= 1e-10 and additive_ok and bypass_ok and baseline_ok,
                }
            cases.append({"case_id": case_id, "source_id": item["case_id"], "condition": condition,
                          "frames": int(desired.size), "source_unchanged": _sha(audio_path) == item["prepared"]["sha256"] and _sha(Path(item["original"]["local_path"])) == item["original"]["sha256"],
                          "short_source_not_repeated": desired.size <= 96000, "settings": records})
    # Same-frequency control demonstrates the manual tradeoff without pretending
    # a semantic classifier can tell wanted bass from identical-frequency hum.
    t = np.arange(48000 * 8, dtype=np.float64) / 48000.0
    wanted = 0.06 * np.sin(2 * np.pi * 100 * t + 0.23)
    controls = {}
    for label, depths in {"full": (12, 12, 12), "second_3dB": (12, 3, 12), "all_off": (0, 0, 0)}.items():
        output, _, _ = render_band_control(wanted, BandControlConfig(band_depths_db=depths))
        controls[label] = {"config_depths": list(depths), "settled_relative_RMSE": float(np.sqrt(np.mean((output[96000:] - wanted[96000:]) ** 2)) / np.sqrt(np.mean(wanted[96000:] ** 2))),
                           "exact_identity": bool(np.array_equal(output, wanted))}
    # Also exclude just order 2 with other requested stages enabled. Their absent
    # spectral evidence should leave this owned, isolated 100 Hz tone unchanged.
    off, _, _ = render_band_control(wanted, BandControlConfig(band_depths_db=(12, 0, 12)))
    controls["only_second_off_other_bands_enabled_exact_identity"] = bool(np.array_equal(off, wanted))
    controls_ok = (controls["second_3dB"]["settled_relative_RMSE"] < controls["full"]["settled_relative_RMSE"]
                   and controls["all_off"]["exact_identity"] and controls["only_second_off_other_bands_enabled_exact_identity"])
    result = {
        "schema_version": 1, "stage": "H2", "scope": "small_real_desired_source_plus_injected_hum_development_study",
        "policy": policy, "source_catalog_sha256": _sha(Path(catalog_path)),
        "source_sha256": {name: _sha(source_root / name) for name in ("band_control.py", "band_processing.py", "band_evaluation.py", "h2_policy.json", "harmonics.py", "harmonic_metrics.py")},
        "case_count": len(cases), "render_count": sum(len(c["settings"]) for c in cases), "cases": cases,
        "owned_wanted_100Hz_control": controls,
        "all_declared_structural_checks_met": controls_ok and all(c["source_unchanged"] and all(r["checks_met"] for r in c["settings"].values()) for c in cases),
        "independent_perceptual_or_natural_noise_quality_verified": False,
        "commercial_release_approved": False, "device_io": False, "audio_played": False,
        "remaining_product_workstreams": 3,
        "remaining_H2": "audible_artifact_review_and_defined_natural_noise_use_conditions_not_universal_preservation",
        "post_first_run_revision": "HTML download-label escaping only; no DSP, source, policy, settings, targets or numerical measurement change. First run preserved.",
    }
    _dump_json(target / "evaluation.json", result)
    _report(result, target / "report.html")
    return result
