"""M2 fixed-filter measurements on the already declared synthetic fixtures."""

import hashlib
import html
from pathlib import Path
from .audio import read_pcm16_wav
from .config import ProcessorConfig
from .evaluation import measure_windows
from .filter import process_audio
from .fixtures import _dump_json, fixture_specs, verify_fixture_suite
from .processing import process_wav


def run_fixed_benchmark(suite: str | Path, output_directory: str | Path) -> dict:
    inputs = Path(suite).resolve(strict=True)
    verified = verify_fixture_suite(inputs)
    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=False)
    case_results = []
    for spec in fixture_specs():
        folder = inputs / spec.case_id
        clean = read_pcm16_wav(folder / "clean.wav")
        samples = read_pcm16_wav(folder / "input.wav")
        # Use the exact decoded input difference, not a separately rounded stem.
        noise = samples - clean
        config = ProcessorConfig(nominal_hum_hz=spec.nominal_hum_hz)
        processing = process_wav(folder / "input.wav", destination / spec.case_id, config)
        decoded = read_pcm16_wav(destination / spec.case_id / "processed.wav")
        floating = process_audio(samples, config)
        quantized_measurements = measure_windows(clean, noise, decoded)
        floating_measurements = measure_windows(clean, noise, floating)
        settled = quantized_measurements["settled"]
        noise_only = spec.kind in ("stationary_target", "moving_target")
        attenuation = settled["reference_error_reduction_db"] if noise_only else None
        case_results.append({
            "case_id": spec.case_id,
            "kind": spec.kind,
            "expected_use": spec.expected_use,
            "config": config.to_dict(),
            "signal_only_attenuation_db": attenuation,
            "stationary_10db_target_met": None if spec.kind != "stationary_target" else attenuation is not None and attenuation >= 10.0,
            "preservation_guarantee": False,
            "float_measurements": floating_measurements,
            "decoded_wav_measurements": quantized_measurements,
            "output": processing["output"],
            "pcm16_quantization_max_abs_error": processing["pcm16_quantization_max_abs_error"],
            "input_unchanged": hashlib.sha256((folder / "input.wav").read_bytes()).hexdigest() == processing["input"]["sha256"],
            "frame_count_preserved": processing["input"]["frames"] == processing["output"]["frames"],
        })
    stationary = [item for item in case_results if item["kind"] == "stationary_target"]
    result = {
        "schema_version": 1,
        "stage": "M2",
        "engine": "fixed_finite_depth_notch",
        "scope": "owned_synthetic_development_conditions_not_independent_real_world_validation",
        "fixture_verification": verified,
        "case_count": len(case_results),
        "stationary_noise_only_cases": len(stationary),
        "stationary_noise_only_target_met": all(item["stationary_10db_target_met"] for item in stationary),
        "all_input_files_unchanged": all(item["input_unchanged"] for item in case_results),
        "all_frame_counts_preserved": all(item["frame_count_preserved"] for item in case_results),
        "auto_tracking_implemented": False,
        "real_time_device_performance_verified": False,
        "manual_target_selection_required": True,
        "phase_or_group_delay_removed": False,
        "cases": case_results,
        "interpretation": [
            "Only target-only RMS reduction is described as signal attenuation.",
            "Output-minus-clean for mixtures is restoration error, not isolated noise attenuation.",
            "Same-frequency desired tones are damaged; no automatic semantic separation is claimed.",
            "Moving-tone shortfalls remain recorded; tracking belongs to M3, not this fixed baseline.",
            "First two seconds are retained in startup and whole-record measurements.",
        ],
        "remaining_development_milestones": ["M3", "M4", "M5"],
    }
    _dump_json(destination / "benchmark.json", result)
    _write_html(destination / "report.html", result)
    return result


def _write_html(path: Path, report: dict) -> None:
    rows = []
    for case in report["cases"]:
        attenuation = case["signal_only_attenuation_db"]
        reduction = "not a noise-only case" if attenuation is None else f"{attenuation:.2f} dB"
        error = case["decoded_wav_measurements"]["settled"]["output_error_rmse"]
        rows.append(
            f"<tr><td>{html.escape(case['case_id'])}</td><td>{html.escape(case['kind'])}</td>"
            f"<td>{reduction}</td><td>{error:.7f}</td>"
            f"<td><a href='{html.escape(case['case_id'], quote=True)}/processed.wav'>Processed WAV</a></td></tr>"
        )
    document = (
        "<!doctype html><html lang='en'><meta charset='utf-8'><title>Hum DSP M2 fixed-filter report</title>"
        "<style>body{font:16px system-ui;max-width:1200px;margin:40px auto;padding:0 20px;color:#16222a}"
        "table{border-collapse:collapse;width:100%;font-size:14px}td,th{border:1px solid #ccd5da;padding:10px;text-align:left}"
        ".note{padding:16px;background:#fff4da;line-height:1.6}code{background:#eef2f5}</style>"
        "<h1>Hum DSP — M2 fixed-filter results</h1>"
        "<p class='note'>Synthetic development fixtures only. No microphone or speaker was used. "
        "This is a manually selected fixed filter, not automatic noise recognition, frequency tracking, "
        "spatial cancellation or real-world audio-quality validation. Desired tones at 50/60 Hz are attenuated too.</p>"
        f"<p>Cases: {report['case_count']}. Stationary noise-only 10 dB target met: "
        f"{str(report['stationary_noise_only_target_met']).lower()}. "
        "First 2 seconds remain in JSON startup/full-record measurements; table shows settled measurements.</p>"
        "<table><thead><tr><th>Fixture</th><th>Condition</th><th>Noise-only RMS attenuation</th>"
        "<th>Raw output–clean RMSE</th><th>Output</th></tr></thead><tbody>"
        + "".join(rows) + "</tbody></table>"
        "<p>Mixture RMSE includes wanted-signal damage; component-fit projection is not total noise suppression. "
        "No gain matching or post-hoc alignment was applied.</p><p><a href='benchmark.json'>Full JSON evidence</a></p></html>"
    )
    with path.open("x", encoding="utf-8", newline="\n") as target:
        target.write(document)

