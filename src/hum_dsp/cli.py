"""Offline fixture and fixed-filter commands. No devices or network access."""

import argparse
import json
from pathlib import Path

from .audio import SAMPLE_RATE_HZ, read_pcm16_wav
from .fixtures import export_fixture_suite, verify_fixture_suite, _dump_json
from .config import ProcessorConfig
from .processing import process_wav
from .benchmark import run_fixed_benchmark
from .tracked_processing import process_tracked_wav
from .tracking_benchmark import run_tracking_benchmark
from .guarded_processing import process_guarded_wav
from .safety_benchmark import run_safety_benchmark


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Hum DSP: fixed or opt-in limited-band tracking for manually selected 50/60 Hz; no device I/O.")
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate-fixtures", help="Create a new local synthetic development suite (not denoised output)")
    generate.add_argument("--output", type=Path, required=True)
    validate = commands.add_parser("validate-input", help="Check mono 48 kHz PCM16 WAV without changing it")
    validate.add_argument("input", type=Path)
    verify = commands.add_parser("verify-fixtures", help="Check generated fixture integrity and input contracts, not denoising quality")
    verify.add_argument("suite", type=Path)
    verify.add_argument("--report", type=Path, help="Write a new JSON verification report; never overwrite")
    process = commands.add_parser("process", help="Write a fixed-filter processed WAV and report to a new directory; no playback")
    process.add_argument("input", type=Path)
    process.add_argument("--output-dir", type=Path, required=True)
    process.add_argument("--hum-hz", type=float, choices=(50, 60), required=True, help="Explicit target frequency; wanted tones here will also change")
    process.add_argument("--attenuation-db", type=float, default=12.0)
    process.add_argument("--block-size", type=int, default=480)
    benchmark = commands.add_parser("benchmark-fixed", help="Evaluate the fixed filter on all sixteen declared synthetic development cases")
    benchmark.add_argument("suite", type=Path)
    benchmark.add_argument("--output-dir", type=Path, required=True)
    tracked = commands.add_parser("process-tracked", help="Opt-in causal limited-band center tracking; no playback or automatic noise semantics")
    tracked.add_argument("input", type=Path)
    tracked.add_argument("--output-dir", type=Path, required=True)
    tracked.add_argument("--hum-hz", type=float, choices=(50, 60), required=True)
    tracked.add_argument("--attenuation-db", type=float, default=12.0)
    tracked.add_argument("--block-size", type=int, default=480)
    tracking_benchmark = commands.add_parser("benchmark-tracked", help="Compare center tracking against preserved M2 WAVs on the frozen synthetic suite")
    tracking_benchmark.add_argument("suite", type=Path)
    tracking_benchmark.add_argument("--baseline", type=Path, required=True)
    tracking_benchmark.add_argument("--output-dir", type=Path, required=True)
    guarded = commands.add_parser("process-guarded", help="Opt-in qualified tracked processing with soft attack/release; not a hardware safety or semantic noise guarantee")
    guarded.add_argument("input", type=Path)
    guarded.add_argument("--output-dir", type=Path, required=True)
    guarded.add_argument("--hum-hz", type=float, choices=(50, 60), required=True)
    guarded.add_argument("--attenuation-db", type=float, default=12.0)
    guarded.add_argument("--block-size", type=int, default=480)
    harmonic = commands.add_parser("process-harmonics", help="Opt-in independent evidence-gated harmonic bands; wanted content here also changes")
    harmonic.add_argument("input", type=Path)
    harmonic.add_argument("--output-dir", type=Path, required=True)
    harmonic.add_argument("--hum-hz", type=float, choices=(50, 60), required=True)
    harmonic.add_argument("--orders", type=int, nargs="+", default=[1, 2, 3], help="Selected sorted unique harmonic orders in 1..6 (default: 1 2 3)")
    harmonic.add_argument("--attenuation-db", type=float, default=12.0)
    harmonic.add_argument("--block-size", type=int, default=480)
    bands = commands.add_parser("process-bands", help="Offline manual attenuation for three bands; no automatic wanted-content protection")
    bands.add_argument("input", type=Path)
    bands.add_argument("--output-dir", type=Path, required=True)
    bands.add_argument("--hum-hz", type=float, choices=(50, 60), required=True)
    bands.add_argument("--band-depths-db", type=float, nargs=3, default=[6, 3, 0], metavar=("BASE", "SECOND", "THIRD"))
    bands.add_argument("--block-size", type=int, default=480)
    compatible = commands.add_parser("process-compatible", help="Explicit 44.1/48 kHz mono/stereo PCM16 WAV mode; linked filters and recorded conversion, up to 60 seconds")
    compatible.add_argument("input", type=Path)
    compatible.add_argument("--output-dir", type=Path, required=True)
    compatible.add_argument("--hum-hz", type=float, choices=(50, 60), required=True)
    compatible.add_argument("--band-depths-db", type=float, nargs=3, default=[6, 3, 0], metavar=("BASE", "SECOND", "THIRD"))
    compatible.add_argument("--block-size", type=int, default=480)
    long_file = commands.add_parser("process-long", help="Continuous-state 44.1/48 kHz mono/stereo PCM16 WAV, up to 10 minutes, with bounded explicit conversion")
    long_file.add_argument("input", type=Path)
    long_file.add_argument("--output-dir", type=Path, required=True)
    long_file.add_argument("--hum-hz", type=float, choices=(50, 60), required=True)
    long_file.add_argument("--band-depths-db", type=float, nargs=3, default=[6, 3, 0], metavar=("BASE", "SECOND", "THIRD"))
    long_file.add_argument("--block-size", type=int, default=4096)
    precision = commands.add_parser("process-precision", help="Same-format PCM16/PCM24/Float32 WAV, 44.1/48 kHz mono/stereo, bounded 600s/128MiB; no 16-bit intermediate")
    precision.add_argument("input", type=Path)
    precision.add_argument("--output-dir", type=Path, required=True)
    precision.add_argument("--hum-hz", type=float, choices=(50, 60), required=True)
    precision.add_argument("--band-depths-db", type=float, nargs=3, default=[6, 3, 0], metavar=("BASE", "SECOND", "THIRD"))
    precision.add_argument("--block-size", type=int, default=4096)
    compressed = commands.add_parser("process-compressed", help="Opt-in MP3 or AAC M4A import to decoded Float32 WAV; optional PyAV 19, no lossy re-encoding")
    compressed.add_argument("input", type=Path)
    compressed.add_argument("--output-dir", type=Path, required=True)
    compressed.add_argument("--hum-hz", type=float, choices=(50, 60), required=True)
    compressed.add_argument("--band-depths-db", type=float, nargs=3, default=[6, 3, 0], metavar=("BASE", "SECOND", "THIRD"))
    compressed.add_argument("--block-size", type=int, default=4096)
    compressed_eval = commands.add_parser("evaluate-compressed", help="Repository-only owned MP3/AAC import contract checks, not real-noise or perceptual quality approval")
    compressed_eval.add_argument("--output-dir", type=Path, required=True)
    safety = commands.add_parser("benchmark-guarded", help="Evaluate M4 controls with preserved M3 evidence and ten additional synthetic guard cases")
    safety.add_argument("suite", type=Path)
    safety.add_argument("--baseline", type=Path, required=True)
    safety.add_argument("--output-dir", type=Path, required=True)
    local = commands.add_parser("serve", help="Open a loopback-only local processing UI; no automatic browser launch or audio playback")
    local.add_argument("--port", type=int, default=0, help="Local port; 0 chooses a free port (default)")
    local_evaluation = commands.add_parser("evaluate-local", help="Repository-only M5 check: compare local tool output to actual CLI on owned synthetic examples; no devices")
    local_evaluation.add_argument("--output-dir", type=Path, required=True)
    harmonic_benchmark = commands.add_parser("benchmark-harmonics", help="Compare selective harmonic processing and oracle wanted-signal distortion on twelve owned development cases")
    harmonic_benchmark.add_argument("--output-dir", type=Path, required=True)
    band_eval = commands.add_parser("evaluate-bands", help="Repository-only declared CC0 real-source plus injected-hum study; not a perceptual quality approval")
    band_eval.add_argument("catalog", type=Path)
    band_eval.add_argument("--output-dir", type=Path, required=True)
    compatible_eval = commands.add_parser("evaluate-compatible", help="Repository-only native WAV/stereo/conversion contract check, not noise-removal quality approval")
    compatible_eval.add_argument("--output-dir", type=Path, required=True)
    compatible_eval.add_argument("--piano-source", type=Path, help="Optional exact hash-pinned CC0 Emotional Piano source from the retained H2 files")
    long_eval = commands.add_parser("evaluate-long", help="Repository-only continuous long WAV contract checks; not sound-quality approval")
    long_eval.add_argument("--output-dir", type=Path, required=True)
    long_compatible_eval = commands.add_parser("evaluate-long-compatible", help="Repository-only long 44.1/48 kHz conversion and continuity contracts, not sound-quality approval")
    long_compatible_eval.add_argument("--output-dir", type=Path, required=True)
    precision_eval = commands.add_parser("evaluate-precision", help="Repository-only same-format PCM24/Float32 WAV contracts with optional test-only SciPy reference reader")
    precision_eval.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "serve":
            from .local_app import serve_local
            serve_local(args.port)
            return 0
        elif args.command == "evaluate-local":
            from .local_evaluation import run_local_evaluation
            result = run_local_evaluation(args.output_dir)
            payload = {"status": "local_tool_scoped_evaluation_complete", "case_count": result["case_count"], "actual_cli_parity_count": result["actual_cli_parity_count"], "scoped_checks_met": result["all_scoped_checks_met"], "report": str(args.output_dir / "report.html"), "independent_real_audio_quality_verified": False}
        elif args.command == "benchmark-harmonics":
            from .harmonic_benchmark import run_harmonic_benchmark
            result = run_harmonic_benchmark(args.output_dir)
            payload = {"status": "harmonic_development_comparison_complete", "case_count": result["case_count"], "development_targets_met": result["all_development_targets_met"], "report": str(args.output_dir / "report.html"), "independent_real_audio_quality_verified": False}
        elif args.command == "evaluate-bands":
            from .band_evaluation import run_band_evaluation
            result = run_band_evaluation(args.catalog, args.output_dir)
            payload = {"status": "manual_band_real_source_development_study_complete", "case_count": result["case_count"], "structural_checks_met": result["all_declared_structural_checks_met"], "report": str(args.output_dir / "report.html"), "independent_perceptual_quality_verified": False}
        elif args.command == "evaluate-compatible":
            from .compatible_evaluation import run_compatible_evaluation
            result = run_compatible_evaluation(args.output_dir, args.piano_source)
            payload = {"status": "WAV_compatibility_contract_evaluation_complete", "case_count": result["case_count"], "contract_checks_met": result["all_declared_contract_checks_met"], "report": str(args.output_dir / "report.html"), "perceptual_quality_verified": False}
        elif args.command == "evaluate-long":
            from .long_evaluation import evaluate_long
            result = evaluate_long(args.output_dir)
            payload = {"status": "continuous_long_WAV_contract_evaluation_complete", "case_count": result["case_count"], "contract_checks_met": result["all_declared_contract_checks_met"], "report": str(args.output_dir / "report.html"), "perceptual_quality_verified": False}
        elif args.command == "evaluate-long-compatible":
            from .long_compatible_evaluation import evaluate_long_compatible
            result = evaluate_long_compatible(args.output_dir)
            payload = {"status": "long_44k_WAV_conversion_contract_evaluation_complete", "case_count": result["case_count"], "contract_checks_met": result["all_declared_contract_checks_met"], "report": str(args.output_dir / "report.html"), "perceptual_quality_verified": False}
        elif args.command == "evaluate-precision":
            from .precision_evaluation import evaluate_precision
            result = evaluate_precision(args.output_dir)
            payload = {"status": "native_precision_WAV_contract_evaluation_complete", "case_count": result["case_count"], "contract_checks_met": result["all_declared_contract_checks_met"], "report": str(args.output_dir / "report.html"), "perceptual_quality_verified": False}
        elif args.command == "evaluate-compressed":
            from .compressed_evaluation import evaluate_compressed
            result = evaluate_compressed(args.output_dir)
            payload = {"status": "explicit_compressed_import_contract_evaluation_complete", "case_count": result["case_count"], "contract_checks_met": result["all_declared_contract_checks_met"], "report": str(args.output_dir / "report.html"), "perceptual_quality_verified": False}
        elif args.command == "generate-fixtures":
            result = export_fixture_suite(args.output)
            payload = {"status": "fixtures_created_without_processing", "case_count": result["case_count"], "manifest": str(args.output / "manifest.json"), "processor_applied": False}
        elif args.command == "validate-input":
            samples = read_pcm16_wav(args.input)
            payload = {"status": "supported_input", "frames": int(samples.size), "sample_rate_hz": SAMPLE_RATE_HZ, "duration_seconds": samples.size / SAMPLE_RATE_HZ, "processor_applied": False}
        elif args.command == "verify-fixtures":
            payload = verify_fixture_suite(args.suite)
            if args.report is not None:
                _dump_json(args.report, payload)
        elif args.command == "process":
            result = process_wav(args.input, args.output_dir, ProcessorConfig(nominal_hum_hz=args.hum_hz, max_attenuation_db=args.attenuation_db, block_size_frames=args.block_size))
            payload = {"status": "fixed_filter_output_created", "output_wav": str(args.output_dir / "processed.wav"), "report": str(args.output_dir / "report.json"), "frames": result["output"]["frames"], "device_io": False, "reference_quality_metrics": None}
        elif args.command == "benchmark-fixed":
            result = run_fixed_benchmark(args.suite, args.output_dir)
            payload = {"status": "synthetic_fixed_filter_benchmark_complete", "case_count": result["case_count"], "stationary_noise_only_target_met": result["stationary_noise_only_target_met"], "report": str(args.output_dir / "report.html"), "remaining_development_milestones": result["remaining_development_milestones"]}
        elif args.command == "process-tracked":
            result = process_tracked_wav(args.input, args.output_dir, ProcessorConfig(nominal_hum_hz=args.hum_hz, max_attenuation_db=args.attenuation_db, block_size_frames=args.block_size))
            payload = {"status": "tracked_filter_output_created", "output_wav": str(args.output_dir / "processed.wav"), "tracking": str(args.output_dir / "tracking.json"), "report": str(args.output_dir / "report.json"), "first_acquired_sample": result["tracking"]["first_acquired_sample"], "device_io": False}
        elif args.command == "process-guarded":
            result = process_guarded_wav(args.input, args.output_dir, ProcessorConfig(nominal_hum_hz=args.hum_hz, max_attenuation_db=args.attenuation_db, block_size_frames=args.block_size))
            payload = {"status": "guarded_filter_output_created", "output_wav": str(args.output_dir / "processed.wav"), "guard": str(args.output_dir / "guard.json"), "tracking": str(args.output_dir / "tracking.json"), "report": str(args.output_dir / "report.json"), "device_io": False}
        elif args.command == "process-harmonics":
            from .harmonics import HarmonicConfig
            from .harmonic_processing import process_harmonic_wav
            result = process_harmonic_wav(args.input, args.output_dir, HarmonicConfig(nominal_hum_hz=args.hum_hz, orders=tuple(args.orders), max_attenuation_db=args.attenuation_db, block_size_frames=args.block_size))
            payload = {"status": "harmonic_filter_output_created", "output_wav": str(args.output_dir / "processed.wav"), "trace": str(args.output_dir / "harmonics.json"), "report": str(args.output_dir / "report.json"), "device_io": False, "reference_quality_metrics": result["reference_quality_metrics"]}
        elif args.command == "process-bands":
            from .band_control import BandControlConfig
            from .band_processing import process_band_wav
            result = process_band_wav(args.input, args.output_dir, BandControlConfig(nominal_hum_hz=args.hum_hz, band_depths_db=tuple(args.band_depths_db), block_size_frames=args.block_size))
            payload = {"status": "manual_per_band_output_created", "output_wav": str(args.output_dir / "processed.wav"), "trace": str(args.output_dir / "bands.json"), "report": str(args.output_dir / "report.json"), "device_io": False, "reference_quality_metrics": result["reference_quality_metrics"]}
        elif args.command == "process-compatible":
            from .band_control import BandControlConfig
            from .compatible_processing import process_compatible_wav
            result = process_compatible_wav(args.input, args.output_dir, BandControlConfig(nominal_hum_hz=args.hum_hz, band_depths_db=tuple(args.band_depths_db), block_size_frames=args.block_size))
            payload = {"status": "compatible_WAV_output_created", "output_wav": str(args.output_dir / "processed.wav"), "trace": str(args.output_dir / "compatible.json"), "report": str(args.output_dir / "report.json"), "device_io": False, "reference_quality_metrics": result["reference_quality_metrics"]}
        elif args.command == "process-long":
            from .band_control import BandControlConfig
            from .long_compatible_processing import process_long_compatible_wav
            result = process_long_compatible_wav(args.input, args.output_dir, BandControlConfig(nominal_hum_hz=args.hum_hz, band_depths_db=tuple(args.band_depths_db), block_size_frames=args.block_size))
            payload = {"status": "continuous_long_WAV_output_created", "output_wav": str(args.output_dir / "processed.wav"), "trace": str(args.output_dir / "long_trace.jsonl"), "report": str(args.output_dir / "report.json"), "device_io": False, "reference_quality_metrics": result["reference_quality_metrics"]}
        elif args.command == "process-precision":
            from .band_control import BandControlConfig
            from .precision_processing import process_precision_wav
            result = process_precision_wav(args.input, args.output_dir, BandControlConfig(nominal_hum_hz=args.hum_hz, band_depths_db=tuple(args.band_depths_db), block_size_frames=args.block_size))
            payload = {"status": "same_format_precision_WAV_output_created", "output_wav": str(args.output_dir / "processed.wav"), "trace": str(args.output_dir / "long_trace.jsonl"), "report": str(args.output_dir / "report.json"), "sample_format": result["output"]["sample_format"], "device_io": False, "reference_quality_metrics": result["reference_quality_metrics"]}
        elif args.command == "process-compressed":
            from .band_control import BandControlConfig
            from .compressed_processing import process_compressed_file
            result = process_compressed_file(args.input, args.output_dir, BandControlConfig(nominal_hum_hz=args.hum_hz, band_depths_db=tuple(args.band_depths_db), block_size_frames=args.block_size))
            payload = {"status": "decoded_Float32_WAV_and_DSP_output_created", "decoded_input_wav": str(args.output_dir / "decoded_input.wav"), "output_wav": str(args.output_dir / "processed.wav"), "report": str(args.output_dir / "report.json"), "source_codec": result["compressed_source"]["source_codec"], "lossy_reencoding": False, "device_io": False}
        elif args.command == "benchmark-guarded":
            result = run_safety_benchmark(args.suite, args.baseline, args.output_dir)
            payload = {"status": "synthetic_guard_comparison_complete", "retained_case_count": result["retained_case_count"], "safety_case_count": result["safety_case_count"], "development_targets_met": result["m4_development_targets_met"], "report": str(args.output_dir / "report.html"), "remaining_development_milestones": ["M5"] if result["m4_development_targets_met"] else ["M4", "M5"]}
        else:
            result = run_tracking_benchmark(args.suite, args.baseline, args.output_dir)
            payload = {"status": "synthetic_tracking_comparison_complete", "case_count": result["case_count"], "development_targets_met": result["m3_development_targets_met"], "report": str(args.output_dir / "report.html"), "remaining_development_milestones": ["M4", "M5"] if result["m3_development_targets_met"] else ["M3", "M4", "M5"]}
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        parser.exit(2, f"Error: {exc}\n")
    print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))
    return 0

