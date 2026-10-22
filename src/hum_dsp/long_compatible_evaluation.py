"""Fixed H3c conversion/continuity cases; not perceptual or denoising evidence."""

import hashlib
import html
import json
from pathlib import Path
from time import perf_counter
import wave

import numpy as np

from .band_control import BandControlConfig
from .compatible_processing import process_compatible_wav
from .long_processing import process_long_wav
from .long_evaluation import _trace_summary


CASE_IDS = (
    "short44_mono_legacy_parity", "short44_stereo_legacy_parity",
    "short44_antiphase_legacy_parity", "long44_65s_stereo_chunk_parity",
    "long44_125s_mono_continuity", "long44_600s_identity", "short48_unchanged_dispatch",
)


def _sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as source:
        while block := source.read(65536):
            result.update(block)
    return result.hexdigest()


def _preflight():
    source = Path(__file__).parent
    previous = source.parents[1] / "outputs/h3b_20261021_v1/evaluation.json"
    old = json.loads(previous.read_text(encoding="utf-8"))
    frozen = {**old["preserved_source_sha256"], **old["source_sha256"]}
    if len(frozen) != 33:
        raise ValueError("H3c requires the full 33-file frozen H3b engine/policy record")
    for name, digest in frozen.items():
        if Path(name).name != name or _sha(source / name) != digest:
            raise ValueError("Historical source changed; H3c evaluation refused")
    policy = json.loads((source / "h3c_policy.json").read_text(encoding="utf-8"))
    if policy.get("stage") != "H3c_continuous_long_44100Hz_WAV":
        raise ValueError("H3c policy missing or wrong stage")
    return source, frozen, policy


def _write_owned(path, rate, seconds, channels, *, antiphase=False, identity=False):
    if rate not in (44100, 48000) or channels not in (1, 2) or not 0 < seconds <= 600:
        raise ValueError("Invalid declared fixture specification")
    frames = round(rate * seconds)
    if frames < 1:
        raise ValueError("Fixture must contain at least one frame")
    with Path(path).open("xb") as handle, wave.open(handle, "wb") as output:
        output.setnchannels(channels)
        output.setsampwidth(2)
        output.setframerate(rate)
        output.setnframes(frames)
        for start in range(0, frames, 4096):
            indices = np.arange(start, min(frames, start + 4096), dtype=np.int64)
            if identity:
                codes = ((indices * 37 + 7919) % 65536 - 32768).astype("<i2")[:, None]
                codes = np.repeat(codes, channels, axis=1)
            else:
                t = indices.astype(np.float64) / rate
                hum = sum(level * np.sin(2 * np.pi * 50 * order * t + .37 * order)
                          for order, level in ((1, .03), (2, .018), (3, .01)))
                left = hum + .04 * np.sin(2 * np.pi * 233 * t + .2)
                values = left[:, None]
                if channels == 2:
                    right = -left if antiphase else -.8 * hum + .025 * np.sin(2 * np.pi * 311 * t + .71)
                    values = np.column_stack((left, right))
                codes = np.rint(values * 32768).astype("<i2")
            output.writeframesraw(codes.tobytes())
    return {"sample_rate_hz": rate, "frames": frames, "channels": channels,
            "duration_seconds": frames / rate, "antiphase": antiphase, "identity_codes": identity}


def _pcm(path):
    result = hashlib.sha256()
    with wave.open(str(path), "rb") as audio:
        metadata = {"sample_rate_hz": audio.getframerate(), "channels": audio.getnchannels(), "frames": audio.getnframes()}
        while block := audio.readframes(4096):
            result.update(block)
    return result.hexdigest(), metadata


def _run(folder, label, fixture, depths, block_size):
    from .long_compatible_processing import process_long_compatible_wav
    config = BandControlConfig(band_depths_db=depths, block_size_frames=block_size)
    report = process_long_compatible_wav(folder / "input.wav", folder / label, config)
    pcm_hash, metadata = _pcm(folder / label / "processed.wav")
    active = bool(config.orders)
    engine_frames = (fixture["frames"] * 48000 + fixture["sample_rate_hz"] - 1) // fixture["sample_rate_hz"]
    trace = _trace_summary(folder / label / "long_trace.jsonl", engine_frames, active)
    algorithm = report["algorithm"]
    checks = {
        "native_rate_channels_frame_count_exact": all(metadata[key] == fixture[key] for key in metadata),
        "source_hash_matches": report["input"]["sha256"] == _sha(folder / "input.wav"),
        "all_global_48k_trace_clocks_exact": trace["all_global_event_clocks_exact"],
        "trace_count_matches_report": trace["event_count"] == report["band_trace"]["event_count"],
        "FFT_count_matches_global_clock": algorithm["shared_original_input_fft_count"] == (trace["event_count"] - 1 if active else 0),
        "no_whole_audio_or_trace_retention": report["resource_limits"]["whole_audio_retained_in_memory"] is False and report["resource_limits"]["whole_trace_retained_in_memory"] is False,
        "no_automatic_normalization_or_device_IO": report["output_gain_normalization_applied"] is False and report["device_io"] is False,
    }
    if active:
        checks["audio_codes_actually_changed"] = report["processor_applied"] is True
        checks["observed_engine_blocks_match_config"] = report["resource_limits"]["max_engine_block_frames_observed"] == block_size
        checks["first_acquisition_at_global_2_seconds"] = trace["first_acquired_samples"] == [96000] * len(config.orders)
        if fixture["duration_seconds"] >= 60:
            checks["no_reset_at_60_seconds"] = trace["checkpoint_seen"]["60_seconds"] and trace["acquisition_preserved_across_60_second_boundary"]
        if fixture["duration_seconds"] >= 120:
            checks["observation_at_120_seconds"] = trace["checkpoint_seen"]["120_seconds"]
    else:
        checks["zero_depth_exact_native_PCM_identity"] = pcm_hash == _pcm(folder / "input.wav")[0]
        checks["zero_depth_no_FFT_or_trace"] = algorithm["shared_original_input_fft_count"] == 0 and trace["event_count"] == 0
    return {"label": label, "relative_directory": f"{folder.name}/{label}", "metadata": metadata,
            "output_PCM16_sha256": pcm_hash, "trace": trace, "checks": checks,
            "all_checks_met": all(checks.values()), "processing_seconds": report["timing"]["file_processing_seconds"],
            "conversion": report.get("conversion"), "resource_limits": report["resource_limits"]}


def _case(target, name, rate, seconds, channels, *, antiphase=False, bypass=False, partition=False, legacy=False):
    folder = target / name
    folder.mkdir()
    fixture = _write_owned(folder / "input.wav", rate, seconds, channels, antiphase=antiphase, identity=bypass)
    depths = (0, 0, 0) if bypass else (6, 3, 0)
    runs = [_run(folder, "blocks_4096", fixture, depths, 4096)]
    checks = {}
    if partition:
        runs.append(_run(folder, "blocks_480", fixture, depths, 480))
        checks["actual_chunk_partitions_exact_PCM"] = runs[0]["output_PCM16_sha256"] == runs[1]["output_PCM16_sha256"]
        checks["actual_chunk_partitions_exact_global_clocks"] = runs[0]["trace"]["event_clocks_sha256"] == runs[1]["trace"]["event_clocks_sha256"]
    if legacy:
        config = BandControlConfig(band_depths_db=depths, block_size_frames=4096)
        if rate == 44100:
            process_compatible_wav(folder / "input.wav", folder / "frozen_H3a", config)
            old = folder / "frozen_H3a/processed.wav"
        else:
            process_long_wav(folder / "input.wav", folder / "frozen_H3b", config)
            old = folder / "frozen_H3b/processed.wav"
        checks["frozen_legacy_exact_PCM"] = runs[0]["output_PCM16_sha256"] == _pcm(old)[0]
    return {"case_id": name, "fixture": fixture, "runs": runs, "checks": checks,
            "all_checks_met": all(checks.values()) and all(run["all_checks_met"] for run in runs)}


def _declared_cases(target):
    return [
        _case(target, CASE_IDS[0], 44100, 3.4, 1, legacy=True),
        _case(target, CASE_IDS[1], 44100, 3.4, 2, legacy=True),
        _case(target, CASE_IDS[2], 44100, 3.4, 2, antiphase=True, legacy=True),
        _case(target, CASE_IDS[3], 44100, 65, 2, partition=True),
        _case(target, CASE_IDS[4], 44100, 125, 1),
        _case(target, CASE_IDS[5], 44100, 600, 1, bypass=True),
        _case(target, CASE_IDS[6], 48000, 3.4, 2, legacy=True),
    ]


def evaluate_long_compatible(output_directory):
    target = Path(output_directory)
    if target.exists() or target.is_symlink():
        raise FileExistsError("Use a new evaluation directory")
    source, frozen, policy = _preflight()
    names = ("stream_resampling.py", "long_compatible_processing.py", "long_compatible_evaluation.py", "h3c_policy.json")
    current = {name: _sha(source / name) for name in names}
    started = perf_counter()
    target.mkdir(parents=True, exist_ok=False)
    cases = _declared_cases(target)
    if [case["case_id"] for case in cases] != list(CASE_IDS):
        raise ValueError("Incomplete fixed case sequence")
    historical_unchanged = all(_sha(source / name) == value for name, value in frozen.items())
    current_unchanged = all(_sha(source / name) == value for name, value in current.items())
    result = {"schema_version": 1, "stage": "H3c", "case_count": len(cases), "cases": cases,
              "policy": policy, "source_sha256": current, "preserved_source_sha256": frozen,
              "historical_source_count": len(frozen), "historical_sources_unchanged": historical_unchanged,
              "current_sources_unchanged_during_run": current_unchanged,
              "all_declared_contract_checks_met": historical_unchanged and current_unchanged and all(c["all_checks_met"] for c in cases),
              "evaluation_seconds": perf_counter() - started,
              "fixture_scope": "owned_deterministic_mathematical_signals_not_natural_noise",
              "perceptual_quality_verified": False, "commercial_approval": False,
              "whole_process_RSS_or_device_realtime_verified": False,
              "audio_played_or_recorded": False, "external_publication": False}
    with (target / "evaluation.json").open("x", encoding="utf-8") as output:
        output.write(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    rows = []
    for case in cases:
        for run in case["runs"]:
            directory = html.escape(run["relative_directory"], quote=True)
            values = (case["case_id"], case["fixture"]["duration_seconds"], run["label"], run["trace"]["event_count"], run["all_checks_met"])
            rows.append("<tr>" + "".join("<td>" + html.escape(str(v)) + "</td>" for v in values) + f'<td><a href="{directory}/report.json">JSON</a></td></tr>')
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>H3c long 44.1kHz WAV contracts</title><style>body{font:16px system-ui;background:#f5f7f1;color:#173b2b;margin:32px}main{max-width:1200px;margin:auto}p{line-height:1.6}table{border-collapse:collapse;background:white}td,th{padding:10px;border:1px solid #cbd5c5}.scroll{overflow:auto}</style><main><h1>H3c: continuous long 44.1 kHz WAV</h1><p>Seven fixed mathematical cases exercise native length, true-file-edge FIR behavior, continuous 48 kHz DSP state, exact chunk partition PCM, frozen short-engine parity and 600-second zero-depth identity.</p><p>44.1 kHz active audio is explicitly converted to 48 kHz and back using the frozen H3a kernel. FIR boundary and high-frequency changes remain; no clipping, gain fitting or post-hoc alignment. A preliminary conversion pass selects the full-file RMS reference channel. No complete audio or trace is retained. This is offline processing, not microphone/device streaming.</p><p>These are compatibility contracts, not proof of perceptual quality, natural-noise removal, total-process memory limits, real-time latency or commercial readiness. No audio was played or recorded.</p><p><a href="evaluation.json">Complete policy, checks and source hashes</a></p><div class="scroll"><table><tr><th>Case</th><th>Seconds</th><th>Partition</th><th>48k events</th><th>Checks</th><th>Report</th></tr>'''
    with (target / "report.html").open("x", encoding="utf-8") as output:
        output.write(page + "".join(rows) + "</table></div></main></html>")
    return result
