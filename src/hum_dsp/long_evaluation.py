"""Reproducible H3b file/clock contracts, not denoising or listening evidence.

Owned mathematical fixtures are generated a block at a time. Long results and
JSONL events are inspected sequentially; no complete long audio or trace is
loaded. The fixed cases exercise real durations, not manually advanced clocks.
"""

import hashlib
import html
import json
import math
from pathlib import Path
import struct
from time import perf_counter
import wave

import numpy as np

from .band_control import BandControlConfig
from .band_processing import process_band_wav
from .compatible_processing import process_compatible_wav
from .long_processing import inspect_long_wav, process_long_wav


SAMPLE_RATE_HZ = 48000
OBSERVATION_FRAMES = 96000
UPDATE_FRAMES = 4800
CASE_IDS = (
    "owned_65s_stereo_chunk_parity", "owned_125s_mono_continuity",
    "owned_600s_mono_identity", "owned_short_mono_legacy_parity",
    "owned_short_stereo_legacy_parity",
)


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(65536):
            digest.update(block)
    return digest.hexdigest()


def _preflight():
    source = Path(__file__).parent
    historical = source.parents[1] / "outputs/h3_20261021_v1/evaluation.json"
    old = json.loads(historical.read_text(encoding="utf-8"))
    preserved = {**old["preserved_source_sha256"], **old["source_sha256"]}
    if len(preserved) != 29:
        raise ValueError("H3b requires the complete frozen 29-file H3a source record")
    for name, digest in preserved.items():
        if (type(name) is not str or Path(name).name != name or type(digest) is not str
                or len(digest) != 64 or _sha(source / name) != digest):
            raise ValueError("preserved H1/H2/H3a source changed; H3b evaluation refused")
    policy = json.loads((source / "h3b_policy.json").read_text(encoding="utf-8"))
    if policy.get("stage") != "H3b_bounded_continuous_long_WAV":
        raise ValueError("H3b frozen policy stage is invalid")
    return source, policy, preserved, _sha(historical)


def _write_owned(path, seconds, channels, *, identity_codes=False):
    """Write a new classic PCM16 fixture using <=4096 generated frames at once."""
    frames = round(SAMPLE_RATE_HZ * seconds)
    if not 1 <= frames <= 600 * SAMPLE_RATE_HZ or channels not in (1, 2):
        raise ValueError("invalid owned long fixture duration or channels")
    with Path(path).open("xb") as handle:
        with wave.open(handle, "wb") as wav:
            wav.setnchannels(channels)
            wav.setsampwidth(2)
            wav.setframerate(SAMPLE_RATE_HZ)
            wav.setnframes(frames)
            for start in range(0, frames, 4096):
                indices = np.arange(start, min(start + 4096, frames), dtype=np.int64)
                if identity_codes:
                    codes = ((indices * 37 + 7919) % 65536 - 32768).astype("<i2")
                    codes = np.repeat(codes[:, None], channels, axis=1)
                else:
                    t = indices.astype(np.float64) / SAMPLE_RATE_HZ
                    hum = sum(amplitude * np.sin(2 * np.pi * 50 * order * t + .37 * order)
                              for order, amplitude in ((1, .03), (2, .018), (3, .01)))
                    left = hum + .04 * np.sin(2 * np.pi * 233 * t + .2)
                    values = left[:, None]
                    if channels == 2:
                        right = -.8 * hum + .025 * np.sin(2 * np.pi * 311 * t + .71)
                        values = np.column_stack((left, right))
                    codes = np.rint(values * 32768)
                    if not np.isfinite(codes).all() or np.any(codes < -32768) or np.any(codes > 32767):
                        raise ValueError("owned mathematical fixture cannot be encoded without clipping")
                    codes = codes.astype("<i2")
                wav.writeframesraw(codes.tobytes())
    return {"frames": frames, "channels": channels, "sample_rate_hz": SAMPLE_RATE_HZ,
            "duration_seconds": frames / SAMPLE_RATE_HZ,
            "generation": "deterministic_PCM16_code_cycle" if identity_codes else
                          "50_100_150_Hz_hum_plus_233_Hz_left_311_Hz_right_owned_sines",
            "maximum_generation_block_frames": 4096}


def _pcm_digest(path):
    metadata = inspect_long_wav(path)
    digest, consumed = hashlib.sha256(), 0
    with wave.open(str(path), "rb") as wav:
        while payload := wav.readframes(4096):
            digest.update(payload)
            consumed += len(payload) // (2 * metadata["channels"])
    if consumed != metadata["frames"]:
        raise ValueError("incomplete evaluation PCM scan")
    return digest.hexdigest(), {key: metadata[key] for key in
                                ("frames", "channels", "sample_rate_hz", "duration_seconds")}


def _trace_summary(path, frames, active):
    """Check every streamed event clock while retaining only small counters."""
    count, clock_hash = 0, hashlib.sha256()
    last_clock, last_kind = None, None
    acquired_indices = None
    acquisition_preserved = True
    checkpoint_seen = {"60_seconds": False, "120_seconds": False}
    with Path(path).open("r", encoding="utf-8") as handle:
        while line := handle.readline(1024 * 1024 + 1):
            if len(line) > 1024 * 1024:
                raise ValueError("evaluation JSONL event exceeds its bounded line limit")
            event = json.loads(line)
            clock = event.get("sample_index")
            when = event.get("time_seconds")
            expected = 0 if count == 0 else OBSERVATION_FRAMES + (count - 1) * UPDATE_FRAMES
            kind = "initial_dry" if count == 0 else "observation"
            if (not active or type(clock) is not int or clock != expected or clock > frames
                    or type(when) not in (int, float) or not math.isfinite(when)
                    or when != clock / SAMPLE_RATE_HZ or event.get("event_kind") != kind):
                raise ValueError("JSONL trace lost the declared continuous global clock")
            clock_hash.update(struct.pack("<qd", clock, float(when)))
            if count:
                acquisition = [band.get("first_acquired_sample") for band in event["bands"]]
                if acquired_indices is None:
                    acquired_indices = acquisition
                if clock >= 60 * SAMPLE_RATE_HZ:
                    acquisition_preserved = acquisition_preserved and acquisition == acquired_indices
            for label, seconds in (("60_seconds", 60), ("120_seconds", 120)):
                checkpoint_seen[label] = checkpoint_seen[label] or clock == seconds * SAMPLE_RATE_HZ
            last_clock, last_kind = clock, kind
            count += 1
    expected_count = (0 if not active else 1 + (0 if frames < OBSERVATION_FRAMES else
                                              1 + (frames - OBSERVATION_FRAMES) // UPDATE_FRAMES))
    if count != expected_count:
        raise ValueError("JSONL trace is missing events or contains unexpected events")
    return {
        "event_count": count, "event_clocks_sha256": clock_hash.hexdigest(),
        "last_sample_index": last_clock, "last_event_kind": last_kind,
        "expected_event_count": expected_count,
        "all_global_event_clocks_exact": True,
        "checkpoint_seen": checkpoint_seen,
        "first_acquired_samples": acquired_indices,
        "acquisition_preserved_across_60_second_boundary": acquisition_preserved if active else None,
        "complete_trace_retained_in_memory": False,
    }


def _run(folder, label, config, expected, source_sha):
    destination = folder / label
    result = process_long_wav(folder / "input.wav", destination, config)
    pcm_sha, metadata = _pcm_digest(destination / "processed.wav")
    trace = _trace_summary(destination / "long_trace.jsonl", expected["frames"], bool(config.orders))
    algorithm, resources = result["algorithm"], result["resource_limits"]
    checks = {
        "source_unchanged": _sha(folder / "input.wav") == source_sha,
        "native_frames_channels_rate_preserved": all(metadata[key] == expected[key]
                                                      for key in ("frames", "channels", "sample_rate_hz")),
        "global_JSONL_event_schedule_exact": trace["all_global_event_clocks_exact"],
        "trace_count_matches_report": result["band_trace"]["event_count"] == trace["event_count"],
        "FFT_count_matches_global_clock": algorithm["shared_original_input_fft_count"] ==
                                           (trace["event_count"] - 1 if config.orders else 0),
        "bounded_disk_reads": resources["observed_max_read_bytes"] <= 16384,
        "actual_engine_blocks_obey_configuration": (resources.get("max_engine_block_frames_observed") ==
                                                       min(config.block_size_frames, expected["frames"]))
                                                      if config.orders else None,
        "whole_audio_and_trace_not_retained": resources["whole_audio_retained_in_memory"] is False and
                                               resources["whole_trace_retained_in_memory"] is False,
        "one_latest_engine_event_only": algorithm.get("retained_trace_records") == 1 if config.orders else None,
        "fixed_two_second_engine_history": algorithm.get("retained_audio_frames") == OBSERVATION_FRAMES
                                             if config.orders else None,
        "active_audio_codes_actually_changed": result["processor_applied"] is True if config.orders else None,
    }
    if config.orders:
        checks["first_acquisition_at_global_two_seconds"] = (
            trace["first_acquired_samples"] == [OBSERVATION_FRAMES] * len(config.orders))
        if expected["frames"] >= 60 * SAMPLE_RATE_HZ:
            checks["observation_at_60_seconds"] = trace["checkpoint_seen"]["60_seconds"]
            checks["acquisition_not_reset_at_60_seconds"] = trace["acquisition_preserved_across_60_second_boundary"]
        if expected["frames"] >= 120 * SAMPLE_RATE_HZ:
            checks["observation_at_120_seconds"] = trace["checkpoint_seen"]["120_seconds"]
    else:
        original_pcm, _ = _pcm_digest(folder / "input.wav")
        checks["zero_depth_exact_PCM16_identity"] = pcm_sha == original_pcm
        checks["owned_classic_WAV_container_byte_identity"] = result["output"]["sha256"] == source_sha
        checks["zero_depth_no_FFT_or_events"] = algorithm["shared_original_input_fft_count"] == 0 and trace["event_count"] == 0
    return {
        "label": label, "relative_directory": f"{folder.name}/{label}",
        "config": config.to_dict(), "metadata": metadata,
        "output_container_sha256": result["output"]["sha256"], "output_PCM16_sha256": pcm_sha,
        "reference_channel_index": result["stereo_policy"]["reference_channel_index"],
        "shared_original_input_fft_count": algorithm["shared_original_input_fft_count"],
        "trace": trace, "checks": checks,
        "all_declared_checks_met": all(value is not False for value in checks.values()),
        "file_processing_seconds": result["timing"]["file_processing_seconds"],
        "resource_limits": resources,
    }


def _declared_case(target, case_id, seconds, channels, *, bypass=False, chunk_parity=False, legacy=False):
    folder = target / case_id
    folder.mkdir()
    expected = _write_owned(folder / "input.wav", seconds, channels, identity_codes=bypass)
    source_sha = _sha(folder / "input.wav")
    depths = (0, 0, 0) if bypass else (6, 3, 0)
    large = BandControlConfig(band_depths_db=depths, block_size_frames=4096)
    runs = [_run(folder, "blocks_4096", large, expected, source_sha)]
    checks = {}
    if chunk_parity:
        small = BandControlConfig(band_depths_db=depths, block_size_frames=480)
        runs.append(_run(folder, "blocks_480", small, expected, source_sha))
        checks["actual_4096_vs_480_engine_block_sizes_observed"] = (
            runs[0]["resource_limits"].get("max_engine_block_frames_observed") == 4096 and
            runs[1]["resource_limits"].get("max_engine_block_frames_observed") == 480)
        checks["chunk_partition_exact_PCM16_identity"] = runs[0]["output_PCM16_sha256"] == runs[1]["output_PCM16_sha256"]
        checks["chunk_partition_exact_global_JSONL_event_times"] = (
            runs[0]["trace"]["event_clocks_sha256"] == runs[1]["trace"]["event_clocks_sha256"] and
            runs[0]["trace"]["event_count"] == runs[1]["trace"]["event_count"])
    if legacy:
        process_compatible_wav(folder / "input.wav", folder / "preserved_H3a", large)
        old_digest, _ = _pcm_digest(folder / "preserved_H3a/processed.wav")
        checks["short_48k_frozen_H3a_exact_PCM16_identity"] = runs[0]["output_PCM16_sha256"] == old_digest
        if channels == 1:
            process_band_wav(folder / "input.wav", folder / "preserved_H2", large)
            h2_digest, _ = _pcm_digest(folder / "preserved_H2/processed.wav")
            checks["short_48k_mono_frozen_H2_exact_PCM16_identity"] = runs[0]["output_PCM16_sha256"] == h2_digest
    return {"case_id": case_id, "fixture": expected, "input_sha256": source_sha,
            "runs": runs, "checks": checks,
            "all_declared_checks_met": all(run["all_declared_checks_met"] for run in runs) and
                                        all(value is not False for value in checks.values())}


def _run_declared_cases(target):
    return [
        _declared_case(target, CASE_IDS[0], 65, 2, chunk_parity=True),
        _declared_case(target, CASE_IDS[1], 125, 1),
        _declared_case(target, CASE_IDS[2], 600, 1, bypass=True),
        _declared_case(target, CASE_IDS[3], 3.4, 1, legacy=True),
        _declared_case(target, CASE_IDS[4], 3.4, 2, legacy=True),
    ]


def _report(result, path):
    rows = []
    for case in result["cases"]:
        for run in case["runs"]:
            directory = html.escape(run["relative_directory"], quote=True)
            cells = (case["case_id"], run["label"], run["metadata"]["duration_seconds"],
                     run["metadata"]["channels"], run["shared_original_input_fft_count"],
                     run["trace"]["event_count"], run["all_declared_checks_met"])
            rows.append("<tr>" + "".join(f"<td>{html.escape(str(value))}</td>" for value in cells) +
                        f'<td><a href="{directory}/report.json">JSON</a> · <a href="{directory}/processed.wav">WAV</a></td></tr>')
    text = """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>H3b continuous long-WAV contracts</title>
<style>body{font:16px system-ui;background:#f5f7f1;color:#173b2b;margin:32px}main{max-width:1250px;margin:auto}p{line-height:1.6}table{border-collapse:collapse;background:white;width:100%;font-size:14px}td,th{padding:10px;border:1px solid #cbd5c5}a{color:#25654a}.scroll{overflow-x:auto}</style><main>
<h1>H3b: continuous long-file WAV processing</h1><p>48 kHz mono/stereo PCM16 WAV, up to ten minutes. Offline bounded disk passes select a fixed original-input reference channel, then continuously process independent channel filter states. No reset at 60-second boundaries, no stereo summing, no automatic resampling, clipping or normalization.</p>
<p>Fixed owned-maths checks: real 65-second active stereo with actual 4096 versus 480-frame engine chunks; real 125-second active mono through the 120-second boundary; real 600-second all-zero-depth PCM identity; short 48 kHz mono/stereo equivalence to frozen H2/H3a.</p>
<p>These are file, clock, partition and bypass contracts. They do not establish improved denoising accuracy, perceptual quality, naturally noisy audio performance, device real-time latency or total-process memory usage. Wanted same-frequency content can still be attenuated. No audio was played or recorded.</p><p><a href="evaluation.json">Complete checks, predeclared policy and source hashes</a></p><div class="scroll"><table><thead><tr><th>Case</th><th>Run</th><th>Seconds</th><th>Channels</th><th>FFTs</th><th>Events</th><th>Checks</th><th>Manual download</th></tr></thead><tbody>"""
    with Path(path).open("x", encoding="utf-8") as handle:
        handle.write(text + "".join(rows) + "</tbody></table></div><p>Commercial and perceptual approval: NO. Historical engines remain unchanged. No device, microphone, cloud or external publication was used.</p></main></html>")


def evaluate_long(output_directory):
    """Run the five fixed H3b integration cases once into a NEW directory."""
    target = Path(output_directory)
    if target.exists() or target.is_symlink():
        raise FileExistsError("evaluation directory must be new; previous results are never replaced")
    source, policy, preserved, historical_sha = _preflight()
    source_names = ("stream_bands.py", "long_processing.py", "long_evaluation.py", "h3b_policy.json")
    sources = {name: _sha(source / name) for name in source_names}
    started = perf_counter()
    target.mkdir(parents=True, exist_ok=False)
    cases = _run_declared_cases(target)
    if [case["case_id"] for case in cases] != list(CASE_IDS):
        raise ValueError("evaluation did not execute the complete fixed H3b case sequence")
    historical_unchanged = all(_sha(source / name) == digest for name, digest in preserved.items())
    current_unchanged = all(_sha(source / name) == digest for name, digest in sources.items())
    result = {
        "schema_version": 1, "stage": "H3b", "case_count": len(cases), "cases": cases,
        "policy": policy, "policy_sha256": sources["h3b_policy.json"],
        "source_sha256": sources, "preserved_source_sha256": preserved,
        "historical_source_count": len(preserved),
        "historical_evaluation_record": {"relative_file": "outputs/h3_20261021_v1/evaluation.json",
                                         "sha256": historical_sha},
        "historical_sources_unchanged": historical_unchanged,
        "evaluation_sources_unchanged_during_run": current_unchanged,
        "all_declared_contract_checks_met": historical_unchanged and current_unchanged and
                                            all(case["all_declared_checks_met"] for case in cases),
        "total_evaluation_seconds": perf_counter() - started,
        "fixture_scope": "owned_deterministic_mathematical_signals_not_natural_audio",
        "complete_long_audio_or_trace_loaded_for_evaluation": False,
        "perceptual_or_natural_noise_accuracy_verified": False,
        "whole_process_memory_limit_or_device_realtime_verified": False,
        "commercial_approval": False, "audio_played_or_recorded": False,
        "cloud_or_external_publication": False,
    }
    serialized = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    with (target / "evaluation.json").open("x", encoding="utf-8") as handle:
        handle.write(serialized + "\n")
    _report(result, target / "report.html")
    return result
