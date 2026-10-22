"""Predeclared native-precision WAV contracts, not listening/quality evidence."""

import hashlib
import html
import json
from pathlib import Path
import struct
from time import perf_counter

import numpy as np

from .band_control import BandControlConfig
from .long_evaluation import _trace_summary


FORMATS = {"PCM16": (1, 16, "signed_PCM16_little_endian"),
           "PCM24": (1, 24, "signed_PCM24_little_endian"),
           "Float32": (3, 32, "IEEE_FLOAT32_little_endian")}
CASE_IDS = tuple(f"active_{fmt}_{rate}_{channels}ch" for fmt in ("PCM24", "Float32")
                 for rate in (44100, 48000) for channels in (1, 2)) + (
    "quiet_sub16LSB_PCM24", "bypass_PCM24_endpoints", "bypass_Float32_signed_zero",
    "long65s_PCM24_partition", "maximum600s_Float32_bypass",
    "PCM16_44100_regression", "PCM16_48000_regression")


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        while block := source.read(65536):
            digest.update(block)
    return digest.hexdigest()


def _preflight():
    source = Path(__file__).parent
    old = json.loads((source.parents[1] / "outputs/h3c_20261021_v2/evaluation.json").read_text(encoding="utf-8"))
    frozen = {**old["preserved_source_sha256"], **old["source_sha256"]}
    if len(frozen) != 37 or any(Path(name).name != name or _sha(source / name) != digest for name, digest in frozen.items()):
        raise ValueError("The 37 historical DSP/policy files must remain unchanged")
    policy = json.loads((source / "h3d_policy.json").read_text(encoding="utf-8"))
    if policy.get("stage") != "H3d_same_format_precision_WAV":
        raise ValueError("Missing frozen H3d policy")
    return source, frozen, policy


def _pack_codes(codes, bits):
    if bits == 16:
        return codes.astype("<i2").tobytes()
    unsigned = (codes.reshape(-1).astype(np.int32) & 0xFFFFFF).astype(np.uint32)
    packed = np.column_stack((unsigned & 255, (unsigned >> 8) & 255, (unsigned >> 16) & 255))
    return packed.astype(np.uint8).tobytes()


def _write_owned(path, fmt, rate, seconds, channels, *, kind="sines"):
    """Independent struct writer, <=4096 generated frames, no product encoder."""
    if fmt not in FORMATS or rate not in (44100, 48000) or channels not in (1, 2) or not 0 < seconds <= 600:
        raise ValueError("Invalid fixed fixture specification")
    if (kind not in ("sines", "quiet", "float_bits", "integer_endpoints")
            or kind == "float_bits" and fmt != "Float32"
            or kind == "integer_endpoints" and fmt != "PCM24"
            or kind == "quiet" and channels != 1):
        raise ValueError("Invalid fixed fixture kind/format combination")
    frames = round(seconds * rate)
    if frames < 1:
        raise ValueError("Fixture must contain at least one frame")
    tag, bits, sample_format = FORMATS[fmt]
    alignment = channels * (bits // 8)
    data_bytes = frames * alignment
    format_payload = struct.pack("<HHIIHH", tag, channels, rate, rate * alignment, alignment, bits)
    if tag == 3:
        format_payload += b"\x00\x00"
    chunks = b"fmt " + struct.pack("<I", len(format_payload)) + format_payload
    if tag == 3:
        chunks += b"fact" + struct.pack("<II", 4, frames)
    header = b"RIFF" + struct.pack("<I", 4 + len(chunks) + 8 + data_bytes + data_bytes % 2) + b"WAVE" + chunks + b"data" + struct.pack("<I", data_bytes)
    if len(header) + data_bytes + data_bytes % 2 > 128 * 1024 * 1024:
        raise ValueError("Fixture exceeds the product size cap")
    with Path(path).open("xb") as output:
        output.write(header)
        for start in range(0, frames, 4096):
            indices = np.arange(start, min(frames, start + 4096), dtype=np.int64)
            t = indices.astype(np.float64) / rate
            if kind == "float_bits":
                pattern = np.array([0, 0x80000000, 0x3F000000, 0xBF000000, 0x3F7FFFFF, 0xBF800000, 1, 0x80000001], dtype="<u4")
                payload = np.repeat(pattern[indices % len(pattern), None], channels, axis=1).astype("<u4").tobytes()
            elif kind == "integer_endpoints":
                pattern = np.array([-8388608, -8388607, -256, -1, 0, 1, 255, 256, 8388606, 8388607], dtype=np.int32)
                payload = _pack_codes(np.repeat(pattern[indices % len(pattern), None], channels, axis=1), bits)
            else:
                hum = sum(level * np.sin(2 * np.pi * 50 * order * t + .37 * order) for order, level in ((1, .03), (2, .018), (3, .01)))
                left = (1e-6 * np.sin(2 * np.pi * 233 * t) if kind == "quiet" else hum + .04 * np.sin(2 * np.pi * 233 * t + .2))
                values = left[:, None] if channels == 1 else np.column_stack((left, -.8 * hum + .025 * np.sin(2 * np.pi * 311 * t + .71)))
                payload = values.astype("<f4").tobytes() if tag == 3 else _pack_codes(np.rint(values * (1 << (bits - 1))), bits)
            output.write(payload)
        if data_bytes % 2:
            output.write(b"\x00")
    return {"format": fmt, "sample_format": sample_format, "bits_per_sample": bits,
            "sample_rate_hz": rate, "channels": channels, "frames": frames,
            "duration_seconds": frames / rate, "kind": kind}


def _payload(path):
    """Independent bounded RIFF scan/hash, separate from product decoder."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        header = source.read(12)
        if header[:4] != b"RIFF" or header[8:] != b"WAVE":
            raise ValueError("Invalid reference RIFF header")
        fmt = None
        while chunk := source.read(8):
            tag, count = struct.unpack("<4sI", chunk)
            if tag == b"fmt ":
                fields = source.read(min(count, 40))
                if len(fields) < 16:
                    raise ValueError("Incomplete reference format")
                code, channels, rate, _, align, bits = struct.unpack_from("<HHIIHH", fields)
                fmt = {"sample_rate_hz": rate, "channels": channels, "bits_per_sample": bits, "format_tag": code, "alignment": align}
                source.seek(count - len(fields) + count % 2, 1)
            elif tag == b"data":
                if fmt is None:
                    raise ValueError("Reference data precedes format")
                remaining = count
                while remaining:
                    block = source.read(min(65536, remaining))
                    if not block:
                        raise ValueError("Incomplete reference data")
                    digest.update(block)
                    remaining -= len(block)
                return digest.hexdigest(), {**fmt, "frames": count // fmt["alignment"], "data_bytes": count}
            else:
                source.seek(count + count % 2, 1)
    raise ValueError("Missing reference data")


def _short_oracle(path, fixture):
    # Independent parser only on short files, never materializes long audio.
    from scipy.io import wavfile
    rate, decoded = wavfile.read(path)
    expected_dtype = {"PCM16": np.dtype("int16"), "PCM24": np.dtype("int32"), "Float32": np.dtype("float32")}[fixture["format"]]
    channels = 1 if decoded.ndim == 1 else decoded.shape[1]
    return bool(rate == fixture["sample_rate_hz"] and len(decoded) == fixture["frames"] and channels == fixture["channels"] and decoded.dtype == expected_dtype and np.isfinite(decoded).all())


def _run(folder, label, fixture, depths, block_size=4096):
    from .precision_processing import process_precision_wav
    config = BandControlConfig(band_depths_db=depths, block_size_frames=block_size)
    report = process_precision_wav(folder / "input.wav", folder / label, config)
    payload_hash, metadata = _payload(folder / label / "processed.wav")
    active = bool(config.orders)
    engine_frames = (fixture["frames"] * 48000 + fixture["sample_rate_hz"] - 1) // fixture["sample_rate_hz"]
    trace = _trace_summary(folder / label / "long_trace.jsonl", engine_frames, active)
    checks = {"native_rate_channels_frames_bits_exact": all(metadata[key] == fixture[key] for key in ("sample_rate_hz", "channels", "frames", "bits_per_sample")),
              "same_native_sample_format": report["output"]["sample_format"] == fixture["sample_format"],
              "source_hash_unchanged": report["input"]["sha256"] == _sha(folder / "input.wav"),
              "global_48k_trace_clock": trace["all_global_event_clocks_exact"],
              "FFT_count_matches": report["algorithm"]["shared_original_input_fft_count"] == (trace["event_count"] - 1 if active else 0),
              "no_whole_audio_or_trace_retained": report["resource_limits"]["whole_audio_retained_in_memory"] is False and report["resource_limits"]["whole_trace_retained_in_memory"] is False,
              "no_normalization_or_device_IO": report["output_gain_normalization_applied"] is False and report["device_io"] is False}
    if fixture["format"] != "PCM16":
        checks["no_PCM16_intermediate_or_dither"] = report["precision_io"]["PCM16_intermediate_used"] is False and report["precision_io"]["dither_applied"] is False
        checks["native_quantization_error_reported"] = bool(isinstance(report["native_quantization_max_abs_error"], (int, float)) and np.isfinite(report["native_quantization_max_abs_error"]))
    if fixture["duration_seconds"] <= 3.4:
        checks["independent_SciPy_short_reader"] = _short_oracle(folder / label / "processed.wav", fixture)
    if not active:
        checks["zero_depth_raw_payload_exact_including_sign_bits"] = payload_hash == _payload(folder / "input.wav")[0]
        checks["zero_depth_no_FFT_or_trace"] = report["algorithm"]["shared_original_input_fft_count"] == 0 and trace["event_count"] == 0
    elif fixture["kind"] == "quiet":
        checks["sub16LSB_payload_preserved_without16bit_intermediate"] = payload_hash == _payload(folder / "input.wav")[0] and report["output"]["peak_abs"] > 0
    else:
        checks["active_output_codes_changed"] = report["processor_applied"] is True
        checks["actual_engine_block_partition"] = report["resource_limits"]["max_engine_block_frames_observed"] == block_size
        if fixture["duration_seconds"] >= 60:
            checks["60s_clock_and_acquisition_not_reset"] = trace["checkpoint_seen"]["60_seconds"] and trace["acquisition_preserved_across_60_second_boundary"]
    return {"label": label, "relative_directory": f"{folder.name}/{label}", "metadata": metadata,
            "output_payload_sha256": payload_hash, "trace": trace, "checks": checks,
            "all_checks_met": all(checks.values()), "processing_seconds": report["timing"]["file_processing_seconds"],
            "quantization_max_abs_error": report.get("native_quantization_max_abs_error", report.get("pcm16_quantization_max_abs_error")),
            "resource_limits": report["resource_limits"]}


def _case(target, name, fmt, rate, seconds, channels, *, kind="sines", bypass=False, partition=False, legacy=False):
    folder = target / name
    folder.mkdir()
    fixture = _write_owned(folder / "input.wav", fmt, rate, seconds, channels, kind=kind)
    depths = (0, 0, 0) if bypass else (6, 3, 0)
    runs = [_run(folder, "blocks_4096", fixture, depths)]
    checks = {}
    if partition:
        runs.append(_run(folder, "blocks_480", fixture, depths, 480))
        checks["actual_4096_480_payload_exact"] = runs[0]["output_payload_sha256"] == runs[1]["output_payload_sha256"]
        checks["actual_4096_480_global_clock_exact"] = runs[0]["trace"]["event_clocks_sha256"] == runs[1]["trace"]["event_clocks_sha256"]
    if legacy:
        from .long_compatible_processing import process_long_compatible_wav
        process_long_compatible_wav(folder / "input.wav", folder / "frozen_H3c", BandControlConfig(block_size_frames=4096))
        checks["PCM16_frozen_H3c_payload_exact"] = runs[0]["output_payload_sha256"] == _payload(folder / "frozen_H3c/processed.wav")[0]
    return {"case_id": name, "fixture": fixture, "runs": runs, "checks": checks,
            "all_checks_met": all(checks.values()) and all(run["all_checks_met"] for run in runs)}


def _declared_cases(target):
    result = [_case(target, f"active_{fmt}_{rate}_{channels}ch", fmt, rate, 3.4, channels)
              for fmt in ("PCM24", "Float32") for rate in (44100, 48000) for channels in (1, 2)]
    result.extend((
        _case(target, CASE_IDS[8], "PCM24", 48000, 3.4, 1, kind="quiet"),
        _case(target, CASE_IDS[9], "PCM24", 48000, .11, 1, kind="integer_endpoints", bypass=True),
        _case(target, CASE_IDS[10], "Float32", 44100, .11, 2, kind="float_bits", bypass=True),
        _case(target, CASE_IDS[11], "PCM24", 44100, 65, 2, partition=True),
        _case(target, CASE_IDS[12], "Float32", 48000, 600, 1, kind="float_bits", bypass=True),
        _case(target, CASE_IDS[13], "PCM16", 44100, 3.4, 1, legacy=True),
        _case(target, CASE_IDS[14], "PCM16", 48000, 3.4, 2, legacy=True),
    ))
    return result


def evaluate_precision(output_directory):
    target = Path(output_directory)
    if target.exists() or target.is_symlink():
        raise FileExistsError("Use a new evaluation directory")
    source, frozen, policy = _preflight()
    current = {name: _sha(source / name) for name in ("precision_wav.py", "precision_processing.py", "precision_evaluation.py", "h3d_policy.json")}
    started = perf_counter()
    target.mkdir(parents=True, exist_ok=False)
    cases = _declared_cases(target)
    if [case["case_id"] for case in cases] != list(CASE_IDS):
        raise ValueError("Incomplete fixed precision case sequence")
    historical_unchanged = all(_sha(source / name) == digest for name, digest in frozen.items())
    current_unchanged = all(_sha(source / name) == digest for name, digest in current.items())
    result = {"schema_version": 1, "stage": "H3d", "case_count": len(cases), "cases": cases,
              "policy": policy, "source_sha256": current, "preserved_source_sha256": frozen,
              "historical_source_count": len(frozen), "historical_sources_unchanged": historical_unchanged,
              "current_sources_unchanged_during_run": current_unchanged,
              "all_declared_contract_checks_met": historical_unchanged and current_unchanged and all(case["all_checks_met"] for case in cases),
              "evaluation_seconds": perf_counter() - started,
              "fixture_scope": "owned_mathematical_sources_not_natural_noise",
              "independent_reader_scope": "SciPy_only_on_short_at_most3.4s_outputs_no_SciPy_in_runtime_DSP",
              "complete_long_audio_or_trace_loaded": False,
              "perceptual_quality_verified": False, "commercial_approval": False,
              "audio_played_or_recorded": False, "external_publication": False}
    with (target / "evaluation.json").open("x", encoding="utf-8") as output:
        output.write(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    rows = []
    for case in cases:
        for run in case["runs"]:
            directory = html.escape(run["relative_directory"], quote=True)
            values = (case["case_id"], case["fixture"]["sample_format"], case["fixture"]["duration_seconds"], run["label"], run["all_checks_met"])
            rows.append("<tr>" + "".join("<td>" + html.escape(str(value)) + "</td>" for value in values) + f'<td><a href="{directory}/report.json">JSON</a></td></tr>')
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>H3d native-precision WAV contracts</title><style>body{font:16px system-ui;background:#f5f7f1;color:#173b2b;margin:32px}main{max-width:1250px;margin:auto}p{line-height:1.6}table{border-collapse:collapse;background:white}td,th{padding:10px;border:1px solid #cbd5c5}.scroll{overflow:auto}</style><main><h1>H3d: same-format PCM24 and Float32 WAV</h1><p>Fifteen fixed owned-maths cases. Same native rate, channel count, frame count and sample format. Packed PCM24 and Float32 do not pass through PCM16. Zero-depth bypass copies validated raw payload bytes, including Float32 negative zero. Whole-container metadata is not preserved.</p><p>44.1 kHz active input uses the frozen streaming FIR and DSP path; 48 kHz does not resample. Float input must be finite and normalized to [-1,1). Output rounding is explicit; unrepresentable output is refused without clipping, gain normalization or dithering.</p><p>600 seconds and 128 MiB are simultaneous limits: high-bit-depth stereo may hit the byte cap sooner. These compatibility checks do not establish perceptual quality, natural-noise accuracy, device real-time performance or commercial readiness. No sound was played or recorded.</p><p><a href="evaluation.json">Complete policy, source hashes and checks</a></p><div class="scroll"><table><tr><th>Case</th><th>Native format</th><th>Seconds</th><th>Partition</th><th>Checks</th><th>Report</th></tr>'''
    with (target / "report.html").open("x", encoding="utf-8") as output:
        output.write(page + "".join(rows) + "</table></div></main></html>")
    return result
