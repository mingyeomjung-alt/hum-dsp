"""H3a format/conversion/stereo contracts, not new denoising accuracy evidence."""

import hashlib
import html
import json
from pathlib import Path
import wave

import numpy as np

from .band_control import BandControlConfig
from .band_processing import process_band_wav
from .compatible_processing import process_compatible_wav, read_compatible_wav
from .fixtures import _dump_json


PIANO_SHA256 = "9661f81d37c59f230b324b830ab68c0482336af3ec117c92e73108ffb4095f15"


def _sha(path):
    digest = hashlib.sha256()
    total = 0
    with Path(path).open("rb") as stream:
        while chunk := stream.read(65536):
            total += len(chunk)
            if total > 32 * 1024 * 1024:
                raise ValueError("evaluation hash input exceeds 32 MiB")
            digest.update(chunk)
    return digest.hexdigest()


def _write(path, values, rate):
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] not in (1, 2) or not np.isfinite(array).all():
        raise ValueError("invalid owned evaluation audio")
    codes = np.rint(array * 32768)
    if np.any(codes < -32768) or np.any(codes > 32767):
        raise ValueError("owned audio cannot be encoded without saturation")
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(array.shape[1])
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(codes.astype("<i2").tobytes())


def _preflight():
    source = Path(__file__).parent
    project = source.parents[1]
    h1 = json.loads((project / "outputs/h1_20261021_v2/benchmark.json").read_text(encoding="utf-8"))
    h2 = json.loads((project / "outputs/h2_20261021_v2/evaluation.json").read_text(encoding="utf-8"))
    previous = {**h1["legacy_source_sha256"], **h1["source_sha256"], **h2["source_sha256"]}
    if any(Path(name).name != name or _sha(source / name) != value for name, value in previous.items()):
        raise ValueError("preserved H1/H2 sources changed; H3 comparison refused")
    return source, json.loads((source / "h3_policy.json").read_text(encoding="utf-8")), previous


def _cases():
    for rate in (44100, 48000):
        # Deliberately not divisible by the sample-rate ratio; retain final-frame
        # rounding behavior rather than choosing favorable exact-ratio lengths.
        t = np.arange(round(rate * 3.4) + 1, dtype=np.float64) / rate
        noise = sum(0.02 * np.sin(2 * np.pi * 50 * order * t + 0.37 * order) for order in (1, 2, 3))
        left = 0.06 * np.sin(2 * np.pi * 233 * t) + noise
        right = 0.04 * np.sin(2 * np.pi * 311 * t + 0.3) + 0.6 * noise
        for channels in (1, 2):
            values = left[:, None] if channels == 1 else np.column_stack((left, right))
            for label, depths in (("bypass", (0, 0, 0)), ("gentle", (6, 3, 0))):
                yield f"owned_{rate}_{channels}ch_{label}", rate, values, depths, None
        if rate == 44100:
            yield "owned_44100_identical", rate, np.column_stack((left, left)), (6, 3, 0), "identical"
            yield "owned_44100_antiphase", rate, np.column_stack((left, -left)), (6, 3, 0), "antiphase"
    yield "owned_44100_60s_bypass", 44100, np.zeros((44100 * 60, 2)), (0, 0, 0), None
    t = np.arange(48000 * 60, dtype=np.float64) / 48000.0
    long_source = 0.04 * np.sin(2 * np.pi * 233 * t) + 0.02 * np.sin(2 * np.pi * 50 * t + 0.37)
    yield "owned_48000_60s_gentle", 48000, long_source[:, None], (6, 3, 0), None


def _run_case(folder, config, expected_relation=None):
    source = folder / "input.wav"
    before = _sha(source)
    native, info = read_compatible_wav(source)
    report = process_compatible_wav(source, folder / "result", config)
    output, out_info = read_compatible_wav(folder / "result/processed.wav")
    trace = json.loads((folder / "result/compatible.json").read_text(encoding="utf-8"))
    band = trace["band_control_bundle"]
    acquired = any(record["qualified"] for event in band.get("analysis_trace", [])[1:] for record in event["bands"])
    checks = {
        "source_unchanged": _sha(source) == before,
        "native_frames_channels_rate_preserved": all(info[key] == out_info[key] for key in ("sample_rate_hz", "channels", "frames", "duration_seconds")) and output.shape == native.shape,
        "finite_valid_output": bool(np.isfinite(output).all() and np.all(output >= -1) and np.all(output < 1)),
        "zero_depth_native_PCM16_identity": bool(np.array_equal(output, native)) if not config.orders else None,
        "zero_depth_no_FFT_or_FIR": (report["algorithm"]["shared_original_input_fft_count"] == 0 and report["conversion"]["to_engine"]["conversion_applied"] is False) if not config.orders else None,
        "identical_channels_remain_identical": bool(np.array_equal(output[:, 0], output[:, 1])) if expected_relation == "identical" else None,
        "antiphase_zero_sum_preserved": bool(np.array_equal(output[:, 0], -output[:, 1])) if expected_relation == "antiphase" else None,
        "antiphase_hum_acquired_without_summing_channels": acquired if expected_relation == "antiphase" else None,
    }
    if info["sample_rate_hz"] == 48000 and info["channels"] == 1:
        legacy = folder / "preserved_H2"
        process_band_wav(source, legacy, config)
        checks["48k_mono_H2_output_byte_equivalence"] = (legacy / "processed.wav").read_bytes() == (folder / "result/processed.wav").read_bytes()
    return {
        "case_id": folder.name, "input_sha256": before,
        "output_sha256": _sha(folder / "result/processed.wav"),
        "input": info, "output": out_info, "config": config.to_dict(),
        "checks": checks, "all_declared_checks_met": all(value is not False for value in checks.values()),
        "file_processing_seconds": report["timing"]["file_processing_seconds"],
        "reference_channel_index": report["stereo_policy"]["reference_channel_index"],
        "shared_original_input_fft_count": report["algorithm"]["shared_original_input_fft_count"],
        "conversion": report["conversion"],
        "native_raw_output_minus_input_RMSE": float(np.sqrt(np.mean((output - native) ** 2))),
        "native_error_is_not_noise_reduction_or_perceptual_quality": True,
        "PCM16_quantization_max_abs_error": report["pcm16_quantization_max_abs_error"],
    }


def _report(result, path):
    rows = []
    for case in result["cases"]:
        name = html.escape(case["case_id"], quote=True)
        rows.append("<tr>" + "".join(f"<td>{html.escape(str(value))}</td>" for value in (
            case["case_id"], case["input"]["sample_rate_hz"], case["input"]["channels"], case["input"]["frames"],
            case["config"]["band_depths_db"], case["all_declared_checks_met"], case["native_raw_output_minus_input_RMSE"]))
            + f'<td><a href="{name}/result/processed.wav">WAV</a> · <a href="{name}/result/report.json">JSON</a></td></tr>')
    text = """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>H3a WAV compatibility checks</title>
<style>body{font:16px system-ui;background:#f5f7f1;color:#173b2b;margin:40px}main{max-width:1250px;margin:auto}p{line-height:1.6}table{border-collapse:collapse;background:white;width:100%;font-size:14px}td,th{padding:10px;border:1px solid #cbd5c5}a{color:#25654a}.scroll{overflow-x:auto}</style><main>
<h1>H3a: native WAV and linked-stereo compatibility</h1>
<p>Explicit mode: 44.1/48 kHz mono/stereo PCM16 WAV, at most 60 seconds and32 MiB. Native frame count, sample rate and channel order are retained. All-zero attenuation skips conversion and FFT and preserves PCM16 audio codes, not arbitrary WAV metadata chunks. Other formats and long-file streaming remain unsupported.</p>
<p>Nonzero44.1 kHz processing uses a fixed NumPy polyphase FIR to48 kHz and back. Conversion can change boundaries, near-Nyquist content and peaks; there is no gain fitting, normalization, clipping or post-hoc alignment. A first-maximum full-file RMS reference channel drives frozen filters shared across both channels. No L+R summing, downmix, crossfeed or separate-channel tracking. Quieter-channel-only hum can be missed; this is not a perceptual stereo guarantee or a streaming path.</p>
<p>These are format, frame, bypass and channel-relation checks, not new denoising accuracy or listening evidence. Output-minus-input RMSE is not noise reduction. Real-piano cases, when supplied, use the fixed publicly licensed source; injected hum is not naturally noisy audio.</p>
<p><a href="evaluation.json">Complete results, frozen policy and source hashes</a></p><div class="scroll"><table><thead><tr><th>Case</th><th>Hz</th><th>Channels</th><th>Frames</th><th>Depths dB</th><th>Checks</th><th>Raw RMS error</th><th>Manual download</th></tr></thead><tbody>"""
    path.write_text(text + "".join(rows) + "</tbody></table></div><p>Commercial and perceptual quality approval: NO. No sound was played or recorded. Historical engines unchanged.</p></main></html>", encoding="utf-8")


def run_compatible_evaluation(output_directory, piano_source=None):
    target = Path(output_directory)
    if target.exists():
        raise FileExistsError("evaluation directory must be new")
    source_root, policy, preserved = _preflight()
    original = None
    if piano_source is not None:
        original = Path(piano_source)
        if _sha(original) != PIANO_SHA256:
            raise ValueError("public piano must match the frozen source hash")
        audio, metadata = read_compatible_wav(original)
        if metadata["sample_rate_hz"] != 44100 or metadata["channels"] != 2:
            raise ValueError("public source format changed")
    target.mkdir(parents=True, exist_ok=False)
    cases = []
    for case_id, rate, values, depths, relation in _cases():
        folder = target / case_id
        folder.mkdir()
        _write(folder / "input.wav", values, rate)
        cases.append(_run_case(folder, BandControlConfig(band_depths_db=depths), relation))
    piano = {"supplied": original is not None, "license": "CC0-1.0", "creator": "triangelx",
             "license_page": "https://freesound.org/people/triangelx/sounds/189175/",
             "source_url": "https://upload.wikimedia.org/wikipedia/commons/7/70/Emotional_piano.wav"}
    if original is not None:
        for label, add_noise in (("real_piano_native", False), ("real_piano_injected_50", True)):
            folder = target / label
            folder.mkdir()
            values = audio.copy()
            if add_noise:
                t = np.arange(values.shape[0]) / 44100.0
                noise = sum(np.sqrt(2) * 0.012 * np.sin(2 * np.pi * 50 * n * t + 0.37 * n) for n in (1, 2, 3))
                values += noise[:, None]
            _write(folder / "input.wav", values, 44100)
            cases.append(_run_case(folder, BandControlConfig()))
        piano["original_sha256"] = PIANO_SHA256
        piano["original_unchanged"] = _sha(original) == PIANO_SHA256
    historical_preserved = all(_sha(source_root / name) == digest for name, digest in preserved.items())
    result = {
        "schema_version": 1, "stage": "H3a", "scope": "bounded_WAV_compatibility_and_linked_channel_contracts_not_noise_quality",
        "policy": policy, "case_count": len(cases), "cases": cases, "public_piano": piano,
        "preserved_source_sha256": preserved,
        "historical_sources_preserved": historical_preserved,
        "source_sha256": {name: _sha(source_root / name) for name in ("compat_resampling.py", "compatible_processing.py", "compatible_evaluation.py", "h3_policy.json")},
        "all_declared_contract_checks_met": historical_preserved and all(case["all_declared_checks_met"] for case in cases) and piano.get("original_unchanged", True),
        "audio_played_or_recorded": False, "device_io": False, "perceptual_or_natural_noise_quality_verified": False,
        "commercial_release_approved": False, "remaining_product_workstreams": 3,
        "remaining_H3": "longer_recordings_streaming_and_other_formats_not_implemented",
    }
    _dump_json(target / "evaluation.json", result)
    _report(result, target / "report.html")
    return result
