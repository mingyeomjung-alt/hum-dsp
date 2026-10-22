"""Fixed owned-codec compatibility contracts, not a decoder or quality oracle."""

import hashlib
import html
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .band_control import BandControlConfig
from .precision_evaluation import _payload


CASE_IDS = tuple(f"{codec}_{rate}_{channels}ch" for codec in ("MP3", "M4A")
                 for rate in (44100, 48000) for channels in (1, 2))
CURRENT_SOURCES = ("compressed_decode.py", "_compressed_worker.py", "compressed_processing.py",
                   "compressed_evaluation.py", "h3e_policy.json")


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        while block := source.read(65536):
            digest.update(block)
    return digest.hexdigest()


def _require_av():
    try:
        import av
    except ImportError as exc:
        raise RuntimeError("H3e evaluation requires the optional PyAV compressed-audio dependency") from exc
    return av


def _preflight():
    _require_av()
    source = Path(__file__).parent
    prior_path = source.parents[1] / "outputs/h3d_20261021_v1/evaluation.json"
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    frozen = {**prior["preserved_source_sha256"], **prior["source_sha256"]}
    if len(frozen) != 41 or any(Path(name).name != name or _sha(source / name) != digest for name, digest in frozen.items()):
        raise ValueError("The 41 historical DSP/policy files must remain unchanged")
    policy = json.loads((source / "h3e_policy.json").read_text(encoding="utf-8"))
    if policy.get("stage") != "H3e_compressed_import_to_Float32_WAV":
        raise ValueError("Missing frozen H3e policy")
    return source, frozen, policy


def _write_owned_compressed(path, codec, rate, seconds, channels):
    """Encode bounded mathematical fixtures, never user recordings.

    Fixture encoder and runtime decoder share PyAV/FFmpeg. This is deliberately
    NOT an independent decoder comparison or a quality/codec fidelity metric.
    """
    if (codec not in ("MP3", "M4A") or rate not in (44100, 48000)
            or channels not in (1, 2) or not 0 < seconds <= 3.4):
        raise ValueError("Invalid fixed compressed fixture specification")
    target = Path(path)
    expected_extension = ".mp3" if codec == "MP3" else ".m4a"
    if target.suffix.lower() != expected_extension:
        raise ValueError("Fixture suffix must match its declared codec container")
    if target.exists() or target.is_symlink():
        raise FileExistsError("Fixture output must be new")
    frames = round(rate * seconds)
    if frames < 1:
        raise ValueError("Fixture must contain at least one frame")
    av = _require_av()
    layout = "mono" if channels == 1 else "stereo"
    encoder_name = "libmp3lame" if codec == "MP3" else "aac"
    # Exclusive file creation avoids clobbering an existing file through av.open.
    with target.open("xb") as handle, av.open(handle, "w", format="mp3" if codec == "MP3" else "mp4") as output:
        stream = output.add_stream(encoder_name, rate=rate)
        stream.layout = layout
        stream.bit_rate = 128000 if channels == 1 else 192000
        stream.codec_context.format = "fltp"
        for start in range(0, frames, 4096):
            indices = np.arange(start, min(frames, start + 4096), dtype=np.float64)
            t = indices / rate
            hum = sum(level * np.sin(2 * np.pi * 50 * order * t + .37 * order)
                      for order, level in ((1, .03), (2, .018), (3, .01)))
            left = hum + .04 * np.sin(2 * np.pi * 233 * t + .2)
            samples = left[None, :] if channels == 1 else np.vstack((left, -.8 * hum + .025 * np.sin(2 * np.pi * 311 * t + .71)))
            frame = av.AudioFrame.from_ndarray(samples.astype(np.float32), format="fltp", layout=layout)
            frame.sample_rate = rate
            frame.pts = start
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode(None):
            output.mux(packet)
    return {"codec": codec, "sample_rate_hz": rate, "channels": channels,
            "preencode_frames": frames, "preencode_duration_seconds": frames / rate,
            "fixture_encoder": "PyAV/FFmpeg:" + encoder_name,
            "encoder_and_runtime_decoder_share_backend": True,
            "preencode_frame_identity_after_lossy_codec_claimed": False}


def _run(folder, label, fixture, depths):
    from .compressed_processing import process_compressed_file
    from .precision_processing import process_precision_wav
    config = BandControlConfig(band_depths_db=depths, block_size_frames=4096)
    extension = ".mp3" if fixture["codec"] == "MP3" else ".m4a"
    original = folder / ("input" + extension)
    source_hash = _sha(original)
    destination = folder / label
    report = process_compressed_file(original, destination, config)
    decoded = destination / "decoded_input.wav"
    processed = destination / "processed.wav"
    decoded_hash, decoded_metadata = _payload(decoded)
    output_hash, output_metadata = _payload(processed)
    reference = folder / (label + "_frozen_precision")
    reference_report = process_precision_wav(decoded, reference, config)
    compressed = report["compressed_source"]
    active = bool(config.orders)
    checks = {
        "compressed_source_unchanged": source_hash == _sha(original) == compressed["source_sha256"],
        "source_bytes_exact": compressed["source_bytes"] == original.stat().st_size,
        "lossy_decode_disclosed": compressed["lossy_decode"] is True and compressed["original_sample_identity_guaranteed"] is False,
        "delay_padding_backend_scope_disclosed": compressed["decoder_delay_padding_handled_by_backend"] is True,
        "decoded_float32_without_PCM16": decoded_metadata["format_tag"] == 3 and decoded_metadata["bits_per_sample"] == 32,
        "decoded_native_rate_channels": all(decoded_metadata[key] == fixture[key] for key in ("sample_rate_hz", "channels")),
        "decoded_positive_bounded_frames": 0 < decoded_metadata["frames"] <= fixture["sample_rate_hz"] * 600,
        "decoded_hash_matches_report": _sha(decoded) == compressed["decoded"]["sha256"],
        "decoded_framecount_matches_report": decoded_metadata["frames"] == compressed["decoded"]["frames"],
        "output_rate_channels_frames_bits_match_decoded": all(output_metadata[key] == decoded_metadata[key] for key in ("sample_rate_hz", "channels", "frames", "bits_per_sample", "format_tag")),
        "frozen_precision_output_container_exact": _sha(processed) == _sha(reference / "processed.wav"),
        "frozen_precision_trace_exact": _sha(destination / "long_trace.jsonl") == _sha(reference / "long_trace.jsonl"),
        "same_precision_processing_decision": report["processor_applied"] == reference_report["processor_applied"],
        "no_PCM16_intermediate_or_dither": report["precision_io"]["PCM16_intermediate_used"] is False and report["precision_io"]["dither_applied"] is False,
        "no_gain_normalization_or_device_IO": report["output_gain_normalization_applied"] is False and report["device_io"] is False,
        "no_whole_audio_or_trace_retained": report["resource_limits"]["whole_audio_retained_in_memory"] is False and report["resource_limits"]["whole_trace_retained_in_memory"] is False,
        "conversion_history_present": bool(report.get("conversion_pipeline")),
    }
    if not active:
        checks["bypass_matches_decoded_payload_not_compressed_bytes"] = output_hash == decoded_hash
        checks["bypass_no_FFT"] = report["algorithm"]["shared_original_input_fft_count"] == 0
    return {"label": label, "relative_directory": f"{folder.name}/{label}",
            "decoded_metadata": decoded_metadata, "output_metadata": output_metadata,
            "compressed_source_sha256": source_hash, "decoded_payload_sha256": decoded_hash,
            "output_payload_sha256": output_hash, "checks": checks,
            "all_checks_met": all(checks.values())}


def _case(target, name, codec, rate, channels):
    folder = target / name
    folder.mkdir()
    fixture = _write_owned_compressed(folder / ("input.mp3" if codec == "MP3" else "input.m4a"), codec, rate, 3.4, channels)
    runs = [_run(folder, "active_6_3_0", fixture, (6, 3, 0)),
            _run(folder, "bypass_0_0_0", fixture, (0, 0, 0))]
    checks = {"separate_active_bypass_decode_payload_exact": runs[0]["decoded_payload_sha256"] == runs[1]["decoded_payload_sha256"]}
    return {"case_id": name, "fixture": fixture, "runs": runs, "checks": checks,
            "all_checks_met": all(checks.values()) and all(run["all_checks_met"] for run in runs)}


def _declared_cases(target):
    return [_case(target, f"{codec}_{rate}_{channels}ch", codec, rate, channels)
            for codec in ("MP3", "M4A") for rate in (44100, 48000) for channels in (1, 2)]


def evaluate_compressed(output_directory):
    target = Path(output_directory)
    if target.exists() or target.is_symlink():
        raise FileExistsError("Use a new compressed evaluation directory")
    source, frozen, policy = _preflight()
    current = {name: _sha(source / name) for name in CURRENT_SOURCES}
    started = perf_counter()
    target.mkdir(parents=True, exist_ok=False)
    cases = _declared_cases(target)
    if [case["case_id"] for case in cases] != list(CASE_IDS):
        raise ValueError("Incomplete fixed compressed case sequence")
    historical_unchanged = all(_sha(source / name) == digest for name, digest in frozen.items())
    current_unchanged = all(_sha(source / name) == digest for name, digest in current.items())
    result = {"schema_version": 1, "stage": "H3e", "case_count": len(cases), "cases": cases,
              "policy": policy, "source_sha256": current, "preserved_source_sha256": frozen,
              "historical_source_count": len(frozen), "historical_sources_unchanged": historical_unchanged,
              "current_sources_unchanged_during_run": current_unchanged,
              "all_declared_contract_checks_met": historical_unchanged and current_unchanged and all(case["all_checks_met"] for case in cases),
              "evaluation_seconds": perf_counter() - started,
              "fixture_scope": "owned_mathematical_sources_encoded_with_PyAV_FFmpeg",
              "decoder_independence_claimed": False,
              "codec_original_sample_identity_claimed": False,
              "perceptual_quality_verified": False, "commercial_approval": False,
              "audio_played_or_recorded": False, "external_publication": False}
    with (target / "evaluation.json").open("x", encoding="utf-8") as output:
        output.write(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    rows = []
    for case in cases:
        for run in case["runs"]:
            values = (case["case_id"], run["label"], run["decoded_metadata"]["frames"], run["all_checks_met"])
            link = html.escape(run["relative_directory"], quote=True)
            rows.append("<tr>" + "".join("<td>" + html.escape(str(value)) + "</td>" for value in values) + f'<td><a href="{link}/report.json">JSON</a></td></tr>')
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>H3e compressed import contracts</title><style>body{font:16px system-ui;background:#f5f7f1;color:#173b2b;margin:32px}main{max-width:1100px;margin:auto}p{line-height:1.6}table{border-collapse:collapse;background:white}td,th{padding:10px;border:1px solid #cbd5c5}.scroll{overflow:auto}</style><main><h1>H3e: MP3 and AAC/M4A import</h1><p>Eight fixed owned mathematical fixtures, each processed with active 6/3/0 dB controls and zero-depth bypass. Compressed input is decoded to normalized Float32 WAV, then passed through the frozen native-precision DSP. Output is WAV, not recompressed audio.</p><p>Frame preservation is measured against the decoded WAV, not the signal before lossy encoding. Zero-depth output preserves decoded payload bytes, not the compressed source bytes. The fixture encoder and runtime decoder share PyAV/FFmpeg: these tests are not an independent decoder comparison, a perceptual-quality metric or a natural-noise improvement claim.</p><p>No audio was played or recorded. No device, cloud service or commercial approval was enabled.</p><p><a href="evaluation.json">Fixed policy, source hashes and complete checks</a></p><div class="scroll"><table><tr><th>Case</th><th>Controls</th><th>Decoded frames</th><th>Checks</th><th>Report</th></tr>'''
    with (target / "report.html").open("x", encoding="utf-8") as output:
        output.write(page + "".join(rows) + "</table></div></main></html>")
    return result
