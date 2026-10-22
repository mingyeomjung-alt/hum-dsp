"""Explicit lossy-file import followed by the frozen native WAV processor.

The A/B input is the decoded Float32 WAV, not an encoded-file identity claim.
No codec re-encoding, device I/O, automatic gain, clipping or cloud operation.
"""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import tempfile
from time import perf_counter

from .band_control import BandControlConfig


MAX_SOURCE_BYTES = 32 * 1024 * 1024


def source_digest(path):
    from .compressed_decode import _safe_path
    # Reject remote paths and every reparse ancestor before any source read.
    source = _safe_path(path)
    before = source.stat()
    if not 0 < before.st_size <= MAX_SOURCE_BYTES:
        raise ValueError("Compressed input must be nonempty and at most 32 MiB")
    digest, total = hashlib.sha256(), 0
    with source.open("rb") as stream:
        while block := stream.read(32768):
            total += len(block)
            if total > MAX_SOURCE_BYTES:
                raise ValueError("Compressed source grew beyond 32 MiB")
            digest.update(block)
    if total != before.st_size or source.stat().st_size != total:
        raise ValueError("Compressed source changed while hashing")
    return digest.hexdigest()


def process_compressed_file(input_path, output_directory, config=None, *, source_format=None):
    """Create decoded input, processed WAV, trace and disclosure in a new folder."""
    from .compressed_decode import decode_compressed, _safe_path
    from .precision_processing import process_precision_wav

    selected = BandControlConfig(block_size_frames=4096) if config is None else config
    if type(selected) is not BandControlConfig:
        raise ValueError("config must be a validated BandControlConfig")
    selected = BandControlConfig(**asdict(selected))
    source, target = Path(input_path), Path(output_directory)
    if target.exists() or target.is_symlink():
        raise FileExistsError("Output directory must be new; never replace prior results")
    source = _safe_path(source)
    target = _safe_path(target, output=True)
    original_sha = source_digest(source)
    started = perf_counter()
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".hum_compressed_", dir=target.parent) as temporary:
        stage = Path(temporary)
        delivery = stage / "delivery"
        delivery.mkdir()
        decoded = delivery / "decoded_input.wav"
        imported = decode_compressed(source, decoded, source_format=source_format)
        if imported["source_sha256"] != original_sha:
            raise ValueError("Compressed source changed before decoding")
        report = process_precision_wav(decoded, stage / "dsp", selected)
        if report["input"]["sha256"] != imported["decoded"]["sha256"]:
            raise ValueError("Decoded input changed before DSP")
        report = dict(report)
        report["stage"] = "H3e_explicit_compressed_import_to_Float32_WAV"
        report["compressed_source"] = imported
        report["conversion_pipeline"] = [
            {"step": "lossy_decode_to_Float32_WAV", "source_format": imported["source_format"],
             "codec": imported["source_codec"], "lossy_input": True,
             "decoder_delay_padding_handled_by_backend": True,
             "original_pre_encoding_frame_count_guaranteed": False,
             "sample_rate_conversion_applied": False, "downmix_applied": False,
             "PCM16_intermediate_used": False},
            {"step": "frozen_precision_WAV_processor", "engine": report["engine"],
             "conversion": report["conversion"], "processing_bypassed": report["processing_bypassed"]},
        ]
        report["compressed_import"] = {
            "comparison_A": "decoded_Float32_WAV_not_original_compressed_bytes",
            "output_format": "IEEE_FLOAT32_WAV_no_lossy_reencoding",
            "all_zero_depth_meaning": "identity_of_decoded_WAV_payload_not_compressed_or_pre_encoding_audio",
            "metadata_preserved": False, "decoder_process_is_OS_security_sandbox": False,
            "whole_compressed_audio_loaded_by_wrapper": False,
            "compressed_source_byte_limit": MAX_SOURCE_BYTES,
            "decoded_container_byte_limit": 128 * 1024 * 1024,
            "decoded_duration_limit_seconds": 600,
            "decode_deadline_seconds": 120,
            "decoded_Float32_WAV_written_to_disk": True,
        }
        # The frozen WAV processor writes no float *intermediate*. This adapter
        # does write a decoded Float32 input; do not inherit a broader false claim.
        report["resource_limits"] = dict(report["resource_limits"])
        report["resource_limits"].update({
            "temporary_float_audio_written": True,
            "resampler_float_intermediate_written": False,
            "decoder_process_memory_RSS_bound_claimed": False,
            "scope": "DSP_buffer_bounds_plus_explicit_decoded_Float32_WAV_not_native_decoder_RSS",
        })
        report["long_file"] = dict(report["long_file"])
        report["long_file"].update({"temporary_float_audio_written": True,
                                    "decoded_Float32_input_written": True,
                                    "resampler_float_intermediate_written": False})
        report["quality_notice"] = (
            "MP3/AAC decoding is a lossy-source import, not restoration. A/B compares the decoded WAV "
            "with DSP output. Zero depths preserve decoded samples, not original encoded bytes or "
            "pre-encoding samples. Codec delay/padding follows the installed backend. "
            "Sound quality, natural-hum removal and all malformed-file detection are not established."
        )
        report["inherited_precision_engine_limits"] = {
            "scope": "frozen_WAV_processor_only_compressed_import_happens_before_it",
            "limitations": list(report["limitations"]),
        }
        report["limitations"] = [item for item in report["limitations"]
                                 if not item.startswith("MP3/AAC,")] + [
            "Compressed import is a separate adapter: only MP3 and single-stream AAC-in-M4A at "
            "44100/48000 Hz canonical mono/stereo; no ALAC, artwork/multistream or automatic downmix.",
            "Decoded Float32 must be finite in [-1,1). Codec overshoot is refused, not clipped or normalized.",
            "Compressed-to-WAV frame count is backend-decoded count; pre-encoding length and gapless "
            "identity are not guaranteed. Some truncated inputs accepted by the decoder may remain undetected.",
            "PyAV is an optional codec adapter, not our DSP algorithm; its native codec build and distribution "
            "rights require separate review before a bundled or commercial release.",
        ]
        report["timing"] = dict(report["timing"])
        report["timing"]["file_processing_seconds"] = perf_counter() - started
        report["timing"]["includes_compressed_decoding"] = True
        for name in ("processed.wav", "long_trace.jsonl"):
            (stage / "dsp" / name).rename(delivery / name)
        with (delivery / "report.json").open("x", encoding="utf-8") as output:
            output.write(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        if source_digest(source) != original_sha:
            raise ValueError("Compressed source changed during processing")
        if target.exists() or target.is_symlink():
            raise FileExistsError("Output appeared during processing; never replace it")
        delivery.rename(target)
    return report
