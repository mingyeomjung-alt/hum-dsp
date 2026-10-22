"""Explicit bounded WAV compatibility wrapper around the unchanged H2 engine.

Only 44.1/48 kHz mono/stereo PCM16 WAV is supported. The new wrapper does not
broaden any historical reader or engine contract. Stereo uses one fixed analysis
channel and exactly the same frozen filters in both channels; there is no downmix,
crossfeed, per-channel drift, normalization, clipping, playback or device access.
"""

from dataclasses import asdict
import hashlib
import io
import json
from pathlib import Path
import struct
from time import perf_counter
import wave

import numpy as np

from .audio import MAX_FILE_BYTES
from .band_control import BandControlConfig, render_band_control, render_band_control_replay
from .compat_resampling import convert_sample_rate
from .fixtures import _dump_json
from .processing import _input_sha256, _levels


SUPPORTED_SAMPLE_RATES_HZ = (44100, 48000)
SUPPORTED_CHANNELS = (1, 2)
ALGORITHM = "bounded_native_wav_linked_stereo_band_control_v1"


def _validate_native(samples, sample_rate_hz):
    if type(sample_rate_hz) is not int or sample_rate_hz not in SUPPORTED_SAMPLE_RATES_HZ:
        raise ValueError("compatible WAV supports 44100 or 48000 Hz only")
    array = np.asarray(samples)
    if (array.ndim != 2 or array.shape[1] not in SUPPORTED_CHANNELS
            or not 1 <= array.shape[0] <= 60 * sample_rate_hz):
        raise ValueError("compatible audio must be 1..60 seconds of mono/stereo frames")
    if array.dtype.kind not in "fiu":
        raise ValueError("compatible audio must contain real numeric samples")
    result = np.array(array, dtype=np.float64, copy=True)
    if not np.isfinite(result).all():
        raise ValueError("compatible audio must contain finite samples")
    if np.any(result < -1.0) or np.any(result >= 1.0):
        raise ValueError("compatible audio must be in [-1, 1); no clipping or normalization")
    return result


def _validate_container(raw, frames, channels, rate):
    """Reject ambiguous or partial-frame RIFF data that ``wave`` may round down."""
    if len(raw) < 12 or raw[:4] != b"RIFF" or raw[8:12] != b"WAVE":
        raise ValueError("invalid RIFF WAV container")
    riff_end = struct.unpack_from("<I", raw, 4)[0] + 8
    if riff_end != len(raw):
        raise ValueError("truncated or trailing RIFF WAV container bytes")
    position = 12
    fmt_seen = data_seen = False
    layout_metadata = None
    while position < riff_end:
        if position + 8 > riff_end:
            raise ValueError("truncated WAV chunk header")
        tag, size = struct.unpack_from("<4sI", raw, position)
        start = position + 8
        end = start + size
        padded_end = end + size % 2
        if padded_end > riff_end:
            raise ValueError("truncated WAV chunk payload")
        if tag == b"fmt ":
            if fmt_seen or size < 16:
                raise ValueError("ambiguous or incomplete WAV format chunk")
            fmt_seen = True
            format_tag, native_channels, native_rate, byte_rate, alignment, bits = struct.unpack_from("<HHIIHH", raw, start)
            if (native_channels, native_rate, alignment, byte_rate, bits) != (channels, rate, channels * 2, rate * channels * 2, 16):
                raise ValueError("inconsistent PCM16 WAV format alignment or byte rate")
            if format_tag == 0xFFFE:
                pcm_guid = b"\x01\x00\x00\x00\x00\x00\x10\x00\x80\x00\x00\xaa\x00\x38\x9b\x71"
                if size < 40 or struct.unpack_from("<H", raw, start + 16)[0] < 22 or struct.unpack_from("<H", raw, start + 18)[0] != 16 or raw[start + 24:start + 40] != pcm_guid:
                    raise ValueError("unsupported extensible WAV format")
                channel_mask = struct.unpack_from("<I", raw, start + 20)[0]
                accepted_masks = (0, 0x4) if channels == 1 else (0, 0x3)
                if channel_mask not in accepted_masks:
                    raise ValueError("only canonical mono/stereo extensible WAV channel masks are supported")
            elif format_tag != 1:
                raise ValueError("uncompressed PCM16 WAV format required")
            else:
                channel_mask = None
            layout_metadata = {"wav_format_tag": format_tag, "channel_mask": channel_mask,
                               "channel_order": ["front_center"] if channels == 1 else ["front_left", "front_right"],
                               "canonical_layout_assumption": "unspecified_masks_and_classic_PCM_use_canonical_mono_or_left_right_order"}
        elif tag == b"data":
            if data_seen or not fmt_seen:
                raise ValueError("ambiguous WAV data chunk")
            data_seen = True
            if size % (channels * 2) or size != frames * channels * 2:
                raise ValueError("WAV payload contains an incomplete PCM16 frame")
        position = padded_end
    if not fmt_seen or not data_seen:
        raise ValueError("WAV format and data chunks are required")
    return layout_metadata


def read_compatible_wav(path):
    """Read bounded PCM16 WAV as float64 ``frames x channels`` plus metadata."""
    source = Path(path)
    if source.stat().st_size > MAX_FILE_BYTES:
        raise ValueError("WAV file exceeds the 32 MiB container limit")
    with source.open("rb") as handle:
        raw_bytes = handle.read(MAX_FILE_BYTES + 1)
    if len(raw_bytes) > MAX_FILE_BYTES:
        raise ValueError("WAV file exceeds the 32 MiB container limit")
    try:
        with io.BytesIO(raw_bytes) as raw:
            with wave.open(raw, "rb") as wav:
                channels, rate = wav.getnchannels(), wav.getframerate()
                if (channels not in SUPPORTED_CHANNELS
                        or rate not in SUPPORTED_SAMPLE_RATES_HZ
                        or wav.getsampwidth() != 2 or wav.getcomptype() != "NONE"):
                    raise ValueError("only mono/stereo 44100/48000 Hz uncompressed PCM16 WAV is supported")
                frames = wav.getnframes()
                if not 1 <= frames <= 60 * rate:
                    raise ValueError("WAV duration must be greater than 0 and at most 60 seconds")
                layout = _validate_container(raw_bytes, frames, channels, rate)
                payload = wav.readframes(frames)
                if len(payload) != frames * channels * 2:
                    raise ValueError("truncated WAV payload")
    except (wave.Error, EOFError) as exc:
        raise ValueError(f"invalid compatible PCM16 WAV: {exc}") from exc
    samples = np.frombuffer(payload, dtype="<i2").reshape(frames, channels)
    audio = _validate_native(samples.astype(np.float64) / 32768.0, rate)
    return audio, {"sample_rate_hz": rate, "channels": channels, "frames": frames,
                   "sample_format": "signed_PCM16_little_endian", "duration_seconds": frames / rate,
                   **layout}


def _encoded_wav(samples, sample_rate_hz):
    """Validate and quantize *before* any output path can be created."""
    audio = _validate_native(samples, sample_rate_hz)
    codes = np.rint(audio * 32768.0)
    # A float just below +1 may round to 32768. Reject it explicitly instead of
    # silently applying endpoint saturation, a limiter, or a gain adjustment.
    if np.any(codes < -32768) or np.any(codes > 32767):
        raise ValueError("output cannot be represented as PCM16 without clipping")
    payload = codes.astype("<i2").tobytes()
    stream = io.BytesIO()
    with wave.open(stream, "wb") as wav:
        wav.setnchannels(audio.shape[1])
        wav.setsampwidth(2)
        wav.setframerate(sample_rate_hz)
        wav.writeframes(payload)
    quantized = codes / 32768.0
    return stream.getvalue(), quantized


def _channel_levels(samples):
    return [{"channel_index": index, **_levels(samples[:, index])}
            for index in range(samples.shape[1])]


def _bypass_conversion(rate, frames):
    return {"conversion_applied": False, "reason": "all_band_depths_zero_native_PCM16_identity",
            "source_rate_hz": rate, "target_rate_hz": rate,
            "source_frames": frames, "target_frames": frames}


def process_compatible_wav(input_path, output_directory, config=None):
    """Process a supported WAV, preserving native rate, count and channel order.

    Nonzero depths resample to the unchanged 48 kHz engine domain when needed.
    The first maximum full-file converted-channel RMS chooses one reference for
    all channel decisions, without summing channels. Its processed output and
    frozen trace are then reused for linked channel processing. This can miss
    weak noise occurring only in a quieter other channel; it is not a perceptual
    stereo guarantee. All-zero depths bypass analysis and resampling entirely.
    """
    selected = BandControlConfig() if config is None else config
    if type(selected) is not BandControlConfig:
        raise ValueError("config must be a validated BandControlConfig")
    selected = BandControlConfig(**asdict(selected))
    source, target = Path(input_path), Path(output_directory)
    if target.exists():
        raise FileExistsError("output directory must be new; previous files are never replaced")
    original_sha = _input_sha256(source)
    native, metadata = read_compatible_wav(source)
    if _input_sha256(source) != original_sha:
        raise ValueError("input changed while being read")
    frames, channels = native.shape
    rate = metadata["sample_rate_hz"]
    started = perf_counter()
    if not selected.orders:
        floating = native.copy()
        forward = _bypass_conversion(rate, frames)
        backward = _bypass_conversion(rate, frames)
        reference_index = None
        analysis_rms = []
        bundle = {}
        algorithm = {"algorithm": ALGORITHM, "processing_bypassed": True,
                     "shared_original_input_fft_count": 0, "analysis_trace_event_count": 0,
                     "analysis": None, "config": selected.to_dict(),
                     "native_PCM16_code_identity": True}
    else:
        engine_input, forward = convert_sample_rate(native, rate, 48000)
        engine_input = _validate_native(engine_input, 48000)
        if engine_input.shape[1] != channels:
            raise ValueError("sample-rate conversion changed channel count")
        analysis_rms = [float(np.sqrt(np.mean(engine_input[:, index] ** 2)))
                        for index in range(channels)]
        reference_index = int(np.argmax(analysis_rms))
        processed_reference, description, bundle = render_band_control(
            engine_input[:, reference_index], selected)
        processed = np.empty_like(engine_input)
        for index in range(channels):
            processed[:, index] = (processed_reference if index == reference_index else
                                   render_band_control_replay(engine_input[:, index], selected, bundle))
        floating, backward = convert_sample_rate(processed, 48000, rate, target_frames=frames)
        algorithm = {"algorithm": ALGORITHM, "processing_bypassed": False,
                     "shared_original_input_fft_count": description["shared_original_input_fft_count"],
                     "analysis_trace_event_count": description["analysis_trace_event_count"],
                     "native_PCM16_code_identity": False,
                     "config": selected.to_dict(), "analysis": description}
    floating = _validate_native(floating, rate)
    if floating.shape != native.shape:
        raise ValueError("compatible processing changed native frame or channel count")
    encoded, decoded = _encoded_wav(floating, rate)
    elapsed = perf_counter() - started
    if _input_sha256(source) != original_sha:
        raise ValueError("input changed while processing")
    conversions = {"to_engine": forward, "to_native": backward,
                   "engine_sample_rate_hz": 48000,
                   "native_sample_rate_hz": rate,
                   "native_frame_count_preserved": True}
    stereo_policy = {
        "mode": "linked_frozen_filters_no_downmix_no_crossfeed",
        "reference_channel_index": reference_index,
        "reference_selection": "first_maximum_full_file_engine_domain_raw_RMS",
        "reference_selection_uses_entire_file": bool(selected.orders),
        "engine_channel_rms": analysis_rms,
        "same_trace_and_coefficients_all_channels": True,
        "independent_channel_frequency_tracking": False,
        "perceptual_stereo_preservation_verified": False,
        "limitation": "A quieter other channel's weak hum may be missed; opposite-polarity channels are never summed for detection.",
    }
    trace = {"schema_version": 1, "algorithm": ALGORITHM,
             "config": selected.to_dict(), "conversion": conversions,
             "stereo_policy": stereo_policy, "band_control_bundle": bundle}
    report = {
        "schema_version": 1, "stage": "H3_bounded_WAV_compatibility",
        "engine": ALGORITHM, "config": selected.to_dict(), "algorithm": algorithm,
        "processor_applied": bool(np.any(decoded != native)),
        "processing_bypassed": not selected.orders,
        "band_trace": {"file_name": "compatible.json"},
        "conversion": conversions, "stereo_policy": stereo_policy,
        "device_io": False, "output_gain_normalization_applied": False,
        "posthoc_alignment_applied": False, "reference_quality_metrics": None,
        "quality_notice": "Format compatibility and linked filters are not proof of improved sound quality. Wanted same-frequency content can still be attenuated.",
        "input": {"file_name": source.name, "sha256": original_sha, **metadata,
                  **_levels(native), "channel_levels": _channel_levels(native)},
        "output": {"file_name": "processed.wav", "sha256": hashlib.sha256(encoded).hexdigest(),
                   **metadata, "wav_format_tag": 1, "channel_mask": None,
                   **_levels(decoded), "channel_levels": _channel_levels(decoded)},
        "floating_output": {**_levels(floating), "channel_levels": _channel_levels(floating)},
        "pcm16_quantization_max_abs_error": float(np.max(np.abs(decoded - floating))),
        "pcm16_rounding": "nearest_ties_to_even_reject_unrepresentable_endpoint_no_saturation",
        "timing": {"file_processing_seconds": elapsed, "duration_seconds": frames / rate,
                   "includes_device_driver_or_playback": False,
                   "real_time_device_performance_verified": False},
        "limitations": [
            "Explicit compatibility mode only: 44100/48000 Hz, mono/stereo, uncompressed PCM16 WAV, at most 60 seconds and 32 MiB.",
            "No MP3/AAC/float/24-bit decode and no long-file streaming implementation.",
            "Only audio samples, native rate and channel order are retained; non-audio WAV chunks are not copied to the new output.",
            "Only canonical mono/front-center or stereo/front-left/front-right layouts are accepted. Unspecified masks assume canonical order; output is classic PCM without an extensible channel-mask chunk.",
            "Nonzero 44100 Hz processing includes fixed FIR conversion to 48000 Hz and back; boundary/interpolation differences remain.",
            "A full-file RMS choice fixes one analysis channel for the entire offline file; quieter-channel-only hum can be missed.",
            "Identical frozen linear filters avoid independent channel wander but are not a perceptual phase/image preservation guarantee.",
            "Wanted audio at the selected frequencies is not distinguishable from hum. No automatic quality approval is provided.",
            "No normalization, clipping, limiter, playback, microphone capture, spatial cancellation, cloud or network use.",
        ],
    }
    # All signal, encoding and metadata validation completes before any output.
    json.dumps((report, trace), allow_nan=False)
    target.mkdir(parents=True, exist_ok=False)
    (target / "processed.wav").write_bytes(encoded)
    _dump_json(target / "compatible.json", trace)
    _dump_json(target / "report.json", report)
    return report
