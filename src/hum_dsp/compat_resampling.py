"""Bounded offline sample-rate compatibility, with no gain fitting or SciPy runtime.

The fixed design is a symmetric Kaiser (beta=5) windowed sinc of
``64 * max(up, down) + 1`` taps.  Its cutoff is the smaller Nyquist limit,
and its DC gain after conceptual zero insertion is ``up``.  These standard
FIR design choices are declared here, before compatibility measurements.

There is no zero-inserted audio buffer.  At output frame n the compensated
convolution uses h[n*down + half_length - j*up].  This both removes the
linear-phase FIR delay and makes the mapping identical for every channel.
Outside the original file, input samples are explicitly zero.  Boundary
transients remain real output; this module does not conceal them by fitting,
normalizing, clipping, crossfading, or shifting the result.
"""

from __future__ import annotations

import hashlib
import math
from numbers import Integral
from typing import Any

import numpy as np


SUPPORTED_RATES_HZ = (44100, 48000)
MAX_DURATION_SECONDS = 60
OUTPUT_CHUNK_FRAMES = 1024
KAISER_BETA = 5.0
HALF_LENGTH_MULTIPLIER = 32
# Array payload bound for per-chunk indexing, coefficients, samples, and output.
# This excludes validated contiguous input storage, returned output,
# Python/NumPy allocator overhead, and the small fixed coefficient table.
# It is not a process RSS guarantee.
TEMPORARY_ARRAY_PAYLOAD_UPPER_BYTES = 4 * 1024 * 1024


def _integer(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _rate(value: object, name: str) -> int:
    result = _integer(value, name)
    if result not in SUPPORTED_RATES_HZ:
        raise ValueError(f"{name} must be 44100 or 48000 Hz")
    return result


def _validated_samples(samples: object, source_rate_hz: int) -> tuple[np.ndarray, bool]:
    try:
        original = np.asarray(samples)
    except (ValueError, TypeError, OverflowError) as error:
        raise ValueError("samples must be a rectangular real numeric array") from error
    if original.ndim == 1:
        was_mono_vector = True
    elif original.ndim == 2 and original.shape[1] in (1, 2):
        was_mono_vector = False
    else:
        raise ValueError("samples must have shape (frames,), (frames, 1), or (frames, 2)")
    if original.shape[0] < 1:
        raise ValueError("samples must not be empty")
    if original.shape[0] > source_rate_hz * MAX_DURATION_SECONDS:
        raise ValueError("sample-rate conversion is limited to 60 seconds")
    if original.dtype.kind not in "fiu":
        raise ValueError("samples must contain real numeric values, not booleans or objects")
    # Validate in bounded blocks before float conversion, including wider floats.
    for start in range(0, original.shape[0], OUTPUT_CHUNK_FRAMES):
        block = original[start:start + OUTPUT_CHUNK_FRAMES]
        if not np.isfinite(block).all() or np.any(block < -1) or np.any(block >= 1):
            raise ValueError("samples must be finite normalized audio in [-1, 1)")
    # Convert a strided caller view at most once.  Otherwise np.take may make
    # an implicit full-input contiguous copy on every output chunk.
    converted = np.ascontiguousarray(original, dtype=np.float64)
    return converted.reshape((-1, 1)) if was_mono_vector else converted, was_mono_vector


def _kernel(up: int, down: int) -> tuple[np.ndarray, int]:
    """Generate a standard fixed low-pass directly from its analytic expression."""
    maximum = max(up, down)
    half_length = HALF_LENGTH_MULTIPLIER * maximum
    positions = np.arange(-half_length, half_length + 1, dtype=np.float64)
    cutoff = 1.0 / maximum
    coefficients = cutoff * np.sinc(cutoff * positions)
    coefficients *= np.kaiser(2 * half_length + 1, KAISER_BETA)
    coefficients /= np.sum(coefficients, dtype=np.float64)
    coefficients *= up
    return coefficients, half_length


def convert_sample_rate(
    samples: object,
    source_rate_hz: int,
    target_rate_hz: int,
    *,
    target_frames: int | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Convert 44.1/48 kHz mono/stereo offline using a fixed rational FIR.

    The default length is exactly ceil(input_frames * up / down).  A caller
    restoring the product's 44.1 ->48 ->44.1 kHz round trip may request one
    fewer frame.  The inverse round trip can need two frames trimmed and is
    intentionally NOT supported by this one-frame override.  No other crop/pad
    is accepted.  Equal-rate conversion returns an exact float64 copy and requires
    the unchanged length.  The raw converted result may exceed [-1, 1); the
    file-processing layer must refuse overshoot before quantization.
    """
    source_rate = _rate(source_rate_hz, "source_rate_hz")
    target_rate = _rate(target_rate_hz, "target_rate_hz")
    source, was_mono_vector = _validated_samples(samples, source_rate)
    input_frames, channels = source.shape
    divisor = math.gcd(source_rate, target_rate)
    up, down = target_rate // divisor, source_rate // divisor
    expected_frames = (input_frames * up + down - 1) // down
    if target_frames is None:
        output_frames = expected_frames
    else:
        output_frames = _integer(target_frames, "target_frames")
        allowed = (expected_frames,) if up == down else (expected_frames, expected_frames - 1)
        if output_frames < 1 or output_frames not in allowed:
            raise ValueError("target_frames must equal the default length or, for conversion, one fewer")
    # A duration-limit input cannot create a longer-than-limit default output.
    if output_frames > target_rate * MAX_DURATION_SECONDS:
        raise ValueError("sample-rate conversion output exceeds 60 seconds")

    metadata: dict[str, Any] = {
        "algorithm": "bounded_kaiser_windowed_sinc_polyphase_v1",
        "source_rate_hz": source_rate,
        "target_rate_hz": target_rate,
        "up": up,
        "down": down,
        "input_frames": input_frames,
        "output_frames": output_frames,
        "default_output_frames": expected_frames,
        "explicit_target_frames": target_frames is not None,
        "discarded_final_frames": expected_frames - output_frames,
        "explicit_frame_override_maximum_trim": 1 if up != down else 0,
        "supported_product_round_trip": "44100_to_48000_to_44100_with_original_frame_count",
        "inverse_round_trip_two_frame_trim_supported": False,
        "channels": channels,
        "maximum_duration_seconds": MAX_DURATION_SECONDS,
        "maximum_output_chunk_frames": OUTPUT_CHUNK_FRAMES,
        "temporary_array_payload_upper_bytes_excluding_input_output": TEMPORARY_ARRAY_PAYLOAD_UPPER_BYTES,
        "validated_contiguous_input_storage_bytes_excluded_from_chunk_bound": int(source.nbytes),
        "returned_output_storage_bytes_excluded_from_chunk_bound": output_frames * channels * 8,
        "temporary_memory_bound_is_not_process_rss": True,
        "same_kernel_and_time_mapping_for_every_channel": True,
        "channel_mixing_applied": False,
        "automatic_gain_normalization_applied": False,
        "gain_fitting_applied": False,
        "clipping_applied": False,
        "peak_limiting_applied": False,
        "outside_input_boundary": "zero_extension",
        "boundary_transients_removed": False,
        "linear_phase_fir_delay_compensated": True,
        "output_time_origin_seconds": 0.0,
        "operation": "offline_centered_FIR_not_realtime_streaming",
    }
    if up == down:
        output = source.copy()
        metadata.update({
            "identity_copy": True,
            "filter_tap_count": 0,
            "filter_coefficients_sha256_float64_le": None,
            "kaiser_beta": None,
            "cutoff_relative_to_upsampled_nyquist": None,
            "fixed_filter_dc_gain": 1.0,
            "uncompensated_fir_delay_upsampled_frames": 0,
            "uncompensated_fir_delay_input_frames": 0.0,
            "uncompensated_fir_delay_output_frames": 0.0,
            "maximum_input_taps_per_output_frame": 1,
        })
    else:
        coefficients, half_length = _kernel(up, down)
        taps_per_phase = (len(coefficients) + up - 1) // up
        # The only filter-sized padding is <= up-1 zeros, never an audio buffer.
        phase_coefficients = np.zeros((up, taps_per_phase), dtype=np.float64)
        for phase in range(up):
            branch = coefficients[phase::up]
            phase_coefficients[phase, :len(branch)] = branch
        offsets = np.arange(taps_per_phase, dtype=np.int64)
        output = np.empty((output_frames, channels), dtype=np.float64)
        for start in range(0, output_frames, OUTPUT_CHUNK_FRAMES):
            stop = min(start + OUTPUT_CHUNK_FRAMES, output_frames)
            positions = np.arange(start, stop, dtype=np.int64) * down + half_length
            centers, phases = np.divmod(positions, up)
            indices = centers[:, None] - offsets[None, :]
            valid = (indices >= 0) & (indices < input_frames)
            weights = phase_coefficients[phases]
            weights *= valid
            selected = np.take(source, indices, axis=0, mode="clip")
            output[start:stop] = np.einsum("ij,ijk->ik", weights, selected, optimize=False)
        metadata.update({
            "identity_copy": False,
            "filter_tap_count": len(coefficients),
            "filter_coefficients_sha256_float64_le": hashlib.sha256(
                coefficients.astype("<f8", copy=False).tobytes()
            ).hexdigest(),
            "kaiser_beta": KAISER_BETA,
            "half_length_multiplier": HALF_LENGTH_MULTIPLIER,
            "cutoff_relative_to_upsampled_nyquist": 1.0 / max(up, down),
            "fixed_filter_dc_gain": float(up),
            "actual_coefficient_sum": float(np.sum(coefficients)),
            "uncompensated_fir_delay_upsampled_frames": half_length,
            "uncompensated_fir_delay_input_frames": half_length / up,
            "uncompensated_fir_delay_output_frames": half_length / down,
            "maximum_input_taps_per_output_frame": taps_per_phase,
        })
    for start in range(0, len(output), OUTPUT_CHUNK_FRAMES):
        if not np.isfinite(output[start:start + OUTPUT_CHUNK_FRAMES]).all():
            raise ValueError("sample-rate conversion produced non-finite values")
    return (output[:, 0] if was_mono_vector else output), metadata
