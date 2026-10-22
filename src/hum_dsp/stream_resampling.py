"""Bounded continuous 44.1/48 kHz conversion using the frozen H3a FIR.

The centered FIR needs lookahead. A block call returns only output whose future
input is already available; ``finish`` applies zero extension at the *actual*
file end. Caller block edges, including minute boundaries, are never padded.
There is one global rational-phase clock and a small retained input overlap.
This is offline streaming computation, not an audio-device latency guarantee.
"""

from __future__ import annotations

import hashlib
import math
from numbers import Integral

import numpy as np

from .compat_resampling import (
    HALF_LENGTH_MULTIPLIER, KAISER_BETA, OUTPUT_CHUNK_FRAMES,
    SUPPORTED_RATES_HZ, TEMPORARY_ARRAY_PAYLOAD_UPPER_BYTES, _kernel,
)


ALGORITHM = "bounded_continuous_centered_kaiser_polyphase_v1"
MAX_DURATION_SECONDS = 600
MAX_INPUT_BLOCK_FRAMES = 4096
MAX_RETURNED_BLOCK_FRAMES = 4608


def _integer(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    return int(value)


class StreamingResampler:
    """Known-length centered FIR with fixed-size overlap, no full audio storage.

    Input is strictly frames-by-channels, with one or two independent channels.
    Invalid public input is rejected before mutating the stream. An internal
    computation failure permanently faults the instance, never returns dry
    fallback audio, and requires a new instance.
    """

    def __init__(self, source_rate_hz: int, target_rate_hz: int, input_frames: int,
                 channels: int = 1, target_frames: int | None = None) -> None:
        source = _integer(source_rate_hz, "source_rate_hz")
        target = _integer(target_rate_hz, "target_rate_hz")
        count = _integer(input_frames, "input_frames")
        selected_channels = _integer(channels, "channels")
        if source not in SUPPORTED_RATES_HZ or target not in SUPPORTED_RATES_HZ:
            raise ValueError("sample rates must be 44100 or 48000 Hz")
        if not 1 <= count <= source * MAX_DURATION_SECONDS:
            raise ValueError("input_frames must represent more than zero and at most 600 seconds")
        if selected_channels not in (1, 2):
            raise ValueError("channels must be integer 1 or 2")
        divisor = math.gcd(source, target)
        up, down = target // divisor, source // divisor
        default = (count * up + down - 1) // down
        selected = default if target_frames is None else _integer(target_frames, "target_frames")
        allowed = (default,) if up == down else (default, default - 1)
        if selected < 1 or selected not in allowed:
            raise ValueError("target_frames must equal the default length or, for conversion, one fewer")
        if selected > target * MAX_DURATION_SECONDS:
            raise ValueError("output exceeds 600 seconds")
        self._source_rate = source
        self._target_rate = target
        self._input_frames = count
        self._channels = selected_channels
        self._up, self._down = up, down
        self._default_output_frames = default
        self._output_frames = selected
        self._explicit_target_frames = target_frames is not None
        self._received = self._emitted = 0
        self._finished = False
        self._fault = None
        self._buffer_start = 0
        self._buffer = np.empty((0, selected_channels), dtype=np.float64)
        self._maximum_buffer_frames = self._maximum_retained_frames = 0
        self._maximum_returned_frames = 0
        self._half_length = 0
        self._taps_per_phase = 1
        self._coefficients = self._phase_coefficients = None
        self._offsets = np.arange(1, dtype=np.int64)
        self._kernel_hash = None
        if up != down:
            coefficients, half = _kernel(up, down)
            taps = (len(coefficients) + up - 1) // up
            phases = np.zeros((up, taps), dtype=np.float64)
            for phase in range(up):
                branch = coefficients[phase::up]
                phases[phase, :len(branch)] = branch
            self._coefficients = coefficients
            self._phase_coefficients = phases
            self._offsets = np.arange(taps, dtype=np.int64)
            self._half_length, self._taps_per_phase = half, taps
            self._kernel_hash = hashlib.sha256(
                coefficients.astype("<f8", copy=False).tobytes()
            ).hexdigest()

    @property
    def output_frames(self) -> int:
        return self._output_frames

    @property
    def input_frames(self) -> int:
        return self._input_frames

    def _ensure_open(self) -> None:
        if self._fault is not None:
            raise ValueError("resampler is faulted; construct a new instance")
        if self._finished:
            raise ValueError("resampler is finished; no further input or finish is allowed")

    def _validate_state(self) -> None:
        if (type(self._received) is not int or type(self._emitted) is not int
                or not 0 <= self._received <= self._input_frames
                or not 0 <= self._emitted <= self._output_frames
                or type(self._buffer_start) is not int
                or not 0 <= self._buffer_start <= self._received
                or not isinstance(self._buffer, np.ndarray)
                or self._buffer.dtype != np.float64
                or self._buffer.shape != (self._received - self._buffer_start, self._channels)
                or not np.isfinite(self._buffer).all()):
            raise ValueError("internal rational clock or retained input is invalid")
        if self._buffer.shape[0] > MAX_INPUT_BLOCK_FRAMES + self._taps_per_phase + 2:
            raise ValueError("internal retained input exceeds the fixed overlap bound")
        available = (self._received if self._up == self._down else
                     max(0, (self._received * self._up - self._half_length - 1)
                         // self._down + 1))
        expected_emitted = (self._output_frames if self._finished else
                            min(self._output_frames, available))
        if self._emitted != expected_emitted or (self._finished and self._received != self._input_frames):
            raise ValueError("internal global output phase clock is inconsistent")

    def _latch(self, error: Exception) -> None:
        self._fault = {"reason": "internal_continuous_resampling_failure",
                       "exception_type": type(error).__name__}

    def _render(self, stop: int) -> np.ndarray:
        count = stop - self._emitted
        if count < 0 or count > MAX_RETURNED_BLOCK_FRAMES:
            raise ValueError("internal returned block exceeds its fixed frame bound")
        output = np.empty((count, self._channels), dtype=np.float64)
        first = self._emitted
        for start in range(first, stop, OUTPUT_CHUNK_FRAMES):
            end = min(start + OUTPUT_CHUNK_FRAMES, stop)
            positions = np.arange(start, end, dtype=np.int64) * self._down + self._half_length
            centers, phases = np.divmod(positions, self._up)
            indices = centers[:, None] - self._offsets[None, :]
            valid = (indices >= 0) & (indices < self._input_frames)
            local = indices - self._buffer_start
            if (np.any(local[valid] < 0) or np.any(local[valid] >= len(self._buffer))):
                raise ValueError("internal FIR requested unavailable real input")
            weights = self._phase_coefficients[phases]
            weights *= valid
            selected = np.take(self._buffer, local, axis=0, mode="clip")
            output[start - first:end - first] = np.einsum(
                "ij,ijk->ik", weights, selected, optimize=False)
        if not np.isfinite(output).all():
            raise ValueError("continuous sample-rate conversion produced non-finite output")
        self._emitted = stop
        self._maximum_returned_frames = max(self._maximum_returned_frames, count)
        return output

    def _discard_consumed_overlap(self) -> None:
        if self._emitted >= self._output_frames:
            keep_from = self._received
        else:
            next_center = (self._emitted * self._down + self._half_length) // self._up
            keep_from = min(self._received, max(0, next_center - self._taps_per_phase + 1))
        if not self._buffer_start <= keep_from <= self._received:
            raise ValueError("internal FIR overlap moved backwards")
        # Copy the small tail: a view would keep the entire prior input block alive.
        self._buffer = self._buffer[keep_from - self._buffer_start:].copy()
        self._buffer_start = keep_from
        self._maximum_retained_frames = max(self._maximum_retained_frames, len(self._buffer))

    def process_block(self, samples: object) -> np.ndarray:
        """Consume 1..4096 frames and return only fully supported output frames."""
        self._ensure_open()
        try:
            original = np.asarray(samples)
        except (ValueError, TypeError, OverflowError) as error:
            raise ValueError("samples must be a rectangular real numeric array") from error
        if (original.ndim != 2 or original.shape[1] != self._channels
                or not 1 <= original.shape[0] <= MAX_INPUT_BLOCK_FRAMES
                or original.dtype.kind not in "fiu"):
            raise ValueError("samples must have 1..4096 real frames and configured channels")
        if (not np.isfinite(original).all() or np.any(original < -1)
                or np.any(original >= 1)):
            raise ValueError("samples must be finite normalized audio in [-1, 1)")
        block = np.array(original, dtype=np.float64, copy=True)
        if (not np.isfinite(block).all() or np.any(block < -1) or np.any(block >= 1)):
            raise ValueError("samples cannot be represented as normalized finite float64 audio")
        try:
            self._validate_state()
        except Exception as error:
            self._latch(error)
            raise ValueError("internal continuous resampling failed; construct a new instance") from error
        if self._received + len(block) > self._input_frames:
            raise ValueError("input exceeds the declared input_frames")
        try:
            if self._up == self._down:
                self._received += len(block)
                self._emitted += len(block)
                self._buffer_start = self._received
                self._maximum_returned_frames = max(self._maximum_returned_frames, len(block))
                return block
            self._buffer = np.concatenate((self._buffer, block), axis=0)
            self._received += len(block)
            self._maximum_buffer_frames = max(self._maximum_buffer_frames, len(self._buffer))
            # floor((n*down + half)/up) < received. No EOF padding yet.
            available = max(0, (self._received * self._up - self._half_length - 1)
                            // self._down + 1)
            stop = min(self._output_frames, available)
            output = self._render(stop)
            self._discard_consumed_overlap()
            self._validate_state()
            return output
        except Exception as error:
            self._latch(error)
            raise ValueError("internal continuous resampling failed; construct a new instance") from error

    def finish(self) -> np.ndarray:
        """Finish once, extending only beyond the declared true file boundaries."""
        self._ensure_open()
        try:
            self._validate_state()
        except Exception as error:
            self._latch(error)
            raise ValueError("internal continuous resampling failed; construct a new instance") from error
        if self._received != self._input_frames:
            raise ValueError("all declared input_frames must be supplied before finish")
        try:
            output = (np.empty((0, self._channels), dtype=np.float64)
                      if self._up == self._down else self._render(self._output_frames))
            self._discard_consumed_overlap()
            self._finished = True
            self._validate_state()
            return output
        except Exception as error:
            self._latch(error)
            raise ValueError("internal continuous resampling failed; construct a new instance") from error

    def describe(self) -> dict:
        """Return bounded diagnostics, never an audio buffer or complete history."""
        identity = self._up == self._down
        next_position = self._emitted * self._down + self._half_length
        return {
            "algorithm": ALGORITHM,
            "source_rate_hz": self._source_rate, "target_rate_hz": self._target_rate,
            "up": self._up, "down": self._down,
            "channels": self._channels, "input_frames": self._input_frames,
            "input_frames_received": self._received,
            "output_frames": self._output_frames, "output_frames_expected": self._output_frames,
            "output_frames_emitted": self._emitted,
            "default_output_frames": self._default_output_frames,
            "explicit_target_frames": self._explicit_target_frames,
            "discarded_final_frames": self._default_output_frames - self._output_frames,
            "explicit_frame_override_maximum_trim": 0 if identity else 1,
            "supported_product_round_trip": "44100_to_48000_to_44100_with_original_frame_count",
            "inverse_round_trip_two_frame_trim_supported": False,
            "finished": self._finished, "fault": None if self._fault is None else dict(self._fault),
            "global_next_output_index": self._emitted,
            "global_next_output_upsampled_position": next_position,
            "global_next_polyphase_index": next_position % self._up,
            "retained_input_frames": len(self._buffer),
            "retained_input_start_index": self._buffer_start,
            "maximum_retained_overlap_frames": self._maximum_retained_frames,
            "maximum_combined_input_buffer_frames": self._maximum_buffer_frames,
            "input_buffer_frame_upper_bound": MAX_INPUT_BLOCK_FRAMES + self._taps_per_phase + 2,
            "maximum_input_block_frames": MAX_INPUT_BLOCK_FRAMES,
            "maximum_output_block_frames": MAX_RETURNED_BLOCK_FRAMES,
            "observed_maximum_returned_block_frames": self._maximum_returned_frames,
            "maximum_output_computation_chunk_frames": OUTPUT_CHUNK_FRAMES,
            "temporary_array_payload_upper_bytes_excluding_input_output": TEMPORARY_ARRAY_PAYLOAD_UPPER_BYTES,
            "temporary_memory_bound_is_not_process_rss": True,
            "maximum_duration_seconds": MAX_DURATION_SECONDS,
            "identity_copy": identity,
            "filter_tap_count": 0 if identity else len(self._coefficients),
            "filter_coefficients_sha256_float64_le": self._kernel_hash,
            "kaiser_beta": None if identity else KAISER_BETA,
            "half_length_multiplier": None if identity else HALF_LENGTH_MULTIPLIER,
            "cutoff_relative_to_upsampled_nyquist": None if identity else 1.0 / max(self._up, self._down),
            "fixed_filter_dc_gain": 1.0 if identity else float(self._up),
            "actual_coefficient_sum": 1.0 if identity else float(np.sum(self._coefficients)),
            "uncompensated_fir_delay_upsampled_frames": self._half_length,
            "uncompensated_fir_delay_input_frames": self._half_length / self._up,
            "uncompensated_fir_delay_output_frames": self._half_length / self._down,
            "maximum_input_taps_per_output_frame": self._taps_per_phase,
            "linear_phase_fir_delay_compensated": True,
            "output_time_origin_seconds": 0.0,
            "outside_input_boundary": "zero_extension_only_at_true_file_edges",
            "caller_block_boundaries_zero_padded": False,
            "single_global_phase_clock": True,
            "boundary_transients_removed": False,
            "same_kernel_and_time_mapping_for_every_channel": True,
            "channel_mixing_applied": False,
            "automatic_gain_normalization_applied": False,
            "gain_fitting_applied": False, "clipping_applied": False,
            "peak_limiting_applied": False,
            "operation": "offline_centered_FIR_with_lookahead_not_device_realtime",
            "memory_scope": "fixed_kernel_overlap_and_current_block_excludes_caller_outputs_allocator_RSS",
        }
