"""Causal, bounded, center-frequency-only extension of the frozen M2 notch.

Candidate v1 observes the preceding 2 seconds through a Hann window, every
0.1 seconds on a GLOBAL sample clock. A zero-padded FFT and a local log-power
parabola estimate one sinusoidal peak in the manually selected 50/60 Hz band.
These are fixed development choices, not learned or fixture-dependent values.

Before acquisition and when an observation is insufficient, the filter stays
at the nominal/last center: the absence of an eligible peak does not bypass it.
An initial acquisition can jump inside the allowed band. Later changes are
slew-limited. The M2 depth and 1 Hz pole bandwidth are unchanged. Its DFII-T
state survives coefficient changes; there is no crossfade, gain normalization,
limiter, clipping, lookahead, or claim that transitions are inaudible.

The eligibility/confidence values describe spectral concentration heuristically;
they are neither calibrated probabilities nor evidence that a tone is unwanted.
Wanted content at the selected frequency is inherently confusable with hum.
"""

import copy
import math

import numpy as np

from .audio import validate_audio
from .config import ProcessorConfig
from .filter import MAX_BLOCK_FRAMES, POLE_DESIGN_BANDWIDTH_HZ


OBSERVATION_FRAMES = 96000
UPDATE_FRAMES = 4800
FFT_FRAMES = 524288
MIN_COHERENT_POWER_FRACTION = 0.02
MIN_PEAK_AMPLITUDE = 1e-7
ESTIMATE_EDGE_TOLERANCE_HZ = 0.001


def _coefficients(frequency: float, config: dict) -> tuple[tuple, tuple]:
    """The exact M2 design, with only its center frequency allowed to change."""
    omega = 2.0 * math.pi * frequency / config["sample_rate_hz"]
    beta = math.tan(math.pi * POLE_DESIGN_BANDWIDTH_HZ / config["sample_rate_hz"])
    gain = 1.0 / (1.0 + beta)
    a = (1.0, -2.0 * gain * math.cos(omega), 2.0 * gain - 1.0)
    full_notch_b = (gain, -2.0 * gain * math.cos(omega), gain)
    center_gain = 10.0 ** (-config["max_attenuation_db"] / 20.0)
    b = tuple(center_gain * denominator + (1.0 - center_gain) * numerator
              for denominator, numerator in zip(a, full_notch_b, strict=True))
    return b, a


class TrackedHumSuppressor:
    """Stateful bounded tracker, independent of caller block boundaries.

    Updates occur AFTER the sample_index input frames in an event have been
    consumed; the changed coefficients first affect output at that zero-based
    index. Only past input is used. Invalid blocks are rejected before any
    history, trace, counter, coefficients, or filter state is updated.
    """

    def __init__(self, config: ProcessorConfig) -> None:
        if type(config) is not ProcessorConfig:
            raise ValueError("config must be a validated ProcessorConfig")
        self._config = ProcessorConfig(**config.to_dict()).to_dict()
        self._window = np.hanning(OBSERVATION_FRAMES)
        self._window_sum = float(np.sum(self._window))
        self._window_square_sum = float(np.dot(self._window, self._window))
        self._frequency_step = self._config["sample_rate_hz"] / FFT_FRAMES
        lower = self._config["nominal_hum_hz"] - self._config["tracking_half_width_hz"]
        upper = self._config["nominal_hum_hz"] + self._config["tracking_half_width_hz"]
        # A one-bin margin permits interpolation at the inclusive band edges.
        self._search_indices = np.arange(
            max(1, math.ceil(lower / self._frequency_step - 1.0)),
            min(FFT_FRAMES // 2 - 1, math.floor(upper / self._frequency_step + 1.0)) + 1,
            dtype=np.int64,
        )
        self.reset()

    def reset(self) -> None:
        """Restart at fixed nominal frequency, empty history and zero IIR state."""
        self._center = self._config["nominal_hum_hz"]
        self._b, self._a = _coefficients(self._center, self._config)
        self._state = (0.0, 0.0)
        self._history = np.zeros(OBSERVATION_FRAMES, dtype=np.float64)
        self._history_position = 0
        self._samples_processed = 0
        self._next_update = OBSERVATION_FRAMES
        self._acquired = False
        self._first_acquired_sample = None
        self._trace = []
        self._maximum_slew = 0.0

    def get_trace(self) -> list[dict]:
        """Independent JSON-safe events; sample_index is an input frame count."""
        return copy.deepcopy(self._trace)

    def describe(self) -> dict:
        return {
            "algorithm": "causal_hann_peak_slew_finite_depth_notch_v1",
            "tracking_enabled": self._config["max_attenuation_db"] != 0.0,
            "processing_bypassed": self._config["max_attenuation_db"] == 0.0,
            "candidate_revision": 1,
            "config": dict(self._config),
            "observation_frames": OBSERVATION_FRAMES,
            "observation_seconds": OBSERVATION_FRAMES / self._config["sample_rate_hz"],
            "update_frames": UPDATE_FRAMES,
            "update_seconds": UPDATE_FRAMES / self._config["sample_rate_hz"],
            "fft_frames": FFT_FRAMES,
            "estimation_method": "hann_fft_log_power_quadratic_peak",
            "minimum_coherent_power_fraction": MIN_COHERENT_POWER_FRACTION,
            "minimum_peak_amplitude": MIN_PEAK_AMPLITUDE,
            "estimate_edge_tolerance_hz": ESTIMATE_EDGE_TOLERANCE_HZ,
            "confidence_note": "Heuristic coherent spectral power fraction; not probability or proof of noise",
            "frequency_estimate_time_reference": "2-second trailing window; no prediction or future input",
            "before_acquisition": "fixed_nominal_notch",
            "insufficient_observation": "hold_last_center_not_bypass",
            "initial_acquisition_jump_allowed": True,
            "coefficient_transition": "deterministic_0.1s_updates_slew_after_initial_acquisition_continuous_df2t_state",
            "coefficient_transition_note": "No crossfade; transition audibility is not guaranteed",
            "pole_design_bandwidth_hz": POLE_DESIGN_BANDWIDTH_HZ,
            "bandwidth_note": "1 Hz defines the full-notch poles, not the finite-depth -3 dB bandwidth",
            "explicit_delay_samples": 0,
            "delay_note": "No lookahead; phase/group delay and two-second initial acquisition are not zero",
            "filter_frequency_hz": self._center,
            "acquired": self._acquired,
            "first_acquired_sample": self._first_acquired_sample,
            "samples_processed": self._samples_processed,
            "trace_event_count": len(self._trace),
            "maximum_post_acquisition_slew_hz_per_s": self._maximum_slew,
            "a": list(self._a),
            "b": list(self._b),
        }

    def _append_history(self, block: np.ndarray) -> None:
        position = self._history_position
        first = min(block.size, OBSERVATION_FRAMES - position)
        self._history[position:position + first] = block[:first]
        if first < block.size:
            self._history[:block.size - first] = block[first:]
        self._history_position = (position + block.size) % OBSERVATION_FRAMES

    def _estimate(self) -> tuple[float | None, float, str]:
        position = self._history_position
        ordered = np.concatenate((self._history[position:], self._history[:position]))
        windowed = ordered * self._window
        weighted_power = float(np.dot(windowed, windowed) / self._window_square_sum)
        if weighted_power <= 0.0:
            return None, 0.0, "insufficient_energy_hold"
        spectrum = np.fft.rfft(windowed, n=FFT_FRAMES)
        powers = np.abs(spectrum) ** 2
        peak_bin = int(self._search_indices[int(np.argmax(powers[self._search_indices]))])
        peak_amplitude = 2.0 * math.sqrt(float(powers[peak_bin])) / self._window_sum
        fraction = (0.5 * peak_amplitude**2) / weighted_power
        confidence = float(np.clip(fraction, 0.0, 1.0))
        if peak_amplitude < MIN_PEAK_AMPLITUDE or fraction < MIN_COHERENT_POWER_FRACTION:
            return None, confidence, "insufficient_concentration_hold"
        logarithms = np.log(np.maximum(powers[peak_bin - 1:peak_bin + 2], np.finfo(float).tiny))
        left, middle, right = map(float, logarithms)
        denominator = left - 2.0 * middle + right
        offset = 0.0 if denominator >= -1e-18 else 0.5 * (left - right) / denominator
        offset = float(np.clip(offset, -1.0, 1.0))
        estimate = (peak_bin + offset) * self._frequency_step
        lower = self._config["nominal_hum_hz"] - self._config["tracking_half_width_hz"]
        upper = self._config["nominal_hum_hz"] + self._config["tracking_half_width_hz"]
        if not lower - ESTIMATE_EDGE_TOLERANCE_HZ <= estimate <= upper + ESTIMATE_EDGE_TOLERANCE_HZ:
            return None, confidence, "peak_outside_selected_band_hold"
        return float(np.clip(estimate, lower, upper)), confidence, "eligible_peak"

    def _update(self) -> None:
        estimate, confidence, status = self._estimate()
        previous = self._center
        first_acquisition = False
        if estimate is not None:
            if not self._acquired:
                self._center = estimate
                self._acquired = True
                self._first_acquired_sample = self._samples_processed
                first_acquisition = True
                status = "initial_acquisition"
            else:
                elapsed = UPDATE_FRAMES / self._config["sample_rate_hz"]
                limit = self._config["max_slew_hz_per_s"] * elapsed
                difference = estimate - self._center
                self._center += float(np.clip(difference, -limit, limit))
                self._maximum_slew = max(self._maximum_slew, abs(self._center - previous) / elapsed)
                status = "tracking" if abs(difference) <= limit else "tracking_slew_limited"
            self._b, self._a = _coefficients(self._center, self._config)
        self._trace.append({
            "sample_index": self._samples_processed,
            "time_seconds": self._samples_processed / self._config["sample_rate_hz"],
            "estimated_frequency_hz": estimate,
            "filter_frequency_hz": self._center,
            "previous_filter_frequency_hz": previous,
            "confidence": confidence,
            "confidence_is_probability": False,
            "status": status,
            "initial_acquisition": first_acquisition,
            "acquired": self._acquired,
            "slew_limited_after_initial_acquisition": True,
        })

    def process_block(self, samples: object) -> np.ndarray:
        try:
            source = np.asarray(samples)
        except (TypeError, ValueError) as exc:
            raise ValueError("block must be a real mono vector") from exc
        if source.ndim != 1 or not 1 <= source.size <= MAX_BLOCK_FRAMES:
            raise ValueError("block must contain 1..4096 mono samples")
        block = validate_audio(source)
        if self._config["max_attenuation_db"] == 0.0:
            # Explicit OFF is an exact identity with no estimation, acquisition,
            # observation history or trace. Configuration is immutable, so this
            # instance cannot later activate with an inconsistent clock.
            self._samples_processed += block.size
            return block
        output = np.empty_like(block)
        offset = 0
        while offset < block.size:
            count = min(block.size - offset, self._next_update - self._samples_processed)
            part = block[offset:offset + count]
            b0, b1, b2 = self._b
            _, a1, a2 = self._a
            state1, state2 = self._state
            for index, value in enumerate(part):
                sample = float(value)
                filtered = b0 * sample + state1
                state1 = b1 * sample - a1 * filtered + state2
                state2 = b2 * sample - a2 * filtered
                output[offset + index] = filtered
            if not (math.isfinite(state1) and math.isfinite(state2)):
                raise ValueError("tracking filter produced nonfinite state")
            self._state = (state1, state2)
            self._append_history(part)
            self._samples_processed += count
            offset += count
            if self._samples_processed == self._next_update:
                self._update()
                self._next_update += UPDATE_FRAMES
        if not np.isfinite(output).all():
            raise ValueError("tracking filter produced nonfinite output")
        return output


def process_tracked_audio(samples: object, config: ProcessorConfig | None = None, *, return_trace: bool = False):
    """Fresh causal processing of <=60 seconds, preserving length and amplitude.

    Explicit depth 0 is exact bypass and does not observe/acquire a frequency.
    No measured file or truth is consulted. With return_trace=True, return
    (float64_output, independent_JSON_safe_event_list), otherwise output alone.
    """
    if type(return_trace) is not bool:
        raise ValueError("return_trace must be a bool")
    selected = ProcessorConfig() if config is None else config
    engine = TrackedHumSuppressor(selected)
    audio = validate_audio(samples)
    block_size = engine.describe()["config"]["block_size_frames"]
    output = np.empty_like(audio)
    for start in range(0, audio.size, block_size):
        stop = min(start + block_size, audio.size)
        output[start:stop] = engine.process_block(audio[start:stop])
    return (output, engine.get_trace()) if return_trace else output
