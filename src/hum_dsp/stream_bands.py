"""Bounded-state 48 kHz per-band DSP across continuous mono/stereo blocks.

The unchanged H1 detector observes one fixed ORIGINAL input channel. Independent
per-channel H1 filter states receive the same decisions; stereo is never summed.
Depths are the explicit H2 controls, not automatic wanted/noise classification.
Only a two-second reference ring, filter state and ONE latest event are retained.
An optional caller-owned sink can stream events elsewhere; sink storage is not
part of the engine's memory bound. No whole-file buffer or trace list is kept.

This is a bounded streaming computation API, not a device/real-time guarantee.
It neither clips nor normalizes floating output. A file writer must reject an
unrepresentable output before publishing it. A failed call never returns dry
audio as success; internal or sink failures latch the instance permanently.
"""

from dataclasses import asdict
import copy
import json
import math

import numpy as np

from .band_control import BandControlConfig
from .harmonics import (
    ALGORITHM as H1_ALGORITHM,
    ATTACK_FRAMES, ESTIMATE_EDGE_TOLERANCE_HZ, FFT_FRAMES, MAX_BLOCK_FRAMES,
    MIN_COHERENT_POWER_FRACTION, MIN_PEAK_AMPLITUDE, OBSERVATION_FRAMES,
    QUALIFICATION_HOLD_FRAMES, RELEASE_FRAMES, UPDATE_FRAMES,
    HarmonicConfig, _Band, _safe,
)


ALGORITHM = "bounded_continuous_per_band_reference_channel_v1"
MAX_STREAM_FRAMES = 48000 * 600
_EVENT_ONLY = (
    "estimated_frequency_hz", "confidence", "confidence_is_probability", "status",
    "qualified", "reason", "previous_filter_frequency_hz", "initial_acquisition",
    "target_changed", "mix_before",
)


class StreamBandControl:
    """Continuous 1..4096-frame, frames-by-channels float64 processing.

    Public block validation is transactional. The ten-minute clock and fixed
    reference channel cannot change mid-stream. Every observation at N consumed
    frames first changes output N onward, including at a caller block boundary.
    """

    def __init__(self, config: BandControlConfig, channels: int = 1,
                 reference_channel_index: int = 0, trace_sink=None) -> None:
        if type(config) is not BandControlConfig:
            raise ValueError("config must be a validated BandControlConfig")
        selected = BandControlConfig(**asdict(config))
        if type(channels) is not int or channels not in (1, 2):
            raise ValueError("channels must be integer 1 or 2")
        if (type(reference_channel_index) is not int
                or not 0 <= reference_channel_index < channels):
            raise ValueError("reference_channel_index must select an existing input channel")
        if trace_sink is not None and not callable(trace_sink):
            raise ValueError("trace_sink must be callable or None")
        self._config = selected.to_dict()
        self._channels = channels
        self._reference_channel_index = reference_channel_index
        self._trace_sink = trace_sink
        self._bypass = not selected.orders
        self._bands = []
        for _ in range(channels):
            current = []
            for order in selected.orders:
                stage = HarmonicConfig(
                    nominal_hum_hz=selected.nominal_hum_hz,
                    max_attenuation_db=selected.band_depths_db[order - 1],
                    orders=(order,), block_size_frames=selected.block_size_frames,
                ).to_dict()
                current.append(_Band(order, stage))
            self._bands.append(current)
        self._samples_processed = self._fft_count = self._trace_event_count = 0
        self._next_update = OBSERVATION_FRAMES
        self._latest_event = None
        self._fault = None
        self._emitting = False
        self._maximum_input_peak = self._maximum_output_peak = 0.0
        self._history_position = 0
        self._history = self._window = None
        self._window_sum = self._window_square_sum = None
        if not self._bypass:
            self._history = np.zeros(OBSERVATION_FRAMES, dtype=np.float64)
            self._window = np.hanning(OBSERVATION_FRAMES)
            self._window_sum = float(np.sum(self._window))
            self._window_square_sum = float(np.dot(self._window, self._window))
            try:
                self._emit([[band.snapshot() for band in channel] for channel in self._bands],
                           "initial_dry")
            except Exception as exc:
                self._latch(exc)
                raise ValueError("stream trace sink failed during initialization") from exc

    def _latch(self, exc: Exception) -> None:
        self._fault = {"sample_index": _safe(self._samples_processed),
                       "reason": "internal_stream_processing_failure",
                       "exception_type": type(exc).__name__}

    def _emit(self, channel_bands: list[list[dict]], kind: str) -> None:
        event = {
            "schema_version": 1, "algorithm": ALGORITHM,
            "sample_index": self._samples_processed,
            "time_seconds": self._samples_processed / 48000.0,
            "event_kind": kind, "config": copy.deepcopy(self._config),
            "channels": self._channels,
            "reference_channel_index": self._reference_channel_index,
            "bands": copy.deepcopy(channel_bands[self._reference_channel_index]),
            "channel_bands": channel_bands,
        }
        json.dumps(event, allow_nan=False)
        self._latest_event = event
        self._trace_event_count += 1
        if self._trace_sink is not None:
            self._emitting = True
            try:
                self._trace_sink(copy.deepcopy(event))
            finally:
                self._emitting = False

    def _validate_state(self) -> None:
        if (type(self._samples_processed) is not int
                or not 0 <= self._samples_processed <= MAX_STREAM_FRAMES):
            raise ValueError("internal continuous sample clock is invalid")
        expected = (0 if self._samples_processed < OBSERVATION_FRAMES else
                    1 + (self._samples_processed - OBSERVATION_FRAMES) // UPDATE_FRAMES)
        expected = 0 if self._bypass else expected
        if type(self._fft_count) is not int or self._fft_count != expected:
            raise ValueError("internal continuous FFT clock is invalid")
        if (type(self._trace_event_count) is not int
                or self._trace_event_count != (0 if self._bypass else expected + 1)):
            raise ValueError("internal continuous trace clock is invalid")
        if (type(self._next_update) is not int or self._next_update !=
                OBSERVATION_FRAMES + expected * UPDATE_FRAMES):
            raise ValueError("internal continuous observation clock is invalid")
        if (not isinstance(self._bands, list) or len(self._bands) != self._channels
                or any(not isinstance(channel, list) or
                       [band.order for band in channel] != self._config["orders"]
                       for channel in self._bands)):
            raise ValueError("internal continuous selected bands are inconsistent")
        for channel in self._bands:
            for band in channel:
                band.validate(self._samples_processed)
                if band.depth != self._config["band_depths_db"][band.order - 1]:
                    raise ValueError("internal continuous band depth is inconsistent")
        if self._bypass:
            if self._history is not None or self._window is not None or self._latest_event is not None:
                raise ValueError("bypassed stream must retain no history or events")
        else:
            for array in (self._history, self._window):
                if (not isinstance(array, np.ndarray) or array.shape != (OBSERVATION_FRAMES,)
                        or array.dtype != np.float64 or not np.isfinite(array).all()):
                    raise ValueError("internal continuous history or window is invalid")
            if (type(self._history_position) is not int or
                    self._history_position != self._samples_processed % OBSERVATION_FRAMES):
                raise ValueError("internal continuous ring position is invalid")
            last_clock = 0 if expected == 0 else OBSERVATION_FRAMES + (expected - 1) * UPDATE_FRAMES
            if (type(self._latest_event) is not dict
                    or self._latest_event.get("sample_index") != last_clock):
                raise ValueError("internal latest continuous event is invalid")
        for peak in (self._maximum_input_peak, self._maximum_output_peak):
            if type(peak) is not float or not math.isfinite(peak) or peak < 0.0:
                raise ValueError("internal continuous peak is invalid")

    def _append_history(self, original_reference: np.ndarray) -> None:
        position = self._history_position
        first = min(original_reference.size, OBSERVATION_FRAMES - position)
        self._history[position:position + first] = original_reference[:first]
        if first < original_reference.size:
            self._history[:original_reference.size - first] = original_reference[first:]
        self._history_position = (position + original_reference.size) % OBSERVATION_FRAMES

    def _estimate_band(self, band: _Band, powers: np.ndarray, weighted_power: float) -> tuple:
        # Keep H1's exact arithmetic and evidence rules, not a new detector.
        if weighted_power <= 0.0:
            return None, 0.0, "insufficient_energy_hold"
        peak = int(band.search[int(np.argmax(powers[band.search]))])
        amplitude = 2.0 * math.sqrt(float(powers[peak])) / self._window_sum
        fraction = 0.5 * amplitude**2 / weighted_power
        confidence = float(np.clip(fraction, 0.0, 1.0))
        if amplitude < MIN_PEAK_AMPLITUDE or fraction < MIN_COHERENT_POWER_FRACTION:
            return None, confidence, "insufficient_concentration_hold"
        left, middle, right = map(float, np.log(np.maximum(
            powers[peak - 1:peak + 2], np.finfo(float).tiny)))
        denominator = left - 2.0 * middle + right
        offset = 0.0 if denominator >= -1e-18 else 0.5 * (left - right) / denominator
        estimate = (peak + float(np.clip(offset, -1.0, 1.0))) * 48000.0 / FFT_FRAMES
        if not band.lower - ESTIMATE_EDGE_TOLERANCE_HZ <= estimate <= band.upper + ESTIMATE_EDGE_TOLERANCE_HZ:
            return None, confidence, "peak_outside_selected_band_hold"
        return float(np.clip(estimate, band.lower, band.upper)), confidence, "eligible_peak"

    def _observe(self) -> None:
        position = self._history_position
        ordered = np.concatenate((self._history[position:], self._history[:position]))
        windowed = ordered * self._window
        weighted_power = float(np.dot(windowed, windowed) / self._window_square_sum)
        spectrum = np.fft.rfft(windowed, n=FFT_FRAMES)
        powers = np.abs(spectrum) ** 2
        if not math.isfinite(weighted_power) or not np.isfinite(powers).all():
            raise ValueError("shared continuous spectral observation is nonfinite")
        reference = self._bands[self._reference_channel_index]
        evidence = [self._estimate_band(band, powers, weighted_power) for band in reference]
        decisions = []
        for channel in self._bands:
            decisions.append([
                band.observe(estimate, confidence, status, self._samples_processed)
                for band, (estimate, confidence, status) in zip(channel, evidence, strict=True)
            ])
        self._fft_count += 1
        self._emit(decisions, "observation")

    def process_block(self, samples: object) -> np.ndarray:
        """Process a validated frames×channels block, without changing input."""
        try:
            array = np.asarray(samples)
        except (TypeError, ValueError) as exc:
            raise ValueError("stream block must be a real frames-by-channels array") from exc
        if (array.ndim != 2 or array.shape[1] != self._channels
                or not 1 <= array.shape[0] <= MAX_BLOCK_FRAMES or array.dtype.kind not in "fiu"):
            raise ValueError("stream block must have 1..4096 real numeric frames and configured channels")
        block = np.array(array, dtype=np.float64, copy=True)
        if (not np.isfinite(block).all() or np.any(block < -1.0) or np.any(block >= 1.0)):
            raise ValueError("stream samples must be finite in [-1, 1); no automatic normalization")
        if self._emitting:
            raise ValueError("trace sink must not recursively process input")
        if self._fault is not None:
            raise ValueError("stream processor is faulted; construct a new instance")
        # Validate the internal clock before using it for the public length cap.
        # A corrupted clock is an internal latched failure, not an input error.
        try:
            self._validate_state()
        except Exception as exc:
            self._latch(exc)
            raise ValueError("internal continuous processing failed; construct a new instance") from exc
        if self._samples_processed + block.shape[0] > MAX_STREAM_FRAMES:
            raise ValueError("continuous stream exceeds the ten-minute limit")
        output = np.empty_like(block)
        offset = 0
        try:
            while offset < block.shape[0]:
                count = (block.shape[0] - offset if self._bypass else
                         min(block.shape[0] - offset, self._next_update - self._samples_processed))
                if count <= 0:
                    raise ValueError("internal continuous boundary did not advance")
                part = block[offset:offset + count]
                result = part.copy()
                if not self._bypass:
                    for index, channel in enumerate(self._bands):
                        rendered = part[:, index]
                        for band in channel:
                            rendered = band.process(rendered)
                        result[:, index] = rendered
                    self._append_history(part[:, self._reference_channel_index])
                if result.shape != part.shape or not np.isfinite(result).all():
                    raise ValueError("continuous cascade produced invalid output")
                output[offset:offset + count] = result
                self._maximum_input_peak = max(self._maximum_input_peak, float(np.max(np.abs(part))))
                self._maximum_output_peak = max(self._maximum_output_peak, float(np.max(np.abs(result))))
                self._samples_processed += count
                offset += count
                if not self._bypass and self._samples_processed == self._next_update:
                    self._observe()
                    self._next_update += UPDATE_FRAMES
            self._validate_state()
        except Exception as exc:
            self._latch(exc)
            raise ValueError("internal continuous processing failed; construct a new instance") from exc
        return output

    def describe(self) -> dict:
        channels = []
        for channel_index, channel in enumerate(self._bands):
            current_bands = []
            for position, band in enumerate(channel):
                current = band.snapshot()
                for name in _EVENT_ONLY:
                    current.pop(name)
                current["last_observation"] = None
                if self._latest_event is not None and self._latest_event["event_kind"] == "observation":
                    current["last_observation"] = {
                        "sample_index": self._latest_event["sample_index"],
                        "time_seconds": self._latest_event["time_seconds"],
                        **copy.deepcopy(self._latest_event["channel_bands"][channel_index][position]),
                    }
                current_bands.append(current)
            channels.append({"channel_index": channel_index, "bands": current_bands})
        result = {
            "algorithm": ALGORITHM, "h1_primitives_algorithm": H1_ALGORITHM,
            "config": copy.deepcopy(self._config), "channels": self._channels,
            "reference_channel_index": self._reference_channel_index,
            "samples_processed": self._samples_processed,
            "processing_bypassed": self._bypass,
            "shared_original_input_fft_count": self._fft_count,
            "trace_event_count": self._trace_event_count,
            "latest_event": copy.deepcopy(self._latest_event),
            "bands": copy.deepcopy(channels[self._reference_channel_index]["bands"]),
            "channel_bands": channels,
            "maximum_input_peak": self._maximum_input_peak,
            "maximum_output_peak": self._maximum_output_peak,
            "fault": copy.deepcopy(self._fault),
            "retained_trace_records": 0 if self._latest_event is None else 1,
            "retained_audio_frames": 0 if self._history is None else OBSERVATION_FRAMES,
            "memory_scope": "fixed_reference_history_window_filter_state_latest_event_excludes_caller_sink_storage",
            "policy": {
                "maximum_frames": MAX_STREAM_FRAMES, "maximum_seconds": 600,
                "maximum_block_frames": MAX_BLOCK_FRAMES,
                "observation_frames": OBSERVATION_FRAMES, "update_frames": UPDATE_FRAMES,
                "fft_frames": FFT_FRAMES, "attack_frames": ATTACK_FRAMES,
                "release_frames": RELEASE_FRAMES,
                "qualification_hold_frames": QUALIFICATION_HOLD_FRAMES,
                "analysis_input": "original_fixed_reference_channel_no_stereo_sum_no_cascade_feedback",
                "other_channels": "independent_filter_states_same_reference_channel_decisions",
                "event_timing": "N_consumed_frames_affects_output_N_onward",
                "depth_selection": "manual_explicit_unchanged_H2_depths",
                "wanted_tone_protection": "not_guaranteed_same_frequency_wanted_content_can_be_suppressed",
                "confidence": "unchanged_H1_heuristic_not_probability_or_noise_semantics",
                "event_sink": "independent_JSON_safe_event_copy_no_full_trace_retained",
                "output_peak_handling": "measured_only_no_clipping_normalization_or_limiter",
                "device_or_real_time_processing_claim": False,
                "internal_failure": "latched_raise_no_successful_fallback",
            },
        }
        return _safe(result) if self._fault is not None else result
