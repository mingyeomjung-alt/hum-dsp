"""Offline per-band depth controls using unchanged H1 detection and replay.

Pass one records H1 decisions from the original mixture, once, for the nonzero
bands at 12 dB. Its output is discarded. Pass two derives each selected stage's
coefficients at its explicitly selected depth and replays the frozen decisions
in ascending order. This is an offline two-pass API, not a streaming processor.
The decisions are causal, but replay requires the entire input and trace first.

Lower depth or switching a band off can reduce damage; neither distinguishes
wanted tones from hum at the same frequency. No automatic noise classification,
quality guarantee, clipping, normalization, device I/O or truth-driven tuning is
performed. Trace validation checks structure and causal consistency, not the
authenticity of externally supplied evidence or its acoustic source identity.
"""

from dataclasses import asdict, dataclass
import copy
import json
import math
import numbers

import numpy as np

from .audio import validate_audio
from .harmonics import (
    HarmonicConfig, _coefficients, render_harmonic, render_harmonic_replay,
)


ALGORITHM = "offline_per_band_harmonic_depth_replay_v1"


def _real(value: object, name: str) -> float:
    if (not isinstance(value, numbers.Real) or isinstance(value, (bool, np.bool_))
            or not math.isfinite(value)):
        raise ValueError(f"{name} must be a finite real number")
    return float(value)


@dataclass(frozen=True)
class BandControlConfig:
    sample_rate_hz: int = 48000
    nominal_hum_hz: float = 50.0
    band_depths_db: tuple[float, float, float] = (6.0, 3.0, 0.0)
    block_size_frames: int = 480

    def __post_init__(self) -> None:
        if type(self.sample_rate_hz) is not int or self.sample_rate_hz != 48000:
            raise ValueError("per-band processing supports 48000 Hz only")
        if type(self.block_size_frames) is not int or not 1 <= self.block_size_frames <= 4096:
            raise ValueError("block_size_frames must be an integer in [1, 4096]")
        nominal = _real(self.nominal_hum_hz, "nominal_hum_hz")
        if nominal not in (50.0, 60.0):
            raise ValueError("nominal_hum_hz must be 50 or 60")
        if type(self.band_depths_db) not in (tuple, list) or len(self.band_depths_db) != 3:
            raise ValueError("band_depths_db must contain exactly three depths")
        depths = tuple(_real(depth, "band depth") for depth in self.band_depths_db)
        if any(not 0.0 <= depth <= 12.0 for depth in depths):
            raise ValueError("every band depth must be in [0, 12] dB")
        object.__setattr__(self, "nominal_hum_hz", nominal)
        object.__setattr__(self, "band_depths_db", depths)

    @property
    def orders(self) -> tuple[int, ...]:
        return tuple(index + 1 for index, depth in enumerate(self.band_depths_db) if depth > 0.0)

    @property
    def max_attenuation_db(self) -> float:
        return max(self.band_depths_db)

    def to_dict(self) -> dict:
        validated = BandControlConfig(**asdict(self))
        result = asdict(validated)
        result["band_depths_db"] = list(validated.band_depths_db)
        result["orders"] = list(validated.orders)
        result["max_attenuation_db"] = validated.max_attenuation_db
        return result


def _selected(config: object) -> BandControlConfig:
    if type(config) is not BandControlConfig:
        raise ValueError("config must be a validated BandControlConfig")
    # Even a frozen dataclass may have been deliberately changed with
    # object.__setattr__. Revalidate before it can drive rendering or replay.
    return BandControlConfig(**asdict(config))


def _analysis_config(config: BandControlConfig) -> HarmonicConfig | None:
    if not config.orders:
        return None
    return HarmonicConfig(sample_rate_hz=config.sample_rate_hz,
                          nominal_hum_hz=config.nominal_hum_hz,
                          max_attenuation_db=12.0, orders=config.orders,
                          block_size_frames=config.block_size_frames)


def _derive_stages(config: BandControlConfig, analysis_trace: list[dict]) -> list[dict]:
    stages = []
    for position, order in enumerate(config.orders):
        depth = config.band_depths_db[order - 1]
        selected = HarmonicConfig(sample_rate_hz=config.sample_rate_hz,
                                  nominal_hum_hz=config.nominal_hum_hz,
                                  max_attenuation_db=depth, orders=(order,),
                                  block_size_frames=config.block_size_frames)
        selected_dict = selected.to_dict()
        trace = []
        for original in analysis_trace:
            event = copy.deepcopy(original)
            band = copy.deepcopy(original["bands"][position])
            b, a = _coefficients(band["filter_frequency_hz"], depth)
            band["a"], band["b"] = list(a), list(b)
            event["config"] = copy.deepcopy(selected_dict)
            event["bands"] = [band]
            trace.append(event)
        stages.append({"order": order, "attenuation_db": depth,
                       "config": selected_dict, "trace": trace})
    return stages


def _bundle(config: BandControlConfig, frames: int, analysis_trace: list[dict]) -> dict:
    analysis = _analysis_config(config)
    return {"schema_version": 1, "algorithm": ALGORITHM, "source_frames": frames,
            "config": config.to_dict(),
            "analysis_config": None if analysis is None else analysis.to_dict(),
            "analysis_trace": copy.deepcopy(analysis_trace),
            "stages": _derive_stages(config, analysis_trace)}


def _strict_match(actual: object, expected: object, path: str = "trace_bundle") -> None:
    """Exact JSON shape/types; no bool, NaN, tolerant depth or extra fields."""
    if type(expected) is dict:
        if type(actual) is not dict or actual.keys() != expected.keys():
            raise ValueError(f"{path} has invalid fields")
        for name in expected:
            _strict_match(actual[name], expected[name], f"{path}.{name}")
    elif type(expected) is list:
        if type(actual) is not list or len(actual) != len(expected):
            raise ValueError(f"{path} has invalid cardinality")
        for index, (value, reference) in enumerate(zip(actual, expected, strict=True)):
            _strict_match(value, reference, f"{path}[{index}]")
    elif type(actual) is not type(expected) or actual != expected:
        raise ValueError(f"{path} has an invalid value or type")


def _render_stages(source: np.ndarray, config: BandControlConfig, bundle: dict) -> np.ndarray:
    output = source.copy()
    for stage in bundle["stages"]:
        # Public replay independently checks the complete derived H1 trace,
        # including coefficients, clocks, evidence, ramps and exact EOF binding.
        selected = HarmonicConfig(**stage["config"])
        output = render_harmonic_replay(output, selected, stage["trace"])
    if not np.isfinite(output).all():
        raise ValueError("per-band replay produced nonfinite output")
    return output


def render_band_control(samples: object, config: BandControlConfig | None = None) -> tuple[np.ndarray, dict, dict]:
    """Offline render of 1..60 s, returning raw floats, descriptor and trace.

    The H1 analysis output is deliberately discarded. Nonzero stages all use
    decisions from the same original mixture, not preceding filtered stages.
    All-zero depth bypass performs no FFT and returns no analysis/stage events.
    """
    source = validate_audio(samples)
    selected = _selected(BandControlConfig() if config is None else config)
    analysis = _analysis_config(selected)
    analysis_description = None
    analysis_trace = []
    if analysis is not None:
        _, analysis_description, analysis_trace = render_harmonic(source, analysis)
    bundle = _bundle(selected, source.size, analysis_trace)
    output = _render_stages(source, selected, bundle)
    descriptor = {
        "algorithm": ALGORITHM, "config": selected.to_dict(),
        "samples_processed": int(source.size),
        "processing_bypassed": not selected.orders,
        "processing_mode": "offline_two_pass_original_mixture_analysis_then_frozen_replay",
        "streaming_api": False, "real_time_processing_claim": False,
        "shared_original_input_fft_count": 0 if analysis_description is None else analysis_description["shared_original_input_fft_count"],
        "analysis_trace_event_count": len(analysis_trace),
        "analysis": analysis_description,
        "bands": [{"order": order, "nominal_frequency_hz": selected.nominal_hum_hz * order,
                   "attenuation_db": depth, "enabled": depth > 0.0}
                  for order, depth in enumerate(selected.band_depths_db, start=1)],
        "maximum_input_peak": float(np.max(np.abs(source))),
        "maximum_output_peak": float(np.max(np.abs(output))),
        "policy": {
            "analysis_depth_db": 12.0,
            "analysis_input": "unchanged_original_mixture_once_for_all_nonzero_bands",
            "analysis_output": "discarded_not_used_as_processed_output",
            "decisions": "unchanged_H1_confidence_tracking_qualification_and_ramps",
            "replay_order": "ascending_nonzero_orders",
            "depth_selection": "manual_explicit_no_truth_driven_tuning",
            "wanted_tone_protection": "not_guaranteed_same_frequency_wanted_content_can_be_suppressed",
            "output_peak_handling": "measured_only_no_clipping_normalization_or_limiter",
            "trace_binding": "configuration_and_exact_frame_count_not_acoustic_source_authentication",
        },
    }
    json.dumps((descriptor, bundle), allow_nan=False)
    return output, descriptor, bundle


def render_band_control_replay(samples: object, config: BandControlConfig, trace_bundle: object) -> np.ndarray:
    """Replay frozen mixture decisions on equal-length references, without FFT.

    This validates configuration, schema, causal H1 evidence and all coefficient
    derivations before replay. It cannot authenticate arbitrary external traces.
    References must share the original decoded frame boundary; they are never
    aligned, gain-fitted, re-estimated or used to retune parameters here.
    """
    source = validate_audio(samples)
    selected = _selected(config)
    if type(trace_bundle) is not dict:
        raise ValueError("trace_bundle must be a dictionary")
    analysis = _analysis_config(selected)
    analysis_trace = trace_bundle.get("analysis_trace")
    if analysis is None:
        _strict_match(trace_bundle, _bundle(selected, source.size, []))
        return source.copy()
    # Full public H1 replay is used as a validator. Its returned samples are
    # discarded, exactly as in pass one, and no private engine state is changed.
    render_harmonic_replay(source, analysis, analysis_trace)
    expected = _bundle(selected, source.size, analysis_trace)
    _strict_match(trace_bundle, expected)
    return _render_stages(source, selected, expected)
