"""Opt-in guarded offline processing, never a speaker mute or device path."""

import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .audio import SAMPLE_RATE_HZ, read_pcm16_wav, validate_audio, write_pcm16_wav
from .config import ProcessorConfig
from .fixtures import _dump_json
from .guarded import GuardedHumSuppressor
from .processing import _input_sha256, _levels


def _controls(controls, frames):
    if not isinstance(controls, (tuple, list)) or len(controls) > 256:
        raise ValueError("controls must be at most 256 ordered (sample_index, bool) events")
    result = []
    previous = -1
    for event in controls:
        if not isinstance(event, (tuple, list)) or len(event) != 2:
            raise ValueError("each control must be (sample_index, enabled)")
        index, enabled = event
        if type(index) is not int or not previous < index < frames or index < 0 or type(enabled) is not bool:
            raise ValueError("controls must be strictly increasing in-range integer indices and bool states")
        previous = index
        result.append((index, enabled))
    return tuple(result)


def render_guarded(samples, config: ProcessorConfig, *, controls=()):
    source = validate_audio(samples)
    if type(config) is not ProcessorConfig:
        raise ValueError("config must be a ProcessorConfig")
    selected = ProcessorConfig(**config.to_dict())
    events = _controls(controls, source.size)
    engine = GuardedHumSuppressor(selected)
    output = np.empty_like(source)
    cursor, control_cursor = 0, 0
    while cursor < source.size:
        while control_cursor < len(events) and events[control_cursor][0] == cursor:
            engine.set_enabled(events[control_cursor][1])
            control_cursor += 1
        stop = min(source.size, cursor + selected.block_size_frames)
        if control_cursor < len(events):
            stop = min(stop, events[control_cursor][0])
        block = np.asarray(engine.process_block(source[cursor:stop]))
        if block.ndim != 1 or block.shape != source[cursor:stop].shape:
            raise ValueError("guarded processor changed block shape or sample count")
        output[cursor:stop] = validate_audio(block)
        cursor = stop
    description, guard, tracking = engine.describe(), engine.get_guard_trace(), engine.get_tracking_trace()
    json.dumps((description, guard, tracking), allow_nan=False)
    return output, description, guard, tracking


def process_guarded_wav(input_path, output_directory, config=None) -> dict:
    selected = ProcessorConfig() if config is None else config
    if type(selected) is not ProcessorConfig:
        raise ValueError("config must be a ProcessorConfig")
    selected = ProcessorConfig(**selected.to_dict())
    source, target = Path(input_path), Path(output_directory)
    if target.exists():
        raise FileExistsError("output directory must be new; existing files are never replaced")
    input_sha = _input_sha256(source)
    samples = read_pcm16_wav(source)
    if _input_sha256(source) != input_sha:
        raise ValueError("input changed while it was being read")
    started = perf_counter()
    floating, description, guard, tracking = render_guarded(samples, selected)
    elapsed = perf_counter() - started
    if _input_sha256(source) != input_sha:
        raise ValueError("input changed while processing")
    target.mkdir(parents=True, exist_ok=False)
    output_path = target / "processed.wav"
    write_pcm16_wav(output_path, floating)
    decoded = read_pcm16_wav(output_path)
    for name, events in (("guard", guard), ("tracking", tracking)):
        _dump_json(target / f"{name}.json", {"schema_version": 1, "sample_rate_hz": SAMPLE_RATE_HZ, "events": events})
    report = {
        "schema_version": 1,
        "stage": "M4_distortion_stability_controls",
        "engine": "guarded_tracked_finite_depth_notch",
        "config": selected.to_dict(),
        "algorithm": description,
        "processing_requested": selected.max_attenuation_db > 0,
        "processor_applied": bool(np.any(floating != samples)),
        "guard": {"file_name": "guard.json", "event_count": len(guard)},
        "tracking": {"file_name": "tracking.json", "event_count": len(tracking)},
        "device_io": False,
        "output_gain_normalization_applied": False,
        "posthoc_alignment_applied": False,
        "reference_quality_metrics": None,
        "input": {"file_name": source.name, "sha256": input_sha, "frames": int(samples.size), "sample_rate_hz": SAMPLE_RATE_HZ, **_levels(samples)},
        "output": {"file_name": "processed.wav", "sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(), "frames": int(decoded.size), "sample_rate_hz": SAMPLE_RATE_HZ, **_levels(decoded)},
        "floating_output": _levels(floating),
        "pcm16_quantization_max_abs_error": float(np.max(np.abs(decoded - floating))),
        "timing": {"file_processing_seconds": elapsed, "real_time_device_performance_verified": False},
        "limitations": [
            "Two eligible observations and a soft attack delay useful attenuation; effectiveness can be lower than always-on M3.",
            "Loss recovery includes the trailing 2-second estimator, evidence hold and 250 ms release; not immediate.",
            "The gate does not know whether a selected same-frequency tone is wanted.",
            "Soft OFF is not emergency mute. Digital guard controls are not a physical acoustic safety guarantee.",
            "No clipping, normalization, microphone, playback or cloud operation. Internal faults raise an error instead of returning successful output.",
        ],
    }
    _dump_json(target / "report.json", report)
    return report

