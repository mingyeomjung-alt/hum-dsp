"""Guard envelope reconstruction and objective transition evidence."""

import math

import numpy as np

from .audio import SAMPLE_RATE_HZ


def reconstruct_mix(events, frames):
    if type(frames) is not int or frames <= 0:
        raise ValueError("frames must be positive")
    mix = np.zeros(frames)
    hard = np.zeros(frames)
    cursor, start_mix, target, length, position = 0, 0.0, 0.0, 0, 0
    current = 0.0
    previous_event = -1

    def fill(stop):
        nonlocal cursor, position, current
        count = stop - cursor
        if count:
            if length:
                indices = np.minimum(np.arange(position + 1, position + count + 1), length)
                values = start_mix + (target - start_mix) * (1.0 - np.cos(np.pi * indices / length)) / 2.0
                values[indices == length] = target
                position = min(length, position + count)
            else:
                values = np.full(count, target)
            mix[cursor:stop] = values
            hard[cursor:stop] = target
            current = float(values[-1])
            cursor = stop

    for event in events:
        at = event["sample_index"]
        if type(at) is not int or not 0 <= at <= frames or at < previous_event:
            raise ValueError("guard event clock is invalid")
        previous_event = at
        if not event["target_changed"]:
            continue
        fill(at)
        start_mix, target, length = event["ramp_start_mix"], event["target_mix"], event["ramp_total_frames"]
        if not (math.isfinite(start_mix) and 0 <= start_mix <= 1 and target in (0.0, 1.0) and type(length) is int and length > 0):
            raise ValueError("invalid envelope state")
        if abs(start_mix - current) > 1e-10:
            raise ValueError("guard retarget would jump its envelope")
        position = 0
    fill(frames)
    return mix, hard


def _high_frequency_power(signal, event_indices):
    powers = []
    for at in event_indices:
        start, stop = max(0, at - 2400), min(signal.size, at + 2400)
        if stop - start < 64:
            continue
        segment = signal[start:stop]
        window = np.hanning(segment.size)
        spectrum = np.fft.rfft(segment * window)
        frequencies = np.fft.rfftfreq(segment.size, 1 / SAMPLE_RATE_HZ)
        selected = (frequencies >= 500) & (frequencies <= 20000)
        powers.append(float(np.sum(np.abs(spectrum[selected])**2) / (segment.size * np.sum(window**2))))
    return None if not powers else max(powers)


def measure_guard(source, raw, guarded, guard_trace):
    source, raw, guarded = [np.asarray(value, dtype=np.float64) for value in (source, raw, guarded)]
    if source.ndim != 1 or source.size == 0 or raw.shape != source.shape or guarded.shape != source.shape:
        raise ValueError("signals must be equal-length nonempty vectors")
    if not all(np.isfinite(value).all() for value in (source, raw, guarded)):
        raise ValueError("signals must be finite")
    mix, hard_mix = reconstruct_mix(guard_trace, int(source.size))
    reconstruction = source + mix * (raw - source)
    changes = [event["sample_index"] for event in guard_trace if event["target_changed"]]
    residual, hard_residual = guarded - source, hard_mix * (raw - source)
    indices = np.flatnonzero(mix > 0)
    final_dry = int(indices[-1] + 1) if indices.size else 0
    return {
        "reconstruction_max_error": float(np.max(np.abs(guarded - reconstruction))),
        "mix_min": float(np.min(mix)),
        "mix_max": float(np.max(mix)),
        "maximum_mix_sample_step": float(np.max(np.abs(np.diff(np.concatenate(([0.0], mix)))))),
        "theoretical_attack_step_bound": math.pi / (2 * 4800),
        "target_change_count": len(changes),
        "final_dry_from_sample": None if final_dry == source.size else final_dry,
        "maximum_residual_first_difference": float(np.max(np.abs(np.diff(residual)))) if source.size > 1 else 0.0,
        "hard_switch_maximum_residual_first_difference": float(np.max(np.abs(np.diff(hard_residual)))) if source.size > 1 else 0.0,
        "transition_residual_500_to_20000hz_max_window_power": _high_frequency_power(residual, changes),
        "hard_switch_residual_500_to_20000hz_max_window_power": _high_frequency_power(hard_residual, changes),
        "metric_scope": "output_minus_input_residual_not_perceptual_click_or_audibility_measure",
        "whole_record_error_from_input_rmse": float(np.sqrt(np.mean(residual**2))),
        "no_processing_entire_record": bool(np.all(mix == 0)),
    }
