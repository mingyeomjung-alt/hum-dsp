"""Causal, fixed-frequency, finite-depth hum filter for the M2 baseline.

This is not a hum detector or an adaptive tracker. A user-selected 50/60 Hz
center and a fixed 1 Hz pole-design bandwidth define one stable biquad. The
finite-depth transfer function is G + (1-G)*H_notch; it does not globally lower
the signal, normalize its peak, limit it or clip it. Floating-point output can
temporarily exceed the PCM16 input range and must be checked before encoding.

The notch coefficients follow the standard Orfanidis second-order notch
design (also used by scipy.signal.iirnotch). The direct-form II transposed
sample kernel below is independent of SciPy; NumPy is the only dependency.
"""

import math

import numpy as np

from .audio import validate_audio
from .config import ProcessorConfig


POLE_DESIGN_BANDWIDTH_HZ = 1.0
MAX_BLOCK_FRAMES = 4096


class FixedHumSuppressor:
    """Stateful fixed notch with exact state continuity across valid blocks.

    There is no lookahead or explicitly buffered sample delay. This does NOT
    imply zero phase shift, zero group delay, or immediate steady-state notch
    attenuation: the causal IIR has frequency-dependent phase and transients.
    Invalid input never updates the filter state or sample counter.
    """

    def __init__(self, config: ProcessorConfig) -> None:
        if type(config) is not ProcessorConfig:
            raise ValueError("config must be a validated ProcessorConfig")
        # Snapshot and revalidate even a frozen dataclass whose guard was bypassed.
        # Later changes to the caller's object cannot retune this fixed filter.
        validated = ProcessorConfig(**config.to_dict())
        self._config_values = validated.to_dict()
        self._bypass = validated.max_attenuation_db == 0.0
        omega = 2.0 * math.pi * validated.nominal_hum_hz / validated.sample_rate_hz
        beta = math.tan(math.pi * POLE_DESIGN_BANDWIDTH_HZ / validated.sample_rate_hz)
        gain = 1.0 / (1.0 + beta)
        a = (1.0, -2.0 * gain * math.cos(omega), 2.0 * gain - 1.0)
        full_notch_b = (gain, -2.0 * gain * math.cos(omega), gain)
        center_gain = 10.0 ** (-validated.max_attenuation_db / 20.0)
        self._a = a
        self._b = tuple(
            center_gain * denominator + (1.0 - center_gain) * numerator
            for denominator, numerator in zip(a, full_notch_b, strict=True)
        )
        self.reset()

    def reset(self) -> None:
        """Return to zero initial conditions without changing coefficients."""
        self._state = (0.0, 0.0)
        self._samples_processed = 0

    def describe(self) -> dict:
        """Return JSON-safe design/state metadata, never a mutable state handle."""
        return {
            "algorithm": "fixed_finite_depth_notch_df2_transposed_v1",
            "tracking_enabled": False,
            "pole_design_bandwidth_hz": POLE_DESIGN_BANDWIDTH_HZ,
            "bandwidth_note": "1 Hz defines the full-notch poles, not the finite-depth -3 dB bandwidth",
            "explicit_delay_samples": 0,
            "delay_note": "No lookahead; phase/group delay and convergence time are not zero",
            "config": dict(self._config_values),
            "b": list(self._b),
            "a": list(self._a),
            "samples_processed": self._samples_processed,
        }

    def process_block(self, samples: object) -> np.ndarray:
        """Process 1..4096 finite real mono samples in the input range [-1, 1)."""
        try:
            source = np.asarray(samples)
        except (TypeError, ValueError) as exc:
            raise ValueError("block must be a real mono vector") from exc
        if source.ndim != 1 or not 1 <= source.size <= MAX_BLOCK_FRAMES:
            raise ValueError("block must contain 1..4096 mono samples")
        block = validate_audio(source)
        if self._bypass:
            # Do not depend on finite-precision IIR pole-zero cancellation for
            # the exact 0 dB identity path. validate_audio already made a copy.
            self._samples_processed += block.size
            return block

        b0, b1, b2 = self._b
        _, a1, a2 = self._a
        state1, state2 = self._state
        output = np.empty(block.size, dtype=np.float64)
        for index, value in enumerate(block):
            sample = float(value)
            filtered = b0 * sample + state1
            state1 = b1 * sample - a1 * filtered + state2
            state2 = b2 * sample - a2 * filtered
            output[index] = filtered

        # Commit only after the complete calculation has succeeded. No output
        # clipping is used to conceal nonfinite values or ordinary overshoot.
        if not np.isfinite(output).all() or not (math.isfinite(state1) and math.isfinite(state2)):
            raise ValueError("filter produced nonfinite output; state was not advanced")
        self._state = (state1, state2)
        self._samples_processed += block.size
        return output


def process_audio(samples: object, config: ProcessorConfig | None = None) -> np.ndarray:
    """Process 1 sample up to 60 seconds of mono audio with continuous state.

    Each invocation creates a fresh fixed filter. It does not reuse state from
    a previous file. The output has the same length and no gain normalization.
    """
    selected = ProcessorConfig() if config is None else config
    engine = FixedHumSuppressor(selected)
    audio = validate_audio(samples)
    block_size = engine.describe()["config"]["block_size_frames"]
    output = np.empty_like(audio)
    for start in range(0, audio.size, block_size):
        stop = min(start + block_size, audio.size)
        output[start:stop] = engine.process_block(audio[start:stop])
    return output
