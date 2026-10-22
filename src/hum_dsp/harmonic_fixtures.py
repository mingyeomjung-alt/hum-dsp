"""Owned mathematical H1 cases for selective 50/60 Hz harmonic processing.

All labels are known synthetic ground truth, not information an audio-only
processor receives. In particular, a desired 100 Hz tone is observationally
identical to unwanted second-harmonic hum at the same phase and amplitude.
There is no recording, external dataset, device access or random generation.
"""

from dataclasses import dataclass

import numpy as np

from .audio import SAMPLE_RATE_HZ, validate_audio


@dataclass(frozen=True)
class HFixture:
    case_id: str
    nominal_hum_hz: float
    clean: np.ndarray
    noise: np.ndarray
    input_audio: np.ndarray
    kind: str
    notes: str


_SECONDS = 8
_NOISE_RMS_PER_ORDER = 0.012
_PHASES = (0.37, 1.13, -0.61)
_NOTICE = "Owned synthetic mathematical development case, not real-world validation. "


def _tone(time: np.ndarray, hz: float, amplitude: float, phase: float) -> np.ndarray:
    return amplitude * np.sin(2.0 * np.pi * hz * time + phase)


def _program(time: np.ndarray) -> np.ndarray:
    return sum((_tone(time, hz, amplitude, phase) for hz, amplitude, phase in (
        (233.0, 0.06, 0.21), (419.0, 0.04, -0.43), (811.0, 0.02, 0.83),
    )), start=np.zeros_like(time))


def _hum(time: np.ndarray, nominal: float, orders: tuple[int, ...] = (1, 2, 3)) -> np.ndarray:
    amplitude = np.sqrt(2.0) * _NOISE_RMS_PER_ORDER
    return sum((_tone(time, nominal * order, amplitude, _PHASES[order - 1])
                for order in orders), start=np.zeros_like(time))


def _fixture(suffix: str, nominal: float, clean: np.ndarray, noise: np.ndarray,
             kind: str, notes: str) -> HFixture:
    clean = validate_audio(clean)
    noise = validate_audio(noise)
    mixed = validate_audio(clean + noise)
    for values in (clean, noise, mixed):
        values.setflags(write=False)
    return HFixture(f"synthetic_h1_{suffix}", nominal, clean, noise, mixed, kind,
                    _NOTICE + notes)


def harmonic_fixtures() -> tuple[HFixture, ...]:
    """Return the frozen twelve readonly float64, eight-second, 48 kHz cases.

    WAV evaluation must quantize/decode at the shared PCM16 boundary first.
    Component replay uses those decoded references and decisions frozen from
    their decoded mixture, not separate clean/noise detection decisions.
    """
    time = np.arange(_SECONDS * SAMPLE_RATE_HZ, dtype=np.float64) / SAMPLE_RATE_HZ
    zeros = np.zeros_like(time)
    program = _program(time)
    cases = []
    for nominal in (50.0, 60.0):
        hum = _hum(time, nominal)
        cases.append(_fixture(
            f"compound_stationary_noise_only_{int(nominal)}", nominal, zeros, hum,
            "stationary_noise_only",
            "Orders 1, 2 and 3 each have RMS 0.012 and phases 0.37, 1.13 and -0.61; total input is noise only.",
        ))
    for nominal in (50.0, 60.0):
        cases.append(_fixture(
            f"compound_stationary_mixture_{int(nominal)}", nominal, program, _hum(time, nominal),
            "stationary_mixture",
            "Desired 233, 419 and 811 Hz tones have peak amplitudes 0.06, 0.04 and 0.02. Unwanted orders 1, 2 and 3 each have RMS 0.012; oracle replay must report desired-signal error separately from residual noise.",
        ))
    cases.extend((
        _fixture(
            "fundamental_only_50", 50.0, zeros, _hum(time, 50.0, (1,)), "fundamental_only",
            "Only the 50 Hz fundamental exists; absent second and third harmonic bands should remain dry. This does not require the active fundamental band to remain dry.",
        ),
        _fixture(
            "missing_fundamental_60", 60.0, zeros, _hum(time, 60.0, (2, 3)), "missing_fundamental",
            "Only 120 and 180 Hz unwanted tones exist, each RMS 0.012; independent harmonic detection must not require a measurable 60 Hz fundamental.",
        ),
        _fixture(
            "clean_program_50", 50.0, program, zeros, "clean_program",
            "Desired 233, 419 and 811 Hz tones only, outside all selected search bands. This negative control should remain exactly dry after the PCM16 boundary.",
        ),
        _fixture(
            "neighboring_desired_50", 50.0,
            _tone(time, 101.5, 0.04, 0.67) + _tone(time, 233.0, 0.06, 0.21)
            + _tone(time, 419.0, 0.04, -0.43), zeros, "neighboring_desired_limit",
            "Desired 101.5 Hz tone falls inside the second-harmonic search band, with desired 233 and 419 Hz tones. KNOWN SEMANTIC LIMIT: detection may suppress wanted audio; this is not an exact-dry preservation pass criterion.",
        ),
        _fixture(
            "wanted_harmonic_50", 50.0, _tone(time, 100.0, np.sqrt(2.0) * 0.012, _PHASES[1]),
            zeros, "wanted_harmonic_limit",
            "Desired 100 Hz tone only, waveform-identical to unwanted second-harmonic hum. KNOWN SEMANTIC LIMIT: wanted-tone damage must be measured and disclosed, not called successful noise removal.",
        ),
        _fixture(
            "silence_50", 50.0, zeros, zeros, "silence",
            "Exact silence is a negative control and must remain exactly dry. Noise attenuation in dB is undefined, not positive infinity.",
        ),
    ))

    fade_start = 4 * SAMPLE_RATE_HZ
    fade_frames = round(0.02 * SAMPLE_RATE_HZ)
    fade_end = fade_start + fade_frames
    envelope = np.ones_like(time)
    envelope[fade_start:fade_end] = 0.5 * (
        1.0 + np.cos(np.pi * np.arange(fade_frames, dtype=np.float64) / fade_frames)
    )
    envelope[fade_end:] = 0.0
    cases.append(_fixture(
        "abrupt_end_50", 50.0, zeros, _hum(time, 50.0) * envelope, "abrupt_end",
        "All three unwanted orders stop with a 20 ms raised-cosine fade from 4.0 to 4.02 s, followed by exact silence. Source offset is sample 192960; recovery includes estimator history and guard release, not release time alone.",
    ))

    # Integrating base frequency produces a phase-continuous 0.02 Hz/s drift.
    cycles = 49.8 * time + 0.5 * 0.02 * np.square(time)
    drifting = sum((np.sqrt(2.0) * _NOISE_RMS_PER_ORDER
                    * np.sin(2.0 * np.pi * order * cycles + _PHASES[order - 1])
                    for order in (1, 2, 3)), start=zeros.copy())
    cases.append(_fixture(
        "drift_compound_50", 50.0, zeros, drifting, "compound_drift",
        "Phase-continuous base frequency 49.8 + 0.02*t Hz for eight seconds; orders 1, 2 and 3 drift proportionally with RMS approximately 0.012 each. All remain within their selected bands and per-order slew limits.",
    ))
    return tuple(cases)
