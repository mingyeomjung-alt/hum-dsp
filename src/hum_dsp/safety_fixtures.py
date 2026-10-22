"""Owned mathematical M4 safety inputs; not real-world audio validation.

These cases freeze event times, control changes and desired/noise labels before
the safety benchmark is run.  Labels are ground truth for measurements, not
information available to an audio-only processor.  In particular, an unwanted
50 Hz sinusoid cannot be distinguished from the same wanted sinusoid by DSP
alone.  No input/output, random generation, recordings or device access occurs.
"""

from dataclasses import dataclass

import numpy as np

from .audio import SAMPLE_RATE_HZ, validate_audio


@dataclass(frozen=True)
class SafetyFixture:
    case_id: str
    nominal_hum_hz: float
    clean: np.ndarray
    noise: np.ndarray
    input_audio: np.ndarray
    controls: tuple[tuple[int, bool], ...]
    noise_end_sample: int | None
    expected_final_dry: bool
    expect_exact_dry_everywhere: bool
    disable_interval: tuple[int, int] | None
    notes: str


_PHASE = 0.37
_AMPLITUDE = np.sqrt(2.0) * 0.02
_NOTICE = "Owned synthetic mathematical development case, not real-world validation. "


def _time(seconds: int) -> np.ndarray:
    return np.arange(seconds * SAMPLE_RATE_HZ, dtype=np.float64) / SAMPLE_RATE_HZ


def _tone(time: np.ndarray, hz: float, amplitude: float = _AMPLITUDE) -> np.ndarray:
    return amplitude * np.sin(2.0 * np.pi * (hz * time) + _PHASE)


def _fixture(
    suffix: str,
    nominal_hum_hz: float,
    clean: np.ndarray,
    noise: np.ndarray,
    *,
    expected_final_dry: bool,
    expect_exact_dry_everywhere: bool = False,
    controls: tuple[tuple[int, bool], ...] = (),
    noise_end_sample: int | None = None,
    disable_interval: tuple[int, int] | None = None,
    notes: str,
) -> SafetyFixture:
    clean = validate_audio(clean)
    noise = validate_audio(noise)
    mixed = validate_audio(clean + noise)
    for array in (clean, noise, mixed):
        array.setflags(write=False)
    return SafetyFixture(
        f"synthetic_m4_{suffix}", nominal_hum_hz, clean, noise, mixed,
        controls, noise_end_sample, expected_final_dry,
        expect_exact_dry_everywhere, disable_interval, _NOTICE + notes,
    )


def safety_fixtures() -> tuple[SafetyFixture, ...]:
    """Return the ten frozen 48 kHz mono M4 safety cases as readonly float64.

    Times are source sample coordinates.  The below-PCM-floor case deliberately
    contains nonzero floating-point samples that become exact zero through the
    project's PCM16 round-to-nearest input boundary.  Its exact-dry expectation
    applies to that decoded PCM16 input, not a noise-free semantic claim.
    """
    time10 = _time(10)
    time12 = _time(12)
    time8 = _time(8)
    zero10 = np.zeros_like(time10)
    zero12 = np.zeros_like(time12)
    zero8 = np.zeros_like(time8)

    fade_frames = round(0.02 * SAMPLE_RATE_HZ)
    end_start = 4 * SAMPLE_RATE_HZ
    end_sample = end_start + fade_frames
    ends_envelope = np.ones_like(time10)
    # The analytic raised cosine is 1 at t=4 and 0 at t=4.02.  Sampling the
    # interior plus an explicit zero tail avoids a discontinuous hard cutoff.
    ends_envelope[end_start:end_sample] = 0.5 * (
        1.0 + np.cos(np.pi * np.arange(fade_frames) / fade_frames)
    )
    ends_envelope[end_sample:] = 0.0

    appears_start = 3 * SAMPLE_RATE_HZ
    appears_end = appears_start + fade_frames
    appears_envelope = np.zeros_like(time10)
    appears_envelope[appears_start:appears_end] = 0.5 * (
        1.0 - np.cos(np.pi * np.arange(fade_frames) / fade_frames)
    )
    appears_envelope[appears_end:] = 1.0

    # Integrate frequency piecewise instead of multiplying time by the new
    # frequency, which would cause a phase jump at t=5.  The 1.6 Hz jump is
    # intentionally outside the slow-changing scope established in M3.
    step_cycles = 50.8 * np.minimum(time12, 5.0) + 49.2 * np.maximum(time12 - 5.0, 0.0)
    step_noise = _AMPLITUDE * np.sin(2.0 * np.pi * step_cycles + _PHASE)
    impulse = np.zeros_like(time8)
    impulse[3 * SAMPLE_RATE_HZ] = 0.5
    disabled = 4 * SAMPLE_RATE_HZ
    reenabled = 6 * SAMPLE_RATE_HZ

    return (
        _fixture(
            "tone_ends_50", 50.0, zero10, _tone(time10, 50.0) * ends_envelope,
            expected_final_dry=True, noise_end_sample=end_sample,
            notes="50 Hz noise ends using a 20 ms raised-cosine fade from 4.0 to 4.02 s; the remaining input is exact silence.",
        ),
        _fixture(
            "tone_appears_60", 60.0, zero10, _tone(time10, 60.0) * appears_envelope,
            expected_final_dry=False,
            notes="Exact silence until 3 s, followed by a 20 ms raised-cosine fade into stationary 60 Hz noise; delayed safe acquisition is expected.",
        ),
        _fixture(
            "soft_toggle_50", 50.0, zero12, _tone(time12, 50.0),
            expected_final_dry=False,
            controls=((disabled, False), (reenabled, True)),
            disable_interval=(disabled + 12000, reenabled),
            notes="Continuous 50 Hz noise; disable at 4 s and re-enable at 6 s. The exact-dry interval starts 12000 frames after disabling to allow a bounded soft transition.",
        ),
        _fixture(
            "inband_step_50", 50.0, zero12, step_noise,
            expected_final_dry=True,
            notes="Phase-continuous frequency step 50.8 to 49.2 Hz at 5 s: both are in-band, but the abrupt jump is outside M3 slow-drift scope. Center-estimate mismatch should return safely to dry, not claim tracking success.",
        ),
        _fixture(
            "silence", 50.0, zero8, zero8,
            expected_final_dry=True, expect_exact_dry_everywhere=True,
            notes="Exact silence is a negative control, not a defined dB-reduction measurement.",
        ),
        _fixture(
            "impulse", 50.0, impulse, zero8,
            expected_final_dry=True, expect_exact_dry_everywhere=True,
            notes="One desired amplitude-0.5 sample at 3 s; a transient is not stationary hum and must not trigger filtering.",
        ),
        _fixture(
            "desired_tone_50", 50.0, _tone(time8, 50.0), zero8,
            expected_final_dry=False,
            notes="Desired 50 Hz tone only, with 0.02 RMS and phase 0.37. This is NOT a preservation pass criterion: it is waveform-identical to unwanted 50 Hz hum and documents the known semantic limitation.",
        ),
        _fixture(
            "below_pcm_floor_60", 60.0, zero8, _tone(time8, 60.0, amplitude=1e-9),
            expected_final_dry=True, expect_exact_dry_everywhere=True,
            notes="60 Hz amplitude 1e-9: the decoded PCM16 input is all zero. Exact-dry expectations refer to that decoded input, not the original nonzero floating-point reference.",
        ),
        _fixture(
            "dc", 50.0, np.full(time8.shape, 0.1, dtype=np.float64), zero8,
            expected_final_dry=True, expect_exact_dry_everywhere=True,
            notes="Desired constant DC level 0.1; it is not 50 Hz hum and must remain dry.",
        ),
        _fixture(
            "nearby_desired_60", 60.0, _tone(time8, 63.0), zero8,
            expected_final_dry=True, expect_exact_dry_everywhere=True,
            notes="Desired 63 Hz tone only, with 0.02 RMS and phase 0.37; it is outside the configured 60 Hz +/-1 Hz tracking band.",
        ),
    )
