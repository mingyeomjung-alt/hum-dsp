"""Bounded PCM16 WAV I/O. No resampling, mixing, playback or normalization."""

from pathlib import Path
import wave

import numpy as np

SAMPLE_RATE_HZ = 48000
MAX_FRAMES = SAMPLE_RATE_HZ * 60
MAX_FILE_BYTES = 32 * 1024 * 1024
PCM16_STEP = 1.0 / 32768.0


def validate_audio(samples: object) -> np.ndarray:
    try:
        array = np.asarray(samples)
    except (TypeError, ValueError) as exc:
        raise ValueError("audio must be a real mono vector") from exc
    if array.ndim != 1 or not 1 <= array.size <= MAX_FRAMES:
        raise ValueError("audio must contain 1..2880000 mono samples (maximum 60 seconds)")
    if array.dtype.kind not in "fiu":
        raise ValueError("audio must contain real numeric samples, not bool, complex or objects")
    result = np.array(array, dtype=np.float64, copy=True)
    if not np.isfinite(result).all():
        raise ValueError("audio must contain finite samples")
    if np.any(result < -1.0) or np.any(result >= 1.0):
        raise ValueError("audio samples must be in [-1, 1); no automatic normalization")
    return result


def write_pcm16_wav(path: str | Path, samples: object, *, overwrite: bool = False) -> None:
    array = validate_audio(samples)
    if type(overwrite) is not bool:
        raise ValueError("overwrite must be a bool")
    # Round to nearest, ties to even. The largest positive PCM16 code is 32767.
    codes = np.clip(np.rint(array * 32768.0), -32768, 32767).astype("<i2")
    destination = Path(path)
    with destination.open("wb" if overwrite else "xb") as raw:
        with wave.open(raw, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(SAMPLE_RATE_HZ)
            wav.writeframes(codes.tobytes())


def read_pcm16_wav(path: str | Path) -> np.ndarray:
    source = Path(path)
    if source.stat().st_size > MAX_FILE_BYTES:
        raise ValueError("WAV file exceeds the 32 MiB container limit")
    try:
        with source.open("rb") as raw:
            with wave.open(raw, "rb") as wav:
                if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getcomptype()) != (1, 2, SAMPLE_RATE_HZ, "NONE"):
                    raise ValueError("only uncompressed mono 48000 Hz PCM16 WAV is supported")
                frames = wav.getnframes()
                if not 1 <= frames <= MAX_FRAMES:
                    raise ValueError("WAV duration must be greater than 0 and at most 60 seconds")
                payload = wav.readframes(frames)
                if len(payload) != frames * 2:
                    raise ValueError("truncated WAV payload")
    except (wave.Error, EOFError) as exc:
        raise ValueError(f"invalid PCM16 WAV: {exc}") from exc
    return validate_audio(np.frombuffer(payload, dtype="<i2").astype(np.float64) / 32768.0)

