"""Strict, serializable v0.1 processing configuration (not an active engine)."""

from dataclasses import asdict, dataclass
import math
import numbers


def _finite_real(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a finite real number")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite")
    return converted


@dataclass(frozen=True)
class ProcessorConfig:
    sample_rate_hz: int = 48000
    nominal_hum_hz: float = 50.0
    max_attenuation_db: float = 12.0
    tracking_half_width_hz: float = 1.0
    max_slew_hz_per_s: float = 0.05
    block_size_frames: int = 480

    def __post_init__(self) -> None:
        if type(self.sample_rate_hz) is not int or self.sample_rate_hz != 48000:
            raise ValueError("v0.1 supports sample_rate_hz=48000 only")
        if type(self.block_size_frames) is not int or not 1 <= self.block_size_frames <= 4096:
            raise ValueError("block_size_frames must be an integer in [1, 4096]")
        frequency = _finite_real(self.nominal_hum_hz, "nominal_hum_hz")
        if frequency not in (50.0, 60.0):
            raise ValueError("nominal_hum_hz must be 50 or 60")
        depth = _finite_real(self.max_attenuation_db, "max_attenuation_db")
        if not 0.0 <= depth <= 12.0:
            raise ValueError("max_attenuation_db must be in [0, 12]")
        width = _finite_real(self.tracking_half_width_hz, "tracking_half_width_hz")
        if not 0.0 < width <= 1.0:
            raise ValueError("tracking_half_width_hz must be in (0, 1]")
        slew = _finite_real(self.max_slew_hz_per_s, "max_slew_hz_per_s")
        if not 0.0 < slew <= 0.05:
            raise ValueError("max_slew_hz_per_s must be in (0, 0.05]")
        for name in ("nominal_hum_hz", "max_attenuation_db", "tracking_half_width_hz", "max_slew_hz_per_s"):
            object.__setattr__(self, name, float(getattr(self, name)))

    def to_dict(self) -> dict:
        # Revalidate at the serialization boundary as well as construction.
        # A frozen dataclass is not a security barrier against object.__setattr__.
        return asdict(ProcessorConfig(**asdict(self)))

