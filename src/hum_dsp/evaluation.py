"""Keep startup behavior visible alongside the settled analysis interval."""

from numbers import Integral

from .metrics import measure_output


def measure_windows(clean, noise, output, *, settling_samples=96000) -> dict:
    if isinstance(settling_samples, bool) or not isinstance(settling_samples, Integral) or settling_samples < 1:
        raise ValueError("settling_samples must be a positive integer")
    # Validate all arrays first; whole-record measurements are always retained.
    whole = measure_output(clean, noise, output)
    count = whole["total_sample_count"]
    boundary = min(int(settling_samples), count)
    startup = measure_output(clean[:boundary], noise[:boundary], output[:boundary])
    settled = None if boundary == count else measure_output(clean, noise, output, start_sample=boundary)
    return {
        "schema_version": 1,
        "settling_samples": int(settling_samples),
        "whole_record": whole,
        "startup": startup,
        "settled": settled,
        "settled_undefined_reason": "no_samples_after_settling" if settled is None else None,
        "processor_performance_claim": False,
    }

