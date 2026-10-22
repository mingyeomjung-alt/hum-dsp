"""Explicit local persistent-family candidate CLI; no playback or network."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hum_dsp.persistent_processing import process_persistent_file
from hum_dsp.sdk import HumDSPError


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Opt-in offline persistent 50/60 Hz family suppression; not a sound-quality approval."
    )
    parser.add_argument("input", help="Local PCM16/PCM24/Float32 WAV, mono/stereo 44.1/48 kHz, up to 60 seconds")
    parser.add_argument("--output-dir", required=True, help="New local directory; never overwritten")
    parser.add_argument("--hum-hz", required=True, type=float, choices=(50.0, 60.0))
    parser.add_argument("--attenuation-db", type=float, default=12.0, help="Maximum depth in [0,18] dB (default: 12)")
    arguments = parser.parse_args(argv)
    try:
        result = process_persistent_file(
            arguments.input, output_dir=arguments.output_dir, hum_hz=arguments.hum_hz,
            attenuation_db=arguments.attenuation_db,
        )
    except HumDSPError as error:
        print(json.dumps({"status": "error", "error": error.to_dict()}, allow_nan=False), file=sys.stderr)
        return 2
    print(json.dumps(result.to_dict(), allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
