"""Run this isolated prototype with an existing Python + NumPy installation."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from hum_dsp.cli import main

if __name__ == "__main__":
    raise SystemExit(main())

