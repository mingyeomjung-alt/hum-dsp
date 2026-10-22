"""Small local-file SDK for the existing, explicit Hum DSP processors.

This module only validates developer-facing options and packages the artifacts
already produced by the file engines. It does not detect noise automatically,
certify sound quality, change DSP policy, play audio, or open audio devices.
"""

from copy import deepcopy
from dataclasses import dataclass
import json
import os
from pathlib import Path
import stat
from typing import Any, Literal


__all__ = ["HumDSPError", "ProcessResult", "process_file"]


class HumDSPError(Exception):
    """A stable SDK error code and English recovery message.

    The original exception is available as ``__cause__`` for local debugging.
    Messages intentionally omit paths and raw decoder/operating-system errors.
    """

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(message)

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code, "message": self.message}


@dataclass(frozen=True)
class ProcessResult:
    """Paths to committed files and a detached copy of their actual report.

    Frozen fields prevent accidental path reassignment. ``report`` is a normal
    detached dictionary for convenient inspection, not an immutable certificate.
    ``to_dict`` returns another detached copy suitable for ``json.dumps``.
    """

    output_wav: Path
    report_path: Path
    report: dict[str, Any]
    trace_path: Path | None = None
    decoded_input_wav: Path | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "report", deepcopy(self.report))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "status": "completed",
            "output_wav": str(self.output_wav),
            "report_path": str(self.report_path),
            "trace_path": None if self.trace_path is None else str(self.trace_path),
            "decoded_input_wav": (
                None if self.decoded_input_wav is None else str(self.decoded_input_wav)
            ),
            "report": deepcopy(self.report),
        }


def _path_value(value: object, name: str) -> Path:
    try:
        raw = os.fspath(value)
        if (not isinstance(raw, str) or not raw or "\x00" in raw
                or "://" in raw or raw.startswith(("//", "\\\\"))):
            raise ValueError("a local text filesystem path is required")
        return Path(os.path.abspath(raw))
    except (OSError, TypeError, ValueError) as error:
        raise HumDSPError(
            "invalid_options", f"{name} must be a local text filesystem path; URLs and network shares are unsupported."
        ) from error


def _local_path(value: object, name: str) -> Path:
    path = _path_value(value, name)
    try:
        # Do not follow existing symbolic links or Windows reparse ancestors.
        # This is input hygiene, not a sandbox or a concurrent-path guarantee.
        for item in reversed((path, *path.parents)):
            try:
                info = item.lstat()
            except FileNotFoundError:
                continue
            if (stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0)
                    & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 1024)):
                raise ValueError("symbolic links and reparse paths are unsupported")
            if item != path and not stat.S_ISDIR(info.st_mode):
                raise ValueError("path ancestors must be directories")
        return path
    except (OSError, TypeError, ValueError) as error:
        raise HumDSPError(
            "invalid_options",
            f"{name} must be a local filesystem path without symbolic links or reparse paths.",
        ) from error


def process_file(
    input_path: str | os.PathLike[str],
    *,
    output_dir: str | os.PathLike[str],
    hum_hz: float,
    band_depths_db: tuple[float, float, float] = (6.0, 3.0, 0.0),
    mode: Literal["precision", "compressed"] = "precision",
    block_size_frames: int = 4096,
) -> ProcessResult:
    """Process one local file into a *new* directory; never replace results.

    ``hum_hz`` is required and must be 50 or 60. Depths explicitly select
    attenuation of the fundamental and two harmonics, each from 0 to 12 dB;
    three zero depths use the existing validated audio-payload bypass.

    ``precision`` accepts WAV PCM16/packed PCM24/normalized Float32, canonical
    mono/stereo, 44.1/48 kHz, at most 600 seconds and 128 MiB. ``compressed``
    must be selected explicitly for MP3/AAC-M4A, requires optional PyAV 19.x
    and an existing output parent, and imports at most 32 MiB to Float32 WAV.
    Codec decoding and its existing limits/disclosures remain unchanged.

    The engine performs final container/sample validation and publishes its
    existing all-or-nothing output. No mode, profile or hum frequency is chosen
    automatically. A completed result is not proof of improved sound quality.
    """
    try:
        if mode not in ("precision", "compressed") or type(mode) is not str:
            raise ValueError("unknown processing mode")
        from .band_control import BandControlConfig

        config = BandControlConfig(
            nominal_hum_hz=hum_hz,
            band_depths_db=band_depths_db,
            block_size_frames=block_size_frames,
        )
    except (TypeError, ValueError) as error:
        raise HumDSPError(
            "invalid_options",
            "Use mode precision or compressed, hum_hz 50 or 60, three depths in [0, 12] dB, "
            "and an integer block_size_frames in [1, 4096].",
        ) from error
    except ImportError as error:
        raise HumDSPError("dependency_unavailable", "A processing dependency is unavailable; check this installation.") from error

    # Reject lexical remote paths and reparse ancestors before dereferencing
    # anything. An existing plain output is refused without touching its files.
    target = _local_path(output_dir, "output_dir")
    if target.exists():
        raise HumDSPError("output_exists", "Choose a new output directory; existing results are never replaced.")
    source = _local_path(input_path, "input_path")
    try:
        source_info = source.stat()
    except FileNotFoundError as error:
        raise HumDSPError("input_not_found", "Input file was not found; provide an existing local audio file.") from error
    except OSError as error:
        raise HumDSPError("unsupported_input", "Input file could not be read; check its local permissions.") from error
    if not stat.S_ISREG(source_info.st_mode):
        raise HumDSPError("unsupported_input", "Input must be a regular local audio file, not a directory or device.")
    allowed = (".wav",) if mode == "precision" else (".mp3", ".m4a")
    if source.suffix.lower() not in allowed:
        raise HumDSPError(
            "unsupported_input",
            "Precision mode requires WAV; MP3/AAC-M4A requires explicit compressed mode.",
        )
    if mode == "compressed" and not target.parent.is_dir():
        raise HumDSPError("invalid_options", "Compressed mode requires an existing local output parent directory.")

    if mode == "precision":
        try:
            from .precision_wav import inspect_precision_wav

            inspect_precision_wav(source)
        except ImportError as error:
            raise HumDSPError("dependency_unavailable", "A processing dependency is unavailable; check this installation.") from error
        except ValueError as error:
            raise HumDSPError(
                "unsupported_input",
                "WAV header is unsupported or invalid; use canonical mono/stereo PCM16, PCM24 or Float32 "
                "at 44.1/48 kHz within 600 seconds and 128 MiB.",
            ) from error
        except OSError as error:
            raise HumDSPError("processing_failed", "Input could not be inspected; check access and concurrent changes.") from error

    try:
        if mode == "precision":
            from .precision_processing import process_precision_wav

            actual_report = process_precision_wav(source, target, config)
        else:
            from .compressed_decode import decoder_status
            from .compressed_processing import process_compressed_file

            if not decoder_status()["available"]:
                raise HumDSPError("dependency_unavailable", "Compressed mode requires the optional PyAV 19.x decoder.")
            actual_report = process_compressed_file(source, target, config)

        # Bind the convenience result to the report actually committed by the
        # engine, not to a newly invented SDK report or a quality assertion.
        report_path = target / "report.json"
        committed_report = json.loads(report_path.read_text(encoding="utf-8"))
        if not isinstance(committed_report, dict) or committed_report != actual_report:
            raise RuntimeError("committed report does not match engine result")
        output_wav, trace_path = target / "processed.wav", target / "long_trace.jsonl"
        decoded_input = target / "decoded_input.wav" if mode == "compressed" else None
        if not output_wav.is_file() or not trace_path.is_file() or (
            decoded_input is not None and not decoded_input.is_file()
        ):
            raise RuntimeError("committed processing artifacts are incomplete")
        return ProcessResult(output_wav, report_path, committed_report, trace_path, decoded_input)
    except HumDSPError:
        raise
    except FileExistsError as error:
        raise HumDSPError("output_exists", "Choose a new output directory; existing results are never replaced.") from error
    except ImportError as error:
        raise HumDSPError("dependency_unavailable", "A processing dependency is unavailable; check this installation.") from error
    except (OSError, RuntimeError, ValueError) as error:
        # ValueError also denotes changed inputs, sample/intermediate range
        # refusals and engine invariants, not just an unsupported file header.
        raise HumDSPError(
            "processing_failed",
            "Processing was refused or could not complete; inspect the chained exception locally. "
            "Check sample-range limits and concurrent file changes as well as the selected mode.",
        ) from error
