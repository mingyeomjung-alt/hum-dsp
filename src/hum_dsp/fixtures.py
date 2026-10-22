"""Owned synthetic development fixtures, not human recordings or model data."""

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath

import numpy as np

from .audio import PCM16_STEP, SAMPLE_RATE_HZ, read_pcm16_wav, validate_audio, write_pcm16_wav
from .config import ProcessorConfig


@dataclass(frozen=True)
class FixtureSpec:
    case_id: str
    nominal_hum_hz: float
    kind: str
    expected_use: str
    duration_seconds: float = 6.0
    clean_kind: str = "program"
    noise_kind: str = "none"
    direction: int = 1


@dataclass(frozen=True)
class Fixture:
    spec: FixtureSpec
    clean: np.ndarray
    noise: np.ndarray
    mixed: np.ndarray
    instantaneous_noise_hz: np.ndarray | None


def fixture_specs() -> tuple[FixtureSpec, ...]:
    cases = [
        FixtureSpec("synthetic_silence", 50.0, "negative_control", "No sound; no dB improvement is defined.", clean_kind="silence"),
        FixtureSpec("synthetic_clean_program", 50.0, "negative_control", "Desired synthetic multitone only; inspect unintended changes."),
    ]
    for nominal in (50.0, 60.0):
        suffix = str(int(nominal))
        cases.extend([
            FixtureSpec(f"synthetic_fixed_{suffix}_noise_only", nominal, "stationary_target", "Measure selected stationary tone attenuation.", clean_kind="silence", noise_kind="fixed"),
            FixtureSpec(f"synthetic_fixed_{suffix}_mixture", nominal, "stationary_mixture", "Measure raw restoration error and identifiable component diagnostics.", noise_kind="fixed"),
            FixtureSpec(f"synthetic_drift_{suffix}_noise_only", nominal, "moving_target", "Full +/-1 Hz sweep; compare the same-depth, same-bandwidth fixed baseline.", duration_seconds=44.0, clean_kind="silence", noise_kind="drift", direction=1 if nominal == 50.0 else -1),
            FixtureSpec(f"synthetic_drift_{suffix}_mixture", nominal, "moving_mixture", "Full sweep mixed with desired synthetic multitone.", duration_seconds=44.0, noise_kind="drift", direction=1 if nominal == 50.0 else -1),
            FixtureSpec(f"synthetic_nearby_desired_{suffix}", nominal, "negative_control", "Desired tone 3 Hz above target; not ground-truth noise.", clean_kind="nearby"),
            FixtureSpec(f"synthetic_same_frequency_desired_{suffix}", nominal, "semantic_ambiguity", "Same waveform as target-only case, but labeled desired: automatic semantic separation is not claimed.", clean_kind="target_tone"),
            FixtureSpec(f"synthetic_collinear_{suffix}_mixture", nominal, "component_ambiguity", "Desired and noise references are proportional; separate fitted gains must be withheld.", clean_kind="target_tone", noise_kind="fixed"),
        ])
    return tuple(cases)


def noise_frequency_and_phase(spec: FixtureSpec, time: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if spec.noise_kind == "drift":
        ramp_time = np.clip(time - 2.0, 0.0, 40.0)
        tail_time = np.maximum(time - 42.0, 0.0)
        start_hz = spec.nominal_hum_hz - float(spec.direction)
        rate = float(spec.direction) * 0.05
        frequency = start_hz + rate * ramp_time
        # Integrate the 2s hold + linear 40s ramp + final 2s hold.
        cycles = start_hz * time + 0.5 * rate * ramp_time**2 + rate * 40.0 * tail_time
    else:
        frequency = np.full(time.shape, spec.nominal_hum_hz)
        cycles = spec.nominal_hum_hz * time
    return frequency, 2.0 * np.pi * cycles + 0.37


def make_fixture(spec: FixtureSpec) -> Fixture:
    if spec not in fixture_specs():
        raise ValueError("fixture must be one of the frozen M1 specifications")
    samples = round(spec.duration_seconds * SAMPLE_RATE_HZ)
    time = np.arange(samples, dtype=np.float64) / SAMPLE_RATE_HZ
    if spec.clean_kind == "silence":
        clean = np.zeros(samples)
    elif spec.clean_kind == "program":
        envelope = 0.75 + 0.25 * np.sin(2.0 * np.pi * 1.7 * time)
        clean = envelope * sum(
            0.035 * np.sin(2.0 * np.pi * frequency * time + index * 0.23)
            for index, frequency in enumerate((173.0, 317.0, 641.0, 1193.0))
        )
    elif spec.clean_kind == "nearby":
        clean = np.sqrt(2.0) * 0.02 * np.sin(2.0 * np.pi * (spec.nominal_hum_hz + 3.0) * time + 0.37)
    elif spec.clean_kind == "target_tone":
        # Match the exact arithmetic used for a fixed noise tone, not merely
        # its mathematical expression: ambiguity controls must be identical.
        clean = np.sqrt(2.0) * 0.02 * np.sin(2.0 * np.pi * (spec.nominal_hum_hz * time) + 0.37)
    else:
        raise ValueError("unknown clean fixture kind")
    if spec.noise_kind == "none":
        noise = np.zeros(samples)
        frequency = None
    else:
        frequency, phase = noise_frequency_and_phase(spec, time)
        amplitude = 0.01 if spec.kind == "component_ambiguity" else 0.02
        noise = np.sqrt(2.0) * amplitude * np.sin(phase)
    clean = validate_audio(clean)
    noise = validate_audio(noise)
    mixed = validate_audio(clean + noise)
    for array in (clean, noise, mixed, frequency):
        if array is not None:
            array.setflags(write=False)
    return Fixture(spec, clean, noise, mixed, frequency)


def _dump_json(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as destination:
        json.dump(value, destination, ensure_ascii=False, indent=2, allow_nan=False, sort_keys=True)
        destination.write("\n")


def export_fixture_suite(output: str | Path) -> dict:
    """Create a new local suite. Existing output directories are never reused."""
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=False)
    protocol_path = Path(__file__).with_name("protocol.json")
    protocol_bytes = protocol_path.read_bytes()
    protocol = json.loads(protocol_bytes)
    case_records = []
    for spec in fixture_specs():
        fixture = make_fixture(spec)
        case_directory = destination / spec.case_id
        case_directory.mkdir()
        for name, array in (("clean", fixture.clean), ("noise", fixture.noise), ("input", fixture.mixed)):
            write_pcm16_wav(case_directory / f"{name}.wav", array)
        decoded_clean = read_pcm16_wav(case_directory / "clean.wav")
        decoded_noise = read_pcm16_wav(case_directory / "noise.wav")
        decoded_input = read_pcm16_wav(case_directory / "input.wav")
        sum_error = float(np.max(np.abs(decoded_input - decoded_clean - decoded_noise)))
        if sum_error > 1.5 * PCM16_STEP:
            raise ValueError("fixture failed the declared quantization relationship")
        ground_truth = {
            "schema_version": 1,
            "spec": asdict(spec),
            "source": "owned_synthetic_math_not_real_speech_or_music",
            "evaluation_role": "development_not_independent_validation",
            "sample_rate_hz": SAMPLE_RATE_HZ,
            "frames": int(fixture.mixed.size),
            "peak_before_pcm16": float(np.max(np.abs(fixture.mixed))),
            "noise_hz_min": None if fixture.instantaneous_noise_hz is None else float(np.min(fixture.instantaneous_noise_hz)),
            "noise_hz_max": None if fixture.instantaneous_noise_hz is None else float(np.max(fixture.instantaneous_noise_hz)),
            "pcm16_sum_error_max": sum_error,
            "pcm16_sum_error_bound": 1.5 * PCM16_STEP,
            "measurement_noise_reference": "decoded_input_minus_decoded_clean",
            "noise_wav_role": "intended_noise_quantized_separately_not_exact_decoded_sum_reference",
            "processor_applied": False,
        }
        _dump_json(case_directory / "truth.json", ground_truth)
        files = []
        for file_path in sorted(case_directory.iterdir()):
            contents = file_path.read_bytes()
            files.append({"path": file_path.relative_to(destination).as_posix(), "bytes": len(contents), "sha256": hashlib.sha256(contents).hexdigest()})
        case_records.append({"case_id": spec.case_id, "kind": spec.kind, "frames": int(fixture.mixed.size), "files": files})
    manifest = {
        "schema_version": 2,
        "project": "hum_dsp",
        "milestone": "M1",
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
        "default_configuration": ProcessorConfig().to_dict(),
        "case_count": len(case_records),
        "cases": case_records,
        "scope_notice": "Synthetic development inputs only. Generation does not process audio or measure noise reduction, human-audio quality, or real-time device performance.",
        "processor_applied": False,
        "implementation_status_reference": "project_milestones_not_fixture_manifest",
    }
    _dump_json(destination / "protocol.json", protocol)
    _dump_json(destination / "manifest.json", manifest)
    return manifest


def verify_fixture_suite(output: str | Path) -> dict:
    """Verify suite contents against the current M1 contract, not engine quality."""
    directory = Path(output).resolve(strict=True)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    source_protocol = Path(__file__).with_name("protocol.json").read_bytes()
    if manifest.get("protocol_sha256") != hashlib.sha256(source_protocol).hexdigest():
        raise ValueError("fixture suite belongs to a different or modified protocol")
    if json.loads((directory / "protocol.json").read_text(encoding="utf-8")) != json.loads(source_protocol):
        raise ValueError("copied protocol differs from the package protocol")
    specs = {spec.case_id: spec for spec in fixture_specs()}
    if manifest.get("case_count") != len(specs) or len(manifest.get("cases", [])) != len(specs):
        raise ValueError("fixture case count does not match the M1 contract")
    seen = set()
    verified_files = 0
    largest_sum_error = 0.0
    for case in manifest["cases"]:
        case_id = case.get("case_id")
        if case_id not in specs or case_id in seen:
            raise ValueError("unknown or duplicated fixture case")
        seen.add(case_id)
        expected_files = {f"{case_id}/{name}" for name in ("clean.wav", "noise.wav", "input.wav", "truth.json")}
        records = case.get("files", [])
        if len(records) != 4 or {record["path"] for record in records} != expected_files:
            raise ValueError("unexpected fixture file inventory")
        for record in records:
            relative = PurePosixPath(record["path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("fixture path must be local to the suite")
            path = (directory / relative).resolve(strict=True)
            if not path.is_relative_to(directory):
                raise ValueError("fixture path escapes its suite")
            contents = path.read_bytes()
            if len(contents) != record["bytes"] or hashlib.sha256(contents).hexdigest() != record["sha256"]:
                raise ValueError(f"fixture integrity mismatch: {record['path']}")
            verified_files += 1
        case_directory = directory / case_id
        truth = json.loads((case_directory / "truth.json").read_text(encoding="utf-8"))
        if truth.get("spec") != asdict(specs[case_id]) or truth.get("processor_applied") is not False:
            raise ValueError("fixture truth does not match its frozen mathematical specification")
        clean, noise, mixed = [read_pcm16_wav(case_directory / f"{name}.wav") for name in ("clean", "noise", "input")]
        expected_frames = round(specs[case_id].duration_seconds * SAMPLE_RATE_HZ)
        if any(array.size != expected_frames for array in (clean, noise, mixed)):
            raise ValueError("fixture duration differs from its mathematical specification")
        sum_error = float(np.max(np.abs(mixed - clean - noise)))
        largest_sum_error = max(largest_sum_error, sum_error)
        if sum_error > 1.5 * PCM16_STEP:
            raise ValueError("fixture PCM16 sum relationship failed")
    return {
        "schema_version": 2,
        "status": "verified_synthetic_development_fixtures_only",
        "verification_scope": "M1_fixture_integrity_only_not_current_project_implementation_status",
        "protocol_id": manifest["protocol_id"],
        "case_count": len(seen),
        "verified_case_files": verified_files,
        "maximum_pcm16_sum_error": largest_sum_error,
        "pcm16_sum_error_bound": 1.5 * PCM16_STEP,
        "processor_applied": False,
        "noise_reduction_performance_measured": False,
    }

