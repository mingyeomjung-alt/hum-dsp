"""Evaluation-only recorded-hum diagnostics; never a perceptual quality gate.

Natural recording observations have no independent clean reference. Constructed
mixtures have known, separately recorded components, not real-world clean/noisy
pairs. Decisions are frozen from each mixture and replayed without refitting.
"""

import hashlib
import html
import json
import math
from pathlib import Path
from urllib.parse import urlparse

import numpy as np

from .audio import read_pcm16_wav, validate_audio, write_pcm16_wav
from .band_control import BandControlConfig, render_band_control, render_band_control_replay
from .band_evaluation import _catalog
from .fixtures import _dump_json
from .harmonic_metrics import measure_harmonic_components


RATE = 48000
BOUNDARY = 96000
MAX_NOISE_BYTES = 16 * 1024 * 1024


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _policy():
    return {"sample_rate_hz": RATE, "maximum_seconds": 8,
            "settings": {"full": [12, 12, 12], "gentle": [6, 3, 0], "bypass": [0, 0, 0]},
            "nominal_hum_hz": [50, 60], "natural_recording_gain": 0.25,
            "constructed_noise_rms": 0.02, "startup_end_sample": BOUNDARY,
            "post_two_seconds_start_sample": BOUNDARY,
            "selection": "both_nominals_declared_explorations_no_best_setting_selection",
            "adoption_rule": "structural_checks_only_no_quality_pass_or_threshold_tuning"}


def _provenance(noise_path, provenance_path, *, serialized=None):
    path = Path(noise_path)
    if not path.is_file() or not 0 < path.stat().st_size <= MAX_NOISE_BYTES:
        raise ValueError("recorded noise must be a nonempty file of at most 16 MiB")
    data = json.loads(Path(provenance_path).read_bytes() if serialized is None else serialized)
    strings = ("source_id", "title", "creator", "source_page_url", "download_url", "license_url",
               "local_file_name", "sha256", "acquisition_date", "representation", "recording_description")
    if (type(data) is not dict or type(data.get("schema_version")) is not int
            or data["schema_version"] != 1 or data.get("license") != "CC0-1.0"
            or any(not isinstance(data.get(k), str) or not data[k].strip() for k in strings)
            or data["local_file_name"] != path.name
            or data["sha256"] != _sha(path)
            or data["representation"] not in ("public_mp3_preview", "original_recording")
            or type(data.get("limitations")) is not list
            or any(not isinstance(v, str) for v in data["limitations"])
            or data.get("nominal_hum_hz") not in (None, 50, 60)
            or isinstance(data.get("nominal_hum_hz"), bool)):
        raise ValueError("recorded noise provenance, CC0 declaration or actual file hash is invalid")
    for key in ("source_page_url", "download_url", "license_url"):
        parsed = urlparse(data[key])
        if parsed.scheme not in ("https", "http") or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError("source provenance URLs must be public HTTP(S) references")
    return data


def _decode_noise(path, representation="original_recording"):
    """Bounded local decode, arithmetic channel mean, fixed SciPy conversion.

    PyAV is used only by this development study; this adds no product formats.
    The full (at most 60 s) recording is validated before taking its first 8 s.
    """
    import av
    import scipy
    from scipy.signal import resample_poly

    if representation not in ("public_mp3_preview", "original_recording"):
        raise ValueError("recording representation must select MP3 preview or original WAV")
    source = Path(path)
    if not source.is_file() or not 0 < source.stat().st_size <= MAX_NOISE_BYTES:
        raise ValueError("recorded noise must be a nonempty file of at most 16 MiB")
    chunks, rate, channels, count = [], None, None, 0
    try:
        # A binary file object plus a forced elementary format excludes playlist
        # interpretation and external-path/protocol references from media input.
        with source.open("rb") as raw, av.open(raw, mode="r", format="mp3" if representation == "public_mp3_preview" else "wav") as container:
            if len(container.streams.audio) != 1:
                raise ValueError("evaluation recording must have exactly one audio stream")
            converter = None
            for frame in container.decode(container.streams.audio[0]):
                current_rate, current_channels = frame.sample_rate, len(frame.layout.channels)
                if (not 8000 <= current_rate <= 192000 or current_channels not in (1, 2)
                        or (rate is not None and (rate, channels) != (current_rate, current_channels))):
                    raise ValueError("recording rate or channel layout unsupported or changed")
                if converter is None:
                    rate, channels = current_rate, current_channels
                    converter = av.AudioResampler(format="dblp", layout=frame.layout, rate=rate)
                for converted in converter.resample(frame):
                    array = converted.to_ndarray()
                    if array.shape != (channels, converted.samples) or not np.isfinite(array).all():
                        raise ValueError("recording decoder produced invalid or nonfinite samples")
                    count += converted.samples
                    if count > rate * 60:
                        raise ValueError("evaluation recording exceeds 60 seconds")
                    chunks.append(array.mean(axis=0))
            if not chunks:
                raise ValueError("evaluation recording contains no decoded samples")
    except (av.error.FFmpegError, OSError) as exc:
        raise ValueError("recorded noise could not be decoded") from exc
    decoded = np.concatenate(chunks)
    first = decoded[:rate * 8]
    divisor = math.gcd(rate, RATE)
    converted = resample_poly(first, RATE // divisor, rate // divisor, window=("kaiser", 5.0), padtype="constant") if rate != RATE else first.copy()
    result = validate_audio(converted * 0.25)
    return result, {"decoder": "PyAV", "decoder_version": av.__version__, "numpy_version": np.__version__,
                    "scipy_version": scipy.__version__, "forced_decoder_format": "mp3" if representation == "public_mp3_preview" else "wav",
                    "decoded_sample_rate_hz": rate,
                    "decoded_channels": channels, "decoded_frames": count,
                    "selected_source_frames": int(first.size), "prepared_frames": int(result.size),
                    "channel_mix": "arithmetic_mean", "fixed_gain": 0.25,
                    "resampling": "scipy_resample_poly_kaiser_beta5_constant_zero_edges" if rate != RATE else "none",
                    "peak_normalization": False, "repetition": False, "time_stretch": False}


def _rms(array):
    return float(np.sqrt(np.mean(array * array)))


def _db(a, b):
    return float(20 * np.log10(a / b)) if a > 0 and b > 0 else None


def _observed(source, output, nominal):
    """Raw synchronous projection, not noise or desired-content estimation."""
    frequencies = []
    t = np.arange(source.size, dtype=np.float64) / RATE
    for frequency in (nominal, nominal * 2, nominal * 3):
        basis = np.column_stack((np.sin(2 * np.pi * frequency * t), np.cos(2 * np.pi * frequency * t)))
        before = _rms(basis @ np.linalg.lstsq(basis, source, rcond=None)[0])
        after = _rms(basis @ np.linalg.lstsq(basis, output, rcond=None)[0])
        frequencies.append({"frequency_hz": frequency, "input_projection_rms": before,
                            "output_projection_rms": after, "output_to_input_projection_db": _db(after, before)})
    return {"input_rms": _rms(source), "output_rms": _rms(output),
            "output_to_input_rms_db": _db(_rms(output), _rms(source)),
            "raw_output_change_rmse": _rms(output - source), "selected_frequency_projections": frequencies,
            "interpretation": "observed_signal_change_only_not_true_noise_reduction_or_wanted_preservation"}


def _windows(source, output, nominal, components=None):
    result = {"post_two_seconds_unavailable_reason": "source_not_longer_than_two_seconds_no_repetition" if source.size <= BOUNDARY else None,
              "post_two_seconds_not_guaranteed_settled": True}
    for name, selected in (("whole_record", slice(None)), ("startup", slice(0, BOUNDARY)), ("post_two_seconds", slice(BOUNDARY, None))):
        if name == "post_two_seconds" and source.size <= BOUNDARY:
            result[name] = None
            continue
        observation = _observed(source[selected], output[selected], nominal)
        observation["sample_count"] = int(source[selected].size)
        if components is not None:
            desired, noise, replay_desired, replay_noise = (value[selected] for value in components)
            metric = measure_harmonic_components(desired, noise, output[selected], replay_desired, replay_noise)
            metric["interpretation"] = "known_constructed_recorded_components_frozen_replay_before_PCM16_not_field_or_perceptual_quality"
            observation["known_component_diagnostic"] = metric
        result[name] = observation
    return result


def _asset(folder, root, label):
    file = folder / (label + ".wav")
    return {"label": label, "file_name": file.relative_to(root).as_posix(), "sha256": _sha(file)}


def _report(result, path):
    rows = []
    for case in result["cases"]:
        for label, record in case["settings"].items():
            metric = record["measurements"]["post_two_seconds"] or record["measurements"]["whole_record"]
            known = metric.get("known_component_diagnostic", {})
            values = (case["case_id"], label, known.get("noise_attenuation_db"), known.get("clean_relative_error_rmse"), record["checks_met"])
            rows.append("<tr>" + "".join("<td>" + html.escape(str(v)) + "</td>" for v in values) + "</tr>")
    path.write_text("""<!doctype html><html lang="en"><meta charset="utf-8"><title>H2b recorded-hum study</title>
<style>body{font:16px system-ui;max-width:1100px;margin:40px auto;padding:20px}td,th{padding:10px;border:1px solid #ccc}table{border-collapse:collapse}p{line-height:1.6}</style>
<h1>Recorded hum: observations and constructed-mixture diagnostics</h1>
<p>Natural recording: no independent clean reference, so signal changes are NOT proof of noise removal or wanted-content preservation. Constructed piano + recorded-noise mixtures have known components, but are not field clean/noisy pairs. The piano reference may already contain background noise.</p>
<p>Both nominal 50 and 60 Hz, full 12/12/12 dB, gentle 6/3/0 dB and bypass were fixed before this run. No best setting is automatically selected. Post-two-seconds windows are not guaranteed settled. Component metrics are floating replay diagnostics before PCM16 rounding, not decompositions of downloadable files. No listening or commercial quality approval has occurred.</p>
<p><a href="evaluation.json">Measurements / provenance</a> · <a href="review_manifest.json">Manual review manifest</a></p>
<table><tr><th>Case</th><th>Setting</th><th>Known-component attenuation dB</th><th>Desired relative RMSE</th><th>Structural checks only</th></tr>""" + "".join(rows) + "</table></html>", encoding="utf-8")


def run_recorded_hum_evaluation(catalog_path, noise_path, provenance_path, output_directory):
    target, source_root = Path(output_directory), Path(__file__).parent
    if target.exists():
        raise FileExistsError("evaluation output directory must be new")
    policy = _policy()
    provenance_bytes = Path(provenance_path).read_bytes()
    provenance_hash = hashlib.sha256(provenance_bytes).hexdigest()
    provenance = _provenance(noise_path, provenance_path, serialized=provenance_bytes)
    # The actual-media identity is the digest that was validated against the
    # snapshot, not a second, possibly changed read after provenance validation.
    original_noise_hash = provenance["sha256"]
    catalog_bytes = Path(catalog_path).read_bytes()
    catalog_hash = hashlib.sha256(catalog_bytes).hexdigest()
    existing_policy = json.loads((source_root / "h2_policy.json").read_text(encoding="utf-8"))
    catalog, sources = _catalog(catalog_path, existing_policy)
    if catalog != json.loads(catalog_bytes) or _sha(catalog_path) != catalog_hash:
        raise ValueError("prepared catalog changed during provenance validation")
    piano_item, piano_path, piano = next(source for source in sources if source[0]["case_id"] == "piano")
    natural, preparation = _decode_noise(noise_path, provenance["representation"])
    if (_sha(noise_path) != original_noise_hash or _sha(provenance_path) != provenance_hash
            or _sha(catalog_path) != catalog_hash):
        raise ValueError("source or provenance changed during preflight")
    count = min(piano.size, natural.size, 8 * RATE)
    desired, recorded = piano[:count].copy(), natural[:count].copy()
    noise_rms = _rms(recorded)
    if noise_rms <= 0:
        raise ValueError("recorded noise is silent; cannot construct declared RMS mixture")
    mixture_gain = 0.02 / noise_rms
    controlled_noise = validate_audio(recorded * mixture_gain)
    signals = {"recorded_hum_only": (recorded, None),
               "constructed_mixture": (validate_audio(desired + controlled_noise), desired),
               "desired_only": (desired, desired)}
    # Complete preflight before creating any output; existing artifacts untouched.
    target.mkdir(parents=True, exist_ok=False)
    _dump_json(target / "noise_provenance.json", provenance)
    _dump_json(target / "source_catalog.json", catalog)
    write_pcm16_wav(target / "prepared_recorded_noise.wav", recorded)
    cases, review_cases = [], []
    for kind, (unquantized, desired_reference) in signals.items():
        for nominal in policy["nominal_hum_hz"]:
            case_id = f"{kind}_{nominal}"
            folder = target / case_id
            folder.mkdir()
            write_pcm16_wav(folder / "input.wav", unquantized)
            mixture = read_pcm16_wav(folder / "input.wav")
            if desired_reference is not None:
                write_pcm16_wav(folder / "desired_reference.wav", desired_reference)
                noise = mixture - desired_reference
            records = {}
            for label, depths in policy["settings"].items():
                config = BandControlConfig(nominal_hum_hz=nominal, band_depths_db=depths)
                output, description, trace = render_band_control(mixture, config)
                output = validate_audio(output)
                replay = render_band_control_replay(mixture, config, trace)
                components = None
                if desired_reference is not None:
                    components = (desired_reference, noise,
                                  render_band_control_replay(desired_reference, config, trace),
                                  render_band_control_replay(noise, config, trace))
                windows = _windows(mixture, output, nominal, components)
                replay_error = float(np.max(np.abs(output - replay)))
                additive_ok = all(w["known_component_diagnostic"]["component_interpretation_valid"] for w in
                                  (windows["whole_record"], windows["startup"], windows["post_two_seconds"]) if w is not None) if components is not None else True
                write_pcm16_wav(folder / (label + ".wav"), output)
                decoded = read_pcm16_wav(folder / (label + ".wav"))
                bypass_ok = label != "bypass" or (np.array_equal(output, mixture) and description["shared_original_input_fft_count"] == 0 and (folder / "bypass.wav").read_bytes() == (folder / "input.wav").read_bytes())
                _dump_json(folder / (label + "_trace.json"), trace)
                records[label] = {"config": config.to_dict(), "descriptor": description,
                                  "measurements": windows, "output_sha256": _sha(folder / (label + ".wav")),
                                  "replay_mixture_max_abs_error": replay_error,
                                  "component_additivity_check": additive_ok,
                                  "bypass_check": bypass_ok,
                                  "decoded_PCM16_quantization_max_abs_error": float(np.max(np.abs(decoded - output))),
                                  "checks_met": replay_error <= 1e-10 and additive_ok and bypass_ok}
            context = "No independent clean reference; observations cannot establish true noise reduction or wanted preservation." if kind == "recorded_hum_only" else "Known constructed components; not a field clean/noisy pair or perceptual quality result."
            cases.append({"case_id": case_id, "kind": kind, "nominal_hum_hz": nominal,
                          "frames": count, "context": context,
                          "independent_clean_reference": False,
                          "known_components": desired_reference is not None, "settings": records})
            review_cases.append({"case_id": case_id, "title": kind.replace("_", " ").title() + f" / {nominal} Hz",
                                 "kind": kind, "context": context,
                                 "reference": _asset(folder, target, "input"),
                                 "variants": [_asset(folder, target, label) for label in policy["settings"]]})
    unchanged = (_sha(noise_path) == original_noise_hash and _sha(provenance_path) == provenance_hash
                 and _sha(catalog_path) == catalog_hash
                 and all(_sha(audio_path) == item["prepared"]["sha256"]
                         and _sha(item["original"]["local_path"]) == item["original"]["sha256"]
                         for item, audio_path, _ in sources))
    result = {"schema_version": 1, "stage": "H2b", "evaluation_only": True,
              "scope": "recorded_hum_observation_and_constructed_known_component_development_study",
              "policy": policy, "noise_provenance": provenance, "noise_preparation": preparation,
              "constructed_mixture_additional_noise_gain": mixture_gain,
              "constructed_mixture_noise_rms_before_PCM16": _rms(controlled_noise),
              "source_catalog_sha256": catalog_hash, "provenance_sha256": provenance_hash,
              "recorded_noise_sha256": original_noise_hash,
              "source_sha256": {name: _sha(source_root / name) for name in ("recorded_hum_evaluation.py", "band_control.py", "band_processing.py", "harmonics.py", "harmonic_metrics.py", "h2_policy.json")},
              "case_count": len(cases), "render_count": 18, "cases": cases,
              "source_files_unchanged": unchanged,
              "all_declared_structural_checks_met": unchanged and all(r["checks_met"] for c in cases for r in c["settings"].values()),
              "independent_perceptual_or_field_quality_verified": False, "human_review_completed": False,
              "commercial_release_approved": False, "audio_played": False, "device_io": False,
              "remaining_product_workstreams": 2}
    _dump_json(target / "evaluation.json", result)
    _dump_json(target / "review_manifest.json", {"schema_version": 1, "title": "H2b recorded-hum manual review",
               "description": "No audio plays automatically. Listening judgments have not been collected; measurements are not perceptual approval.", "cases": review_cases})
    _report(result, target / "report.html")
    return result
