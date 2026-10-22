"""H2b contracts use owned synthetic fixtures, not new quality evidence."""

import hashlib
import json
import wave

import numpy as np
import pytest

from hum_dsp.audio import read_pcm16_wav, write_pcm16_wav
from hum_dsp import recorded_hum_evaluation as study


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def recording(tmp_path):
    path = tmp_path / "owned.wav"
    t = np.arange(1200) / 48000
    write_pcm16_wav(path, 0.04 * np.sin(2 * np.pi * 50 * t))
    metadata = {"schema_version": 1, "source_id": "owned-test-not-field-audio", "title": "Owned fixture",
                "creator": "test", "source_page_url": "https://example.com/source",
                "download_url": "https://example.com/owned.wav", "license": "CC0-1.0",
                "license_url": "https://creativecommons.org/publicdomain/zero/1.0/",
                "local_file_name": path.name, "sha256": _sha(path), "acquisition_date": "2026-10-21",
                "representation": "original_recording", "recording_description": "Owned synthetic fixture, not a real recording",
                "nominal_hum_hz": None, "limitations": ["Test fixture only"]}
    provenance = tmp_path / "provenance.json"
    provenance.write_text(json.dumps(metadata), encoding="utf-8")
    return path, provenance, metadata


def test_provenance_binds_actual_hash_and_keeps_preview_limitations(recording):
    path, provenance, metadata = recording
    metadata["representation"] = "public_mp3_preview"
    metadata["limitations"].append("Lossy preview, not the original PCM recording")
    provenance.write_text(json.dumps(metadata), encoding="utf-8")
    assert study._provenance(path, provenance) == metadata


@pytest.mark.parametrize("key,value", [
    ("schema_version", True), ("license", "research-only"), ("title", ""),
    ("recording_description", None), ("local_file_name", "../owned.wav"),
    ("sha256", "0" * 64), ("representation", "unknown"), ("limitations", "none"),
    ("nominal_hum_hz", 70), ("nominal_hum_hz", True),
    ("source_page_url", "file:///private"), ("download_url", "https://user:secret@example.com/audio"),
])
def test_provenance_fails_closed(recording, key, value):
    path, provenance, metadata = recording
    metadata[key] = value
    provenance.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError):
        study._provenance(path, provenance)


def test_provenance_cannot_trust_metadata_without_actual_bytes(recording):
    path, provenance, _ = recording
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="hash"):
        study._provenance(path, provenance)


def test_provenance_rejects_oversized_or_empty_file(recording, monkeypatch):
    path, provenance, _ = recording
    monkeypatch.setattr(study, "MAX_NOISE_BYTES", 1)
    with pytest.raises(ValueError, match="16 MiB"):
        study._provenance(path, provenance)
    path.write_bytes(b"")
    with pytest.raises(ValueError, match="16 MiB"):
        study._provenance(path, provenance)


def test_decode_arithmetic_mono_resamples_without_normalizing_or_repetition(tmp_path):
    path = tmp_path / "stereo.wav"
    # Opposed channels must cancel rather than selecting the left channel.
    codes = np.column_stack((np.full(4410, 2000), np.full(4410, -2000))).astype("<i2")
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(44100)
        wav.writeframes(codes.tobytes())
    samples, preparation = study._decode_noise(path)
    assert samples.size == 4800
    assert np.array_equal(samples, np.zeros(4800))
    assert preparation["decoded_frames"] == 4410
    assert preparation["decoded_channels"] == 2
    assert preparation["fixed_gain"] == 0.25
    assert preparation["peak_normalization"] is False
    assert preparation["repetition"] is False
    assert preparation["time_stretch"] is False


def test_decode_takes_first_eight_seconds_without_repeating(tmp_path):
    path = tmp_path / "nine.wav"
    write_pcm16_wav(path, np.full(9 * 48000, 0.1))
    samples, preparation = study._decode_noise(path)
    assert samples.size == 8 * 48000
    assert preparation["decoded_frames"] == 9 * 48000
    assert preparation["selected_source_frames"] == 8 * 48000
    assert np.all(samples == read_pcm16_wav(path)[:samples.size] * 0.25)


def test_decode_invalid_media_is_a_clear_error(tmp_path):
    path = tmp_path / "invalid.mp3"
    path.write_bytes(b"not audio")
    with pytest.raises(ValueError, match="could not be decoded"):
        study._decode_noise(path)


@pytest.mark.parametrize("count", [100, 96000, 96001])
def test_windows_retain_startup_and_post_boundary_without_claiming_settled(count):
    t = np.arange(count) / 48000
    source = 0.1 * np.sin(2 * np.pi * 50 * t)
    result = study._windows(source, source * 0.8, 50)
    assert result["startup"]["sample_count"] == min(count, 96000)
    assert result["post_two_seconds_not_guaranteed_settled"] is True
    if count <= 96000:
        assert result["post_two_seconds"] is None
        assert result["post_two_seconds_unavailable_reason"]
    else:
        assert result["post_two_seconds"]["sample_count"] == 1
        assert result["post_two_seconds_unavailable_reason"] is None
    assert "known_component_diagnostic" not in result["whole_record"]
    assert "not_true_noise_reduction" in result["whole_record"]["interpretation"]


def test_known_component_metrics_are_not_natural_recording_oracles():
    t = np.arange(2400) / 48000
    desired = 0.05 * np.sin(2 * np.pi * 440 * t)
    noise = 0.02 * np.sin(2 * np.pi * 50 * t)
    output = desired * 0.9 + noise * 0.5
    result = study._windows(desired + noise, output, 50, (desired, noise, desired * 0.9, noise * 0.5))
    metric = result["whole_record"]["known_component_diagnostic"]
    assert metric["component_interpretation_valid"] is True
    assert metric["noise_attenuation_db"] == pytest.approx(20 * np.log10(2))
    assert metric["clean_relative_error_rmse"] == pytest.approx(0.1)
    assert "before_PCM16_not_field_or_perceptual_quality" in metric["interpretation"]


@pytest.fixture
def prepared_run(recording, tmp_path, monkeypatch):
    noise_path, provenance_path, _ = recording
    piano_path = tmp_path / "piano.wav"
    t = np.arange(1200) / 48000
    write_pcm16_wav(piano_path, 0.03 * np.sin(2 * np.pi * 440 * t))
    original = tmp_path / "original.bin"
    original.write_bytes(b"owned fixture")
    item = {"case_id": "piano", "prepared": {"sha256": _sha(piano_path)},
            "original": {"local_path": str(original), "sha256": _sha(original)}}
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text('{"owned_test_only":true}', encoding="utf-8")
    monkeypatch.setattr(study, "_catalog", lambda *_: ({"owned_test_only": True}, [(item, piano_path, read_pcm16_wav(piano_path))]))
    target = tmp_path / "new-study"
    return catalog_path, noise_path, provenance_path, target


def test_full_tiny_study_uses_real_frozen_engine_and_preserves_inputs(prepared_run):
    catalog, noise, provenance, target = prepared_run
    original_hashes = {path: _sha(path) for path in (catalog, noise, provenance)}
    result = study.run_recorded_hum_evaluation(*prepared_run)
    assert result["case_count"] == 6
    assert result["render_count"] == 18
    assert result["all_declared_structural_checks_met"] is True
    assert result["source_files_unchanged"] is True
    assert result["audio_played"] is False
    assert result["device_io"] is False
    assert result["human_review_completed"] is False
    assert result["commercial_release_approved"] is False
    assert result["independent_perceptual_or_field_quality_verified"] is False
    assert result["policy"]["settings"] == {"full": [12, 12, 12], "gentle": [6, 3, 0], "bypass": [0, 0, 0]}
    assert result["constructed_mixture_noise_rms_before_PCM16"] == pytest.approx(0.02)
    for case in result["cases"]:
        assert case["frames"] == 1200
        assert case["independent_clean_reference"] is False
        for record in case["settings"].values():
            assert record["checks_met"] is True
            metric = record["measurements"]["whole_record"]
            assert ("known_component_diagnostic" in metric) == (case["kind"] != "recorded_hum_only")
        folder = target / case["case_id"]
        assert (folder / "input.wav").read_bytes() == (folder / "bypass.wav").read_bytes()
    assert {path: _sha(path) for path in original_hashes} == original_hashes
    manifest = json.loads((target / "review_manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["cases"]) == 6
    for case in manifest["cases"]:
        assert [variant["label"] for variant in case["variants"]] == ["full", "gentle", "bypass"]
        assert case["reference"]["label"] == "input"
        for asset in (case["reference"], *case["variants"]):
            assert asset["sha256"] == _sha(target / asset["file_name"])
            assert "\\" not in asset["file_name"]
    report = (target / "report.html").read_text(encoding="utf-8")
    assert "NOT proof of noise removal" in report
    assert "not guaranteed settled" in report
    assert "No listening or commercial quality approval" in report
    assert "autoplay" not in report


def test_study_does_not_overwrite_existing_directory(prepared_run):
    *_, target = prepared_run
    target.mkdir()
    protected = target / "keep.txt"
    protected.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        study.run_recorded_hum_evaluation(*prepared_run)
    assert protected.read_text(encoding="utf-8") == "keep"
    assert len(list(target.iterdir())) == 1


def test_silent_noise_preflight_fails_before_output_creation(prepared_run, monkeypatch):
    *_, target = prepared_run
    monkeypatch.setattr(study, "_decode_noise", lambda *_: (np.zeros(100), {}))
    with pytest.raises(ValueError, match="silent"):
        study.run_recorded_hum_evaluation(*prepared_run)
    assert not target.exists()


def test_wanted_reference_metrics_retain_additivity_failure():
    source = np.full(40, 0.1)
    zero = np.zeros(40)
    result = study._windows(source, source + 0.01, 50, (source, zero, source, zero))
    assert result["whole_record"]["known_component_diagnostic"]["component_interpretation_valid"] is False


@pytest.mark.parametrize("changed_file", ["noise", "provenance", "catalog"])
def test_preflight_rejects_source_metadata_mutation(prepared_run, monkeypatch, changed_file):
    catalog, noise, provenance, target = prepared_run
    real_decode = study._decode_noise

    def mutate_during_decode(*args):
        decoded = real_decode(*args)
        changed = {"noise": noise, "provenance": provenance, "catalog": catalog}[changed_file]
        changed.write_bytes(changed.read_bytes() + b" ")
        return decoded

    monkeypatch.setattr(study, "_decode_noise", mutate_during_decode)
    with pytest.raises(ValueError, match="changed during preflight"):
        study.run_recorded_hum_evaluation(*prepared_run)
    assert not target.exists()


def test_reported_source_digests_bind_initial_metadata_snapshots(prepared_run):
    catalog, noise, provenance, _ = prepared_run
    initial = {"source_catalog_sha256": _sha(catalog), "recorded_noise_sha256": _sha(noise),
               "provenance_sha256": _sha(provenance)}
    result = study.run_recorded_hum_evaluation(*prepared_run)
    assert {key: result[key] for key in initial} == initial
