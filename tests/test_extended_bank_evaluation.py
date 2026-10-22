"""H2c reporting contracts; synthetic fixtures are not performance evidence."""

import copy
import hashlib
import json

import numpy as np
import pytest

from hum_dsp.audio import read_pcm16_wav, write_pcm16_wav
from hum_dsp.band_control import BandControlConfig, render_band_control
from hum_dsp import extended_bank_evaluation as study


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record(noise=1.0, error=.01, level=.02, identity=True):
    return {"measurements": {"post_two_seconds": {"known_component_diagnostic": {
        "noise_attenuation_db": noise, "clean_relative_error_rmse": error,
        "clean_level_change_db": level}}}, "output_exact_input": identity}


def _decision_cases():
    cases = []
    for case_id in (*study.ACTUAL_CASES, "owned_extended_tones_50", "owned_extended_tones_60", "wanted_1260_hz_overlap"):
        actual_mixture = case_id == "constructed_mixture_60"
        synthetic = case_id.startswith("owned_")
        before = .43 if actual_mixture else 1.0
        after = .8 if actual_mixture else 5.0 if synthetic else 1.0
        cases.append({"case_id": case_id, "settings": {
            "baseline": _record(before), "extended": _record(after)}})
    cases[-1]["settings"]["extended"] = _record(0.0, .7, -11.5)
    return cases


def test_development_decision_is_not_default_adoption_or_independent_quality():
    result = study._development_decision(_decision_cases(), study._policy(), True)
    assert all(result["checks"].values())
    assert result["status"] == "development_targets_met_not_adopted"
    assert result["wanted_overlap_harm_diagnostic"]["clean_relative_error_rmse"] == .7
    assert "suppressed identically" in result["harm_warning"]
    assert result["independent_quality_claim"] is False
    assert result["default_engine_changed"] is False
    assert result["commercial_release_approved"] is False


@pytest.mark.parametrize("case_id,field,value", [
    ("constructed_mixture_60", "noise_attenuation_db", .67),
    ("constructed_mixture_60", "clean_relative_error_rmse", .021),
    ("constructed_mixture_60", "clean_level_change_db", -.101),
    ("constructed_mixture_60", "noise_attenuation_db", None),
    ("owned_extended_tones_50", "noise_attenuation_db", 3.9),
    ("owned_extended_tones_60", "noise_attenuation_db", 3.9),
])
def test_any_locked_target_failure_is_retained(case_id, field, value):
    cases = _decision_cases()
    selected = next(case for case in cases if case["case_id"] == case_id)
    selected["settings"]["extended"]["measurements"]["post_two_seconds"]["known_component_diagnostic"][field] = value
    result = study._development_decision(cases, study._policy(), True)
    assert not all(result["checks"].values())
    assert result["status"] == "not_adopted_development_targets_failed"


def test_clean_only_identity_and_structural_check_are_required():
    cases = _decision_cases()
    next(case for case in cases if case["case_id"] == "desired_only_50")["settings"]["extended"]["output_exact_input"] = False
    result = study._development_decision(cases, study._policy(), False)
    assert result["checks"]["actual_desired_only_exact_identity"] is False
    assert result["checks"]["structural_checks"] is False
    assert result["status"] == "not_adopted_development_targets_failed"


@pytest.mark.parametrize("count", [100, 96000, 96001])
def test_windows_keep_startup_and_raw_phase_error(count):
    source = .1 * np.sin(2 * np.pi * 50 * np.arange(count) / 48000)
    desired, noise = source * .8, source * .2
    result = study._windows(source, desired * .9 + noise * .5,
                            (desired, noise, desired * .9, noise * .5))
    assert result["post_two_seconds_not_guaranteed_settled"] is True
    assert result["startup"]["sample_count"] == min(count, 96000)
    assert (result["post_two_seconds"] is None) == (count <= 96000)
    known = result["whole_record"]["known_component_diagnostic"]
    assert known["noise_attenuation_db"] == pytest.approx(20 * np.log10(2))
    assert known["clean_relative_error_rmse"] == pytest.approx(.1)
    assert known["component_interpretation_valid"] is True
    assert "before_PCM16_not_field" in known["interpretation"]


def test_natural_recording_without_reference_has_no_true_noise_metric():
    source = np.full(100, .1)
    result = study._windows(source, source * .5)
    assert "known_component_diagnostic" not in result["whole_record"]
    assert "not_true_noise_reduction" in result["whole_record"]["interpretation"]


def test_owned_fixtures_match_locked_policy_without_clipping_or_repetition():
    policy = study._policy()
    cases = study._synthetic_cases(policy)
    assert len(cases) == 3
    assert [case["nominal_hum_hz"] for case in cases] == [50, 60, 60]
    assert all(case["input"].size == policy["owned_synthetic"]["seconds"] * 48000 for case in cases)
    assert all(np.max(np.abs(case["input"])) < 1 for case in cases)
    assert np.array_equal(cases[-1]["input"], cases[-1]["desired"])
    for case in cases[:2]:
        assert study._rms(case["input"] - case["desired"]) == pytest.approx(.012 * 2, rel=1e-12)


@pytest.fixture
def tiny_baseline(tmp_path, monkeypatch):
    root = tmp_path / "h2b"
    root.mkdir()
    t = np.arange(1200) / 48000
    desired = .03 * np.sin(2 * np.pi * 419 * t)
    cases = []
    for case_id in study.ACTUAL_CASES:
        folder = root / case_id
        folder.mkdir()
        nominal = int(case_id.rsplit("_", 1)[1])
        kind = case_id.rsplit("_", 1)[0]
        noise = .02 * np.sin(2 * np.pi * nominal * t)
        source = noise if kind == "recorded_hum_only" else desired if kind == "desired_only" else desired + noise
        write_pcm16_wav(folder / "input.wav", source)
        if kind != "recorded_hum_only":
            write_pcm16_wav(folder / "desired_reference.wav", desired)
        decoded = read_pcm16_wav(folder / "input.wav")
        baseline, _, _ = render_band_control(decoded, BandControlConfig(nominal_hum_hz=nominal, band_depths_db=(12, 12, 12)))
        write_pcm16_wav(folder / "full.wav", baseline)
        cases.append({"case_id": case_id, "kind": kind, "nominal_hum_hz": nominal,
                      "frames": 1200, "context": "Owned test fixture, not actual data",
                      "settings": {"full": {"output_sha256": _sha(folder / "full.wav")},
                                   "bypass": {"output_sha256": _sha(folder / "input.wav")}}})
    (root / "evaluation.json").write_text(json.dumps({"stage": "H2b", "commercial_release_approved": False, "cases": cases}), encoding="utf-8")
    for name in ("noise_provenance.json", "source_catalog.json"):
        (root / name).write_text('{"owned_test_fixture":true}', encoding="utf-8")
    policy = copy.deepcopy(study._policy())
    policy["owned_synthetic"]["seconds"] = .025
    monkeypatch.setattr(study, "_policy", lambda: policy)

    class Config:
        def __init__(self, **kwargs):
            self.values = kwargs

    def render(source, config):
        return source.copy(), {"owned_identity_fixture": True}, {"frames": source.size}

    def replay(source, config, trace):
        assert trace["frames"] == source.size
        return source.copy()

    monkeypatch.setattr(study, "_engine_api", lambda: (Config, render, replay))
    return root, tmp_path / "h2c"


def test_tiny_report_binds_sources_and_never_claims_quality(tiny_baseline):
    root, target = tiny_baseline
    before = {str(path): _sha(path) for path in root.rglob("*") if path.is_file()}
    result = study.run_extended_bank_evaluation(root, target)
    assert result["case_count"] == 9
    assert result["render_count"] == 18
    assert result["all_declared_structural_checks_met"] is True
    assert result["source_files_unchanged"] is True
    assert result["identities_before"] == result["identities_after"]
    assert result["development_decision"]["status"] == "not_adopted_development_targets_failed"
    for key in ("audio_played", "device_io", "human_review_completed", "independent_perceptual_or_field_quality_verified", "commercial_release_approved", "default_engine_changed"):
        assert result[key] is False
    for case in result["cases"]:
        assert case["frames"] == 1200
        assert case["independent_clean_field_reference"] is False
        for record in case["settings"].values():
            assert record["structural_checks_met"] is True
            assert record["retained_baseline_PCM16_match"] is True
    assert {str(path): _sha(path) for path in root.rglob("*") if path.is_file()} == before
    report = (target / "report.html").read_text(encoding="utf-8")
    assert "NOT an independent evaluation" in report
    assert "before PCM16 export" in report
    assert "No listening" in report
    assert "not guaranteed settled" in report
    assert "autoplay" not in report
    assert json.loads((target / "evaluation.json").read_text(encoding="utf-8"))["case_count"] == 9


def test_existing_output_is_not_changed(tiny_baseline):
    root, target = tiny_baseline
    target.mkdir()
    (target / "keep").write_bytes(b"keep")
    with pytest.raises(FileExistsError):
        study.run_extended_bank_evaluation(root, target)
    assert list(target.iterdir()) == [target / "keep"]
    assert (target / "keep").read_bytes() == b"keep"


@pytest.mark.parametrize("filename", ["input.wav", "full.wav"])
def test_preflight_rejects_mismatched_retained_asset_hash(tiny_baseline, filename):
    root, target = tiny_baseline
    path = root / "constructed_mixture_60" / filename
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="hash"):
        study.run_extended_bank_evaluation(root, target)
    assert not target.exists()


def test_preflight_rejects_reference_boundary_change(tiny_baseline):
    root, target = tiny_baseline
    reference = root / "constructed_mixture_60" / "desired_reference.wav"
    write_pcm16_wav(reference, np.zeros(50), overwrite=True)
    metadata_path = root / "evaluation.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    next(case for case in metadata["cases"] if case["case_id"] == "desired_only_60")["settings"]["bypass"]["output_sha256"] = _sha(reference)
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="PCM boundary"):
        study.run_extended_bank_evaluation(root, target)
    assert not target.exists()


def test_preflight_does_not_accept_changed_desired_reference(tiny_baseline):
    root, target = tiny_baseline
    reference = root / "constructed_mixture_60" / "desired_reference.wav"
    write_pcm16_wav(reference, np.zeros(1200), overwrite=True)
    with pytest.raises(ValueError, match="desired reference hash"):
        study.run_extended_bank_evaluation(root, target)
    assert not target.exists()


def test_baseline_metadata_identity_is_initial_snapshot(tiny_baseline, monkeypatch):
    root, target = tiny_baseline
    real_load = study._load_actual

    def change_after_parse(*args):
        result = real_load(*args)
        path = root / "evaluation.json"
        path.write_bytes(path.read_bytes() + b" ")
        return result

    monkeypatch.setattr(study, "_load_actual", change_after_parse)
    with pytest.raises(ValueError, match="changed during preflight"):
        study.run_extended_bank_evaluation(root, target)
    assert not target.exists()


def test_preflight_requires_all_six_fixed_actual_cases(tiny_baseline):
    root, target = tiny_baseline
    path = root / "evaluation.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata["cases"].pop()
    path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="six fixed"):
        study.run_extended_bank_evaluation(root, target)
    assert not target.exists()
