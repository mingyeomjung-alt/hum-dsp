"""H2d reporting/orchestration contracts, not independent performance evidence."""

import copy
import hashlib
import json

import numpy as np
import pytest

from hum_dsp.audio import read_pcm16_wav, write_pcm16_wav
from hum_dsp import profile_guard_evaluation as study


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record(noise=4.0, error=.01, level=.02, identity=True):
    diagnostic = {"noise_attenuation_db": noise, "clean_relative_error_rmse": error,
                  "clean_level_change_db": level}
    return {"measurements": {window: {"known_component_diagnostic": dict(diagnostic)}
                              for window in ("program", "post_four_seconds")},
            "output_exact_input": identity}


def _decision_cases():
    cases = []
    for case_id in study._policy()["case_ids"]:
        cases.append({"case_id": case_id, "settings": {"baseline": _record(), "profile_guard": _record()}})
    for case in cases:
        if case["case_id"] in study.LIMIT_CASES:
            case["settings"]["profile_guard"] = _record(error=.7, level=-10)
    return cases


def test_development_decision_does_not_hide_hard_limits_or_adopt_default():
    result = study._development_decision(_decision_cases(), study._policy(), True)
    assert all(result["checks"].values())
    assert result["status"] == "development_targets_met_not_adopted"
    assert result["hard_limit_diagnostics"]["contaminated_profile_limit"]["profile_guard"]["clean_relative_error_rmse"] == .7
    assert "growing_noise_60" in result["report_only_cases"]
    assert "recorded_profile_then_piano_60" in result["report_only_cases"]
    for key in ("independent_quality_claim", "default_engine_changed", "commercial_release_approved"):
        assert result[key] is False


@pytest.mark.parametrize("case_id", study.PROTECTED_CASES)
@pytest.mark.parametrize("window", ["program", "post_four_seconds"])
@pytest.mark.parametrize("field,value", [("clean_relative_error_rmse", .021), ("clean_level_change_db", -.101),
                                         ("clean_relative_error_rmse", None)])
def test_each_preregistered_wanted_failure_is_retained(case_id, window, field, value):
    cases = _decision_cases()
    case = next(case for case in cases if case["case_id"] == case_id)
    case["settings"]["profile_guard"]["measurements"][window]["known_component_diagnostic"][field] = value
    result = study._development_decision(cases, study._policy(), True)
    assert not all(result["checks"].values())
    assert result["status"] == "not_adopted_development_targets_failed"


@pytest.mark.parametrize("case_id", study.SELECTIVE_CASES)
@pytest.mark.parametrize("attenuation", [2.99, None])
def test_selective_noise_reduction_is_required_without_threshold_reselection(case_id, attenuation):
    cases = _decision_cases()
    case = next(case for case in cases if case["case_id"] == case_id)
    case["settings"]["profile_guard"]["measurements"]["post_four_seconds"]["known_component_diagnostic"]["noise_attenuation_db"] = attenuation
    result = study._development_decision(cases, study._policy(), True)
    assert result["status"] == "not_adopted_development_targets_failed"


def test_exact_empty_profile_identity_and_structural_gate_cannot_be_skipped():
    cases = _decision_cases()
    case = next(case for case in cases if case["case_id"] == "empty_profile_60")
    case["settings"]["profile_guard"]["output_exact_input"] = False
    result = study._development_decision(cases, study._policy(), False)
    assert result["checks"]["empty_profile_exact_identity"] is False
    assert result["checks"]["structural_checks"] is False


def test_windows_retain_full_program_and_late_onset_and_undefined_zero_wanted():
    count = study.FRAMES
    desired = np.zeros(count)
    desired[study.POST_FOUR:] = .1
    noise = np.full(count, .02)
    output = desired * .9 + noise * .5
    result = study._windows(desired + noise, output, (desired, noise, desired * .9, noise * .5), onset_sample=study.POST_FOUR)
    assert result["post_four_seconds_is_guaranteed_settled"] is False
    assert result["program"]["start_sample"] == 96000
    assert result["program"]["sample_count"] == 288000
    assert result["program_onset"]["sample_count"] == 24000
    assert result["declared_wanted_onset"]["start_sample"] == 192000
    assert result["declared_wanted_onset"]["end_sample_exclusive"] == 216000
    profile = result["profile_startup"]["known_component_diagnostic"]
    assert profile["clean_relative_error_rmse"] is None
    assert profile["undefined_metrics"]["clean_relative_error_rmse"] == "zero_clean_reference"
    known = result["program"]["known_component_diagnostic"]
    assert known["noise_attenuation_db"] == pytest.approx(20 * np.log10(2))
    assert known["clean_relative_error_rmse"] == pytest.approx(.1)
    assert known["component_interpretation_valid"] is True


def test_owned_cases_have_the_locked_prefix_and_expose_profile_contamination():
    cases = study._owned_cases(study._policy())
    assert len(cases) == 8
    assert [case["case_id"] for case in cases] == study._policy()["case_ids"][:-1]
    assert all(case["input"].size == 384000 for case in cases)
    assert all(np.max(np.abs(case["input"])) < 1 for case in cases)
    indexed = {case["case_id"]: case for case in cases}
    assert all(np.all(case["desired"][:96000] == 0) for case in cases[:-1])
    assert np.any(indexed["contaminated_profile_limit"]["desired"][:96000] != 0)
    assert np.all(indexed["empty_profile_60"]["input"][:96000] == 0)
    late = indexed["late_strong_overlap_60"]
    assert np.all(late["desired"][:192000] == 0)
    assert late["wanted_onset_sample"] == 192000
    for nominal in (50, 60):
        case = indexed[f"selective_{nominal}"]
        assert study._rms(case["input"] - case["desired"]) == pytest.approx(.024, rel=1e-11)
    masked = indexed["masked_overlap_limit"]
    # Integer numbers of periods: amplitude masking is exact before PCM rounding.
    assert study._rms(masked["input"][:96000]) == pytest.approx(study._rms(masked["input"][96000:]), rel=1e-11)


@pytest.fixture
def source_study(tmp_path):
    root = tmp_path / "h2b"
    root.mkdir()
    (root / "desired_only_60").mkdir()
    t = np.arange(study.FRAMES) / study.RATE
    write_pcm16_wav(root / "prepared_recorded_noise.wav", .01 * np.sin(2 * np.pi * 60 * t))
    write_pcm16_wav(root / "desired_only_60" / "input.wav", .04 * np.sin(2 * np.pi * 233 * t))
    provenance = {"license": "CC0-1.0", "sha256": "owned_encoded_source", "representation": "owned_test_fixture"}
    (root / "noise_provenance.json").write_text(json.dumps(provenance), encoding="utf-8")
    (root / "source_catalog.json").write_text(json.dumps({"evaluation_only": True, "owned_test_fixture": True,
        "sources": [{"case_id": "piano", "prepared": {"sha256": _sha(root / "desired_only_60" / "input.wav")}}]}), encoding="utf-8")
    metadata = {"stage": "H2b", "commercial_release_approved": False,
                "recorded_noise_sha256": "owned_encoded_source",
                "noise_provenance": provenance,
                "provenance_sha256": _sha(root / "noise_provenance.json"),
                "source_catalog_sha256": _sha(root / "source_catalog.json"),
                "cases": [{"case_id": "recorded_hum_only_60", "settings": {"bypass": {"output_sha256": _sha(root / "prepared_recorded_noise.wav")}}},
                          {"case_id": "desired_only_60", "settings": {"bypass": {"output_sha256": _sha(root / "desired_only_60" / "input.wav")}}}]}
    (root / "evaluation.json").write_text(json.dumps(metadata), encoding="utf-8")
    return root, tmp_path / "h2d"


def test_recorded_case_is_new_constructed_pair_with_no_loop_and_actual_prefix(source_study):
    root, _ = source_study
    metadata = json.loads((root / "evaluation.json").read_bytes())
    case, assets, provenance, catalog = study._load_recorded_case(root, metadata)
    assert len(assets) == 4
    assert case["kind"] == "constructed_recorded_mixture"
    assert case["input"].size == 384000
    assert np.all(case["desired"][:96000] == 0)
    piano = read_pcm16_wav(root / "desired_only_60" / "input.wav")
    assert np.array_equal(case["desired"][96000:], piano[:288000])
    assert study._rms(case["input"] - case["desired"]) == pytest.approx(.02)
    assert case["fixture_preparation"]["repetition_or_time_stretch"] is False
    assert "not an independent" in case["context"]


@pytest.mark.parametrize("name", ["prepared_recorded_noise.wav", "desired_only_60/input.wav", "source_catalog.json", "noise_provenance.json"])
def test_source_asset_digest_mismatch_fails_before_output(source_study, name):
    root, target = source_study
    path = root / name
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="hash"):
        study.run_profile_guard_evaluation(root, target)
    assert not target.exists()


@pytest.mark.parametrize("name,field,value", [("noise_provenance.json", "license", "different-license"),
                                              ("source_catalog.json", "sources", [])])
def test_reserialized_json_copies_require_semantic_metadata_anchors(source_study, name, field, value):
    root, target = source_study
    path = root / name
    data = json.loads(path.read_bytes())
    data[field] = value
    path.write_text(json.dumps(data, indent=5), encoding="utf-8")
    with pytest.raises(ValueError, match="hash/provenance"):
        study.run_profile_guard_evaluation(root, target)
    assert not target.exists()


def test_same_semantics_reserialized_json_is_frozen_by_its_actual_copy_bytes(source_study):
    root, _ = source_study
    metadata = json.loads((root / "evaluation.json").read_bytes())
    for name in ("noise_provenance.json", "source_catalog.json"):
        path = root / name
        path.write_text(json.dumps(json.loads(path.read_bytes()), indent=7, sort_keys=True), encoding="utf-8")
    _, assets, _, _ = study._load_recorded_case(root, metadata)
    for name in ("noise_provenance.json", "source_catalog.json"):
        path = root / name
        assert assets[path] == _sha(path)


def test_existing_output_is_never_overwritten(source_study):
    root, target = source_study
    target.mkdir()
    (target / "keep").write_bytes(b"keep")
    with pytest.raises(FileExistsError):
        study.run_profile_guard_evaluation(root, target)
    assert (target / "keep").read_bytes() == b"keep"
    assert list(target.iterdir()) == [target / "keep"]


def test_complete_orchestration_binds_pcm_components_and_never_claims_quality(source_study, monkeypatch):
    root, target = source_study
    source_before = {str(path): _sha(path) for path in root.rglob("*") if path.is_file()}

    class Config:
        def __init__(self, **kwargs):
            self.values = kwargs

    def render(source, config):
        assert "oracle" not in config.values
        return source.copy(), {"owned_identity_stub": True}, {"frames": source.size}

    def replay(source, config, trace):
        assert trace["frames"] == source.size
        return source.copy()

    monkeypatch.setattr(study, "_engine_api", lambda: (Config, render, replay, Config, render, replay))
    result = study.run_profile_guard_evaluation(root, target)
    assert result["case_count"] == 9
    assert result["render_count"] == 18
    assert result["all_declared_structural_checks_met"] is True
    assert result["source_files_unchanged"] is True
    assert result["identities_before"] == result["identities_after"]
    assert result["development_decision"]["status"] == "not_adopted_development_targets_failed"
    for key in ("audio_played", "device_io", "human_review_completed", "independent_perceptual_or_field_quality_verified",
                "commercial_release_approved", "default_engine_changed", "causal_or_live_processing"):
        assert result[key] is False
    for case in result["cases"]:
        assert case["frames"] == 384000
        assert case["independent_clean_field_reference"] is False
        assert case["profile_interval_from_actual_input"] == [0, 96000]
        assert case["profile_truly_noise_only_in_known_fixture"] == (case["case_id"] != "contaminated_profile_limit")
        source = read_pcm16_wav(target / case["case_id"] / "input.wav")
        desired = read_pcm16_wav(target / case["case_id"] / "desired_reference.wav")
        assert np.max(np.abs(source - (desired + (source - desired)))) <= 1e-15
    assert source_before == {str(path): _sha(path) for path in root.rglob("*") if path.is_file()}
    report = (target / "report.html").read_text(encoding="utf-8")
    for phrase in ("NOT an independent evaluation", "OFFLINE / NONCAUSAL", "before PCM16 export", "No listening", "not guaranteed settled"):
        assert phrase in report
    assert "autoplay" not in report
    assert json.loads((target / "evaluation.json").read_bytes())["case_count"] == 9


def test_metadata_mutation_during_preflight_is_detected(source_study, monkeypatch):
    root, target = source_study
    real_load = study._load_recorded_case

    def change_after_load(*args):
        result = real_load(*args)
        path = root / "evaluation.json"
        path.write_bytes(path.read_bytes() + b" ")
        return result

    monkeypatch.setattr(study, "_load_recorded_case", change_after_load)
    with pytest.raises(ValueError, match="changed during preflight"):
        study.run_profile_guard_evaluation(root, target)
    assert not target.exists()
