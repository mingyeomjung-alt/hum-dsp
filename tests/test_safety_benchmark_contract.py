"""Cheap M4 evidence/preflight contracts; no full DSP benchmark is rerun."""

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

from hum_dsp import guarded, safety_benchmark
from hum_dsp.audio import SAMPLE_RATE_HZ, write_pcm16_wav
from hum_dsp.cli import main
from hum_dsp.config import ProcessorConfig
from hum_dsp.fixtures import fixture_specs


_M3_SOURCES = (
    "tracking.py", "filter.py", "config.py", "tracked_processing.py",
    "tracking_evaluation.py", "tracking_benchmark.py", "m3_candidate.json",
    "protocol.json",
)


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    """Tiny valid containers standing in for already-verified M1/M3 evidence.

    We bypass only the separate expensive M1 suite-integrity routine.  M4's
    source-hash, case inventory, output hash and config checks remain real.
    An accidental transition into DSP is an explicit test failure.
    """
    suite = tmp_path / "suite"
    baseline = tmp_path / "baseline"
    target = tmp_path / "new-output"
    suite.mkdir()
    baseline.mkdir()
    module = Path(safety_benchmark.__file__).parent
    cases = []
    for spec in fixture_specs():
        original = suite / spec.case_id
        prior = baseline / spec.case_id
        original.mkdir()
        prior.mkdir()
        for file in (original / "input.wav", original / "clean.wav", prior / "processed.wav"):
            write_pcm16_wav(file, np.zeros(4, dtype=np.float64))
        cases.append({
            "case_id": spec.case_id,
            "config": ProcessorConfig(nominal_hum_hz=spec.nominal_hum_hz).to_dict(),
            "output": {"sha256": _hash(prior / "processed.wav"), "frames": 4},
        })
    manifest = {
        "stage": "M3", "case_count": 16, "cases": cases,
        "source_sha256": {name: _hash(module / name) for name in _M3_SOURCES},
    }
    manifest_path = baseline / "benchmark.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    (suite / "manifest.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(safety_benchmark, "verify_fixture_suite", lambda path: {"scope": "preflight_contract_test_only"})
    # These are intentionally tiny mocks, not the preserved full-run manifest.
    # Bypass only those two frozen provenance identities so each mutation can
    # reach its own structural check.  Module and WAV hashes stay real.
    policy = json.loads((module / "m4_policy.json").read_text(encoding="utf-8"))
    actual_sha = safety_benchmark._sha

    def fixture_provenance_sha(path):
        resolved = Path(path).resolve()
        if resolved == manifest_path.resolve():
            return policy["frozen_m3_benchmark_sha256"]
        if resolved == (suite / "manifest.json").resolve():
            return policy["frozen_fixture_manifest_sha256"]
        return actual_sha(path)

    monkeypatch.setattr(safety_benchmark, "_sha", fixture_provenance_sha)

    def unexpected_dsp(*args, **kwargs):
        pytest.fail("invalid baseline must be rejected before DSP rendering")

    monkeypatch.setattr(safety_benchmark, "render_guarded", unexpected_dsp)
    return suite, baseline, target, manifest, manifest_path


def _save_manifest(evidence):
    evidence[4].write_text(json.dumps(evidence[3]), encoding="utf-8")


def _assert_preflight_rejection(evidence, error=(OSError, ValueError, KeyError, TypeError)):
    suite, baseline, target, _, _ = evidence
    with pytest.raises(error):
        safety_benchmark.run_safety_benchmark(suite, baseline, target)
    assert not target.exists(), "invalid evidence must not create even a partial run directory"


def test_existing_output_is_refused_without_touching_contents_or_verifying_inputs(evidence, monkeypatch):
    suite, baseline, target, _, _ = evidence
    target.mkdir()
    marker = target / "owner-data.txt"
    marker.write_text("preserve me", encoding="utf-8")
    monkeypatch.setattr(safety_benchmark, "verify_fixture_suite", lambda path: pytest.fail("existing output must be refused first"))
    with pytest.raises(FileExistsError):
        safety_benchmark.run_safety_benchmark(suite, baseline, target)
    assert marker.read_text(encoding="utf-8") == "preserve me"
    assert tuple(target.iterdir()) == (marker,)


@pytest.mark.parametrize("change", ("wrong_stage", "wrong_count", "missing_case", "duplicate_case", "unknown_case", "malformed_json"))
def test_bad_m3_manifest_is_rejected_before_mkdir(evidence, change):
    manifest = evidence[3]
    if change == "wrong_stage":
        manifest["stage"] = "M2"
    elif change == "wrong_count":
        manifest["case_count"] = 15
    elif change == "missing_case":
        manifest["cases"].pop()
    elif change == "duplicate_case":
        manifest["cases"][1] = manifest["cases"][0]
    elif change == "unknown_case":
        manifest["cases"][0]["case_id"] = "not-a-frozen-fixture"
    elif change == "malformed_json":
        evidence[4].write_text("{", encoding="utf-8")
        _assert_preflight_rejection(evidence, ValueError)
        return
    _save_manifest(evidence)
    _assert_preflight_rejection(evidence, ValueError)


@pytest.mark.parametrize("change", ("wrong_digest", "nonlocal_source", "empty_inventory", "missing_source"))
def test_changed_or_incomplete_m3_source_hashes_are_rejected_before_mkdir(evidence, change):
    hashes = evidence[3]["source_sha256"]
    if change == "wrong_digest":
        hashes["tracking.py"] = "0" * 64
    elif change == "nonlocal_source":
        hashes["../tracking.py"] = hashes.pop("tracking.py")
    elif change == "empty_inventory":
        hashes.clear()
    else:
        hashes.pop("config.py")
    _save_manifest(evidence)
    _assert_preflight_rejection(evidence, ValueError)


def test_changed_preserved_wav_is_rejected_before_mkdir(evidence):
    spec = fixture_specs()[-1]
    file = evidence[1] / spec.case_id / "processed.wav"
    write_pcm16_wav(file, np.full(4, 0.1), overwrite=True)
    _assert_preflight_rejection(evidence, ValueError)


def test_changed_manifest_output_hash_is_rejected_before_mkdir(evidence):
    evidence[3]["cases"][-1]["output"]["sha256"] = "f" * 64
    _save_manifest(evidence)
    _assert_preflight_rejection(evidence, ValueError)


def test_changed_manifest_config_is_rejected_before_mkdir(evidence):
    evidence[3]["cases"][0]["config"]["max_attenuation_db"] = 0.0
    _save_manifest(evidence)
    _assert_preflight_rejection(evidence, ValueError)


@pytest.mark.parametrize("which", ("suite", "baseline"))
def test_nonmatching_frozen_manifest_identity_is_rejected_before_mkdir(evidence, monkeypatch, which):
    wrapped = safety_benchmark._sha
    selected = evidence[0] / "manifest.json" if which == "suite" else evidence[4]

    def wrong_manifest_identity(path):
        return "0" * 64 if Path(path).resolve() == selected.resolve() else wrapped(path)

    monkeypatch.setattr(safety_benchmark, "_sha", wrong_manifest_identity)
    _assert_preflight_rejection(evidence, ValueError)


def test_policy_constants_match_guarded_runtime():
    module = Path(safety_benchmark.__file__).parent
    policy = json.loads((module / "m4_policy.json").read_text(encoding="utf-8"))
    metadata = guarded.GuardedHumSuppressor(ProcessorConfig()).describe()["policy"]
    for policy_key, constant, runtime_key in (
        ("qualified_estimate_center_difference_hz_max", guarded.QUALIFICATION_ERROR_HZ, "qualification_frequency_error_hz"),
        ("required_consecutive_qualified_events", guarded.REQUIRED_QUALIFIED_EVENTS, "required_consecutive_qualified_events"),
        ("last_qualified_hold_samples", guarded.QUALIFICATION_HOLD_FRAMES, "qualification_hold_frames"),
        ("attack_samples", guarded.ATTACK_FRAMES, "attack_frames"),
        ("release_samples", guarded.RELEASE_FRAMES, "release_frames"),
    ):
        assert policy[policy_key] == constant == metadata[runtime_key]
    assert policy["initial_mix"] == metadata["initial_mix"] == 0.0
    assert policy["sample_rate_hz"] == SAMPLE_RATE_HZ
    assert policy["envelope"] == metadata["envelope"]
    assert policy["targets"]["soft_disable_exact_dry_after_samples"] == guarded.RELEASE_FRAMES
    assert policy["targets"]["envelope_range"] == [0.0, 1.0]
    assert policy["targets"]["envelope_max_sample_step"] == pytest.approx(math.pi / (2.0 * guarded.ATTACK_FRAMES))
    assert policy["retained_m1_development_case_count"] == len(fixture_specs()) == 16
    assert policy["new_safety_case_count"] == 10
    assert policy["m2_filter_sha256"] == _hash(module / "filter.py")
    assert policy["m3_tracking_sha256"] == _hash(module / "tracking.py")
    assert policy["raw_m3_filter_and_estimator_changed"] is False
    assert policy["targets"]["same_frequency_desired_tone_preservation_guaranteed"] is False
    for key in ("frozen_fixture_manifest_sha256", "frozen_m3_benchmark_sha256"):
        assert len(policy[key]) == 64
        assert all(character in "0123456789abcdef" for character in policy[key])


@pytest.mark.parametrize("arguments", (["--help"], ["benchmark-guarded", "--help"]))
def test_cli_help_advertises_guarded_benchmark(arguments, capsys):
    with pytest.raises(SystemExit) as result:
        main(arguments)
    assert result.value.code == 0
    assert "benchmark-guarded" in capsys.readouterr().out
