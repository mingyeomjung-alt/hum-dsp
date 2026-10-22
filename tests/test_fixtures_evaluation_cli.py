import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from hum_dsp.audio import PCM16_STEP, read_pcm16_wav
from hum_dsp.cli import main
from hum_dsp.evaluation import measure_windows
from hum_dsp.fixtures import export_fixture_suite, fixture_specs, make_fixture, noise_frequency_and_phase, verify_fixture_suite
from hum_dsp.metrics import measure_output


def test_suite_has_sixteen_unique_owned_development_cases():
    specs = fixture_specs()
    assert len(specs) == 16
    assert len({spec.case_id for spec in specs}) == 16
    assert all(spec.case_id.startswith("synthetic_") for spec in specs)
    assert {spec.nominal_hum_hz for spec in specs} == {50.0, 60.0}
    for kind in ("negative_control", "semantic_ambiguity", "component_ambiguity", "moving_target"):
        assert any(spec.kind == kind for spec in specs)


def test_fixture_is_deterministic_readonly_and_preserves_exact_math_sum():
    spec = next(spec for spec in fixture_specs() if spec.case_id == "synthetic_fixed_50_mixture")
    a, b = make_fixture(spec), make_fixture(spec)
    np.testing.assert_array_equal(a.mixed, b.mixed)
    np.testing.assert_array_equal(a.mixed, a.clean + a.noise)
    assert max(abs(a.mixed)) < 0.5
    assert not a.clean.flags.writeable
    assert not a.noise.flags.writeable
    assert not a.mixed.flags.writeable
    assert not a.instantaneous_noise_hz.flags.writeable


def test_unregistered_fixture_spec_cannot_change_frozen_test_conditions():
    with pytest.raises(ValueError):
        make_fixture(replace(fixture_specs()[0], duration_seconds=1.0))


def test_pcm16_quantization_does_not_turn_collinear_content_into_separable_sources():
    spec = next(spec for spec in fixture_specs() if spec.case_id == "synthetic_collinear_50_mixture")
    fixture = make_fixture(spec)
    quantize = lambda array: np.rint(array * 32768).astype(np.int16).astype(np.float64) / 32768
    clean = quantize(fixture.clean)
    mixed = quantize(fixture.mixed)
    measured = measure_output(clean, mixed - clean, mixed)
    assert measured["component_fit"]["status"] == "unidentifiable"
    assert measured["component_fit"]["noise_attenuation_db"] is None
    assert measured["output_error_rmse"] > 0


def test_phase_rotation_is_not_reported_as_raw_noise_reduction():
    time = np.arange(48000) / 48000
    clean = np.zeros_like(time)
    noise = 0.1 * np.sin(2 * np.pi * 50 * time)
    rotated = 0.1 * np.cos(2 * np.pi * 50 * time)
    measured = measure_output(clean, noise, rotated)
    assert measured["reference_error_reduction_db"] == pytest.approx(0, abs=1e-12)
    assert measured["component_fit"]["residual_rms"] > 0.07
    assert measured["component_fit"]["attenuation_interpretation"] == "in_phase_projection_only_not_total_noise_reduction"


@pytest.mark.parametrize("nominal,direction", [(50.0, 1), (60.0, -1)])
def test_full_sweep_has_correct_range_rate_and_integrated_phase(nominal, direction):
    spec = next(spec for spec in fixture_specs() if spec.nominal_hum_hz == nominal and spec.kind == "moving_target")
    time = np.array([0.0, 1.0, 2.0, 12.0, 22.0, 32.0, 42.0, 43.0])
    frequency, phase = noise_frequency_and_phase(spec, time)
    np.testing.assert_allclose(frequency, nominal + direction * np.array([-1, -1, -1, -0.5, 0, 0.5, 1, 1]))
    for at in (1.0, 7.0, 22.0, 37.0, 43.0):
        epsilon = 1e-4
        _, phase_pair = noise_frequency_and_phase(spec, np.array([at - epsilon, at + epsilon]))
        numeric_hz = (phase_pair[1] - phase_pair[0]) / (4 * np.pi * epsilon)
        exact_hz, _ = noise_frequency_and_phase(spec, np.array([at]))
        assert numeric_hz == pytest.approx(exact_hz[0], abs=1e-6)
    fixture = make_fixture(spec)
    assert fixture.mixed.size == 44 * 48000
    assert float(np.min(fixture.instantaneous_noise_hz)) == nominal - 1
    assert float(np.max(fixture.instantaneous_noise_hz)) == nominal + 1
    assert np.max(np.abs(np.diff(fixture.instantaneous_noise_hz))) <= 0.05 / 48000 + 1e-13


@pytest.mark.parametrize("nominal", [50.0, 60.0])
def test_identical_waveform_can_be_noise_or_desired_content(nominal):
    noise_case = next(spec for spec in fixture_specs() if spec.nominal_hum_hz == nominal and spec.kind == "stationary_target")
    desired_case = next(spec for spec in fixture_specs() if spec.nominal_hum_hz == nominal and spec.kind == "semantic_ambiguity")
    noise_fixture, desired_fixture = make_fixture(noise_case), make_fixture(desired_case)
    np.testing.assert_array_equal(noise_fixture.mixed, desired_fixture.mixed)
    assert np.any(noise_fixture.noise != 0)
    assert np.all(desired_fixture.noise == 0)


def test_export_has_exact_hashes_protocol_truth_and_quantization_bounds(tmp_path):
    destination = tmp_path / "suite"
    result = export_fixture_suite(destination)
    assert result["case_count"] == 16
    assert result["processor_applied"] is False
    assert result["implementation_status_reference"] == "project_milestones_not_fixture_manifest"
    assert hashlib.sha256((Path(__file__).parents[1] / "src/hum_dsp/protocol.json").read_bytes()).hexdigest() == result["protocol_sha256"]
    protocol = json.loads((destination / "protocol.json").read_text(encoding="utf-8"))
    assert protocol["measurement"]["startup_report_required"] is True
    assert protocol["measurement"]["output_normalization_allowed"] is False
    assert protocol["targets_not_achieved_results"]["stationary_noise_only_reduction_db_min"] == 10
    for case in result["cases"]:
        assert len(case["files"]) == 4
        for record in case["files"]:
            contents = (destination / record["path"]).read_bytes()
            assert len(contents) == record["bytes"]
            assert hashlib.sha256(contents).hexdigest() == record["sha256"]
        directory = destination / case["case_id"]
        clean, noise, mixed = [read_pcm16_wav(directory / f"{name}.wav") for name in ("clean", "noise", "input")]
        assert np.max(abs(mixed - clean - noise)) <= 1.5 * PCM16_STEP
        truth = json.loads((directory / "truth.json").read_text(encoding="utf-8"))
        assert truth["processor_applied"] is False
        assert truth["measurement_noise_reference"] == "decoded_input_minus_decoded_clean"
    before = (destination / "manifest.json").read_bytes()
    with pytest.raises(FileExistsError):
        export_fixture_suite(destination)
    assert (destination / "manifest.json").read_bytes() == before
    verification = verify_fixture_suite(destination)
    assert verification["case_count"] == 16
    assert verification["verified_case_files"] == 64
    assert verification["noise_reduction_performance_measured"] is False
    target = destination / "synthetic_fixed_50_mixture" / "input.wav"
    tampered = bytearray(target.read_bytes())
    tampered[-1] ^= 1
    target.write_bytes(tampered)
    with pytest.raises(ValueError, match="integrity mismatch"):
        verify_fixture_suite(destination)


def test_startup_errors_are_not_hidden_by_settling_exclusion():
    clean = np.zeros(100)
    noise = np.ones(100) * 0.01
    output = np.zeros(100)
    output[:20] = 0.9
    result = measure_windows(clean, noise, output, settling_samples=20)
    assert result["whole_record"]["output_peak_abs"] == 0.9
    assert result["startup"]["output_peak_abs"] == 0.9
    assert result["settled"]["output_peak_abs"] == 0
    assert result["settled"]["analysis_start_sample"] == 20
    json.dumps(result, allow_nan=False)


def test_short_clip_has_no_fabricated_settled_measurement():
    data = np.zeros(10)
    result = measure_windows(data, data, data)
    assert result["settled"] is None
    assert result["settled_undefined_reason"] == "no_samples_after_settling"
    assert result["startup"]["analyzed_sample_count"] == 10


@pytest.mark.parametrize("value", [0, -1, True, 1.5, None])
def test_invalid_settling_interval_is_rejected(value):
    with pytest.raises(ValueError):
        measure_windows([0], [0], [0], settling_samples=value)


def test_cli_validation_does_not_process_or_rewrite_input(tmp_path, capsys):
    from hum_dsp.audio import write_pcm16_wav
    path = tmp_path / "검사 대상.wav"
    write_pcm16_wav(path, np.array([0.1, -0.1]))
    before = path.read_bytes()
    assert main(["validate-input", str(path)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "supported_input"
    assert result["processor_applied"] is False
    assert path.read_bytes() == before


def test_processing_cli_requires_explicit_target_and_output_directory():
    with pytest.raises(SystemExit) as failure:
        main(["process", "not-a-file.wav"])
    assert failure.value.code == 2
