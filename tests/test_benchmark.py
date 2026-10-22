import json

import numpy as np
import pytest

from hum_dsp import benchmark
from hum_dsp.audio import write_pcm16_wav
from hum_dsp.fixtures import fixture_specs, make_fixture


def test_benchmark_retains_startup_wanted_tone_damage_and_does_not_rename_error_as_noise(tmp_path, monkeypatch):
    selected = tuple(spec for spec in fixture_specs() if spec.case_id in (
        "synthetic_fixed_50_noise_only", "synthetic_fixed_60_noise_only", "synthetic_same_frequency_desired_50"))
    suite = tmp_path / "selected"
    suite.mkdir()
    for spec in selected:
        folder = suite / spec.case_id
        folder.mkdir()
        fixture = make_fixture(spec)
        write_pcm16_wav(folder / "clean.wav", fixture.clean)
        write_pcm16_wav(folder / "input.wav", fixture.mixed)
    monkeypatch.setattr(benchmark, "fixture_specs", lambda: selected)
    monkeypatch.setattr(benchmark, "verify_fixture_suite", lambda path: {"test_scope": "selected_cases"})
    target = tmp_path / "benchmark"
    report = benchmark.run_fixed_benchmark(suite, target)
    assert report["stationary_noise_only_cases"] == 2
    assert report["stationary_noise_only_target_met"] is True
    assert report["all_input_files_unchanged"] is True
    assert report["all_frame_counts_preserved"] is True
    assert report["auto_tracking_implemented"] is False
    assert report["real_time_device_performance_verified"] is False
    assert json.loads((target / "benchmark.json").read_text(encoding="utf-8")) == report
    assert (target / "report.html").is_file()
    for case in report["cases"]:
        for encoding in ("float_measurements", "decoded_wav_measurements"):
            assert case[encoding]["whole_record"]["analyzed_sample_count"] == 6 * 48000
            assert case[encoding]["startup"]["analyzed_sample_count"] == 2 * 48000
            assert case[encoding]["settled"]["analyzed_sample_count"] == 4 * 48000
        if case["kind"] == "stationary_target":
            assert 10 <= case["signal_only_attenuation_db"] <= 12.1
        else:
            assert case["signal_only_attenuation_db"] is None
            assert case["stationary_10db_target_met"] is None
            assert case["preservation_guarantee"] is False
            assert case["decoded_wav_measurements"]["settled"]["output_error_rmse"] > 0.01
    before = (target / "benchmark.json").read_bytes()
    with pytest.raises(FileExistsError):
        benchmark.run_fixed_benchmark(suite, target)
    assert (target / "benchmark.json").read_bytes() == before
