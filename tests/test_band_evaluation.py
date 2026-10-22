"""H2 evaluation contracts without downloaded sources or a full DSP study."""

import copy
import hashlib
import html
import json

import numpy as np
import pytest

from hum_dsp.audio import write_pcm16_wav
from hum_dsp.band_evaluation import _catalog, _metric_window, _report, _windows


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def prepared_catalog(tmp_path):
    """Tiny owned WAVs stand in for public files; hashes are pinned before read."""
    sources = []
    for index, case_id in enumerate(("piano", "phrase", "game")):
        original = tmp_path / (case_id + "_original.bin")
        original.write_bytes(("owned synthetic original " + case_id).encode())
        prepared = tmp_path / (case_id + ".wav")
        write_pcm16_wav(prepared, np.array([0, 1, -1, 2 + index]) / 32768.0)
        sources.append({
            "case_id": case_id,
            "license": "CC0-1.0",
            "original": {"local_path": str(original), "sha256": _digest(original)},
            "prepared": {"file_name": prepared.name, "sha256": _digest(prepared), "frames": 4},
            "preparation": {"gain": 0.25, "peak_normalization": False},
        })
    catalog = {"schema_version": 1, "evaluation_only": True, "sample_rate_hz": 48000,
               "fixed_gain": 0.25, "sources": sources}
    policy = {"source_ids": [item["case_id"] for item in sources],
              "source_sha256": {item["case_id"]: item["original"]["sha256"] for item in sources}}
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(catalog), encoding="utf-8")
    return path, catalog, policy


def _save_catalog(path, catalog):
    path.write_text(json.dumps(catalog), encoding="utf-8")


def test_catalog_loads_only_pinned_owned_fixture_without_changing_files(prepared_catalog):
    path, catalog, policy = prepared_catalog
    before = {file.name: _digest(file) for file in path.parent.iterdir()}
    loaded_catalog, sources = _catalog(path, policy)
    assert loaded_catalog == catalog
    assert [item[0]["case_id"] for item in sources] == policy["source_ids"]
    assert all(samples.size == 4 for _, _, samples in sources)
    assert all(samples.dtype == np.float64 for _, _, samples in sources)
    assert {file.name: _digest(file) for file in path.parent.iterdir()} == before


@pytest.mark.parametrize("field,value", [
    ("schema_version", 2), ("evaluation_only", False),
    ("sample_rate_hz", 44100), ("fixed_gain", 0.5),
    ("fixed_gain", 0), ("fixed_gain", None),
])
def test_catalog_rejects_changed_top_level_contract(prepared_catalog, field, value):
    path, catalog, policy = prepared_catalog
    catalog[field] = value
    _save_catalog(path, catalog)
    with pytest.raises(ValueError, match="fixed prepared"):
        _catalog(path, policy)


@pytest.mark.parametrize("change", ["missing", "order", "wrong_id", "duplicate"])
def test_catalog_requires_exact_source_ids_and_order(prepared_catalog, change):
    path, catalog, policy = prepared_catalog
    if change == "missing":
        catalog["sources"].pop()
    elif change == "order":
        catalog["sources"].reverse()
    elif change == "wrong_id":
        catalog["sources"][0]["case_id"] = "other"
    else:
        catalog["sources"][2] = copy.deepcopy(catalog["sources"][0])
    _save_catalog(path, catalog)
    with pytest.raises(ValueError, match="fixed prepared"):
        _catalog(path, policy)


@pytest.mark.parametrize("change", [
    "missing_license", "other_license", "original_metadata_hash", "gain", "normalization",
    "parent_path", "absolute_path", "different_filename",
])
def test_catalog_rejects_unapproved_source_metadata(prepared_catalog, change):
    path, catalog, policy = prepared_catalog
    item = catalog["sources"][0]
    if change == "missing_license":
        item.pop("license")
    elif change == "other_license":
        item["license"] = "research-only"
    elif change == "original_metadata_hash":
        item["original"]["sha256"] = "0" * 64
    elif change == "gain":
        item["preparation"]["gain"] = 1
    elif change == "normalization":
        item["preparation"]["peak_normalization"] = True
    elif change == "parent_path":
        item["prepared"]["file_name"] = "../piano.wav"
    elif change == "absolute_path":
        item["prepared"]["file_name"] = str(path.parent / "piano.wav")
    else:
        item["prepared"]["file_name"] = "another.wav"
    _save_catalog(path, catalog)
    with pytest.raises(ValueError, match="provenance"):
        _catalog(path, policy)


@pytest.mark.parametrize("target,match", [
    ("original", "original public source hash"),
    ("prepared", "prepared evaluation source hash"),
])
def test_catalog_verifies_actual_file_bytes_not_just_metadata(prepared_catalog, target, match):
    path, catalog, policy = prepared_catalog
    item = catalog["sources"][0]
    file = path.parent / item["prepared"]["file_name"] if target == "prepared" else path.parent / "piano_original.bin"
    file.write_bytes(file.read_bytes() + b"mutation")
    with pytest.raises(ValueError, match=match):
        _catalog(path, policy)


def test_catalog_rejects_original_larger_than_declared_bound(prepared_catalog):
    path, catalog, policy = prepared_catalog
    original = path.parent / "piano_original.bin"
    original.write_bytes(b"x" * (5 * 1024 * 1024 + 1))
    # Even a metadata/policy hash changed consistently must not bypass the bound.
    catalog["sources"][0]["original"]["sha256"] = _digest(original)
    policy["source_sha256"]["piano"] = _digest(original)
    _save_catalog(path, catalog)
    with pytest.raises(ValueError, match="original public source hash"):
        _catalog(path, policy)


def test_catalog_rejects_wrong_declared_frame_count(prepared_catalog):
    path, catalog, policy = prepared_catalog
    catalog["sources"][0]["prepared"]["frames"] = 5
    _save_catalog(path, catalog)
    with pytest.raises(ValueError, match="frame count"):
        _catalog(path, policy)


def _components(size):
    desired = np.full(size, 0.1, dtype=np.float64)
    noise = np.full(size, 0.02, dtype=np.float64)
    desired_replay = desired * 0.9
    noise_replay = noise * 0.5
    output = desired_replay + noise_replay
    return desired, noise, output, desired_replay, noise_replay


@pytest.mark.parametrize("count", [1, 54240, 96000])
def test_short_windows_preserve_whole_startup_and_do_not_fabricate_settled(count):
    arrays = _components(count)
    originals = [array.copy() for array in arrays]
    result = _windows(*arrays)
    assert result["settled"] is None
    assert result["settled_unavailable_reason"] == "source_not_longer_than_two_seconds_not_repeated"
    assert result["startup_end_sample"] == count
    assert result["settled_start_sample"] == 96000
    for key in ("whole_record", "startup"):
        metric = result[key]
        assert metric["total_sample_count"] == count
        assert metric["analyzed_sample_count"] == count
        assert metric["analysis_start_sample"] == 0
        assert metric["component_interpretation_valid"]
        assert metric["clean_relative_error_rmse"] == pytest.approx(0.1)
    for original, array in zip(originals, arrays, strict=True):
        np.testing.assert_array_equal(original, array)


@pytest.mark.parametrize("count", [96001, 120000])
def test_long_windows_preserve_full_and_exact_declared_boundary(count):
    result = _windows(*_components(count))
    assert result["whole_record"]["analyzed_sample_count"] == count
    assert result["whole_record"]["analysis_start_sample"] == 0
    assert result["startup"]["analyzed_sample_count"] == 96000
    assert result["startup"]["total_sample_count"] == 96000
    assert result["settled"]["analysis_start_sample"] == 96000
    assert result["settled"]["analyzed_sample_count"] == count - 96000
    assert result["settled_unavailable_reason"] is None


def test_metrics_keep_known_reference_scope_and_no_fitting():
    metric = _metric_window(*_components(100))
    assert "not_blind_quality" in metric["interpretation"]
    assert metric["alignment_applied"] is False
    assert metric["output_gain_normalization_applied"] is False
    assert metric["component_gain_fitting_applied"] is False
    assert metric["noise_attenuation_db"] == pytest.approx(20 * np.log10(2))


def test_metrics_retain_additivity_failure_not_a_false_quality_pass():
    desired, noise, output, desired_replay, noise_replay = _components(50)
    metric = _metric_window(desired, noise, output + 0.001, desired_replay, noise_replay)
    assert metric["component_interpretation_valid"] is False
    assert metric["decomposition_additivity_max_abs_error"] == pytest.approx(0.001)
    assert metric["component_interpretation_invalid_reason"]


def _report_data(case_id="owned", label="full", settled=None):
    metric = {"noise_attenuation_db": None, "clean_relative_error_rmse": 0.1,
              "clean_level_change_db": -0.5}
    return {"cases": [{"case_id": case_id, "settings": {label: {
        "metrics": {"settled": settled, "whole_record": metric}}}}]}


def test_report_explicitly_separates_float_metrics_pcm16_files_and_quality_claims(tmp_path):
    destination = tmp_path / "report.html"
    _report(_report_data(), destination)
    text = destination.read_text(encoding="utf-8")
    assert "floating renders/replays BEFORE final PCM16 rounding" in text
    assert "not decompositions of the downloadable WAV" in text
    assert "decoded PCM16 raw error and quantization error" in text
    assert "not guaranteed fully settled" in text
    assert "Startup and whole-file results are retained" in text
    assert "NOT natural-noise, perceptual or independent final quality validation" in text
    assert "Commercial quality approval: NO" in text
    assert "whole / too short for settled" in text
    assert "n/a" in text
    assert "autoplay" not in text
    assert "<audio" not in text


def test_report_escapes_text_and_link_attributes(tmp_path):
    case_id = 'owned" onmouseover="alert(1)<script>&'
    label = 'full" onclick="alert(2)<script>&'
    destination = tmp_path / "escaped.html"
    _report(_report_data(case_id, label), destination)
    text = destination.read_text(encoding="utf-8")
    assert html.escape(case_id) in text
    assert html.escape(label) in text
    assert f'href="{html.escape(case_id, quote=True)}/{html.escape(label, quote=True)}.wav"' in text
    assert '<script>' not in text
    assert 'href="owned" onmouseover=' not in text
    assert '/full" onclick=' not in text


def test_report_uses_settled_metric_when_present(tmp_path):
    destination = tmp_path / "settled.html"
    settled = {"noise_attenuation_db": 9.25, "clean_relative_error_rmse": 0.02,
               "clean_level_change_db": -0.01}
    _report(_report_data(settled=settled), destination)
    text = destination.read_text(encoding="utf-8")
    assert "<td>settled</td>" in text
    assert "<td>9.25</td>" in text
    assert "whole / too short for settled" not in text
