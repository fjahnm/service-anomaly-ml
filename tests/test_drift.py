"""Drift diagnostics must stay descriptive, time-aware, reproducible and local."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from service_anomaly_ml.data import file_sha256
from service_anomaly_ml.drift import (
    _focus_keys, analyze, feature_shift, ks_distance,
    population_stability_index, reference_edges,
)
from service_anomaly_ml.multi_series import run_many


def _synthetic_project(tmp_path: Path, n: int = 260) -> Path:
    keys = ["realAWSCloudwatch/a.csv", "realAWSCloudwatch/b.csv", "realAWSCloudwatch/c.csv"]
    labels: dict[str, list] = {}
    ts = pd.date_range("2024-01-01", periods=n, freq="5min")
    for i, key in enumerate(keys):
        rng = np.random.default_rng(10 + i)
        values = 10 + i + np.sin(np.arange(n) / 7) + rng.normal(0, .1, n)
        if i == 0:
            values[215:220] += 8
            labels[key] = [[str(ts[215]), str(ts[219])]]
        else:
            labels[key] = []
        path = tmp_path / "data/raw" / key
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"timestamp": ts, "value": values}).to_csv(path, index=False)
    manifest = tmp_path / "data/labels/combined_windows.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(labels), encoding="utf-8")
    run_many(tmp_path, limit=2, trees=10, report_dir=tmp_path / "artifacts/multi_series")
    return tmp_path


def test_ks_equal_samples_is_zero():
    assert ks_distance([1, 1, 4, 7], [7, 4, 1, 1]) == 0


def test_ks_disjoint_constant_samples_is_one():
    assert ks_distance([0, 0, 0], [3, 3, 3]) == 1


def test_psi_equal_samples_is_zero():
    a = np.repeat([1.0, 2.0, 3.0, 5.0], 25)
    assert population_stability_index(a, a) == pytest.approx(0, abs=1e-12)


def test_psi_constant_reference_can_detect_both_sides():
    ref = np.zeros(80)
    assert len(reference_edges(ref)) == 2
    assert population_stability_index(ref, np.ones(60)) > 1
    assert population_stability_index(ref, -np.ones(60)) > 1


def test_psi_train_only_edges_and_shift():
    train = np.arange(100.0)
    assert np.array_equal(reference_edges(train), reference_edges(train.copy()))
    assert population_stability_index(train, train + 300) > population_stability_index(train, train)


def test_feature_shift_does_not_need_labels():
    result = feature_shift(np.arange(30.0), np.arange(30.0) + 100)
    assert result["outside_train_range_rate"] == 1.0
    assert result["ks_distance"] == 1.0
    assert result["psi"] > 0
    assert result["train_n"] == 30


def test_invalid_values_and_bad_bin_count_fail():
    with pytest.raises(ValueError):
        ks_distance([1, float("nan")], [1, 2])
    with pytest.raises(ValueError):
        population_stability_index([1, 2], [1, 2], bins=1)
    with pytest.raises(ValueError):
        population_stability_index([1, 2], [1, 2], pseudocount=0)


def test_analyze_reconstructs_previous_splits_and_writes_outputs(tmp_path):
    root = _synthetic_project(tmp_path)
    source = root / "artifacts/multi_series/summary.json"
    original_sha = file_sha256(source)
    result = analyze(root)
    assert len(result["series"]) == 2
    assert len(result["per_feature"]) == 2 * 7 * 2
    assert result["methodology"]["no_test_tuning"] is True
    assert result["methodology"]["independent_holdout_claim"] is False
    assert result["provenance"]["source_summary_sha256"] == original_sha
    assert file_sha256(source) == original_sha  # Existing evaluation not overwritten.
    out = root / "artifacts/drift"
    for file in ("summary.json", "per_series.csv", "per_feature.csv", "psi_by_series.png", "DRIFT_REPORT.md"):
        assert (out / file).is_file()
    assert len(list((out / "focus").glob("*.png"))) == 2  # Smoke fallback.
    assert (out / "summary.json").stat().st_size > 1000


def test_changed_source_csv_is_detected(tmp_path):
    root = _synthetic_project(tmp_path)
    path = root / "data/raw/realAWSCloudwatch/a.csv"
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="CSV alterado"):
        analyze(root)


def test_changed_label_manifest_is_detected(tmp_path):
    root = _synthetic_project(tmp_path)
    path = root / "data/labels/combined_windows.json"
    path.write_text(path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 das anotações"):
        analyze(root)


def test_reject_same_output_folder_as_source(tmp_path):
    root = _synthetic_project(tmp_path)
    with pytest.raises(ValueError, match="não pode sobrescrever"):
        analyze(root, report_dir=root / "artifacts/multi_series")


def test_focus_unknown_or_ambiguous_fails():
    rows = [{"dataset": "realAWSCloudwatch/a.csv", "test_value_psi": 1}]
    with pytest.raises(ValueError, match="ausente"):
        _focus_keys(rows, ["missing.csv"])
    assert _focus_keys(rows, ["a.csv", "a.csv"]) == ["realAWSCloudwatch/a.csv"]
