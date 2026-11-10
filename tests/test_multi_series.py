"""Multi-series tests use synthetic data and never fetch from the internet."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from service_anomaly_ml.multi_series import (
    aggregate, partition, run_many, run_one, series_keys,
)
from service_anomaly_ml.features import make_features


def _data(root: Path, n: int = 400) -> list[str]:
    keys = ["realAWSCloudwatch/a.csv", "realAWSCloudwatch/b.csv", "realAWSCloudwatch/c.csv"]
    labels = {}
    rng = np.random.default_rng(11)
    for idx, key in enumerate(keys):
        timestamps = pd.date_range("2024-01-01", periods=n, freq="5min")
        values = (idx + 1) * 10 + np.sin(np.arange(n) / 7) + rng.normal(0, .1, n)
        if idx < 2:
            values[340:347] += 10
            labels[key] = [[str(timestamps[340]), str(timestamps[346])]]
        else:
            labels[key] = []
        target = root / "data/raw" / key
        target.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"timestamp": timestamps, "value": values}).to_csv(target, index=False)
    label_path = root / "data/labels/combined_windows.json"
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text(json.dumps(labels), encoding="utf-8")
    return keys


def test_selection_keeps_zero_window_series_and_deterministic_prefix(tmp_path):
    keys = _data(tmp_path)
    label_path = tmp_path / "data/labels/combined_windows.json"
    assert series_keys(label_path) == keys
    assert series_keys(label_path, limit=2) == keys[:2]
    with pytest.raises(ValueError):
        series_keys(label_path, limit=0)


def test_partition_is_chronological_and_uses_disjoint_rows(tmp_path):
    _data(tmp_path)
    frame = make_features(pd.read_csv(tmp_path / "data/raw/realAWSCloudwatch/a.csv", parse_dates=["timestamp"]))
    train, val, test = partition(frame)
    assert len(train) + len(val) + len(test) == len(frame)
    assert train.timestamp.iloc[-1] < val.timestamp.iloc[0] < test.timestamp.iloc[0]
    assert len(train) == int(len(frame) * .5)


def test_run_one_can_handle_zero_test_windows_without_lying(tmp_path):
    _data(tmp_path)
    result = run_one("realAWSCloudwatch/c.csv", tmp_path, trees=12)
    assert result["test"]["isolation_forest"]["windows_evaluated"] == 0
    assert result["test"]["isolation_forest"]["event_recall"] is None
    assert result["windows_intersect_training"] == 0


def test_aggregate_includes_negative_only_series_in_false_alerts(tmp_path):
    _data(tmp_path)
    report = run_many(tmp_path, trees=12, report_dir=tmp_path / "outputs")
    aggregate_iso = report["aggregate"]["isolation_forest"]
    assert aggregate_iso["series_count"] == 3
    assert aggregate_iso["series_without_test_windows"] == 1
    assert aggregate_iso["test_windows"] == 2
    assert aggregate_iso["micro_event_recall"] is not None
    assert (tmp_path / "outputs/summary.json").is_file()
    assert (tmp_path / "outputs/per_series.csv").is_file()
    assert (tmp_path / "outputs/comparison.png").is_file()
    assert (tmp_path / "outputs/MULTI_SERIES_REPORT.md").is_file()


def test_late_change_cannot_affect_training_threshold(tmp_path):
    _data(tmp_path)
    key = "realAWSCloudwatch/a.csv"
    before = run_one(key, tmp_path, trees=12)
    src = tmp_path / "data/raw" / key
    df = pd.read_csv(src)
    df.loc[len(df) - 10, "value"] += 1000
    df.to_csv(src, index=False)
    after = run_one(key, tmp_path, trees=12)
    assert before["thresholds"] == after["thresholds"]
    assert before["train_end"] == after["train_end"]


def test_missing_series_fails_instead_of_skipping(tmp_path):
    _data(tmp_path)
    (tmp_path / "data/raw/realAWSCloudwatch/b.csv").unlink()
    with pytest.raises(RuntimeError, match="realAWSCloudwatch/b.csv"):
        run_many(tmp_path, trees=12, report_dir=tmp_path / "outputs")


def test_invalid_key_rejected_before_download(tmp_path):
    label_path = tmp_path / "windows.json"
    label_path.write_text(json.dumps({"realAWSCloudwatch/../bad.csv": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="suspeita"):
        series_keys(label_path)
