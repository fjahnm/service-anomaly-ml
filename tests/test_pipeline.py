"""Unit tests using synthetic values only (no public dataset required)."""
import json

import numpy as np
import pandas as pd
import pytest

from service_anomaly_ml.data import load_series, load_windows
from service_anomaly_ml.evaluation import score_alerts
from service_anomaly_ml.features import FEATURE_COLUMNS, make_features


def sample_series(n=100):
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="5min"),
        "value": np.arange(n, dtype=float),
    })


def test_feature_history_and_columns():
    frame = make_features(sample_series())
    assert len(frame) == 88
    assert all(col in frame.columns for col in FEATURE_COLUMNS)
    assert frame.iloc[0]["past_mean_12"] == pytest.approx(np.mean(np.arange(12)))


def test_future_value_does_not_change_past_features():
    original = sample_series()
    modified = original.copy()
    modified.loc[50, "value"] = 9000
    a, b = make_features(original), make_features(modified)
    pd.testing.assert_frame_equal(a.loc[a["timestamp"] < original.loc[50, "timestamp"]],
                                  b.loc[b["timestamp"] < original.loc[50, "timestamp"]])
    # Historical average at the changed timestamp cannot observe its current value.
    past1 = a.loc[a["timestamp"] == original.loc[50, "timestamp"], "past_mean_12"].iloc[0]
    past2 = b.loc[b["timestamp"] == original.loc[50, "timestamp"], "past_mean_12"].iloc[0]
    assert past1 == past2


def test_event_scoring_and_false_episodes():
    times = sample_series(10)["timestamp"]
    flags = np.array([False, True, True, False, False, True, False, False, True, True])
    windows = [(times.iloc[4], times.iloc[6])]
    result = score_alerts(times, flags, windows)
    assert result["windows_evaluated"] == 1
    assert result["windows_detected"] == 1
    assert result["false_alert_points_outside_windows"] == 4
    assert result["false_alert_episodes_outside_windows"] == 2


def test_no_window_has_none_recall():
    times = sample_series(5)["timestamp"]
    result = score_alerts(times, np.zeros(5, dtype=bool), [])
    assert result["event_recall"] is None


def test_length_mismatch_rejected():
    with pytest.raises(ValueError, match="mesmo tamanho"):
        score_alerts(sample_series(3)["timestamp"], np.zeros(2, dtype=bool), [])


def test_nan_rejected(tmp_path):
    f = tmp_path / "input.csv"
    f.write_text("timestamp,value\n2024-01-01 00:00:00,1\n2024-01-01 00:05:00,\n", encoding="utf-8")
    with pytest.raises(ValueError, match="ausentes"):
        load_series(f)


def test_unsorted_rejected(tmp_path):
    f = tmp_path / "input.csv"
    f.write_text("timestamp,value\n2024-01-01 00:05:00,1\n2024-01-01 00:00:00,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="ordem crescente"):
        load_series(f)


def test_windows_loaded(tmp_path):
    f = tmp_path / "labels.json"
    f.write_text(json.dumps({"test.csv": [["2024-01-01 00:00:00", "2024-01-01 01:00:00"]]}))
    assert len(load_windows(f, "test.csv")) == 1


def test_overlapping_windows_rejected(tmp_path):
    f = tmp_path / "labels.json"
    f.write_text(json.dumps({"test.csv": [
        ["2024-01-01 00:00:00", "2024-01-01 01:00:00"],
        ["2024-01-01 00:30:00", "2024-01-01 02:00:00"],
    ]}))
    with pytest.raises(ValueError, match="sobrepostas"):
        load_windows(f, "test.csv")


def test_train_evaluate_without_test_label_leakage(tmp_path):
    """End-to-end on synthetic data; real NAB data must be evaluated by the user."""
    from service_anomaly_ml.train import fit_experiment
    source = sample_series(500)
    source["value"] = 10 + np.sin(np.arange(500) / 11) + np.random.default_rng(1).normal(0, 0.02, 500)
    csv = tmp_path / "data/raw/realAWSCloudwatch/ec2_cpu_utilization_24ae8d.csv"
    labels = tmp_path / "data/labels/combined_windows.json"
    csv.parent.mkdir(parents=True)
    labels.parent.mkdir(parents=True)
    source.to_csv(csv, index=False)
    labels.write_text(json.dumps({"realAWSCloudwatch/ec2_cpu_utilization_24ae8d.csv": [
        [str(source["timestamp"].iloc[450]), str(source["timestamp"].iloc[470])]
    ]}), encoding="utf-8")
    report = fit_experiment(tmp_path, export=False)
    assert report["training"]["feature_rows"] == 488
    assert report["models"]["isolation_forest"]["windows_evaluated"] == 1
    assert report["training"]["train_end"] < report["training"]["test_start"]


def test_training_rejects_annotation_overlap(tmp_path):
    from service_anomaly_ml.train import fit_experiment
    source = sample_series(500)
    csv = tmp_path / "data/raw/realAWSCloudwatch/ec2_cpu_utilization_24ae8d.csv"
    labels = tmp_path / "data/labels/combined_windows.json"
    csv.parent.mkdir(parents=True)
    labels.parent.mkdir(parents=True)
    source.to_csv(csv, index=False)
    labels.write_text(json.dumps({"realAWSCloudwatch/ec2_cpu_utilization_24ae8d.csv": [
        [str(source["timestamp"].iloc[100]), str(source["timestamp"].iloc[110])]
    ]}), encoding="utf-8")
    with pytest.raises(ValueError, match="Treino contém janelas"):
        fit_experiment(tmp_path, export=False)
