"""Regression tests for NAB timestamp irregularities and transparent reports."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from service_anomaly_ml.data_quality import load_nab_series_canonical
from service_anomaly_ml.multi_series import download_series, run_many


def _csv(tmp_path: Path, rows: list[tuple[str, float]]) -> Path:
    path = tmp_path / "data/raw/realAWSCloudwatch/problem.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["timestamp", "value"]).to_csv(path, index=False)
    return path


def test_clean_input_remains_identical(tmp_path):
    path = _csv(tmp_path, [("2024-01-01 00:00:00", 1),
                           ("2024-01-01 00:05:00", 2),
                           ("2024-01-01 00:10:00", 3)])
    df, q = load_nab_series_canonical(path)
    assert df.value.tolist() == [1, 2, 3]
    assert not q["changed"]
    assert q["canonical_rows"] == q["raw_rows"] == 3


def test_out_of_order_and_duplicate_conflict_is_audited(tmp_path):
    path = _csv(tmp_path, [("2024-01-01 00:10:00", 20),
                           ("2024-01-01 00:00:00", 3),
                           ("2024-01-01 00:05:00", 7),
                           ("2024-01-01 00:05:00", 9)])
    source = path.read_bytes()
    df, q = load_nab_series_canonical(path)
    assert df.value.tolist() == [3, 8, 20]
    assert df.timestamp.is_monotonic_increasing
    assert q["raw_rows"] == 4 and q["canonical_rows"] == 3
    assert q["out_of_order_transitions"] == 1
    assert q["duplicate_timestamp_rows_merged"] == 1
    assert q["duplicate_timestamp_groups"] == 1
    assert q["conflicting_duplicate_value_groups"] == 1
    assert path.read_bytes() == source  # Never mutate original benchmark data.


def test_invalid_values_are_rejected_instead_of_silently_dropped(tmp_path):
    path = _csv(tmp_path, [("2024-01-01 00:00:00", 5),
                           ("2024-01-01 00:05:00", float('nan'))])
    with pytest.raises(ValueError, match="ausentes"):
        load_nab_series_canonical(path)


def test_download_can_audit_existing_irregular_csv(tmp_path):
    key = "realAWSCloudwatch/problem.csv"
    _csv(tmp_path, [("2024-01-01 00:10:00", 2),
                    ("2024-01-01 00:00:00", 5),
                    ("2024-01-01 00:00:00", 6)])
    result = download_series(tmp_path, [key])
    assert result["already_present"] == 1
    assert result["normalized_series"] == 1
    assert result["duplicate_timestamp_rows_merged"] == 1
    assert result["out_of_order_transitions"] == 1


def test_full_report_records_data_quality(tmp_path):
    key = "realAWSCloudwatch/problem.csv"
    timestamps = pd.date_range("2024-01-01", periods=420, freq="5min")
    values = [float((i % 12) + 1) for i in range(420)]
    path = _csv(tmp_path, [(str(t), v) for t, v in zip(timestamps, values)])
    df = pd.read_csv(path)
    df.loc[100, 'value'] = 100
    df.loc[101, 'timestamp'] = df.loc[100, 'timestamp']
    df = pd.concat([df.iloc[:110], df.iloc[111:112], df.iloc[110:111], df.iloc[112:]], ignore_index=True)
    df.to_csv(path, index=False)
    lbl = tmp_path / "data/labels/combined_windows.json"
    lbl.parent.mkdir(parents=True, exist_ok=True)
    lbl.write_text(json.dumps({key: []}), encoding="utf-8")
    report = run_many(tmp_path, trees=12, report_dir=tmp_path / "artifacts/multi_series")
    assert report["data_quality"]["series_normalized"] == 1
    assert report["data_quality"]["duplicate_timestamp_rows_merged"] == 1
    assert report["series"][0]["dataset_sha256"]
    assert report["series"][0]["raw_rows"] == 420
    assert report["series"][0]["original_rows"] == 419
    csv = pd.read_csv(tmp_path / "artifacts/multi_series/per_series.csv")
    assert csv.loc[0, "duplicate_timestamp_rows_merged"] == 1
    md = (tmp_path / "artifacts/multi_series/MULTI_SERIES_REPORT.md").read_text(encoding='utf8')
    assert "Input quality and normalization" in md
