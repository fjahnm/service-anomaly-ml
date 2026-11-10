"""Auditable timestamp normalization for the experimental multi-series NAB run.

The original CSV is NEVER modified. Duplicates at the same instant are
consolidated by a deterministic median; out-of-order rows are sorted by time.
This is not official NAB preprocessing. Single-series v0.1 remains strict.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_nab_series_canonical(path: Path) -> tuple[pd.DataFrame, dict]:
    """Parse/validate the raw CSV, return canonical rows and a quality audit."""
    if not path.is_file():
        raise FileNotFoundError(f"CSV não encontrado: {path}")
    raw = pd.read_csv(path)
    if list(raw.columns) != ["timestamp", "value"]:
        raise ValueError("Dataset NAB precisa conter 'timestamp' e 'value', nessa ordem.")
    if raw.empty:
        raise ValueError("Dataset NAB vazio")
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], errors="raise", utc=False)
    raw["value"] = pd.to_numeric(raw["value"], errors="raise")
    if raw.isna().any().any() or not np.isfinite(raw["value"].to_numpy(dtype=float)).all():
        raise ValueError("Dataset NAB contém valores ausentes ou não finitos")
    if raw["timestamp"].dt.tz is not None:
        raise ValueError("Esta versão espera timestamps sem timezone (formato NAB)")

    order_inversions = int((raw["timestamp"].diff() < pd.Timedelta(0)).sum())
    duplicate_rows = int(raw["timestamp"].duplicated().sum())
    groups = raw.groupby("timestamp", sort=True)["value"]
    group_sizes = groups.size()
    duplicate_groups = int((group_sizes > 1).sum())
    conflict_groups = int((groups.nunique() > 1).sum())

    # Sorting is stable and consolidation deterministic, independent of input order.
    # Mean/median is calculated within one timestamp ONLY, not over later timestamps.
    canonical = (raw.sort_values("timestamp", kind="mergesort")
                   .groupby("timestamp", as_index=False, sort=True)["value"].median())
    if canonical.empty or not canonical["timestamp"].is_monotonic_increasing or canonical["timestamp"].duplicated().any():
        raise ValueError("Não foi possível obter uma série temporal válida")
    if not np.isfinite(canonical["value"].to_numpy(dtype=float)).all():
        raise ValueError("Normalização produziu valores não finitos")

    report = {
        "raw_rows": int(len(raw)),
        "canonical_rows": int(len(canonical)),
        "out_of_order_transitions": order_inversions,
        "duplicate_timestamp_rows_merged": duplicate_rows,
        "duplicate_timestamp_groups": duplicate_groups,
        "conflicting_duplicate_value_groups": conflict_groups,
        "changed": bool(order_inversions or duplicate_rows),
        "policy": "stable chronological sort; median of values for identical timestamps",
    }
    return canonical.reset_index(drop=True), report
