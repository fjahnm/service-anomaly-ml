"""Causal feature engineering: only current and historical observations."""
from __future__ import annotations

import numpy as np
import pandas as pd

HISTORY = 12  # 12 * 5 minutes = 1 hour for the selected NAB series
FEATURE_COLUMNS = [
    "value",
    "delta_1",
    "abs_delta_1",
    "past_mean_12",
    "past_std_12",
    "past_median_12",
    "abs_deviation_past_median",
]
BASELINE_COLUMN = "abs_deviation_past_median"


def make_features(series: pd.DataFrame) -> pd.DataFrame:
    """Use shift(1) so rolling aggregates never see the current/future values.

    Rows without a full 12-point history are removed. Do not generate these
    features using a centered rolling window or a global mean of all records.
    """
    values = series["value"].astype(float)
    prior = values.shift(1)
    past_mean = prior.rolling(HISTORY, min_periods=HISTORY).mean()
    past_std = prior.rolling(HISTORY, min_periods=HISTORY).std(ddof=0)
    past_median = prior.rolling(HISTORY, min_periods=HISTORY).median()
    delta = values - prior
    frame = pd.DataFrame({
        "timestamp": series["timestamp"],
        "value": values,
        "delta_1": delta,
        "abs_delta_1": delta.abs(),
        "past_mean_12": past_mean,
        "past_std_12": past_std,
        "past_median_12": past_median,
        "abs_deviation_past_median": (values - past_median).abs(),
    })
    frame = frame.dropna().reset_index(drop=True)
    if frame.empty or not np.isfinite(frame[FEATURE_COLUMNS].to_numpy()).all():
        raise ValueError("Sem observações suficientes ou atributos não finitos.")
    return frame
