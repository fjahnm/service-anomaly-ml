"""NAB window-tolerant event detection (NOT official NAB scoring)."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

Window = tuple[pd.Timestamp, pd.Timestamp]


def in_windows(timestamps: pd.Series, windows: Sequence[Window]) -> np.ndarray:
    out = np.zeros(len(timestamps), dtype=bool)
    for start, end in windows:
        out |= ((timestamps >= start) & (timestamps <= end)).to_numpy(dtype=bool)
    return out


def score_alerts(
    timestamps: pd.Series, predictions: np.ndarray, windows: Sequence[Window]
) -> dict:
    """Report event-level hits and off-window alerts, not inflated pointwise F1.

    Each window contributes at most one event hit. Off-window episodes count
    contiguous alert runs. Official NAB scoring weights timing, duplicates and
    costs differently; these quantities MUST NOT be called a NAB score.
    """
    flags = np.asarray(predictions, dtype=bool)
    if len(flags) != len(timestamps):
        raise ValueError("Previsões e timestamps precisam ter o mesmo tamanho")
    if len(timestamps) == 0:
        raise ValueError("O período avaliado está vazio")
    mask = in_windows(timestamps, windows)
    windows_in_test = [(start, end) for start, end in windows if
                       end >= timestamps.iloc[0] and start <= timestamps.iloc[-1]]
    windows_hit = 0
    for start, end in windows_in_test:
        window_mask = ((timestamps >= start) & (timestamps <= end)).to_numpy(dtype=bool)
        windows_hit += bool(np.any(flags & window_mask))
    false_flags = flags & ~mask
    previous = np.concatenate(([False], false_flags[:-1]))
    false_episodes = int(np.sum(false_flags & ~previous))
    return {
        "windows_evaluated": len(windows_in_test),
        "windows_detected": windows_hit,
        "event_recall": (windows_hit / len(windows_in_test)) if windows_in_test else None,
        "false_alert_points_outside_windows": int(false_flags.sum()),
        "false_alert_episodes_outside_windows": false_episodes,
        "total_alert_points": int(flags.sum()),
        "total_test_points": int(len(flags)),
        "test_alert_rate": round(float(flags.mean()), 6),
        "note": "Métricas experimentais por janela; não equivalem à pontuação oficial NAB nem a F1 ponto a ponto.",
    }
