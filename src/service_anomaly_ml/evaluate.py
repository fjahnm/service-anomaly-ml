"""Recompute metrics from saved predictions; do not retrain or relabel."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .data import default_paths, file_sha256, load_windows
from .evaluation import score_alerts


def evaluate_saved(root: Path) -> dict:
    out = root / "artifacts"
    report = json.loads((out / "experiment.json").read_text(encoding="utf-8"))
    _, windows_path = default_paths(root)
    if file_sha256(windows_path) != report["labels_sha256"]:
        raise ValueError("O arquivo de anotações foi alterado desde o treino.")
    df = pd.read_csv(out / "test_predictions.csv", parse_dates=["timestamp"])
    if not df["timestamp"].is_monotonic_increasing or df["timestamp"].duplicated().any():
        raise ValueError("Arquivo de previsões não está ordenado ou tem timestamps duplicados.")
    if len(df) != report["training"]["test_rows"]:
        raise ValueError("Número de previsões diferente do relatório original.")
    windows = load_windows(windows_path)
    result = {
        "isolation_forest": score_alerts(df["timestamp"], df["isolation_alert"].to_numpy(), windows),
        "rolling_median_baseline": score_alerts(df["timestamp"], df["baseline_alert"].to_numpy(), windows),
    }
    if result != report["models"]:
        raise ValueError("Avaliação recalculada difere do treino; verifique os arquivos.")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify evaluation metrics without retraining")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    result = evaluate_saved(args.root.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("Avaliação confirmada — sem modificar modelo nem parâmetros.")


if __name__ == "__main__":
    main()
