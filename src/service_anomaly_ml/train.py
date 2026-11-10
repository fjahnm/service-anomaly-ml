"""Fit Isolation Forest with a chronological holdout; export reproducible artifacts."""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Works on Windows, CI and headless environments
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data import DATASET_KEY, default_paths, file_sha256, load_series, load_windows
from .evaluation import in_windows, score_alerts
from .features import BASELINE_COLUMN, FEATURE_COLUMNS, make_features


def fit_experiment(root: Path, *, train_fraction: float = 0.70,
                   quantile: float = 0.98, seed: int = 42,
                   export: bool = True) -> dict:
    """Train without labels; evaluate once on later unseen timestamps.

    This first exploration has no labeled validation split or hyperparameter
    search. Do not tune thresholds based on test-window detection results.
    """
    if not (0.5 <= train_fraction <= 0.85):
        raise ValueError("train_fraction deve estar entre 0.5 e 0.85")
    if not (0.90 <= quantile < 1.0):
        raise ValueError("quantile deve estar entre 0.90 e 1.0")
    csv_path, windows_path = default_paths(root)
    raw = load_series(csv_path)
    windows = load_windows(windows_path)
    frame = make_features(raw)
    cutoff = int(len(frame) * train_fraction)
    train, test = frame.iloc[:cutoff], frame.iloc[cutoff:]
    if len(train) < 100 or len(test) < 100:
        raise ValueError("Não há registros suficientes para treino/teste")
    if np.any(in_windows(train["timestamp"], windows)):
        raise ValueError(
            "Treino contém janelas anotadas. Revise o período/estratégia antes de prosseguir."
        )

    model = Pipeline([
        ("scale", StandardScaler()),
        ("isolation_forest", IsolationForest(
            n_estimators=200,
            max_samples="auto",
            contamination="auto",  # Do not rely on sklearn's default threshold
            random_state=seed,
            n_jobs=-1,
        )),
    ])
    model.fit(train[FEATURE_COLUMNS])

    # Invert sklearn's score_samples: bigger number means more anomalous.
    training_scores = -model.score_samples(train[FEATURE_COLUMNS])
    test_scores = -model.score_samples(test[FEATURE_COLUMNS])
    iso_threshold = float(np.quantile(training_scores, quantile))
    baseline_train = train[BASELINE_COLUMN].to_numpy(dtype=float)
    baseline_test = test[BASELINE_COLUMN].to_numpy(dtype=float)
    baseline_threshold = float(np.quantile(baseline_train, quantile))

    # Thresholds are calibrated ONLY on training data, never on test labels.
    iso_flags = test_scores > iso_threshold
    baseline_flags = baseline_test > baseline_threshold

    report = {
        "project": "service-anomaly-ml",
        "dataset": DATASET_KEY,
        "dataset_sha256": file_sha256(csv_path),
        "labels_sha256": file_sha256(windows_path),
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
        "training": {
            "original_rows": len(raw),
            "feature_rows": len(frame),
            "train_rows": len(train),
            "test_rows": len(test),
            "train_start": str(train["timestamp"].iloc[0]),
            "train_end": str(train["timestamp"].iloc[-1]),
            "test_start": str(test["timestamp"].iloc[0]),
            "test_end": str(test["timestamp"].iloc[-1]),
            "train_fraction": train_fraction,
            "quantile": quantile,
            "seed": seed,
            "feature_columns": FEATURE_COLUMNS,
            "method": "Training-only unsupervised threshold calibration; 70/30 chronological split by default",
        },
        "thresholds": {
            "isolation_forest_anomaly_score": iso_threshold,
            "baseline_abs_deviation": baseline_threshold,
        },
        "models": {
            "isolation_forest": score_alerts(test["timestamp"], iso_flags, windows),
            "rolling_median_baseline": score_alerts(test["timestamp"], baseline_flags, windows),
        },
        "caveats": [
            "These metrics are NOT official NAB scores.",
            "Windows are event-timing tolerance intervals, not ground truth for every point.",
            "Two annotated events in a single series are insufficient to claim generalization.",
            "No labels were used to fit the model or choose thresholds.",
            "Test labels should not be used for iterative hyperparameter selection.",
        ],
    }

    if export:
        import skops.io as sio

        out = root / "artifacts"
        out.mkdir(parents=True, exist_ok=True)
        sio.dump(model, out / "isolation_forest.skops")
        (out / "experiment.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        results = pd.DataFrame({
            "timestamp": test["timestamp"].to_numpy(),
            "value": test["value"].to_numpy(),
            "isolation_score": test_scores,
            "isolation_alert": iso_flags.astype(int),
            "baseline_score": baseline_test,
            "baseline_alert": baseline_flags.astype(int),
        })
        results.to_csv(out / "test_predictions.csv", index=False)
        for name, flags in (("isolation_forest", iso_flags), ("rolling_median_baseline", baseline_flags)):
            _save_plot(frame, test, windows, flags, name, out / f"{name}.png")
    return report


def _save_plot(frame: pd.DataFrame, test: pd.DataFrame, windows: list,
               flags: np.ndarray, name: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(frame["timestamp"], frame["value"], linewidth=1, label="CPU (série original)")
    for index, (start, end) in enumerate(windows):
        ax.axvspan(start, end, alpha=0.17, label="Janela NAB" if index == 0 else None)
    ax.axvline(test["timestamp"].iloc[0], linestyle="--", linewidth=1, label="Início do teste")
    if np.any(flags):
        ax.scatter(test.loc[flags, "timestamp"], test.loc[flags, "value"],
                   s=16, label="Alerta no teste", zorder=3)
    ax.set_title(f"{name} | avaliação fora do treino")
    ax.set_xlabel("Timestamp do dataset")
    ax.set_ylabel("CPU — valor original do NAB")
    ax.legend(loc="best")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an Isolation Forest on a NAB CPU series")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--quantile", type=float, default=0.98)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    report = fit_experiment(args.root.resolve(), train_fraction=args.train_fraction,
                            quantile=args.quantile, seed=args.seed)
    t = report["training"]
    print(f"Dataset: {report['dataset']} | {t['original_rows']} linhas")
    print(f"Treino: {t['train_rows']} | Teste: {t['test_rows']}")
    for name, result in report["models"].items():
        print(f"{name}: eventos {result['windows_detected']}/{result['windows_evaluated']} | "
              f"falsos alertas fora das janelas: {result['false_alert_points_outside_windows']} pontos, "
              f"{result['false_alert_episodes_outside_windows']} episódios")
    print("Resultados e gráficos: artifacts/")
    print("Aviso: estas NÃO são métricas oficiais do NAB.")


if __name__ == "__main__":
    main()
