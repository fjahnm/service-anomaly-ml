"""Fixed-protocol multi-series evaluation on the NAB realAWSCloudwatch corpus.

No test-window annotations are used to fit models, choose thresholds, or select
series. A separate unsupervised model is fitted per series. These are NOT NAB
official scores and do NOT test zero-shot cross-service generalization.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from urllib.parse import quote
from urllib.request import Request, urlopen

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data import file_sha256, load_windows
from .data_quality import load_nab_series_canonical
from .evaluation import in_windows, score_alerts
from .features import BASELINE_COLUMN, FEATURE_COLUMNS, make_features

FAMILY = "realAWSCloudwatch/"
LABEL_FILE = Path("data/labels/combined_windows.json")
RAW_DIR = Path("data/raw")
SOURCE_BASE = "https://raw.githubusercontent.com/numenta/NAB"
METHODS = ("isolation_forest", "rolling_median_baseline")


def series_keys(labels_file: Path, *, limit: int | None = None) -> list[str]:
    """Return every AWS series listed by the upstream label manifest.

    A test-sized `limit` is a deterministic prefix of alphabetical keys, never
    selected by evaluation scores. Full run uses all AWS keys, including
    unlabeled-normal series (empty anomaly windows).
    """
    if limit is not None and limit < 1:
        raise ValueError("--limit precisa ser >= 1")
    content = json.loads(labels_file.read_text(encoding="utf-8-sig"))
    if not isinstance(content, dict):
        raise ValueError("Arquivo de janelas precisa ser um objeto JSON")
    keys = sorted(k for k in content if isinstance(k, str) and k.startswith(FAMILY))
    if not keys:
        raise ValueError("Nenhuma série realAWSCloudwatch encontrada em combined_windows.json")
    for key in keys:
        path = PurePosixPath(key)
        if (len(path.parts) != 2 or path.parts[0] != FAMILY.rstrip("/")
                or not path.name.endswith(".csv") or path.name in (".", "..")
                or any(s in key for s in ("\\", "?", "#"))):
            raise ValueError(f"Chave NAB suspeita: {key!r}")
        if not isinstance(content[key], list):
            raise ValueError(f"Anotações com tipo inválido para {key}")
    return keys[:limit] if limit is not None else keys


def _download_file(url: str, target: Path) -> None:
    """Download with a size bound, writing atomically; never overwrite local data."""
    if target.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".partial")
    if temporary.exists():
        temporary.unlink()
    try:
        req = Request(url, headers={"User-Agent": "service-anomaly-ml/0.2"})
        with urlopen(req, timeout=45) as response, temporary.open("wb") as handle:
            total = 0
            while block := response.read(64 * 1024):
                total += len(block)
                if total > 25 * 1024 * 1024:
                    raise ValueError(f"Download muito grande (limite 25 MiB): {url}")
                handle.write(block)
        temporary.replace(target)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def download_series(root: Path, keys: list[str], *, ref: str = "master") -> dict:
    """Fetch only missing source CSV files; preserve all existing local data."""
    if not ref or not all(c.isalnum() or c in "-_." for c in ref):
        raise ValueError("Referência NAB inválida; use um branch ou SHA de commit")
    summary = {"already_present": 0, "downloaded": 0, "normalized_series": 0,
               "duplicate_timestamp_rows_merged": 0, "out_of_order_transitions": 0}
    for key in keys:
        target = root / RAW_DIR / key
        if target.is_file():
            summary["already_present"] += 1
        else:
            url = f"{SOURCE_BASE}/{quote(ref, safe='')}/data/{quote(key, safe='/')}"
            print(f"Baixando {key}", flush=True)
            _download_file(url, target)
            summary["downloaded"] += 1
        # Validate even existing files; audit irregular timestamps without changing CSV.
        _, quality = load_nab_series_canonical(target)
        if quality["changed"]:
            summary["normalized_series"] += 1
            summary["duplicate_timestamp_rows_merged"] += quality["duplicate_timestamp_rows_merged"]
            summary["out_of_order_transitions"] += quality["out_of_order_transitions"]
            print(f"  Auditoria temporal {key}: "
                  f"{quality['out_of_order_transitions']} inversões, "
                  f"{quality['duplicate_timestamp_rows_merged']} linhas repetidas "
                  f"(mediana por instante); original preservado", flush=True)
    return summary


def partition(frame: pd.DataFrame, *, train_fraction: float = .50,
              validation_fraction: float = .15) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not (0.3 <= train_fraction <= .75):
        raise ValueError("train_fraction precisa estar entre 0.30 e 0.75")
    if not (.05 <= validation_fraction <= .30):
        raise ValueError("validation_fraction precisa estar entre 0.05 e 0.30")
    if train_fraction + validation_fraction > .90:
        raise ValueError("Teste precisa representar pelo menos 10% das linhas")
    n = len(frame)
    a = int(n * train_fraction)
    b = int(n * (train_fraction + validation_fraction))
    if min(a, b-a, n-b) < 30:
        raise ValueError("Série muito curta para treino, validação e teste")
    parts = (frame.iloc[:a], frame.iloc[a:b], frame.iloc[b:])
    if not (parts[0]["timestamp"].iloc[-1] < parts[1]["timestamp"].iloc[0]
            < parts[1]["timestamp"].iloc[-1] < parts[2]["timestamp"].iloc[0]):
        raise ValueError("Partições não respeitam a ordem temporal")
    return parts


def _window_count(timestamps: pd.Series, windows: list) -> int:
    start, end = timestamps.iloc[0], timestamps.iloc[-1]
    return sum(wend >= start and wstart <= end for wstart, wend in windows)


def run_one(key: str, root: Path, *, quantile: float = .98,
            seed: int = 42, trees: int = 200,
            train_fraction: float = .50,
            validation_fraction: float = .15) -> dict:
    """Fit on the training prefix; thresholds from training scores ONLY.

    NAB label windows are intentionally opened after fit and calibration so
    they cannot influence sampling, training, or threshold selection.
    Training may contain *unobserved* anomalies; this is an unsupervised study.
    """
    if not (.90 <= quantile < 1):
        raise ValueError("quantile precisa estar entre 0.90 e 1.0")
    if trees < 10 or trees > 1000:
        raise ValueError("trees precisa estar entre 10 e 1000")
    data_path = root / RAW_DIR / key
    original, quality = load_nab_series_canonical(data_path)
    frame = make_features(original)
    train, validation, test = partition(frame, train_fraction=train_fraction,
                                        validation_fraction=validation_fraction)

    model = Pipeline([
        ("scale", StandardScaler()),
        ("isolation_forest", IsolationForest(
            n_estimators=trees, max_samples="auto", contamination="auto",
            random_state=seed, n_jobs=-1)),
    ])
    model.fit(train[FEATURE_COLUMNS])
    training_scores = -model.score_samples(train[FEATURE_COLUMNS])
    threshold = float(np.quantile(training_scores, quantile))
    baseline_threshold = float(np.quantile(train[BASELINE_COLUMN].to_numpy(dtype=float), quantile))
    # Labels are read *only now*, for reporting the frozen evaluation protocol.
    windows = load_windows(root / LABEL_FILE, key)
    outputs = {}
    for part_name, subset in (("validation", validation), ("test", test)):
        scores = -model.score_samples(subset[FEATURE_COLUMNS])
        baseline_scores = subset[BASELINE_COLUMN].to_numpy(dtype=float)
        iso_flags = scores > threshold
        baseline_flags = baseline_scores > baseline_threshold
        window_mask = in_windows(subset["timestamp"], windows)
        outputs[part_name] = {
            "windows_in_partition": _window_count(subset["timestamp"], windows),
            "off_window_observations": int((~window_mask).sum()),
            "isolation_forest": score_alerts(subset["timestamp"], iso_flags, windows),
            "rolling_median_baseline": score_alerts(subset["timestamp"], baseline_flags, windows),
        }

    test_start = test["timestamp"].iloc[0]
    test_end = test["timestamp"].iloc[-1]
    split_windows = sum(start < test_start <= end for start, end in windows)
    return {
        "dataset": key,
        "dataset_sha256": file_sha256(data_path),
        "raw_rows": quality["raw_rows"],
        "original_rows": len(original),  # Existing key means canonical rows in v0.2.1.
        "data_quality": quality,
        "feature_rows": len(frame),
        "train_rows": len(train),
        "validation_rows": len(validation),
        "test_rows": len(test),
        "train_start": str(train["timestamp"].iloc[0]),
        "train_end": str(train["timestamp"].iloc[-1]),
        "validation_start": str(validation["timestamp"].iloc[0]),
        "validation_end": str(validation["timestamp"].iloc[-1]),
        "test_start": str(test_start),
        "test_end": str(test_end),
        "total_labeled_windows": len(windows),
        "windows_intersect_training": _window_count(train["timestamp"], windows),
        "windows_intersect_validation": outputs["validation"]["windows_in_partition"],
        "windows_intersect_test": outputs["test"]["windows_in_partition"],
        "windows_crossing_test_boundary": split_windows,
        "thresholds": {
            "isolation_forest_score": threshold,
            "rolling_median_deviation": baseline_threshold,
        },
        "validation": outputs["validation"],
        "test": outputs["test"],
    }


def aggregate(rows: list[dict], method: str) -> dict:
    n_windows = sum(r["test"][method]["windows_evaluated"] for r in rows)
    detected = sum(r["test"][method]["windows_detected"] for r in rows)
    off_points = sum(r["test"][method]["false_alert_points_outside_windows"] for r in rows)
    off_episodes = sum(r["test"][method]["false_alert_episodes_outside_windows"] for r in rows)
    exposure = sum(r["test"]["off_window_observations"] for r in rows)
    relevant = [r["test"][method]["event_recall"] for r in rows
                if r["test"][method]["event_recall"] is not None]
    return {
        "series_count": len(rows),
        "series_with_test_windows": len(relevant),
        "series_without_test_windows": len(rows) - len(relevant),
        "test_windows": n_windows,
        "detected_test_windows": detected,
        "micro_event_recall": round(detected / n_windows, 6) if n_windows else None,
        "macro_event_recall": round(float(np.mean(relevant)), 6) if relevant else None,
        "off_window_alert_points": off_points,
        "off_window_alert_episodes": off_episodes,
        "off_window_exposure_points": exposure,
        "off_window_alert_points_per_1000_observations": round(
            1000 * off_points / exposure, 4) if exposure else None,
        "test_alert_points": sum(r["test"][method]["total_alert_points"] for r in rows),
        "test_observations": sum(r["test"][method]["total_test_points"] for r in rows),
    }


def _flat_rows(rows: list[dict]) -> list[dict]:
    result = []
    for row in rows:
        entry = {k: row[k] for k in (
            "dataset", "raw_rows", "original_rows", "train_rows", "validation_rows", "test_rows",
            "total_labeled_windows", "windows_intersect_training",
            "windows_intersect_validation", "windows_intersect_test",
            "windows_crossing_test_boundary")}
        quality = row["data_quality"]
        for field in ("out_of_order_transitions", "duplicate_timestamp_rows_merged",
                      "duplicate_timestamp_groups", "conflicting_duplicate_value_groups"):
            entry[field] = quality[field]
        for method in METHODS:
            scores = row["test"][method]
            prefix = "iso" if method == "isolation_forest" else "baseline"
            for key in ("windows_detected", "windows_evaluated", "event_recall",
                        "false_alert_points_outside_windows", "false_alert_episodes_outside_windows",
                        "test_alert_rate"):
                entry[f"{prefix}_{key}"] = scores[key]
        result.append(entry)
    return result


def _write_plot(rows: list[dict], path: Path) -> None:
    dataset_names = [PurePosixPath(r["dataset"]).stem for r in rows]
    indices = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(max(12, len(rows) * .72), 6))
    width = .38
    for offset, method, label in ((-.5, METHODS[0], "Isolation Forest"),
                                  (.5, METHODS[1], "Rolling-median baseline")):
        values = [r["test"][method]["false_alert_points_outside_windows"] for r in rows]
        ax.bar(indices + offset * width, values, width, label=label)
    ax.set_xticks(indices, dataset_names, rotation=70, ha="right")
    ax.set_ylabel("Pontos de alerta fora das janelas anotadas (teste)")
    ax.set_title("NAB realAWSCloudwatch — comparação por série (NÃO é score oficial)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _report_md(report: dict) -> str:
    models = report["aggregate"]
    lines = [
        "# Multi-series experiment — NAB realAWSCloudwatch", "",
        "**Status:** local experimental benchmark, not official NAB scoring.", "",
        f"- Series evaluated: **{len(report['series'])}**",
        f"- NAB labels SHA-256: `{report['provenance']['labels_sha256']}`",
        "- An independent Isolation Forest was trained for EACH series; this does not test unseen-series transfer.",
        "- Strict chronological train/validation/test split; only train scores calibrate thresholds.",
        "- Every AWS series listed in `combined_windows.json` is included unless `--limit` was explicitly used.",
        "- Both models share the quantile policy but have different alert counts; there is no fixed alert quota.",
        "- Irregular timestamps are sorted and duplicate instants consolidated by median ONLY for this experimental run; original CSVs remain unchanged.",
        "- Timestamp normalization changes the sample sequence and is NOT the official NAB preprocessing.",
        "", "## Input quality and normalization", "",
        f"- Series with changed timestamps: **{report['data_quality']['series_normalized']}**",
        f"- Descending timestamp transitions: **{report['data_quality']['out_of_order_transitions']}**",
        f"- Duplicate timestamp rows consolidated: **{report['data_quality']['duplicate_timestamp_rows_merged']}**",
        f"- Conflicting-value timestamp groups: **{report['data_quality']['conflicting_duplicate_value_groups']}**",
        "", "## Aggregate holdout comparison", "",
        "| Metric | Isolation Forest | Rolling median |", "|---|---:|---:|",
    ]
    keys = [
        ("Detected / evaluable test windows", "detected_test_windows", "test_windows"),
        ("Micro event recall", "micro_event_recall", None),
        ("Macro event recall (eligible series only)", "macro_event_recall", None),
        ("Alert points outside windows", "off_window_alert_points", None),
        ("Alert episodes outside windows", "off_window_alert_episodes", None),
        ("Outside-window alert points per 1,000 outside-window observations",
         "off_window_alert_points_per_1000_observations", None),
    ]
    for label, key, denominator in keys:
        vals = []
        for method in METHODS:
            data = models[method]
            value = str(data[key]) if data[key] is not None else "N/A"
            if denominator:
                value += f" / {data[denominator]}"
            vals.append(value)
        lines.append(f"| {label} | {vals[0]} | {vals[1]} |")
    lines.extend([
        "", f"**Series without an annotated window in test:** "
        f"{models[METHODS[0]]['series_without_test_windows']}. They still contribute to false-alert totals.",
        "", "## Series-level results", "",
        "| Dataset | Test windows | Isolation Forest hits | Baseline hits | ISO off-window points | Baseline off-window points |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for r in report["series"]:
        iso, base = r["test"][METHODS[0]], r["test"][METHODS[1]]
        lines.append(f"| `{r['dataset']}` | {iso['windows_evaluated']} | {iso['windows_detected']} | "
                     f"{base['windows_detected']} | {iso['false_alert_points_outside_windows']} | "
                     f"{base['false_alert_points_outside_windows']} |")
    lines.extend([
        "", "## Limitations", "",
        "- NAB windows are tolerance intervals, not exhaustive point-level anomaly truth.",
        "- Median consolidation of duplicate timestamps can change the sample distribution, measured alert rates, and comparability to other NAB experiments.",
        "- Off-window alerts are proxies, not verified false positives.",
        "- No official NAB normalized score, point-level F1, or cross-service transfer performance is claimed.",
        "- Models may learn from anomalous observations in the training prefix; labels never influenced training or thresholds.",
        "- Windows straddling partition boundaries may be partially observable in validation/test.",
        "- Calibration is per-series and fixed at the declared train-score quantile; repeated test-driven tuning is prohibited.",
        "- A full multi-series run does not retroactively update the previously published Hugging Face model.",
        "", "## Provenance", "",
        f"- NAB reference: `{report['provenance']['nab_ref']}` (branch name if not immutable commit)",
        "- Source: https://github.com/numenta/NAB",
        "- Individual input SHA-256 digests are saved in `summary.json`.",
        "", "## Files", "",
        "- `summary.json`: full provenance, per-series scores and aggregates.",
        "- `per_series.csv`: compact per-series comparison.",
        "- `comparison.png`: grouped plot of off-window alert points.",
        "", "## Citation", "",
        "Ahmad, S., Lavin, A., Purdy, S., & Agha, Z. (2017). "
        "*Unsupervised real-time anomaly detection for streaming data.* Neurocomputing. "
        "https://doi.org/10.1016/j.neucom.2017.04.070",
        "",
    ])
    return "\n".join(lines)


def run_many(root: Path, *, limit: int | None = None, quantile: float = .98,
             seed: int = 42, trees: int = 200,
             train_fraction: float = .50, validation_fraction: float = .15,
             report_dir: Path | None = None, nab_ref: str = "master") -> dict:
    labels = root / LABEL_FILE
    keys = series_keys(labels, limit=limit)
    results = []
    for i, key in enumerate(keys, 1):
        print(f"[{i}/{len(keys)}] {key}", flush=True)
        try:
            results.append(run_one(key, root, quantile=quantile, seed=seed,
                                   trees=trees, train_fraction=train_fraction,
                                   validation_fraction=validation_fraction))
        except Exception as exc:
            raise RuntimeError(f"Falha na série {key}: {exc}") from exc
    quality_summary = {
        "series_normalized": sum(r["data_quality"]["changed"] for r in results),
        "raw_rows": sum(r["data_quality"]["raw_rows"] for r in results),
        "canonical_rows": sum(r["data_quality"]["canonical_rows"] for r in results),
        "out_of_order_transitions": sum(r["data_quality"]["out_of_order_transitions"] for r in results),
        "duplicate_timestamp_rows_merged": sum(r["data_quality"]["duplicate_timestamp_rows_merged"] for r in results),
        "conflicting_duplicate_value_groups": sum(r["data_quality"]["conflicting_duplicate_value_groups"] for r in results),
        "normalization_policy": "stable sort then median per identical timestamp; no raw overwrite",
    }
    report = {
        "protocol": "experimental_NAB_AWS_per_series_unsupervised_v0.2.1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "provenance": {"nab_ref": nab_ref, "labels_sha256": file_sha256(labels),
                       "source": "https://github.com/numenta/NAB"},
        "configuration": {
            "family": FAMILY, "selected_series": len(keys),
            "is_full_aws_set": limit is None,
            "train_fraction": train_fraction, "validation_fraction": validation_fraction,
            "test_fraction": round(1 - train_fraction - validation_fraction, 6),
            "threshold_train_quantile": quantile, "trees": trees, "seed": seed,
            "features": FEATURE_COLUMNS,
            "python_version": __import__("platform").python_version(),
            "sklearn_version": sklearn.__version__,
        },
        "data_quality": quality_summary,
        "aggregate": {method: aggregate(results, method) for method in METHODS},
        "series": results,
    }
    out = report_dir or root / "artifacts" / "multi_series"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame(_flat_rows(results)).to_csv(out / "per_series.csv", index=False)
    (out / "MULTI_SERIES_REPORT.md").write_text(_report_md(report), encoding="utf-8")
    _write_plot(results, out / "comparison.png")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the full NAB realAWSCloudwatch family without label leakage")
    parser.add_argument("action", choices=("download", "run"), help="Download missing CSVs or evaluate all series")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Local repository root")
    parser.add_argument("--limit", type=int, default=None, help="SMOKE TEST ONLY: first N alphabetical series")
    parser.add_argument("--nab-ref", default="master", help="NAB branch or immutable commit SHA")
    parser.add_argument("--trees", type=int, default=200)
    parser.add_argument("--quantile", type=float, default=.98)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-fraction", type=float, default=.50)
    parser.add_argument("--validation-fraction", type=float, default=.15)
    parser.add_argument("--report-dir", type=Path, default=None)
    args = parser.parse_args()
    root = args.root.resolve()
    keys = series_keys(root / LABEL_FILE, limit=args.limit)
    if args.action == "download":
        result = download_series(root, keys, ref=args.nab_ref)
        print(f"Séries verificadas: {len(keys)} | Baixadas: {result['downloaded']} "
              f"| Já existiam: {result['already_present']} "
              f"| Séries normalizadas: {result['normalized_series']} "
              f"| Duplicações: {result['duplicate_timestamp_rows_merged']} "
              f"| Inversões: {result['out_of_order_transitions']}")
        return
    # Deliberately do not download during evaluation: missing inputs are a reproducibility error.
    report = run_many(root, limit=args.limit, quantile=args.quantile, seed=args.seed,
                      trees=args.trees, train_fraction=args.train_fraction,
                      validation_fraction=args.validation_fraction,
                      report_dir=args.report_dir, nab_ref=args.nab_ref)
    for method, item in report["aggregate"].items():
        print(f"{method}: eventos {item['detected_test_windows']}/{item['test_windows']} | "
              f"recall micro={item['micro_event_recall']} | "
              f"alertas fora das janelas={item['off_window_alert_points']} pontos / "
              f"{item['off_window_alert_episodes']} episódios")
    dest = args.report_dir or root / "artifacts" / "multi_series"
    print(f"Relatórios: {dest}")
    q = report["data_quality"]
    print(f"Qualidade dos dados: {q['series_normalized']} séries normalizadas, "
          f"{q['duplicate_timestamp_rows_merged']} linhas duplicadas consolidadas "
          f"e {q['out_of_order_transitions']} inversões temporais; originais preservados")
    print("Avaliação experimental; NÃO é o score oficial NAB.")


if __name__ == "__main__":
    main()
