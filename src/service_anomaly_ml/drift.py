"""Auditable, descriptive input-feature drift diagnostics for the NAB AWS experiment.

Runs AFTER the v0.2.1 benchmark. Never selects a model/threshold or fits on
validation/test. PSI bins come from training only, and KS is a *distance* (no
p-values, since time series observations are autocorrelated). Previously
published benchmark scores are linked for exploratory investigation only.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data import file_sha256
from .data_quality import load_nab_series_canonical
from .features import FEATURE_COLUMNS, make_features
from .multi_series import LABEL_FILE, RAW_DIR, partition, series_keys

PROTOCOL = "descriptive_feature_drift_v0.3.0"
PARENT_PROTOCOL = "experimental_NAB_AWS_per_series_unsupervised_v0.2.1"
DEFAULT_FOCUS = (
    "ec2_cpu_utilization_5f5533.csv",
    "rds_cpu_utilization_e47b3b.csv",
    "rds_cpu_utilization_cc0c53.csv",
    "ec2_cpu_utilization_fe7f93.csv",
    "ec2_disk_write_bytes_c0d644.csv",
)


def _finite(values: np.ndarray | pd.Series) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size < 2 or not np.isfinite(array).all():
        raise ValueError("Uma comparação de drift exige >=2 valores finitos em cada partição")
    return array


def reference_edges(reference: np.ndarray | pd.Series, bins: int = 10) -> np.ndarray:
    """Train-only quantile edges, with extra bins for values outside train range.

    A rightward nextafter at max avoids counting the largest training value as
    out-of-support. For a constant reference, below/equal/above are distinct.
    Duplicate quantile cuts are removed instead of inventing fake intervals.
    """
    ref = _finite(reference)
    if not 2 <= bins <= 50:
        raise ValueError("bins deve estar entre 2 e 50")
    upper = np.nextafter(float(np.max(ref)), np.inf)
    if not np.isfinite(upper):
        raise ValueError("Valor máximo próximo de infinito não suportado")
    intermediate = np.quantile(ref, np.linspace(0, 1, bins + 1)[1:-1])
    return np.unique(np.r_[float(np.min(ref)), intermediate, upper])


def _histogram(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    # Each index is a range: (-inf, min), [min, q1), ... [max, +inf).
    indexes = np.searchsorted(edges, values, side="right")
    return np.bincount(indexes, minlength=len(edges) + 1).astype(float)


def population_stability_index(
    reference: np.ndarray | pd.Series,
    observed: np.ndarray | pd.Series,
    *,
    bins: int = 10,
    pseudocount: float = 0.5,
) -> float:
    """Smoothed Jeffrey divergence on train-derived buckets (PSI convention).

    No universal PSI cutoff is applied: binning, sample size and serial
    correlation change its interpretation. Both distributions use SAME cuts.
    """
    ref, cur = _finite(reference), _finite(observed)
    if not np.isfinite(pseudocount) or pseudocount <= 0:
        raise ValueError("pseudocount precisa ser positivo e finito")
    edges = reference_edges(ref, bins)
    ref_counts, cur_counts = _histogram(ref, edges), _histogram(cur, edges)
    p = (ref_counts + pseudocount) / (ref.size + pseudocount * ref_counts.size)
    q = (cur_counts + pseudocount) / (cur.size + pseudocount * cur_counts.size)
    return float(np.sum((q - p) * np.log(q / p)))


def ks_distance(reference: np.ndarray | pd.Series, observed: np.ndarray | pd.Series) -> float:
    """Two-sample empirical CDF distance, intentionally WITHOUT a p-value."""
    ref, cur = np.sort(_finite(reference)), np.sort(_finite(observed))
    # Checking every pooled unique value handles ties and constant features.
    combined = np.union1d(ref, cur)
    ref_cdf = np.searchsorted(ref, combined, side="right") / ref.size
    cur_cdf = np.searchsorted(cur, combined, side="right") / cur.size
    return float(np.max(np.abs(ref_cdf - cur_cdf)))


def feature_shift(reference: np.ndarray | pd.Series, observed: np.ndarray | pd.Series) -> dict:
    """Descriptive metrics; never computes windows, labels, or trained scores."""
    ref, cur = _finite(reference), _finite(observed)
    lo, hi = float(ref.min()), float(ref.max())
    return {
        "psi": round(population_stability_index(ref, cur), 8),
        "ks_distance": round(ks_distance(ref, cur), 8),
        "outside_train_range_rate": round(float(np.mean((cur < lo) | (cur > hi))), 8),
        "train_median": float(np.median(ref)),
        "observed_median": float(np.median(cur)),
        "train_min": lo,
        "train_max": hi,
        "train_n": int(ref.size),
        "observed_n": int(cur.size),
    }


def _validated_parent(root: Path, report_path: Path) -> dict:
    if not report_path.is_file():
        raise FileNotFoundError(f"Relatório multisséries ausente: {report_path}")
    try:
        parent = json.loads(report_path.read_text(encoding="utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("summary.json não contém um JSON UTF-8 válido") from exc
    if parent.get("protocol") != PARENT_PROTOCOL:
        raise ValueError("O diagnóstico exige o summary.json da avaliação v0.2.1")
    config = parent.get("configuration", {})
    if config.get("features") != FEATURE_COLUMNS:
        raise ValueError("Os atributos diferem dos definidos no relatório original")
    rows = parent.get("series")
    if not isinstance(rows, list) or not rows:
        raise ValueError("O relatório anterior não contém séries avaliadas")
    if config.get("selected_series") != len(rows):
        raise ValueError("Contagem de séries inconsistente no relatório anterior")
    labels_path = root / LABEL_FILE
    if file_sha256(labels_path) != parent["provenance"]["labels_sha256"]:
        raise ValueError("SHA-256 das anotações NAB mudou; não misture experimentos")
    expected = series_keys(labels_path)
    keys = [r["dataset"] for r in rows]
    if len(set(keys)) != len(keys) or keys != expected[:len(keys)]:
        raise ValueError("Ordem ou seleção das séries difere do manifest NAB")
    if config.get("is_full_aws_set") and len(keys) != len(expected):
        raise ValueError("Relatório declara conjunto completo, mas faltam séries")
    if not config.get("is_full_aws_set") and len(keys) == len(expected):
        raise ValueError("Relatório declara subconjunto mas inclui todas as séries")
    return parent


def _load_verified_parts(root: Path, item: dict, config: dict):
    key = item["dataset"]
    path = root / RAW_DIR / key
    if file_sha256(path) != item["dataset_sha256"]:
        raise ValueError(f"CSV alterado desde o benchmark: {key}")
    original, quality = load_nab_series_canonical(path)
    if quality != item["data_quality"]:
        raise ValueError(f"Política de normalização mudou: {key}")
    frame = make_features(original)
    train, validation, test = partition(
        frame,
        train_fraction=float(config["train_fraction"]),
        validation_fraction=float(config["validation_fraction"]),
    )
    for name, subset in (("train", train), ("validation", validation), ("test", test)):
        if len(subset) != item[f"{name}_rows"]:
            raise ValueError(f"Tamanho da partição {name} mudou: {key}")
        if str(subset["timestamp"].iloc[0]) != item[f"{name}_start"] or str(subset["timestamp"].iloc[-1]) != item[f"{name}_end"]:
            raise ValueError(f"Limites temporais da partição {name} mudaram: {key}")
    return train, validation, test


def _plot_summary(rows: list[dict], destination: Path):
    # Only univariate raw 'value' PSI. No label-informed threshold is learned.
    sorted_rows = sorted(rows, key=lambda r: r["test_value_psi"], reverse=True)
    labels = [PurePosixPath(r["dataset"]).stem for r in sorted_rows]
    fig, ax = plt.subplots(figsize=(12, max(6, .42 * len(rows))))
    y = np.arange(len(rows))
    ax.barh(y - .19, [r["validation_value_psi"] for r in sorted_rows], .38, label="Validação")
    ax.barh(y + .19, [r["test_value_psi"] for r in sorted_rows], .38, label="Teste")
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("PSI da variável original 'value' (bins definidos no treino)")
    ax.set_title("Mudança de distribuição — magnitude descritiva, não teste de hipótese")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(destination, dpi=150)
    plt.close(fig)


def _plot_focus(key: str, train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame, destination: Path):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), gridspec_kw={"height_ratios": [1.4, 1]})
    for phase, subset in (("Treino", train), ("Validação", validation), ("Teste", test)):
        ax1.plot(subset["timestamp"], subset["value"], linewidth=.85, label=phase)
    ax1.set_title(PurePosixPath(key).stem + " — série original por partição")
    ax1.set_ylabel("Valor bruto")
    ax1.legend()
    ref, cur = train["value"].to_numpy(dtype=float), test["value"].to_numpy(dtype=float)
    values = np.r_[ref, cur]
    edges = np.histogram_bin_edges(values, bins=35)
    ax2.hist(ref, bins=edges, density=True, alpha=.6, label="Treino")
    ax2.hist(cur, bins=edges, density=True, alpha=.6, label="Teste")
    ax2.set_xlabel("Valor bruto")
    ax2.set_ylabel("Densidade (histograma exploratório)")
    ax2.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(destination, dpi=150)
    plt.close(fig)


def _focus_keys(rows: list[dict], requested: list[str] | None) -> list[str]:
    available = {r["dataset"]: r for r in rows}
    names = list(requested) if requested else list(DEFAULT_FOCUS)
    if not requested and not all(any(k.endswith("/" + stem) for k in available) for stem in names):
        # For smoke runs, show the three biggest shifts, not absent default series.
        return [r["dataset"] for r in sorted(rows, key=lambda r: r["test_value_psi"], reverse=True)[:3]]
    result = []
    for name in names:
        matches = [k for k in available if k == name or k.endswith("/" + name)]
        if len(matches) != 1:
            raise ValueError(f"Série de foco ausente/ambígua: {name}")
        if matches[0] not in result:
            result.append(matches[0])
    return result


def _markdown_report(payload: dict) -> str:
    rows = payload["series"]
    top = sorted(rows, key=lambda r: r["test_value_psi"], reverse=True)[:8]
    alerts = sorted(rows, key=lambda r: r["iso_off_window_points"], reverse=True)[:5]
    lines = [
        "# Exploratory feature drift diagnostics — NAB AWS", "",
        "**This is descriptive analysis, not an independent evaluation, model improvement or an official NAB score.**", "",
        f"- Source benchmark: `{payload['provenance']['source_summary']}` (SHA-256 `{payload['provenance']['source_summary_sha256']}`)",
        f"- Examined series: **{len(rows)}**. Same time partitions, same canonicalization and original CSV hashes as v0.2.1.",
        "- Training-derived quantile bins for PSI; empirical KS *distance* (no p-value).",
        "- All seven causal input features, including current `value`, are compared on validation and test against training.",
        "- **No labels, anomaly windows, or model scores are used in drift calculation.** Off-window alert counts come only from the previously frozen benchmark report for post-hoc contextual comparison.",
        "- Time points and rolling features are autocorrelated; metrics are descriptive, not evidence of a statistically significant change.",
        "- PSI is sensitive to binning, smoothing, sample size and ties. **No universal PSI threshold** or causal explanation is asserted.",
        "- Identifying series from previously seen test results is *exploratory* and must not guide claims of independent test performance.",
        "", "## Highest raw-value PSI on test vs training", "",
        "| Dataset | PSI validation | PSI test | KS test | Off-window Isolation Forest points |", "|---|---:|---:|---:|---:|",
    ]
    for r in top:
        lines.append(f"| `{PurePosixPath(r['dataset']).name}` | {r['validation_value_psi']:.3f} | {r['test_value_psi']:.3f} | {r['test_value_ks']:.3f} | {r['iso_off_window_points']} |")
    lines += ["", "## Previously observed most off-window alert points", "",
              "| Dataset | IF points | Baseline points | Test PSI (`value`) |",
              "|---|---:|---:|---:|"]
    for r in alerts:
        lines.append(f"| `{PurePosixPath(r['dataset']).name}` | {r['iso_off_window_points']} | {r['baseline_off_window_points']} | {r['test_value_psi']:.3f} |")
    lines += [
        "", "## Visuals and machine-readable reports", "",
        "- `psi_by_series.png`: raw-value PSI by series and partition.",
        "- `per_series.csv`: raw-value metrics plus exploratory alert context.",
        "- `per_feature.csv`: all seven attributes x validation/test by series.",
        "- `summary.json`: provenance, experiment parameters, metrics and focus selection.",
        "- `focus/`: timeline and train/test value histograms for selected series.",
        "", "## Interpretation", "",
        "A large PSI/KS distance indicates a distributional difference, but does not establish that drift caused false alerts or missed anomalies."
        " Compare each focus plot with the input metric semantics and previous alert metrics; consider seasonal trends, change points, and alternative scales.",
        " Any threshold tuning motivated by this held-out test set needs a **new, genuinely untouched evaluation set**.",
        "", "## Reproduction", "",
        "```powershell", "python -m service_anomaly_ml.drift", "```", "",
    ]
    return "\n".join(lines)


def analyze(root: Path, source_summary: Path | None = None, report_dir: Path | None = None,
            focus: list[str] | None = None) -> dict:
    """Reconstruct same splits, check all data hashes, export descriptive drift.

    Never calls model.fit/predict; no new model is trained and no prior
    experimental files are overwritten. The source summary is read-only.
    """
    root = Path(root).resolve()
    source_summary = Path(source_summary) if source_summary else root / "artifacts/multi_series/summary.json"
    out = Path(report_dir) if report_dir else root / "artifacts/drift"
    if out.resolve() == source_summary.resolve().parent:
        raise ValueError("O diretório de drift não pode sobrescrever o benchmark anterior")
    parent = _validated_parent(root, source_summary)
    config = parent["configuration"]
    per_feature: list[dict] = []
    per_series: list[dict] = []
    frames: dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]] = {}
    for index, item in enumerate(parent["series"], 1):
        key = item["dataset"]
        print(f"[{index}/{len(parent['series'])}] Drift: {key}", flush=True)
        train, val, test = _load_verified_parts(root, item, config)
        frames[key] = (train, val, test)
        comparison = {}
        for phase, sub in (("validation", val), ("test", test)):
            for feature in FEATURE_COLUMNS:
                values = feature_shift(train[feature], sub[feature])
                per_feature.append({"dataset": key, "partition": phase, "feature": feature, **values})
                comparison[(phase, feature)] = values
        test_all = [comparison[("test", f)]["psi"] for f in FEATURE_COLUMNS]
        iso_metrics = item["test"]["isolation_forest"]
        baseline_metrics = item["test"]["rolling_median_baseline"]
        per_series.append({
            "dataset": key,
            "train_rows": len(train), "validation_rows": len(val), "test_rows": len(test),
            "validation_value_psi": comparison[("validation", "value")]["psi"],
            "test_value_psi": comparison[("test", "value")]["psi"],
            "test_value_ks": comparison[("test", "value")]["ks_distance"],
            "test_value_outside_train_range_rate": comparison[("test", "value")]["outside_train_range_rate"],
            "test_max_feature_psi": round(max(test_all), 8),
            "test_mean_feature_psi": round(float(np.mean(test_all)), 8),
            "test_detected_windows_iso": iso_metrics["windows_detected"],
            "test_evaluated_windows_iso": iso_metrics["windows_evaluated"],
            "test_detected_windows_baseline": baseline_metrics["windows_detected"],
            "test_evaluated_windows_baseline": baseline_metrics["windows_evaluated"],
            "iso_off_window_points": iso_metrics["false_alert_points_outside_windows"],
            "baseline_off_window_points": baseline_metrics["false_alert_points_outside_windows"],
            "off_window_exposure": item["test"]["off_window_observations"],
        })
    focused_keys = _focus_keys(per_series, focus)
    payload = {
        "protocol": PROTOCOL,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "provenance": {
            "source_summary": (source_summary.resolve().relative_to(root).as_posix()
                               if source_summary.resolve().is_relative_to(root) else source_summary.name),
            "source_summary_sha256": file_sha256(source_summary),
            "labels_sha256": parent["provenance"]["labels_sha256"],
        },
        "methodology": {
            "features": FEATURE_COLUMNS, "reference": "train", "comparisons": ["validation", "test"],
            "psi_bins": 10, "psi_pseudocount": 0.5,
            "psi_edges": "train quantiles plus out-of-support bins; repeated edges collapsed",
            "ks": "two-sample empirical CDF distance, no p-value",
            "no_test_tuning": True, "independent_holdout_claim": False,
            "full_aws_set": config["is_full_aws_set"],
        },
        "focus_series": focused_keys,
        "series": per_series,
        "per_feature": per_feature,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame(per_feature).to_csv(out / "per_feature.csv", index=False, encoding="utf-8")
    pd.DataFrame(per_series).to_csv(out / "per_series.csv", index=False, encoding="utf-8")
    _plot_summary(per_series, out / "psi_by_series.png")
    (out / "DRIFT_REPORT.md").write_text(_markdown_report(payload), encoding="utf-8")
    focus_dir = out / "focus"
    focus_dir.mkdir(parents=True, exist_ok=True)
    for key in focused_keys:
        _plot_focus(key, *frames[key], focus_dir / (PurePosixPath(key).stem + ".png"))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Exploratory feature drift on frozen NAB AWS multi-series benchmark")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Raiz do projeto")
    parser.add_argument("--input-report", type=Path, default=None, help="summary.json original do benchmark")
    parser.add_argument("--report-dir", type=Path, default=None, help="Destino isolado (padrão artifacts/drift)")
    parser.add_argument("--focus", action="append", default=None, help="CSV ou nome de série para gráficos (repetível)")
    args = parser.parse_args()
    report = analyze(args.root, args.input_report, args.report_dir, args.focus)
    ranked = sorted(report["series"], key=lambda r: r["test_value_psi"], reverse=True)
    print(f"Análise concluída: {len(ranked)} séries, {len(report['per_feature'])} comparações de atributos")
    for row in ranked[:5]:
        print(f"  {PurePosixPath(row['dataset']).name}: PSI(test value)={row['test_value_psi']:.3f}, "
              f"KS={row['test_value_ks']:.3f}, IF off-window pontos={row['iso_off_window_points']}")
    print(f"Relatórios: {args.report_dir or args.root / 'artifacts/drift'}")
    print("Diagnóstico EXPLORATÓRIO; não comprova causalidade nem representa nova avaliação independente.")


if __name__ == "__main__":
    main()
