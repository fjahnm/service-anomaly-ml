# Service Anomaly ML v0.3.0 — Exploratory Feature Drift Diagnostic

**Status:** descriptive, retrospective analysis. This does **not** improve the prior model or provide an independently untouched test evaluation.

## Question

Do input-feature distributions in the 17 NAB `realAWSCloudwatch` series differ between training and later data? Which series should we inspect to understand the concentration of Isolation Forest off-window alert points reported in v0.2.1?

**Observed v0.2.1 benchmark (context only):** both methods hit 11/13 annotated test windows. Isolation Forest produced 3,065 off-window alert points (254 contiguous episodes); the rolling-median baseline produced 718 such points (586 episodes). The top three Isolation Forest series by off-window points were `ec2_cpu_utilization_5f5533.csv` (1,003), `rds_cpu_utilization_e47b3b.csv` (950), and `rds_cpu_utilization_cc0c53.csv` (658). Both models missed the single tested window in `ec2_cpu_utilization_fe7f93.csv` and `ec2_disk_write_bytes_c0d644.csv`. These figures come from a prior experiment, **not** from a new drift run.

## Inputs and reproducibility

Required:

- `data/labels/combined_windows.json` (upstream NAB labels, for provenance verification only)
- 17 local NAB raw CSVs under `data/raw/realAWSCloudwatch/` (never uploaded to this repository)
- `artifacts/multi_series/summary.json` from the completed v0.2.1 run, including original file hashes, original splits and frozen alert metrics

No internet access is used by the drift command; it never downloads data, re-trains the Isolation Forest, evaluates new thresholds, or mutates prior reports. It fails if the current label manifest, raw CSV hashes, canonicalization audit, feature lists, or chronological split boundaries no longer match the original experiment.

## Run on Windows PowerShell

From the project root with `.venv` activated:

```powershell
python -m pip install --no-deps -e .
python -m pytest -q
python -m service_anomaly_ml.drift
```

To run against a different frozen benchmark summary without overwriting it:

```powershell
python -m service_anomaly_ml.drift --input-report artifacts/multi_series/summary.json --report-dir artifacts/drift
```

To select specific focus graphs:

```powershell
python -m service_anomaly_ml.drift --focus ec2_cpu_utilization_5f5533.csv --focus rds_cpu_utilization_e47b3b.csv --focus ec2_disk_write_bytes_c0d644.csv
```

The default focus files, **chosen after viewing v0.2.1 holdout results solely for diagnosis**, are the three high-alert series plus the two series with missed events. They are not a new, independent test sample.

## Temporal partitions and features

Reconstruct exactly the feature engineering and 50%/15%/35% chronological splits from the frozen v0.2.1 experiment. Use all seven causal features in `FEATURE_COLUMNS`:

- `value`
- `delta_1`
- `abs_delta_1`
- `past_mean_12`
- `past_std_12`
- `past_median_12`
- `abs_deviation_past_median`

Rolling statistics use preceding observations only; timestamp normalization is the same stable sort / median-per-instant policy from v0.2.1. The original CSVs remain untouched. Compare **training vs validation** and **training vs test** for every feature, separately for each series. Do not compare pooled raw values across different AWS metrics with incompatible physical units.

## Metrics

### Population Stability Index (PSI)

For the *reference training data*, construct 10 empirical quantile intervals, collapsing duplicate edges. Add two out-of-support intervals, for observations below the training minimum and above its maximum. Constant training values produce below/equal/above categories. Use the **same train-derived edges** on validation and test. Add 0.5 pseudocount to every bin to avoid a logarithm of zero.

\[
\mathrm{PSI} = \sum_{b} (q_b - p_b)\log(q_b/p_b),
\]

where \(p_b\) and \(q_b\) are smoothed train and comparison proportions in bin \(b\). A larger PSI means the histograms differ more. **Do not apply a universal fixed cutoff**; PSI depends on bins, ties, sample sizes and time-series dependence.

### Empirical KS distance

\[
D = \sup_x \left|F_{\mathrm{train}}(x) - F_{\mathrm{later}}(x)\right|.
\]

This is the *descriptive two-sample CDF distance* between 0 and 1. **No KS p-value or significance claim is reported** because time points and adjacent rolling features are autocorrelated.

### Outside-training-support fraction

The fraction of later observations below the training minimum or above the training maximum, computed feature by feature. This can be near zero even if distributions shift within their original range.

## Generated outputs

Located in `artifacts/drift/`:

| Output | Meaning |
|---|---|
| `summary.json` | Provenance SHA-256, metrics, focus selection, configuration and series rows |
| `per_feature.csv` | All seven features × two later partitions × every series |
| `per_series.csv` | Raw-value metrics and context of previously recorded off-window alerts |
| `psi_by_series.png` | Training-referenced raw-value PSI across series (validation/test) |
| `focus/*.png` | Raw-value time plots and train/test histograms for selected focus series |
| `DRIFT_REPORT.md` | Automatically generated ranking and cautious interpretation |

`artifacts/` is ignored by Git. To publish only these derived, public-NAB metrics, **review the output first**, then copy them to `docs/results/drift/` and commit with a clear *exploratory* status.

## Interpretation constraints

- **Distribution drift is not proof of concept drift, model degradation or the cause of off-window alerts.** Detection labels and model scores are not used in the PSI/KS computations; only prior frozen alerts are shown alongside the drift measurements for discussion.
- Shift metrics can be large when data are nearly constant, quantized, trend-driven or strongly seasonal. Compare numbers **within a feature**, inspect charts and avoid interpreting PSI as an alert probability.
- Some annotated events overlap chronological partitions or training segments; this module does not change that prior protocol.
- The 17 series have already been examined to diagnose v0.2.1; any future tuning based on them needs independent held-out data to assess generalization. Do not retrofit thresholds to test outcomes.
- All results remain **experimental and are not official NAB scores**.

## Background references

- Haas & Sibbald, *Measuring data drift with the unstable population indicator*, Data Science (2024), DOI: https://doi.org/10.3233/DS-240059 — discusses PSI/Jeffrey divergence and issues with fixed thresholds.
- Evidently, [Data Drift overview](https://docs-old.evidentlyai.com/presets/data-drift) — background on comparing reference/current feature distributions.
- NAB dataset and labeling documentation: https://github.com/numenta/NAB
