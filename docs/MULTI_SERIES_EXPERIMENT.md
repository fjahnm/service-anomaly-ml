# Service Anomaly ML — NAB multi-series evaluation protocol (v0.2.1)

## Purpose

Evaluate **all** time series listed under `realAWSCloudwatch/` in the official Numenta Anomaly Benchmark `combined_windows.json`, including anomaly-free series. Compare independent unsupervised Isolation Forest models with the same simple rolling-median baseline previously used in experiment 01.

**This is a separate, explicitly experimental evaluation. It is not the official NAB normalized scoring procedure, and does not demonstrate generalization of one pretrained model across unseen services.**

## Data and provenance

- Dataset: https://github.com/numenta/NAB/tree/master/data/realAWSCloudwatch
- Annotation windows: https://github.com/numenta/NAB/blob/master/labels/combined_windows.json
- Citation: Ahmad, S., Lavin, A., Purdy, S., & Agha, Z. (2017). *Unsupervised real-time anomaly detection for streaming data.* Neurocomputing. https://doi.org/10.1016/j.neucom.2017.04.070

The program downloads *only missing* CSV files from NAB. Raw data and labels are ignored by Git; SHA-256 hashes are included in generated results. For immutable provenance, pass a specific NAB commit using `--nab-ref <commit_sha>` when downloading and running.

## Deterministic timestamp normalization (v0.2.1)

The NAB corpus may include timestamps that are not strictly increasing or that repeat,
which the original single-series loader deliberately refuses. The multiseries study
therefore applies a **separate, explicit normalization protocol**:

1. Validate the CSV header, parse timestamps and finite numeric values; missing or
   malformed values remain an error (no silent row drops).
2. Record inversions in the raw row order, repeat timestamps, and conflicting values.
3. Sort chronologically using stable sort. For multiple readings at **the same instant**,
   consolidate readings using the median (the sole rule; no use of future instants).
4. Preserve every original CSV unmodified. Store its SHA-256 and per-series cleaning
   audit in `summary.json`; include counts in `per_series.csv` and the Markdown report.
5. Use only the canonical timestamp sequence for causal features and chronological splits.

**Important:** This preprocessing changes some sample counts and values and is **not NAB's official preprocessing or score**. Do not directly compare these results with official NAB leaderboard numbers.
If a clean NAB dataset variant or a different aggregation policy is used later,
report it as a distinct experiment rather than silently replacing the old results.

## Predeclared protocol

- **All AWS series** found in labels are included; no selection based on detection results. `--limit` is for smoke tests only and is identified as a subset in the JSON report.
- Per series, compute the seven existing causal features using the current observation and **past 12** samples. This is 12 observations, not a universal one-hour window: different series may have different sampling frequencies.
- Split each series chronologically **50% train, 15% validation, 35% test** after feature construction. No shuffling and no feature borrowing across series.
- Fit a **separate** `StandardScaler + IsolationForest` (200 trees, random state 42) per series. Calibrate the isolation score and rolling-median deviation thresholds independently at the **98th percentile of training scores**.
- The validation segment is **reported, not used for tuning** in v0.2.0; labeled windows are only read after model fitting/threshold determination. Training may contain actual anomalies in an unsupervised benchmark. Never use window annotations to select training cutoffs.
- Evaluate held-out test observations only. Event hit = at least one alert inside an annotated time window intersecting test. Count off-window points and consecutive off-window alert episodes. Report number of test windows and series without any test window, as well as windows intersecting split boundaries.
- Aggregated **micro event recall** = total window hits / total test windows. **Macro event recall** = mean of event recall on series that actually have test windows, excluding those with no test-window events. Report off-window alert points per 1,000 off-window test observations (not per hour), including anomaly-free test series in this denominator.
- Do not call off-window alerts verified false positives: NAB windows are tolerance annotations, not exhaustive normal/abnormal labels for every sample.
- Preserve the single-series model and its Hugging Face publication. A multi-series run does not produce a new global ML model or overwrite `artifacts/isolation_forest.skops`.

## Run on Windows (PowerShell)

```powershell
cd "C:\Users\ResTIC55\Desktop\service-anomaly-ml"
.\.venv\Scripts\Activate.ps1
python -m pip install --no-deps -e .
python -m pytest -q
# Verify the downloader with three alphabetically first series:
python -m service_anomaly_ml.multi_series download --limit 3
python -m service_anomaly_ml.multi_series run --limit 3 --trees 20 --report-dir artifacts/multi_series_smoke
# Download every AWS series in the official labels file:
python -m service_anomaly_ml.multi_series download
# Run the predeclared full protocol (200 trees):
python -m service_anomaly_ml.multi_series run
```

`--trees 20` above only accelerates the **smoke test**, not the full experiment. For the complete experiment, leave the default 200 trees in place. Do **not** use the smoke-test metrics as final results.

## Output files

Saved under `artifacts/multi_series/` (ignored by Git):

- `summary.json`: full metadata, input hashes, temporal normalization audit, per-series train/validation/test sizes, model metrics, aggregates.
- `per_series.csv`: concise table for all series.
- `comparison.png`: number of off-window alert points by series, both methods.
- `MULTI_SERIES_REPORT.md`: generated human-readable table, results and limitations.

After verifying the results and ensuring no sensitive information appears, publish copies in `docs/results/multi_series/`. Do not copy model binaries or raw datasets into Git.

## What the evaluation does *not* show

No official NAB score; no pointwise F1; no longitudinal confidence interval; no online deployment; no transfer to an unseen series by a **single model**. With relatively few annotated test windows, micro/macro event recall can have high variance. Running all AWS series provides broader descriptive evidence, not a universal model-quality guarantee.
