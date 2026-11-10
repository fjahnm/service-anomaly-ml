# Exploratory feature drift diagnostics — NAB AWS

**This is descriptive analysis, not an independent evaluation, model improvement or an official NAB score.**

- Source benchmark: `artifacts/multi_series/summary.json` (SHA-256 `9b4b279a2ec050d684cb4b45e425af0176aafbaa2cf1b7bf261672eef0d93167`)
- Examined series: **17**. Same time partitions, same canonicalization and original CSV hashes as v0.2.1.
- Training-derived quantile bins for PSI; empirical KS *distance* (no p-value).
- All seven causal input features, including current `value`, are compared on validation and test against training.
- **No labels, anomaly windows, or model scores are used in drift calculation.** Off-window alert counts come only from the previously frozen benchmark report for post-hoc contextual comparison.
- Time points and rolling features are autocorrelated; metrics are descriptive, not evidence of a statistically significant change.
- PSI is sensitive to binning, smoothing, sample size and ties. **No universal PSI threshold** or causal explanation is asserted.
- Identifying series from previously seen test results is *exploratory* and must not guide claims of independent test performance.

## Highest raw-value PSI on test vs training

| Dataset | PSI validation | PSI test | KS test | Off-window Isolation Forest points |
|---|---:|---:|---:|---:|
| `rds_cpu_utilization_cc0c53.csv` | 0.138 | 6.200 | 0.677 | 658 |
| `ec2_cpu_utilization_5f5533.csv` | 0.432 | 5.168 | 0.706 | 1003 |
| `grok_asg_anomaly.csv` | 0.331 | 4.727 | 0.537 | 170 |
| `rds_cpu_utilization_e47b3b.csv` | 3.191 | 4.519 | 0.802 | 950 |
| `ec2_cpu_utilization_ac20cd.csv` | 0.825 | 3.067 | 0.326 | 2 |
| `ec2_network_in_257a54.csv` | 3.650 | 2.692 | 0.633 | 1 |
| `iio_us-east-1_i-a2eb1cd9_NetworkIn.csv` | 1.370 | 1.640 | 0.339 | 0 |
| `ec2_cpu_utilization_825cc2.csv` | 1.382 | 0.528 | 0.322 | 0 |

## Previously observed most off-window alert points

| Dataset | IF points | Baseline points | Test PSI (`value`) |
|---|---:|---:|---:|
| `ec2_cpu_utilization_5f5533.csv` | 1003 | 1 | 5.168 |
| `rds_cpu_utilization_e47b3b.csv` | 950 | 211 | 4.519 |
| `rds_cpu_utilization_cc0c53.csv` | 658 | 119 | 6.200 |
| `grok_asg_anomaly.csv` | 170 | 105 | 4.727 |
| `ec2_disk_write_bytes_1ef3de.csv` | 64 | 43 | 0.079 |

## Visuals and machine-readable reports

- `psi_by_series.png`: raw-value PSI by series and partition.
- `per_series.csv`: raw-value metrics plus exploratory alert context.
- `per_feature.csv`: all seven attributes x validation/test by series.
- `summary.json`: provenance, experiment parameters, metrics and focus selection.
- `focus/`: timeline and train/test value histograms for selected series.

## Interpretation

A large PSI/KS distance indicates a distributional difference, but does not establish that drift caused false alerts or missed anomalies. Compare each focus plot with the input metric semantics and previous alert metrics; consider seasonal trends, change points, and alternative scales.
 Any threshold tuning motivated by this held-out test set needs a **new, genuinely untouched evaluation set**.

## Reproduction

```powershell
python -m service_anomaly_ml.drift
```
