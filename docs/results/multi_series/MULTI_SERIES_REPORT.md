# Multi-series experiment — NAB realAWSCloudwatch

**Status:** local experimental benchmark, not official NAB scoring.

- Series evaluated: **17**
- NAB labels SHA-256: `1e1fbc4601321aad8d0f8b3784c8134299379f68f6c1f7777565f8ffd57ab6b1`
- An independent Isolation Forest was trained for EACH series; this does not test unseen-series transfer.
- Strict chronological train/validation/test split; only train scores calibrate thresholds.
- Every AWS series listed in `combined_windows.json` is included unless `--limit` was explicitly used.
- Both models share the quantile policy but have different alert counts; there is no fixed alert quota.
- Irregular timestamps are sorted and duplicate instants consolidated by median ONLY for this experimental run; original CSVs remain unchanged.
- Timestamp normalization changes the sample sequence and is NOT the official NAB preprocessing.

## Input quality and normalization

- Series with changed timestamps: **2**
- Descending timestamp transitions: **0**
- Duplicate timestamp rows consolidated: **22**
- Conflicting-value timestamp groups: **1**

## Aggregate holdout comparison

| Metric | Isolation Forest | Rolling median |
|---|---:|---:|
| Detected / evaluable test windows | 11 / 13 | 11 / 13 |
| Micro event recall | 0.846154 | 0.846154 |
| Macro event recall (eligible series only) | 0.818182 | 0.818182 |
| Alert points outside windows | 3065 | 718 |
| Alert episodes outside windows | 254 | 586 |
| Outside-window alert points per 1,000 outside-window observations | 144.2014 | 33.7803 |

**Series without an annotated window in test:** 6. They still contribute to false-alert totals.

## Series-level results

| Dataset | Test windows | Isolation Forest hits | Baseline hits | ISO off-window points | Baseline off-window points |
|---|---:|---:|---:|---:|---:|
| `realAWSCloudwatch/ec2_cpu_utilization_24ae8d.csv` | 2 | 2 | 2 | 36 | 11 |
| `realAWSCloudwatch/ec2_cpu_utilization_53ea38.csv` | 1 | 1 | 1 | 10 | 23 |
| `realAWSCloudwatch/ec2_cpu_utilization_5f5533.csv` | 1 | 1 | 1 | 1003 | 1 |
| `realAWSCloudwatch/ec2_cpu_utilization_77c1ca.csv` | 0 | 0 | 0 | 46 | 46 |
| `realAWSCloudwatch/ec2_cpu_utilization_825cc2.csv` | 0 | 0 | 0 | 0 | 19 |
| `realAWSCloudwatch/ec2_cpu_utilization_ac20cd.csv` | 1 | 1 | 1 | 2 | 7 |
| `realAWSCloudwatch/ec2_cpu_utilization_c6585a.csv` | 0 | 0 | 0 | 42 | 27 |
| `realAWSCloudwatch/ec2_cpu_utilization_fe7f93.csv` | 1 | 0 | 0 | 10 | 29 |
| `realAWSCloudwatch/ec2_disk_write_bytes_1ef3de.csv` | 0 | 0 | 0 | 64 | 43 |
| `realAWSCloudwatch/ec2_disk_write_bytes_c0d644.csv` | 1 | 0 | 0 | 46 | 30 |
| `realAWSCloudwatch/ec2_network_in_257a54.csv` | 0 | 0 | 0 | 1 | 0 |
| `realAWSCloudwatch/ec2_network_in_5abac7.csv` | 1 | 1 | 1 | 20 | 33 |
| `realAWSCloudwatch/elb_request_count_8c0756.csv` | 1 | 1 | 1 | 7 | 14 |
| `realAWSCloudwatch/grok_asg_anomaly.csv` | 1 | 1 | 1 | 170 | 105 |
| `realAWSCloudwatch/iio_us-east-1_i-a2eb1cd9_NetworkIn.csv` | 0 | 0 | 0 | 0 | 0 |
| `realAWSCloudwatch/rds_cpu_utilization_cc0c53.csv` | 2 | 2 | 2 | 658 | 119 |
| `realAWSCloudwatch/rds_cpu_utilization_e47b3b.csv` | 1 | 1 | 1 | 950 | 211 |

## Limitations

- NAB windows are tolerance intervals, not exhaustive point-level anomaly truth.
- Median consolidation of duplicate timestamps can change the sample distribution, measured alert rates, and comparability to other NAB experiments.
- Off-window alerts are proxies, not verified false positives.
- No official NAB normalized score, point-level F1, or cross-service transfer performance is claimed.
- Models may learn from anomalous observations in the training prefix; labels never influenced training or thresholds.
- Windows straddling partition boundaries may be partially observable in validation/test.
- Calibration is per-series and fixed at the declared train-score quantile; repeated test-driven tuning is prohibited.
- A full multi-series run does not retroactively update the previously published Hugging Face model.

## Provenance

- NAB reference: `master` (branch name if not immutable commit)
- Source: https://github.com/numenta/NAB
- Individual input SHA-256 digests are saved in `summary.json`.

## Files

- `summary.json`: full provenance, per-series scores and aggregates.
- `per_series.csv`: compact per-series comparison.
- `comparison.png`: grouped plot of off-window alert points.

## Citation

Ahmad, S., Lavin, A., Purdy, S., & Agha, Z. (2017). *Unsupervised real-time anomaly detection for streaming data.* Neurocomputing. https://doi.org/10.1016/j.neucom.2017.04.070
