---
language: en
library_name: scikit-learn
tags:
- anomaly-detection
- time-series
- isolation-forest
- scikit-learn
- experimental
---

# Service Anomaly ML — AWS CPU Isolation Forest (experimental)

## Model summary

An **unsupervised** `scikit-learn` pipeline using `StandardScaler` and `IsolationForest` to rank unusual CPU-observation patterns in a historical AWS EC2 CPU time series. This is a **research/portfolio prototype**, not a production reliability monitor and not a causal failure detector.

**Source code:** [service-anomaly-ml on GitHub](https://github.com/fjahnm/service-anomaly-ml) (link will work after publication).

## Data and attribution

- Benchmark: [Numenta Anomaly Benchmark (NAB)](https://github.com/numenta/NAB)
- Series: `realAWSCloudwatch/ec2_cpu_utilization_24ae8d.csv`
- Evaluation windows: `labels/combined_windows.json`
- 4,032 original measurements; 12 removed while generating historical features; 2,814 training and 1,206 chronological test observations.
- Upstream data is not redistributed here. Refer to the source repository for attribution and license terms.

**Citation:** Ahmad, S., Lavin, A., Purdy, S., & Agha, Z. (2017). *Unsupervised real-time anomaly detection for streaming data.* Neurocomputing. https://doi.org/10.1016/j.neucom.2017.04.070

## Model details

- Algorithm: `IsolationForest`, **200 trees**, `random_state=42`, with `StandardScaler` fitted on the training portion only.
- Train/test: chronological **70% / 30%** of feature-ready observations; no shuffling.
- Seven input features, in this order: `value`, `delta_1`, `abs_delta_1`, `past_mean_12`, `past_std_12`, `past_median_12`, `abs_deviation_past_median`.
- Rolling historical features use the **previous 12 observations only** (one hour at five-minute sampling); do not calculate features using future samples.
- Anomaly score: `-model.score_samples(features)`; higher is more unusual.
- Alert threshold: **98th percentile of training anomaly scores**. The exact threshold and data SHA-256 hashes are stored in `experiment.json`, not hard-coded here.
- Serialized format: `isolation_forest.skops` (not a Transformers checkpoint).

## Observed evaluation — single series

| Metric | Isolation Forest | Rolling-median baseline |
|---|---:|---:|
| Annotated windows hit | 2/2 | 2/2 |
| Window-level event recall | 1.0 | 1.0 |
| Alert points outside windows | 23 | **9** |
| Off-window alert episodes | 9 | 9 |
| Total test alert points | 46 | 21 |
| Test alert rate | 3.8143% | 1.7413% |

The baseline showed **fewer off-window alert points**, despite matching event recall. The two models had the same nine off-window alert episodes. This does not mean the baseline is universally superior. **Off-window points are annotation-relative proxies, not manually confirmed false positives.**

The event-recall result is based on only **two events** and is **not** official NAB scoring. No pointwise F1/precision, inference latency, production accuracy, or cross-dataset generalization claim is made.

## How to reproduce and use

Use the corresponding GitHub repository to download the input data, build the seven features in the original order, and reproduce the experiment:

```powershell
python -m pip install -e ".[dev]"
python -m service_anomaly_ml.train
python -m service_anomaly_ml.evaluate
```

To inspect an artifact, **only if it came from a source you personally trust and you have verified its provenance**:

```python
import skops.io as sio

path = "isolation_forest.skops"
print(sio.get_untrusted_types(file=path))  # inspect before loading
model = sio.load(path, trusted=["sklearn.tree._tree.Tree"])
```

**Security warning:** `sklearn.tree._tree.Tree` uses raw node indices; malformed or malicious files can cause crashes or memory-safety issues even when that type is explicitly trusted. **Do not automatically trust or load artifacts downloaded from strangers.** The local author-generated artifact was loaded and its scores and alert decisions reproduced using the saved data and thresholds.

For scoring, construct the original seven-feature input using the project code and then compare `-model.score_samples(X)` against the threshold in `experiment.json`. A standalone `value` field is insufficient input: the model requires 12 prior measurements to compute rolling features.

## Intended uses and limitations

**Intended:** methodological demonstrations, educational research, reproducible anomaly-detector comparisons, and feature-engineering experiments.

**Not intended:** production incident paging, financial/medical/security decisions, automatic root-cause diagnosis, cross-dataset claims, or evaluation as a general-purpose LLM. The model has not been validated on other telemetry sources, temporal regimes or infrastructure environments.

**Risks:** false positives, missed incidents, temporal distribution shift, and unsafe deserialization of untrusted model files. Anomalies do not by themselves establish root causes.

## Files

- `isolation_forest.skops` — local-trained model artifact
- `experiment.json` — model settings, dataset and label hashes, thresholds, split and evaluation
- `isolation_forest.png` and `rolling_median_baseline.png` — visualizations

The accompanying source code, methodology, and tests are maintained in the GitHub project.

## License

A license has **not yet been selected by the project author**. Consult the upstream NAB dataset license separately and set the appropriate model repository license before broad redistribution or reuse. Do not assume this model has any license by default.
