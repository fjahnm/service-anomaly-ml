# Experimental Design — Service Anomaly ML

## Hypothesis

An unsupervised Isolation Forest can flag unusual patterns in a cloud CPU series, potentially providing complementary detection behavior to a simple rolling-median deviation baseline.

This is an **exploratory hypothesis**, not a guaranteed improvement. We compare both detectors under thresholds calibrated independently at the same 98th percentile of each model's training score distribution. Similar training alert rates do not guarantee identical rates on the holdout set.

## Scope

- Dataset: one Numenta Anomaly Benchmark (NAB) AWS CPU series (`ec2_cpu_utilization_24ae8d.csv`).
- Ground-truth windows: `combined_windows.json`; exact per-event labels are separately published in `combined_labels.json`.
- Observations: 4,032 in the downloaded dataset, with 12 removed by warm-up for causal features.
- 70% of feature rows for training; remaining 30% held out chronologically. **Neither model is given labels for training.**
- The experiment will refuse to train if the first segment overlaps annotated NAB windows, to avoid silently contaminating the training portion in this simple first iteration.

## Data leakage prevention

All rolling features are computed exclusively with `shift(1)` from previous values, plus the current value and its immediate difference. Splitting happens after deterministic feature construction; no model fitting, global normalization or threshold calibration uses held-out data. `StandardScaler` is inside the training-only sklearn pipeline.

The experiment calculates both model thresholds only from training scores. It does not adjust the threshold based on how many of the two NAB windows were detected. **Repeated tuning against the same test set would make these metrics optimistic.** If tuning is needed, use an additional temporal validation split and preserve a fresh final holdout.

## Evaluation protocol

The primary outcome is event-level window recall: number of NAB windows containing at least one detector alert, divided by the number of evaluable windows. Secondary outcomes are raw alert points outside any window and episodes of contiguous off-window alerts. This is a simple, explicitly named **window-hit protocol**, not the official NAB weighted scoring (which accounts for alert timing and costs).

It is inappropriate to turn all points within the 16+ hour windows into positive per-point ground-truth labels and report conventional classification precision/F1. Windows are broad temporal tolerance regions, not pointwise defect classes.

No official NAB leaderboard result, causal root-cause attribution, production performance or generalization claim will be made from this limited single-series experiment.

## Acceptance criteria

- Reproducible training and JSON metadata with hashes and version info.
- Unit tests for feature causality and evaluation behavior.
- Two detectors compared with a held-out chronological period.
- All obtained metrics presented as observed results, without fabricated precision or recall.
- Public data source attribution and no company/customer information.

## Sources

- NAB repository: https://github.com/numenta/NAB
- NAB data: https://github.com/numenta/NAB/tree/master/data/realAWSCloudwatch
- NAB windows: https://github.com/numenta/NAB/blob/master/labels/combined_windows.json
- NAB point labels: https://github.com/numenta/NAB/blob/master/labels/combined_labels.json
- Official Isolation Forest documentation: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html
- Ahmad et al. (2017): https://doi.org/10.1016/j.neucom.2017.04.070
