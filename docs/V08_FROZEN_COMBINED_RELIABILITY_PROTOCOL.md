# MCGC v0.8 frozen combined-intrinsic reliability confirmation

## Purpose

v0.7 rejected the preregistered hypothesis that trail disagreement alone adds reliable error-detection information beyond the local graph-field margin. A secondary result on the spiral benchmark suggested a smaller but positive signal when the already-defined intrinsic diagnostics were combined.

v0.8 is a **confirmation-only checkpoint** for that secondary result. The feature definitions are frozen before evaluation. No feature selection, coefficient tuning, threshold tuning, or dataset-specific retuning is permitted after test results are inspected.

## Frozen primary hypothesis

The five-feature `combined_intrinsic` risk model has higher test error-detection ROC AUC than the one-feature `local_only` risk model when evaluated on held-out confirmation problems.

Primary paired contrast for every evaluable dataset/seed pair:

`ROC_AUC(combined_intrinsic) - ROC_AUC(local_only)`

The main summary is the distribution of this paired delta over all evaluable confirmation runs, with per-dataset summaries also reported.

## Frozen intrinsic feature sets

No new feature may be added in v0.8.

`local_only`:

- `local_margin_risk`

`combined_intrinsic`:

- `local_margin_risk`
- `final_margin_risk`
- `trail_disagreement_risk`
- `coherence_risk`
- `lost_trail_risk`

The feature semantics and implementation are inherited unchanged from v0.7.

## Frozen MCGC configuration

- `n_neighbors=12`
- `n_walkers=12`
- `max_steps=10`
- `diffusion_steps=40`
- `random_state=17`
- `adaptive_walker_gate=True`

No dataset-specific classifier retuning is allowed.

## Frozen risk learner

Both intrinsic feature sets use exactly the same calibration-only learner:

`StandardScaler + LogisticRegression(class_weight='balanced', C=1.0, max_iter=2000, random_state=71)`

The risk learner is fitted only on the calibration split and evaluated only on the untouched test split.

## Data split

For each dataset and seed:

- 60% classifier training
- 20% risk calibration
- 20% untouched test

The first split uses `random_state=2000 + seed` and the calibration/test split uses `random_state=3000 + seed`, both stratified by class.

A seed is never resplit to obtain a more favorable result. If the calibration error indicator contains only one class, that run is marked `insufficient_calibration_classes` and excluded from ROC-AUC paired comparisons. If the untouched test error indicator contains only one class, Brier and selective-accuracy outputs remain valid but ROC AUC is not included in the primary paired comparison.

## Frozen confirmation datasets

The confirmation set is deliberately broader than v0.7 and contains both harder synthetic geometry and real sklearn datasets requiring no external download.

Synthetic:

1. `moons_hard`: 700 samples, noise 0.30.
2. `circles_hard`: 700 samples, noise 0.18, factor 0.50.
3. `spirals_hard`: 700 samples, two-arm spiral generator inherited from v0.2, noise 0.16.
4. `overlap_hd`: 800 samples from `make_classification`, 8 features, 4 informative, 2 redundant, class separation 0.75, 8% label noise.

Real:

5. `breast_cancer`: sklearn Wisconsin Diagnostic Breast Cancer dataset.
6. `wine`: sklearn Wine dataset.
7. `digits_0_4`: sklearn Digits dataset restricted to classes 0 through 4.

## Frozen seeds

Seeds `0..7` are used for every dataset. For fixed real datasets the seed changes only the train/calibration/test split. For synthetic datasets it also changes dataset generation.

## Standard confidence references

Two external references are reported to contextualize selective reliability. They are **not** part of the primary MCGC paired hypothesis and are not used to tune MCGC:

- distance-weighted 12-NN, risk = `1 - max(predict_proba)`
- standardized logistic regression, risk = `1 - max(predict_proba)`

Both reference classifiers are trained only on the classifier-training split.

## Frozen metrics

Primary:

- test error-detection ROC AUC
- paired ROC-AUC delta `combined_intrinsic - local_only`

Secondary:

- average precision for error detection
- Brier score for calibrated MCGC risk models
- selective classification accuracy at coverages 0.90, 0.80, 0.70, and 0.50
- wins / ties / losses of the paired primary delta
- exact two-sided sign-test p-value over non-tied paired deltas, reported as descriptive confirmation evidence rather than a standalone proof

No metric may be dropped because it is unfavorable.

## Interpretation rule

v0.8 supports an incremental intrinsic-reliability claim only if the combined model shows a consistent positive paired effect beyond `local_only` across the broader confirmation set. A positive result on one dataset family alone is treated as geometry- or dataset-specific evidence, not as a general claim.

A negative or mixed result is a valid checkpoint and must be documented without post-hoc retuning of the frozen five-feature set.
