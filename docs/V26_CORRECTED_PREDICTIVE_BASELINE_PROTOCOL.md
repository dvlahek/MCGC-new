# v2.6 Corrected predictive-baseline audit

Status: **FROZEN BEFORE RESULTS**.

## Purpose

Close the final predictive-comparison issue before manuscript freeze. The original v1.2/v1.3 tuned kNN baseline was fitted on unscaled features, while SVM/logistic used standardized features. This follow-up keeps the original frozen outer splits, MCGC configuration, tuning criterion, and seed sets, but corrects the kNN preprocessing and adds Label Spreading under the same train-only tuning discipline.

## Scope

Two frozen families are reproduced:

1. **v1.2 nonlinear/control family**: `moons_hard`, `circles_hard`, `spirals_hard`, `overlap_hd`, `breast_cancer`.
2. **v1.3 real-data family**: `digits`, `wine`, `breast_cancer`.

Seeds: 0--11. Outer split: 70/30 stratified, `random_state=5000+seed`. Inner tuning: 3-fold stratified CV on outer training only, `random_state=6000+seed`.

## Models

- frozen MCGC local-margin risk, unchanged;
- **scaled tuned kNN**: StandardScaler + kNN, original k/weight grid;
- tuned RBF-SVM, unchanged from v1.2/v1.3;
- tuned Random Forest, unchanged from v1.2/v1.3;
- **scaled tuned Label Spreading**: StandardScaler + LabelSpreading with a frozen kNN-kernel grid.

Label Spreading grid:
- `n_neighbors in {5, 9, 12, 17, 25}`;
- `alpha in {0.2, 0.5, 0.8}`;
- `kernel='knn'`, `max_iter=100`.

No baseline is tuned on outer-test labels.

## Selection and metrics

Each baseline is selected by the same frozen rule used in v1.2: lowest inner-OOF E-AURC, then higher inner-OOF accuracy, then deterministic lexical tie-break.

Primary outer-test metric: E-AURC (lower is better). Also report accuracy, AURC, error-detection ROC-AUC, and selective accuracy at the existing coverage levels.

Paired comparison is defined as `baseline E-AURC - MCGC E-AURC`, so a positive value favours MCGC.

## Inference

Report per-dataset 12-seed W/T/L and exact two-sided sign test. Also report pooled results separately for:

- nonlinear primary family: moons/circles/spirals;
- real-data primary family: digits/wine.

Do not pool the control datasets into a headline superiority claim.

## Interpretation rule

The manuscript will state exactly which baselines MCGC beats, matches, or loses to. No new hyperparameter changes or dataset additions are allowed after seeing v2.6 results.
