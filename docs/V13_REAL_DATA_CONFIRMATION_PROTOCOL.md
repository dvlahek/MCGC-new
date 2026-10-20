# MCGC v1.3 — frozen real-data selective-risk confirmation

## Status

Frozen before viewing any v1.3 result.

The classifier, MCGC risk definition, baseline families, tuning objective, train/test split rule and primary metrics are inherited unchanged from v1.2. This checkpoint tests transfer from the synthetic nonlinear-manifold family to real multiclass/tabular data.

## Scientific question

Does the frozen MCGC local graph-field margin retain competitive selective-risk ranking on previously unused real datasets when conventional baselines receive the same reviewer-grade inner-CV tuning used in v1.2?

This is a transfer/generalization test. It is not a new hyperparameter search for MCGC.

## Datasets

### Primary previously-unseen real-data family

- `digits` — full sklearn handwritten-digits dataset, 10 classes, 64 features.
- `wine` — full sklearn wine-recognition dataset, 3 classes, 13 features.

Neither dataset was used to form the v1.0-v1.2 selective-manifold claim.

### Previously-seen control

- `breast_cancer` — retained only as a continuity/control dataset because its selective-risk behaviour was already observed in v1.0-v1.2.

`breast_cancer` is excluded from the v1.3 primary pooled confirmation claim.

No dataset may be added, removed, relabelled or moved between primary/control status after results are observed.

## Outer protocol

- seeds: `0..11`
- split: stratified 70% train / 30% untouched test
- outer split seed: `5000 + dataset_seed`
- test labels are used only for final evaluation

## Frozen MCGC

Unchanged from v1.0-v1.2:

- `n_neighbors=12`
- `n_walkers=12`
- `max_steps=10`
- `diffusion_steps=40`
- `random_state=17`
- `adaptive_walker_gate=True`
- risk signal: `local_margin_risk`

MCGC receives no v1.3 hyperparameter tuning.

## Tuned baselines

Exactly the v1.2 baseline search spaces and selection rule are retained.

### kNN

- neighbours: `{5, 9, 12, 17, 25}`
- weights: `{uniform, distance}`

### Logistic regression

- standardized features
- `C in {0.1, 1, 10}`

### RBF-SVM

- standardized features
- `C in {0.1, 1, 10}`
- `gamma in {scale, 0.1, 1}`
- risk from negative decision-margin confidence
- no probability calibration

### Random Forest

- 300 trees
- `max_depth in {None, 6, 12}`
- `min_samples_leaf in {1, 3, 5}`
- `max_features in {sqrt, 0.5}`

## Inner tuning

- 3-fold stratified CV on the outer training set only
- inner seed: `6000 + dataset_seed`
- candidate selection objective: minimum OOF E-AURC
- tie-break: greater OOF accuracy, then deterministic candidate key
- selected candidate is refit on the complete outer training set

## Metrics

Primary metric: E-AURC, lower is better.

Required secondary metrics:

- ordinary AURC
- raw accuracy
- error-detection ROC-AUC when evaluable
- selective accuracy at 90%, 80%, 70% and 50% coverage

For every baseline, paired advantage is defined as:

`E-AURC(baseline) - E-AURC(MCGC)`

Positive values favour MCGC.

## Primary v1.3 analysis

Pool the 24 `digits` + `wine` dataset-seed runs and report, separately for tuned kNN, tuned logistic, tuned RBF-SVM and tuned Random Forest:

- mean paired E-AURC advantage
- median paired advantage
- wins/ties/losses
- two-sided sign test
- ordinary AURC paired advantage

Per-dataset results must also be reported. A positive pooled result may not be described as universal if one primary real dataset is clearly negative.

## Claim rule

A positive v1.3 result supports only a real-data transfer statement for the frozen `digits`/`wine` family. It does not establish universal superiority or SOTA.

A mixed or negative v1.3 result does not erase v0.9 XAI faithfulness or the v1.2 nonlinear-manifold selective-risk result. It instead defines the transfer boundary.

## Prohibited after freeze

After any v1.3 test result is viewed, do not:

- tune MCGC;
- change the dataset family;
- change baseline grids;
- change the tuning objective;
- change seeds or split rules;
- introduce a new confidence combination to rescue the result.
