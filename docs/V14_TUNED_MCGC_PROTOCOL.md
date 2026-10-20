# MCGC v1.4 — fair tuned-vs-tuned confirmation protocol

## Purpose

v1.4 tests the reviewer-standard comparison in which **MCGC is tuned under the same nested-validation rules as the competing classifiers**.

This follows v1.2–v1.3, where MCGC was deliberately kept frozen as a conservative stress test while the baselines were tuned. v1.4 does not invalidate those checkpoints. It answers a different question: what happens when all methods receive fair model selection using only the outer-training data?

## Frozen principle

All model-selection decisions in this document are fixed **before inspecting v1.4 outer-test results**.

No grid, dataset, metric, tie-break, seed or claim may be changed after the first complete v1.4 science output is inspected.

## Primary question

Does nested tuning of the existing MCGC architecture improve its outer-test selective-risk performance relative to the frozen MCGC configuration and to the same tuned baselines used in v1.3?

## Primary real-data family

The same unseen real-data family introduced in v1.3 is retained so that the effect of MCGC tuning is directly comparable:

- `digits` — full 10-class sklearn digits dataset
- `wine` — 3-class sklearn wine dataset

Previously inspected `breast_cancer` remains a **control** and is not part of the primary pooled real-data claim.

No dataset may be removed because of an unfavorable v1.4 result.

## Outer evaluation

For every dataset and seed:

- seeds: `0..11`
- split: 70% outer train / 30% untouched outer test
- split seed: `5000 + dataset_seed`
- stratified split

The outer test set is never used for model selection.

## Inner model selection

All tunable models use:

- 3-fold stratified CV on the outer-training set only
- CV seed: `6000 + dataset_seed`
- primary selection criterion: lowest pooled out-of-fold E-AURC
- first tie-break: higher OOF accuracy
- final tie-break: deterministic lexical candidate key

This is the same selection logic used for the tuned baselines in v1.2 and v1.3.

## MCGC tuning budget

The MCGC search contains exactly **18 candidates**, matching the size of the Random-Forest grid and remaining close to the SVM/kNN tuning budgets.

Only structural graph-field hyperparameters are tuned because v1.4 evaluates the existing local graph-field margin as the intrinsic risk score:

- `n_neighbors ∈ {7, 12, 17}`
- `diffusion_alpha ∈ {0.75, 0.85, 0.93}`
- `diffusion_steps ∈ {20, 40}`

Cartesian product: `3 × 3 × 2 = 18` candidates.

The previously frozen configuration `(n_neighbors=12, diffusion_alpha=0.85, diffusion_steps=40)` is included in the grid.

All other MCGC settings remain fixed:

- `n_walkers=12`
- `max_steps=10`
- `temperature=0.35`
- `random_state=17`
- `adaptive_walker_gate=True`
- all score weights and walker mechanics unchanged

Rationale: the primary risk score is `local_margin_risk`, so the first fair MCGC tuning stage targets graph neighborhood scale and evidence-field diffusion rather than opening a much larger post-hoc search over walker/scoring weights.

## Baseline grids

The v1.3 baseline grids are unchanged.

### kNN

- `n_neighbors ∈ {5, 9, 12, 17, 25}`
- `weights ∈ {uniform, distance}`
- 10 candidates

### Logistic regression

- standardized features
- `C ∈ {0.1, 1, 10}`
- 3 candidates

### RBF-SVM

- standardized features
- `C ∈ {0.1, 1, 10}`
- `gamma ∈ {scale, 0.1, 1}`
- 9 candidates
- risk from negative decision-function confidence margin, not calibrated probability

### Random Forest

- 300 trees
- `max_depth ∈ {None, 6, 12}`
- `min_samples_leaf ∈ {1, 3, 5}`
- `max_features ∈ {sqrt, 0.5}`
- 18 candidates

## Risk signals

- tuned MCGC: existing `local_margin_risk`
- kNN / logistic / RF: `1 - max predicted probability`
- RBF-SVM: negative winner-vs-runner decision margin (absolute margin for binary)

No secondary learned risk model is introduced.

## Metrics

Primary metric:

- outer-test E-AURC, lower is better

Mandatory secondary metrics:

- ordinary AURC
- accuracy
- error rate
- error-detection ROC-AUC when evaluable
- selective accuracy at 90%, 80%, 70% and 50% coverage

## Frozen primary contrasts

For every outer-test run, define positive advantage as baseline E-AURC minus tuned-MCGC E-AURC.

Report tuned MCGC against:

1. tuned kNN
2. tuned logistic regression
3. tuned RBF-SVM
4. tuned Random Forest

Also report the paired effect of **tuned MCGC minus the old frozen MCGC configuration** on the same outer splits. For this within-MCGC comparison, improvement means lower E-AURC for tuned MCGC.

## Aggregation

For each dataset and the pooled primary real-data family report:

- mean paired E-AURC advantage
- median paired E-AURC advantage
- W/T/L
- two-sided sign-test p-value
- ordinary AURC counterpart

Per-dataset results must remain visible even when the pooled result is positive.

## Interpretation rules

A positive v1.4 result does **not** erase v1.3. The correct interpretation would be that the previous frozen configuration was conservative and that fair nested MCGC tuning recovers some performance.

A negative result must remain in the project history and closes the hypothesis that simple structural tuning is enough to obtain broad real-data selective-risk superiority.

No universal SOTA claim is allowed from this experiment.

## Reproducibility

The benchmark must save:

- complete protocol metadata
- all inner-CV candidate records for every model
- selected parameters for every dataset×seed run
- all outer-test metrics
- paired summaries

The implementation must not mutate the classifier defaults or alter the core prediction code.
