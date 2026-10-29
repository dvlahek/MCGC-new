# CGEC v3.2 real-data predictive confirmation protocol

Status: **frozen before result inspection**.

Purpose: address the limited breadth of the original real-data predictive comparison without changing the CGEC method, operating configuration, baseline grids, split rule, tuning criterion, or primary selective-risk metric.

## Frozen datasets

The confirmation panel contains eight real PMLB classification datasets that are disjoint from the 12 PMLB datasets used for training-label review and from the original predictive pair (digits, wine):

1. `credit_approval_australia`
2. `haberman`
3. `heart_disease_cleveland`
4. `horse_colic_outcome`
5. `ionosphere`
6. `penguins`
7. `segmentation`
8. `sonar`

The list is fixed before benchmark results are inspected and must not be changed in response to performance.

## CGEC

Field-only paper definition, unchanged:

- `K = 12`
- `alpha = 0.85`
- `n_D = 40`
- `use_walkers = False`
- standardization fitted on the outer training split only
- confidence/risk from the negative top-two local graph-evidence margin

No CGEC parameter is tuned on these datasets.

## Baselines

Use exactly the predictive baseline families and grids already used by the manuscript:

- scaled tuned kNN
- tuned RBF-SVM
- tuned random forest
- scaled tuned Label Spreading

For each outer split, baseline hyperparameters are selected only from the outer training set by 3-fold stratified CV with `random_state = 6000 + seed`, minimizing OOF E-AURC with accuracy as tie-breaker. No grid is modified after seeing confirmation results.

## Outer evaluation

- seeds: `0..11` (12 seeds per dataset)
- split: 70/30 stratified train/test
- split random state: `5000 + seed`
- primary metric: E-AURC (lower is better)
- supporting metrics: raw AURC, error-detection ROC-AUC, classification accuracy

## Inference hierarchy

The primary real-data confirmation unit is the **dataset**. For each dataset and method, metrics are first averaged across the 12 seeds. CGEC-minus-baseline conclusions across the confirmation panel are then summarized from the eight dataset-level mean E-AURC advantages.

Seed-level results (96 dataset-seed runs) are retained and reported descriptively, but they are not treated as 96 independent problem domains.

## Reproducibility safeguards

The benchmark records:

- the frozen dataset list,
- dataset shapes and class counts,
- a content fingerprint of each loaded feature/target array,
- all 12 outer seeds,
- baseline tuning choices,
- per-seed metrics,
- per-dataset mean metrics and paired advantages,
- panel-level dataset-unit and descriptive seed-unit summaries.

The confirmation is interpreted regardless of direction. Datasets are not removed or replaced because CGEC performs poorly on them.
