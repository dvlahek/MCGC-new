# CGEC v3.2 real-data predictive confirmation protocol

Status: **frozen before result inspection**.

Purpose: address the limited breadth of the original real-data predictive comparison without changing the CGEC method, operating configuration, baseline grids, split rule, tuning criterion, or primary selective-risk metric.

## Pre-result implementation and eligibility corrections

The first attempted full run stopped while loading the first dataset because the locally installed `pmlb` package used an older dataset-name registry. No dataset was evaluated and no predictive metric was produced.

Before any result inspection, one protocol inconsistency was identified: the initially listed `sonar` dataset has 208 samples and therefore violates the predeclared eligibility rule requiring at least 250 samples. `sonar` was replaced by `dermatology` solely to restore compliance with the frozen eligibility criteria.

A subsequent loading-only attempt stopped before the dataset/seed evaluation loop because `horse_colic_outcome` contains extensive missingness and only 7 complete cases remain under the predeclared complete-case loading rule. It therefore violates the same minimum-sample criterion after loading. Before any CGEC or baseline result had been produced, `horse_colic_outcome` was replaced by `credit_approval_germany`, which satisfies the frozen sample-size, feature-count, class-count, and minimum-class-size criteria. Neither replacement is based on model performance.

To remove dependence on the locally installed PMLB registry, the executable confirmation loader is pinned to upstream PMLB commit `7c1f4bdc00136dc2e55c87fa6b8ba6e8af6d1a68`.

## Frozen eligibility criteria

A confirmation dataset must satisfy all of the following before evaluation:

- real PMLB classification dataset;
- 250--2500 usable samples after the standard complete-case loading step;
- 2--10 classes;
- at most 100 input features;
- at least 20 usable samples in every class;
- disjoint from the 12 PMLB datasets used for training-label review;
- disjoint from the original predictive pair (`digits`, `wine`);
- not one of the synthetic nonlinear benchmark families.

## Frozen datasets

The final pre-result confirmation panel contains eight real PMLB classification datasets:

1. `credit_approval_australia`
2. `credit_approval_germany`
3. `dermatology`
4. `haberman`
5. `heart_disease_cleveland`
6. `ionosphere`
7. `penguins`
8. `segmentation`

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
- the pinned upstream PMLB commit,
- dataset shapes and class counts after loading,
- a content fingerprint of each loaded feature/target array,
- all 12 outer seeds,
- baseline tuning choices,
- per-seed metrics,
- per-dataset mean metrics and paired advantages,
- panel-level dataset-unit and descriptive seed-unit summaries.

The confirmation is interpreted regardless of direction. Datasets are not removed or replaced because CGEC performs poorly on them.
