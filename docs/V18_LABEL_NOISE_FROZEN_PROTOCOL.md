# MCGC v1.8 — frozen label-noise confirmation protocol

Status: **FROZEN BEFORE CONFIRMATORY RESULTS**

This experiment tests a new hypothesis suggested by exploratory runs: graph evidence may be more useful under corrupted training labels than on clean labels, and the same evidence field may provide a training-label inconsistency signal.

The v1.8 experiment does **not** change the established v0.9/v1.5/v1.7 XAI claims. It is a separate confirmation branch.

## Questions

### Q1. Robust classification

When only outer-training labels are corrupted, does MCGC degrade less than conventional classifiers and a graph-diffusion baseline?

Primary robustness quantity at noise level eta:

`degradation(eta) = performance(0) - performance(eta)`

Report both accuracy and balanced accuracy. Clean outer-test labels are never corrupted.

### Q2. Training-label audit

Can MCGC identify which outer-training labels were synthetically corrupted from disagreement between the observed label and out-of-fold graph evidence?

For an outer-training sample i with observed noisy label y_i, define an out-of-fold local evidence map L_i(c). The frozen audit score is

`audit_i = max_{c != y_i} L_i(c) - L_i(y_i)`

Larger values mean stronger graph-evidence conflict with the observed label.

The score is computed out of fold: sample i is not used to fit the MCGC model that scores i.

Primary audit metrics:

- ROC-AUC
- PR-AUC
- precision@k, with k equal to the known number of synthetically corrupted labels

### Q3. Diffusion specificity

Does MCGC retain any robustness advantage over `sklearn.semi_supervised.LabelSpreading` under the same corrupted outer-training labels?

If Label Spreading matches MCGC, the robustness mechanism must be attributed to graph diffusion/consistency rather than claimed as MCGC-specific. In that case, the MCGC contribution is the integration of that mechanism with contrastive score structure, path/support explanations, feature-level interventions, and explicit label-field disagreement diagnostics.

## Noise model

Primary corruption model: symmetric random class flips on the outer-training set only.

For each selected training sample, replace its label uniformly by one of the other observed classes. The outer-test labels remain untouched.

Frozen full noise levels:

- 0.00
- 0.10
- 0.20
- 0.30

The same corrupted outer-training labels are supplied to every compared method within a dataset/seed/noise unit.

## Data splitting

Outer split:

- 70% outer training
- 30% untouched test
- stratified by the clean labels before corruption
- split seed: `5000 + seed`

Noise seed: `18000 + 1000*seed + round(100*eta)`.

All preprocessing and tuning occur inside the outer-training set.

## Smoke stage

The smoke stage is diagnostic only and does not establish a paper claim.

Dataset: digits.

Seed: 0.

Noise levels: 0.00 and 0.30.

Purpose:

1. verify a correct no-test-leakage implementation,
2. verify Label Spreading as a direct diffusion baseline,
3. verify the out-of-fold label-audit calculation,
4. estimate runtime before the full confirmation run.

## Full confirmation stage

Run only after the smoke implementation passes.

Frozen seeds: 0 through 11 (12 outer splits).

The final real-data dataset list will be locked before the first full confirmatory run. Digits is retained as the continuity dataset. Additional confirmation datasets must not be selected based on v1.8 outcomes.

No dataset may be removed after observing an unfavourable result except for a documented technical failure that makes the predefined protocol impossible to execute.

## Baselines

Required classification baselines:

- distance/uniform tuned kNN
- logistic regression
- RBF SVM
- random forest
- Label Spreading

All baseline tuning uses only the noisy outer-training labels available to that method. No clean training labels are used for hyperparameter selection.

## MCGC variants

Two roles must remain distinct.

### Frozen MCGC

Use the established pre-v1.8 configuration as a non-adaptive continuity baseline.

### Noise-stage MCGC tuning

Any tuned MCGC variant must select hyperparameters only from noisy outer-training labels. Clean pre-corruption labels may be used only to score synthetic corruption recovery after fitting, never to choose hyperparameters.

A result based on oracle selection using clean labels must be marked exploratory and cannot support the operational robustness claim.

## Label-audit anti-leakage rule

The corrupted-label detector must be evaluated out of fold.

For each audit fold:

1. fit MCGC on the remaining noisy outer-training samples,
2. score held-out samples as queries,
3. compare local evidence assigned to their observed label with the strongest rival,
4. concatenate scores across folds,
5. compare against the known synthetic corruption mask.

Do not compute the primary audit AUC from a field fitted on the same sample whose label is being audited.

## Secondary correction experiment

This is predefined as secondary and is run only after the primary metrics are frozen.

At each nonzero noise level, rank training samples by the OOF audit score and select top-k, where k equals the true number of corruptions.

Evaluate separately:

1. **oracle correction** — restore the original clean labels of selected samples,
2. **field-based correction** — replace selected labels by the strongest OOF graph-evidence class.

Refit the classifier and report recovery relative to the corrupted-label model.

The oracle arm measures the value of the ranking if a human/oracle verifies flagged labels. The field-based arm tests automatic correction and must not be conflated with detection performance.

## Statistical unit and reporting

The primary independent reporting unit is dataset × outer seed.

Do not treat multiple queries, audit samples, or noise levels as independent replications when making inferential claims.

Report:

- per-dataset means and medians,
- paired seed-level differences,
- wins/ties/losses,
- descriptive confidence intervals where useful,
- sign tests only for prespecified paired comparisons with sufficient units.

## Claim boundaries

Even if positive, v1.8 does not justify:

- that MCGC is universally more accurate,
- that diffusion robustness is novel,
- that a suspicious real-world label is necessarily wrong,
- that the field identifies annotation errors causally.

Safe positive interpretation if confirmed:

> MCGC uses graph consistency to remain stable under partial training-label corruption and exposes out-of-fold conflicts between observed labels and the fitted graph evidence. The latter provides an inspectable training-data reliability signal in addition to the model's decision-level explanations.

If Label Spreading matches the robustness result, explicitly state that robustness is a graph-diffusion effect and position the MCGC-specific contribution around integration and traceability rather than diffusion itself.
