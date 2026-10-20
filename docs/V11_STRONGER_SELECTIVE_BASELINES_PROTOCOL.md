# MCGC v1.1 — stronger selective-baseline confirmation protocol

## Goal

v1.0 established a frozen positive selective-risk result against distance-weighted kNN confidence on the predefined nonlinear-manifold family. v1.1 asks a stricter question:

> Does the frozen MCGC local graph-field margin remain competitive when compared with stronger nonlinear classifiers and their native confidence signals?

No MCGC classifier parameter, confidence definition, dataset definition, or metric is tuned in v1.1 after observing results.

## Frozen MCGC object

Use the same MCGC configuration and the same local graph-field margin risk as v1.0.

Primary MCGC risk signal:

`local_margin_risk = - local_margin`

No additional path diagnostics, learned risk model, calibration model, or feature combination is permitted.

## Frozen primary datasets

The primary nonlinear-manifold family remains:

- `moons_hard`
- `circles_hard`
- `spirals_hard`

Use exactly the same generators, sample sizes, noise levels, 12 seeds, and 70/30 train/test split logic as v1.0.

Control/transfer datasets remain:

- `overlap_hd`
- `breast_cancer`

## Baselines

Retain the frozen v1.0 references:

1. distance-weighted kNN, `k=12`, confidence `max predict_proba`;
2. logistic regression with StandardScaler, confidence `max predict_proba`.

Add two stronger nonlinear baselines with fixed, conventional settings:

3. **RBF SVM**
   - StandardScaler
   - `SVC(kernel="rbf", C=1.0, gamma="scale", probability=False)`
   - confidence = absolute binary decision-function margin for binary tasks;
   - for multiclass, confidence = gap between the largest and second-largest decision-function values when available.

4. **Random Forest**
   - `RandomForestClassifier(n_estimators=300, max_features="sqrt", random_state=97, n_jobs=-1)`
   - confidence = maximum predicted class probability.

No hyperparameter search is allowed.

## Primary metric

Primary metric remains **E-AURC**, lower is better.

For each baseline `b`, define paired advantage:

`E-AURC(b) - E-AURC(MCGC)`

Positive values favor MCGC.

The main new v1.1 contrasts are:

- RBF-SVM vs MCGC
- Random-Forest vs MCGC

The v1.0 kNN and logistic contrasts are repeated only for continuity/regression checking.

## Secondary metrics

Report for every method:

- raw accuracy;
- ordinary AURC;
- oracle AURC;
- E-AURC;
- error-detection ROC-AUC when both correctness classes are present;
- selective accuracy at fixed coverage 0.90, 0.80, 0.70, and 0.50.

## Statistical reporting

For each dataset and for the pooled 36-run primary manifold family, report paired:

- mean advantage;
- median advantage;
- standard deviation;
- minimum and maximum;
- wins / ties / losses;
- two-sided sign-test p-value.

The pooled 36-run manifold-family result is the main aggregate. Dataset-specific results must also be shown so that one benchmark cannot silently dominate the conclusion.

## Claim discipline

A positive v1.1 result against RBF-SVM and/or Random Forest would strengthen the claim that MCGC local graph-field confidence is specifically useful for nonlinear-manifold selective prediction.

A mixed or negative result would not invalidate v1.0. It would instead delimit the current evidence to superiority over proximity-based kNN confidence, which remains a valid frozen checkpoint.

Do not claim universal selective-classification superiority, universal uncertainty superiority, or SOTA performance from this benchmark alone.
