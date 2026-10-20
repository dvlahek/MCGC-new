# MCGC v0.7 calibrated incremental-reliability protocol

v0.6 showed that the local graph-field margin is a strong intrinsic reliability baseline and that walker trail-score disagreement also ranks errors, especially on moons. v0.7 tests the stricter question: **does walker disagreement add held-out reliability information after controlling for the local margin?**

## Data separation

For each dataset seed, data are split into three disjoint sets:

- 60% classifier training,
- 20% reliability calibration,
- 20% final test.

The MCGC classifier is fitted only on the training split. Reliability features and correctness labels from the calibration split are used to fit the risk models. The final test split is used once for evaluation. No test labels are used for risk-model fitting, model selection, threshold selection or feature selection.

Seeds remain `0..11`. Dataset generation remains identical to the prior synthetic confirmation chain.

## Frozen risk models

Each risk model is a standardised logistic regression with `class_weight="balanced"`, fixed `C=1.0`, `max_iter=2000`, and fixed random state. The target is classification error (`1 = incorrect`).

The four models are fixed before test evaluation:

1. `local_only`
   - `local_margin_risk`

2. `local_plus_final`
   - `local_margin_risk`
   - `final_margin_risk`

3. `local_plus_trail_disagreement`
   - `local_margin_risk`
   - `trail_disagreement_risk`

4. `combined_intrinsic`
   - `local_margin_risk`
   - `final_margin_risk`
   - `trail_disagreement_risk`
   - `coherence_risk`
   - `lost_trail_risk`

No alternative feature set will be substituted after seeing the results.

## Feasibility rule

A seed is evaluable for calibrated reliability only if the calibration split contains at least one correct and one incorrect classifier prediction. If not, that seed is marked `insufficient_calibration_classes` rather than changing the split or pooling test labels into calibration.

## Primary comparison

Primary metric: held-out test ROC AUC for error detection.

The key preregistered contrast is:

```text
AUC(local_plus_trail_disagreement) - AUC(local_only)
```

reported per seed and summarised by mean delta plus wins/ties/losses over evaluable seeds.

A positive result requires the walker-augmented model to show a consistently positive held-out delta, not merely one favourable pooled test result.

## Secondary metrics

- average precision for error detection,
- Brier score for predicted error probability,
- selective accuracy at retained coverages 90%, 80%, 70% and 50%, using the risk model's predicted test error probability,
- comparison with `local_plus_final` and `combined_intrinsic`.

These are descriptive secondary outcomes.

## Claim discipline

A positive v0.7 result would support the narrow claim that walker disagreement contains **complementary reliability information** beyond local graph-field ambiguity. It would not show that walkers improve the class prediction itself.

A null or negative result means the current walker diagnostics should not be sold as adding reliability beyond the graph-field baseline. The intrinsic path explanations can still be useful for traceability, but reliability claims must then be based primarily on graph-field uncertainty.
