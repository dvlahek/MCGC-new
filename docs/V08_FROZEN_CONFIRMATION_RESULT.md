# MCGC v0.8 frozen combined-reliability confirmation

## Status

The frozen v0.8 primary hypothesis is **not confirmed**.

The primary contrast was fixed before evaluation as:

`ROC-AUC(combined_intrinsic) - ROC-AUC(local_only)`

with the v0.7 feature set carried forward unchanged:

- `local_margin_risk`
- `final_margin_risk`
- `trail_disagreement_risk`
- `coherence_risk`
- `lost_trail_risk`

No dataset-specific feature selection or retuning was allowed. The classifier/risk split remained 60% / 20% / 20%, and seeds without both calibration correctness classes were skipped rather than resplit.

## Confirmation summary

Across 53 evaluable paired runs:

- mean delta AUC: `-0.0130355`
- median delta AUC: `-0.00733945`
- wins / ties / losses: `19 / 2 / 32`
- two-sided sign-test p: `0.0919145`

Synthetic runs (n=32):

- mean delta AUC: `-0.00961426`
- median delta AUC: `-0.00845038`
- wins / ties / losses: `10 / 1 / 21`

Real-data runs (n=21):

- mean delta AUC: `-0.0182488`
- median delta AUC: `-0.000493097`
- wins / ties / losses: `9 / 1 / 11`

Every dataset had a negative mean paired delta:

| Dataset | Local ROC-AUC | Combined ROC-AUC | Mean paired delta | W/T/L |
|---|---:|---:|---:|---:|
| moons_hard | 0.809876 | 0.801676 | -0.008200 | 2/1/5 |
| circles_hard | 0.842726 | 0.830308 | -0.012418 | 3/0/5 |
| spirals_hard | 0.629886 | 0.614820 | -0.015066 | 2/0/6 |
| overlap_hd | 0.733875 | 0.731102 | -0.002773 | 3/0/5 |
| breast_cancer | 0.922512 | 0.910714 | -0.011799 | 3/0/5 |
| wine | 0.931412 | 0.911576 | -0.019835 | 1/1/3 |
| digits_0_4 | 0.978365 | 0.954657 | -0.023707 | 5/0/3 |

## Interpretation

The positive secondary spiral signal observed in v0.7 did not replicate under the broader frozen v0.8 confirmation. The combined path-derived diagnostic set should therefore **not** be claimed as a generally improved uncertainty or error-detection model over the local graph-field margin.

This result is treated as a closed negative checkpoint, not as a target for post-hoc tuning.

The project direction after v0.8 is:

1. keep `local_margin_risk` as the strongest currently supported intrinsic reliability signal;
2. keep walker/path diagnostics as explanation objects unless future independent evidence shows predictive reliability value;
3. move the main research emphasis to intrinsic, path-based XAI and explanation faithfulness/stability;
4. avoid claims that walkers improve raw accuracy or standalone uncertainty under the current formulation.

## Next research checkpoint

v0.9 should focus on frozen XAI validation rather than another reliability feature search. Candidate validation targets include:

- exact score-decomposition fidelity;
- deterministic representative-path reproducibility;
- predicted-vs-runner-up contrastive explanation consistency;
- supporting-sample/path perturbation tests against matched controls;
- explanation stability under small input perturbations;
- counterfactual explanation construction and validation.
