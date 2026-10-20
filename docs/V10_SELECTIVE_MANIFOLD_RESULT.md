# MCGC v1.0 — frozen selective-manifold result

## Status

The preregistered v1.0 selective-manifold benchmark is **positive on the frozen nonlinear-manifold family**.

The primary family was fixed before inspecting v1.0 results:

- `moons_hard`
- `circles_hard`
- `spirals_hard`

The primary metric was E-AURC (lower is better), with paired advantages defined as baseline E-AURC minus MCGC E-AURC. No confidence calibration, feature search, or dataset-specific retuning was used.

## Primary result against distance-weighted kNN

Across the 36 frozen manifold runs (3 datasets × 12 seeds), MCGC local graph-field margin achieved lower E-AURC than distance-weighted kNN confidence in:

- 29 wins
- 0 ties
- 7 losses

Paired E-AURC advantage:

- mean: `+0.009157069986134773`
- median: `+0.006353876329219648`
- sign-test p: `0.00031255115754902363`

Ordinary AURC was also favorable:

- 25 wins
- 0 ties
- 11 losses
- mean advantage: `+0.008125534126382374`
- median advantage: `+0.005130841291406594`
- sign-test p: `0.028816719655878842`

This is the main v1.0 result because weighted kNN is the strongest directly relevant local-geometry baseline in the frozen benchmark.

## Per-dataset E-AURC results against weighted kNN

### `moons_hard`

- MCGC mean E-AURC: `0.01880861195218542`
- weighted-kNN mean E-AURC: `0.02645051210056591`
- paired wins/losses: `11/1`

### `circles_hard`

- MCGC mean E-AURC: `0.017257383250261608`
- weighted-kNN mean E-AURC: `0.023252059962603728`
- paired wins/losses: `9/3`

### `spirals_hard`

- MCGC mean E-AURC: `0.19206331971499538`
- weighted-kNN mean E-AURC: `0.20589795281267712`
- paired wins/losses: `9/3`

Thus the pooled result is not produced by only one of the three nonlinear-manifold problems.

## Logistic-regression reference

MCGC also strongly outperformed logistic-regression confidence over the primary family:

- E-AURC: 32 wins / 4 losses, mean advantage `+0.13541330566873053`
- AURC: 35 wins / 1 loss, mean advantage `+0.1938439236106345`

However, this contrast is **not the main superiority claim**, because a linear logistic model is structurally mismatched to strongly nonlinear manifolds, especially concentric circles. The weighted-kNN comparison is therefore the more defensible primary reference.

## Raw accuracy and selective accuracy

The v1.0 result should not be described as universal raw-accuracy superiority.

Examples:

- On `moons_hard`, MCGC mean accuracy was `0.9027777777777778` versus weighted kNN `0.8972222222222223`.
- On `spirals_hard`, MCGC mean accuracy was `0.6226190476190476` versus weighted kNN `0.6305555555555554`.

The contribution is therefore specifically about **confidence/risk ranking and selective prediction on nonlinear manifolds**, not blanket classifier dominance.

On `spirals_hard`, MCGC nevertheless improved selective accuracy at some coverages, for example:

- 50% coverage: MCGC `0.7126984126984127` vs kNN `0.699206349206349`
- 70% coverage: MCGC `0.6740362811791384` vs kNN `0.6723356009070295`
- 80% coverage: MCGC `0.6557539682539683` vs kNN `0.6607142857142857`
- 90% coverage: MCGC `0.6432980599647267` vs kNN `0.642416225749559`

This mixed fixed-coverage picture is consistent with the main claim being lower integrated selective risk rather than uniform dominance at every operating point.

## Transfer/control datasets

The effect is not universal.

On `overlap_hd`, MCGC and weighted kNN were essentially tied in E-AURC on average (mean paired advantage approximately `-0.00006945`, 6 wins / 6 losses).

On `breast_cancer`, MCGC strongly outperformed weighted kNN but was worse than logistic regression, which is appropriate for this more conventional tabular problem.

These controls support a geometry-specific interpretation rather than a universal superiority claim.

## Combined interpretation with v0.9

The current strongest MCGC story has two empirically distinct components:

1. **Faithful intrinsic XAI (v0.9):** representative path nodes are more functionally decision-relevant than matched off-path controls and equal-size nearest-neighbour explanations.
2. **Selective confidence on nonlinear manifolds (v1.0):** local graph-field margin ranks errors better than distance-weighted kNN confidence in the frozen nonlinear-manifold benchmark.

A defensible current positioning is:

> MCGC is an intrinsically traceable graph classifier whose class-conditioned graph explanations identify functionally decision-relevant training samples, while its graph-field margin provides improved selective-risk ranking relative to distance-weighted kNN confidence on nonlinear-manifold problems.

This does **not** establish universal raw-accuracy superiority, universal uncertainty superiority, or superiority to all modern selective-classification methods.

## Next validation priority

The next confirmation step should strengthen the selective-classification comparison rather than tune MCGC:

- keep the MCGC classifier and local-margin confidence frozen;
- add stronger nonlinear baselines such as RBF-SVM confidence and Random Forest probability/margin;
- include several real datasets with nonconvex or manifold-like structure where possible;
- retain E-AURC, ordinary AURC, raw accuracy, and fixed-coverage selective accuracy;
- preserve the current weighted-kNN result as a frozen v1.0 checkpoint.

The goal of the next stage is to determine whether the v1.0 gain is specific to kNN confidence or survives comparison with stronger nonlinear confidence baselines.
