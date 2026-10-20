# MCGC v0.9 — frozen XAI faithfulness protocol

## Goal

After the negative v0.8 combined-reliability confirmation, v0.9 shifts the primary research question from uncertainty improvement to **intrinsic explanation faithfulness**.

The classifier formulation and representative-trail construction are not retuned in v0.9. The goal is to test whether explanations already exposed by MCGC identify training nodes that are functionally important to the fitted decision.

## Frozen hypotheses

### H1 — Exact decomposition fidelity

For every explained query and class, the reported score components must reconstruct the model decision score to numerical tolerance. This is a hard correctness requirement, not a performance metric.

### H2 — Representative-path reproducibility

For a fixed fitted model and query, the deterministic representative path must be identical across repeated explanation calls.

### H3 — Contrastive consistency

For the predicted class `c*` and runner-up class `r`, the reported contrastive component differences must sum exactly to the reported decision margin:

`margin(c*, r) = score(c*) - score(r)`

### H4 — Path-node intervention faithfulness

For an evaluable query, take the existing deterministic representative path for the predicted class. Delete those path nodes from the classifier training set, refit the unchanged MCGC, and re-evaluate the original winner-vs-runner margin.

Repeat the same delete-and-refit intervention with a control set of the same cardinality. Each control node is matched, where possible, to a removed path node by training label and original standardized distance from the query, while excluding all path nodes.

The primary paired quantity is:

`faithfulness_advantage = margin_drop(path_nodes) - margin_drop(matched_off_path_controls)`

Positive values mean that the path explanation identifies nodes whose removal damages the explained decision more than removing equally sized label- and distance-matched off-path controls.

A query is path-evaluable only if the representative winner trail contains at least two distinct training nodes. Ineligible queries are counted and reported, never replaced after looking at outcomes.

### H5 — Proximity-only explanation comparison

As a secondary pre-specified baseline, remove the same number of nearest **off-path** training nodes and refit the model.

Report:

`path_vs_nearest_advantage = margin_drop(path_nodes) - margin_drop(nearest_off_path_nodes)`

A positive value supports the narrower claim that MCGC's path explanation identifies functionally more consequential training nodes than a proximity-only explanation of equal size.

### H6 — Small-perturbation stability

For queries whose prediction is unchanged under a fixed small standardized perturbation, report:

- top-k support overlap;
- representative-path node overlap;
- dominant contrastive-component consistency.

This is secondary. No positive threshold will be selected after seeing results.

## Frozen benchmark scope

To keep delete-and-refit computation practical while covering distinct geometry types, v0.9 uses:

- `moons_hard` — nonlinear manifold;
- `spirals_hard` — long curved manifold;
- `overlap_hd` — overlapping higher-dimensional classes;
- `breast_cancer` — real tabular data.

Dataset seeds are `0..4`. The classifier split uses the existing 70/30 stratified convention with split seed `4000 + dataset_seed`.

For each fitted model, evaluate the first 12 test queries in deterministic test-set order whose winner representative trail has at least two distinct nodes. If fewer than 12 are eligible, report all available eligible queries and the eligibility rate.

No query replacement based on intervention outcome is allowed.

## Classifier settings

Use the existing v0.5+ MCGC settings unchanged:

- `n_neighbors=12`
- `n_walkers=12`
- `max_steps=10`
- `diffusion_steps=40`
- `random_state=17`
- `adaptive_walker_gate=True`

No dataset-specific tuning is allowed.

## Primary reporting

Per dataset and pooled, report:

- eligible / inspected queries;
- mean and median `faithfulness_advantage`;
- wins / ties / losses;
- two-sided sign-test p-value;
- mean and median `path_vs_nearest_advantage`;
- prediction-flip rate after each intervention;
- mean original path length;
- exact decomposition failure count;
- exact contrastive-margin failure count.

The primary v0.9 hypothesis is H4. H5 is the pre-specified comparative XAI baseline.

## Claim discipline

A successful H4 result would support:

> MCGC's intrinsic path explanations identify training nodes that are more functionally important to its decision than label- and locality-matched off-path controls.

If H5 is also positive, a stronger but still narrow statement becomes defensible:

> MCGC path explanations identify decision-relevant training nodes more effectively than equal-size proximity-only explanations for the tested tasks.

Neither result by itself implies that MCGC is fully interpretable, generally causally explanatory, or universally superior to post-hoc XAI methods.
