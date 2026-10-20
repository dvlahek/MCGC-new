# MCGC v0.9 — XAI faithfulness result

## Status

Frozen v0.9 XAI faithfulness confirmation completed.

The classifier and representative-trail construction were unchanged. Each intervention deleted training nodes and refit the same classifier. The primary contrast was

`margin_drop(path_nodes) - margin_drop(label-and-distance-matched off-path nodes)`

and the secondary contrast was

`margin_drop(path_nodes) - margin_drop(equal-size nearest off-path nodes)`.

## Datasets and protocol

Datasets:

- `moons_hard`
- `spirals_hard`
- `overlap_hd`
- `breast_cancer`

Five dataset seeds were used. The split was 70% training / 30% untouched test, with split seed `4000 + dataset_seed`. At most 12 eligible queries per seed were evaluated. A query was eligible only when the deterministic representative path was nontrivial enough for the intervention.

## Pooled query-level results

Across 240 eligible queries:

### Primary matched-control contrast

- mean faithfulness advantage: `+0.0926154467`
- median: `+0.0625828114`
- wins / ties / losses: `202 / 0 / 38`
- two-sided sign-test p-value: `3.377174048e-28`

### Secondary proximity-only contrast

- mean path-vs-nearest advantage: `+0.3593250906`
- median: `+0.3049531662`
- wins / ties / losses: `207 / 0 / 33`
- two-sided sign-test p-value: `5.430695506e-32`

The path-removal prediction-flip rate was `0.29583`, compared with `0.23750` for the matched control and `0.06250` for the nearest-off-path control.

## Per-dataset primary result

All four datasets had positive mean primary contrast:

- `moons_hard`: mean `+0.1566762324`, wins/losses `56/4`
- `spirals_hard`: mean `+0.1088086735`, wins/losses `49/11`
- `overlap_hd`: mean `+0.0430927165`, wins/losses `47/13`
- `breast_cancer`: mean `+0.0618841645`, wins/losses `50/10`

The strongest proximity result was on `spirals_hard`: mean path-vs-nearest advantage `+0.5647923100`, wins/losses `59/1`. Path removal flipped the prediction in `83.33%` of eligible spiral queries, compared with `18.33%` for nearest-off-path removal.

## Exact contrastive fidelity

Contrastive score reconstruction remained exact to floating-point precision. The observed maximum absolute reconstruction errors were on the order of `1e-16` to `4e-16`.

## Cluster-aware robustness check

The pooled query-level sign tests treat within-run queries as if they were independent, which is optimistic because queries are nested inside dataset-seed runs. Therefore the result was also checked at the dataset-seed aggregate level.

There were 20 dataset-seed units (4 datasets x 5 seeds):

- all `20/20` had positive mean primary faithfulness advantage;
- all `20/20` had positive mean path-vs-nearest advantage;
- the two-sided sign-test p-value for 20/20 positives is approximately `1.907348633e-06` for either contrast.

This is the preferred reviewer-resistant significance summary.

## Interpretation

v0.9 provides the first strong positive empirical result for MCGC.

The representative path does not merely provide a visual trace. Removing the training nodes identified by that path changes the original class margin more than removing either:

1. label-and-distance-matched off-path controls, or
2. an equal number of nearest off-path training examples.

This supports the statement that MCGC path explanations identify training examples that are more functionally relevant to the fitted decision than proximity-only explanations.

## Claim discipline

Supported wording:

> MCGC provides intrinsically traceable, class-contrastive graph explanations. In frozen delete-and-refit interventions, nodes selected by its representative paths were consistently more influential on the original class margin than matched off-path controls and equal-size nearest-neighbour explanations.

Not yet supported:

- MCGC is universally more interpretable than all post-hoc XAI methods;
- the path is a complete causal explanation of the underlying data-generating process;
- every query receives a nontrivial path explanation;
- raw predictive accuracy is superior to strong baselines.

## Important limitation

Eligibility was substantially below 100% because many queries had only trivial/short representative paths. The v0.9 claim therefore applies to queries for which a nontrivial representative path exists. This should be reported explicitly rather than hidden.

## Next step

Use v1.0 to test whether MCGC can also obtain a task-level advantage in selective classification on nonlinear/manifold problems, using a frozen risk-coverage protocol and standard confidence baselines. Keep the v0.9 XAI result separate from predictive-performance claims.
