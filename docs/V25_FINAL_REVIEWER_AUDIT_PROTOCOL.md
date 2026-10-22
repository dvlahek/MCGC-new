# MCGC v2.5 final reviewer audit protocol

Status: frozen before result generation.

This branch closes three reviewer-facing questions without opening a new research direction.

## A. Signed winner-support faithfulness control

Parent: `benchmark_xai_faithfulness_v09ter.py` / v0.9-bis family.

Datasets and seeds remain exactly the frozen faithfulness grid: four datasets x five seeds, with the same train/test splits and the same cap on eligible nontrivial-path queries.

For query q with winner c+ and runner-up c-, the signed local contribution of start node j is

`a_j * (D_j^(c+) - D_j^(c-))`.

Only strictly positive contributions are eligible for the signed winner-support comparator. The comparator selects the top `|path|` positive contributors from the complete query start neighbourhood and may overlap the representative path. If fewer than `|path|` positive contributors exist, that query is **ineligible for the signed comparison**. No negative or zero-contribution nodes are used as padding.

Primary signed metric: path margin drop minus signed-control margin drop, where margin drop is `M_full - M_removed` for the original winner-runner pair.

The output must also retain the historical label+distance matched off-path comparison so the manuscript can state the full hierarchy rather than only the strongest comparator.

Inference unit: dataset x seed. Query-level records are descriptive only.

## B. Path-tail stability

Parent: `benchmark_xai_stability_v15bis.py`.

The original full-path stability audit is reproduced unchanged. Two additional quantities are added:

1. **Tail Jaccard**: remove the first path node from both paths and compute Jaccard on the remaining node sets, provided both tails are non-empty.
2. **Same-start conditioned tail Jaccard**: evaluate the tail only when the original and compared path have the same first node.

Both within-query perturbation and nearest same-prediction cross-query controls are retained. Primary inference remains dataset x seed.

This audit tests how much path stability remains after removing the kNN-derived anchor.

## C. Field-only vs full-walker ablation

Datasets, seeds, model settings, and train/test splits are taken from the v1.5 stability suite: digits, wine, spirals_hard, breast_cancer x five seeds, split seed `7000 + seed`.

Two models are fit per unit with identical graph/diffusion parameters and random state:

- `full`: walkers enabled, adaptive walker gate enabled;
- `field_only`: walkers disabled, so prediction uses local evidence only.

Report per dataset x seed:

- accuracy,
- balanced accuracy,
- prediction disagreement rate,
- mean absolute change in top-two score margin when available.

This is an ablation of predictive contribution, not a new tuning exercise. No hyperparameters are changed after results are seen.

## Interpretation lock

- Signed control beats path: path is structural, not a superior influence selector.
- Path beats matched off-path but not signed control: path carries information beyond proximity/label matching, while direct local evidence remains the stronger influence ranking.
- Tail stability remains strong: query-specific path structure is not explained only by the starting kNN anchor.
- Tail stability collapses: manuscript must attribute most stability to the local anchor.
- Full vs field-only shows little predictive difference: walker/gate language remains mechanistic/design-oriented, not a performance claim.
- Full improves selected units only: report selectivity without generalizing beyond the frozen grid.
