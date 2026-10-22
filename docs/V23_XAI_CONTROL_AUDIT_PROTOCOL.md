# MCGC v2.3 XAI control audit

Status: **frozen before execution**.

This protocol responds to two reviewer-style concerns in the current explanation analysis:

1. the v0.9 delete-and-refit comparison against equal-size nearest off-path samples can mix winner- and rival-class samples, and
2. support-set stability may be inherited directly from the fitted k-NN neighbourhood.

The classifier, train/test split rules, seeds, path construction, and query caps remain unchanged from the frozen parent analyses. No threshold or model parameter is tuned from the results below.

## A. v0.9-bis delete-and-refit controls

Parent: `examples/benchmark_xai_faithfulness_v09.py`.

Datasets: `moons_hard`, `spirals_hard`, `overlap_hd`, `breast_cancer`.

Seeds: 0--4.

Per seed, inspect test queries in the same deterministic order until 12 queries with a non-trivial winner representative path are obtained.

For each eligible query, freeze the original winner and runner-up and define

`margin_drop(S) = M_full(q) - M_refit_without_S(q)`.

Compare the representative-path node set with four controls of equal requested size:

- `matched_distance_label`: the original v0.9 label-and-query-distance-matched off-path control.
- `nearest_any`: the original equal-size nearest off-path control, retained only as a historical sensitivity check.
- `nearest_label_composition`: for each path-node label, choose the nearest unused off-path training point with that same label. This preserves the path label composition while making the control as query-local as possible.
- `top_local_contribution`: among exact query start neighbours not on the path, rank nodes by `affinity * abs(D_winner - D_runner)` and remove the top equal-size set. This is a deliberately strong local-evidence baseline. Queries without enough off-path start neighbours are marked ineligible for this comparison only.

Primary contrasts are `margin_drop(path) - margin_drop(control)` for `nearest_label_composition` and `top_local_contribution`.

The primary inference unit is dataset x seed. Query-level results are descriptive. Report mean/median contrast, wins/ties/losses across the 20 dataset x seed units, exact two-sided sign tests, and prediction-flip rates.

Interpretation is frozen in advance:

- path > strong local-evidence control: supports an influential-example claim beyond local evidence ranking;
- path approximately equal: supports a structured evidence-route interpretation but not superiority as an attribution set;
- path < strong local-evidence control: the path is an evidence-route explanation, not an optimal influential-example selector.

## B. v1.5-bis support/path stability control

Parent: `examples/benchmark_xai_stability_v15.py`.

Datasets, seeds, perturbation amplitudes, perturbation repetitions, and the deterministic first-40-query subset are unchanged.

Two changes are introduced:

1. explicitly compute the plain standardized k-NN top-5 support set and verify whether it is identical to the MCGC `supporting_neighbors` set;
2. replace the arbitrary next-same-predicted-class cross-query control with the nearest test query having the same predicted class in standardized feature space.

The same prediction-preserving perturbations are used. Report within-query Jaccard, distance-matched cross-query Jaccard, and their difference for support and representative path.

If MCGC support equals plain k-NN support, support stability will be described as inherited from the local graph rather than as a path-specific MCGC explanation result.

## Claim boundary

Exact score reconstruction is an implementation-fidelity property and is not part of the empirical v2.3 hypothesis test.
