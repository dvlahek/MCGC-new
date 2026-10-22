# MCGC v0.9-ter / v2.4 XAI faithfulness follow-up

Status: frozen before results.

## Motivation

The frozen v0.9-bis reviewer-response audit added a strong local-evidence comparator, but that comparator excluded representative-path nodes from its candidate pool. This can disadvantage the comparator because the strongest local-evidence nodes may themselves lie on the representative path.

This follow-up changes exactly one point: the local-evidence comparator is selected from the complete query neighbourhood, including path nodes.

## Frozen parent protocol

All datasets, splits, seeds, classifier parameters, eligible-query rule, path construction, delete-and-refit procedure, winner/runner pair, query cap, and dataset x seed inference unit are inherited unchanged from `benchmark_xai_faithfulness_v09bis.py`.

Datasets: `moons_hard`, `spirals_hard`, `overlap_hd`, `breast_cancer`.

Seeds: 0--4.

Maximum eligible queries: 12 per dataset x seed unit.

## Unrestricted local-evidence comparator

For each eligible query, let `c+` be the fitted winner and `c-` the fitted runner-up. For every node `j` in the query's fitted k-neighbourhood, compute

`L_j = a_j * abs(D_j^(c+) - D_j^(c-))`,

where `a_j` is the normalized query-start affinity and `D` is the fitted class-contrast field.

Rank all query-neighbourhood nodes by decreasing `L_j`, without excluding nodes that belong to the representative path. Select the top `|path|` unique nodes.

The comparator is therefore allowed to overlap partially or completely with the representative path.

## Primary contrast

For each query:

`C = margin_drop(path) - margin_drop(unrestricted top-local-contribution set)`.

Positive values favor the path; negative values favor the unrestricted local-evidence baseline.

Primary inference remains the mean query-level contrast within each dataset x seed unit, followed by a two-sided exact sign test across available units.

## Additional diagnostics

Report:

- overlap fraction between the path and unrestricted comparator,
- exact-set equality rate,
- query-level and dataset x seed contrasts,
- margin-drop and prediction-flip rates for both interventions.

## Interpretation locked before results

- If the path is consistently stronger, this supports a path-specific functional-faithfulness claim.
- If the two are comparable, the path is interpreted as a structured evidence-route explanation without superior delete-and-refit influence.
- If the unrestricted local-evidence baseline is stronger, the path is not described as an influential-example selector; its role is structural and query-specific.

No threshold, model parameter, dataset, seed, or eligibility rule may be changed after observing results.