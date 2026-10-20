# MCGC v1.5 — frozen XAI stability, multiclass contrastive fidelity, and path eligibility

## Status

Completed frozen confirmation on branch `feature/v1.5-xai-stability`.

The protocol, perturbation scales, dataset suite, seed set, query cap, support size, and primary same-query-vs-same-class control contrasts were frozen before the science run.

Datasets: `digits`, `wine`, `spirals_hard`, `breast_cancer`. Seeds: 0–4.

## Primary stability result

Small feature perturbations were applied at 0.02 and 0.05 training-feature standard deviations. Stability was evaluated only when the perturbed query preserved the original prediction. The primary comparison was the Jaccard similarity of the perturbed explanation to the original explanation minus the similarity to a different same-predicted-class query.

Dataset×seed support-stability advantage:

- mean `+0.8683349141`
- median `+0.8767857143`
- W/T/L `20/0/0`
- two-sided sign-test `p = 1.9073486328125e-06`

Dataset×seed representative-path stability advantage, where a nontrivial path comparison was evaluable:

- mean `+0.8527481131`
- median `+0.9625`
- W/T/L `17/0/0`
- two-sided sign-test `p = 1.52587890625e-05`

Thus both the local supporting-neighbour set and the representative path retain strong query-specific structure under small prediction-preserving perturbations, rather than merely reflecting the predicted class.

## Per-dataset summary

| dataset | accuracy | path eligibility | prediction preservation | support advantage | path advantage |
|---|---:|---:|---:|---:|---:|
| digits | `0.94630` | `0.2400` | `0.99188` | `+0.94246` | `+0.94141` |
| wine | `0.94815` | `0.24815` | `0.99063` | `+0.84083` | `+0.81272` |
| spirals_hard | `0.63238` | `0.22857` | `0.95656` | `+0.77973` | `+0.75703` |
| breast_cancer | `0.94386` | `0.28187` | `0.99938` | `+0.91032` | `+0.89042` |

All five dataset-seed support advantages were positive within every dataset. Path advantage was also positive in every evaluable dataset-seed unit.

## Multiclass contrastive fidelity

The exact `why winner rather than runner-up` decomposition was evaluated across all test queries on the multiclass datasets.

- `digits`: maximum absolute reconstruction error `8.881784197001252e-16`
- `wine`: maximum absolute reconstruction error `4.440892098500626e-16`

This confirms that the contrastive component decomposition reconstructs the requested winner-vs-runner-up class-score difference to floating-point precision in multiclass settings, not only binary examples.

## Path eligibility

Nontrivial representative winner paths occur for roughly one quarter of test queries:

- digits: `24.0%`
- wine: `24.8%`
- spirals_hard: `22.9%`
- breast_cancer: `28.2%`

Eligibility is not a generic marker of easy examples. Across the frozen runs, eligible queries generally have lower margins and lower accuracy than noneligible queries. This is especially clear on breast cancer and spirals, and is also present on digits and wine. The interpretation is that nontrivial paths preferentially occur in more difficult / less locally decisive regions where the query is not already inside a strong terminal evidence basin.

This is an important boundary for the v0.9 faithfulness claim: path-based explanations are available for a minority of queries, but that minority is systematically concentrated in cases where a nontrivial graph trajectory is needed.

## Interpretation

v1.5 is a strong positive XAI confirmation.

The result strengthens v0.9 in three independent ways:

1. **stability:** support sets and representative paths remain strongly query-specific under small prediction-preserving perturbations;
2. **multiclass exactness:** the contrastive score decomposition remains exact to floating-point precision on full multiclass test sets;
3. **eligibility characterization:** nontrivial paths occur in roughly 23–28% of queries and are associated with lower-margin, more difficult cases rather than uniformly easy predictions.

The result does **not** imply that every query has a nontrivial path or that the representative path is the sole causal mechanism of prediction. It supports the narrower claim that, when a path exists, it is a stable trace through the fitted evidence landscape, while the accompanying score decomposition is exact.

## Updated manuscript-level claim

A reviewer-resistant statement supported by v0.9 + v1.5 is:

> MCGC provides intrinsically traceable sample-level explanations whose score decomposition is exact and whose supporting-neighbour and representative-path structure is stable under small prediction-preserving perturbations. For queries admitting a nontrivial representative path, the selected training samples are also interventionally more decision-relevant than matched off-path and proximity-only controls.

This should now be treated as the primary empirical contribution of the project. Selective-risk gains remain secondary and geometry-dependent.
