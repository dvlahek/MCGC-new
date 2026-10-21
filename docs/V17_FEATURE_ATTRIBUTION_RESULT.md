# V1.7 Feature Attribution Result

## Scope

This result freezes the native MCGC feature-attribution analysis on the breast-cancer dataset. The purpose is explanatory, not predictive: no classifier hyperparameter, threshold, benchmark split, or performance objective was tuned in this phase.

The analysis uses two distinct quantities and keeps their interpretation separate.

1. **Paired decision-feature effect** for query `q` and feature `k`:

   `I_k(q) = M(q) - M(q^(-k))`,

   where `M` is the original winner-versus-runner score margin and `q^(-k)` replaces only feature `k` by the corresponding outer-training reference value. The original winner/runner pair is held fixed.

2. **Representative-path geometry share** for graph edge `(i,j)`:

   `rho_ijk = (z_ik-z_jk)^2 / sum_l (z_il-z_jl)^2`.

   These shares sum to one over features for each non-degenerate edge and describe the exact feature composition of standardized squared edge distance. They are not prediction-margin contributions.

Global decision-feature importance is summarized as the outer-test mean absolute intervention effect, then averaged over frozen outer splits:

`GI_k = mean_q |I_k(q)|`.

## Main feature-attribution result

The five-split global ranking is led by `worst texture`, followed by `worst smoothness`, `worst concave points`, `mean texture`, `worst symmetry`, `worst fractal dimension`, `worst radius`, `worst compactness`, `worst concavity`, and `worst perimeter`.

The local representative query demonstrates that the intervention effect is signed rather than merely ranked by magnitude: some features increase the original winner-runner margin when retained, while others oppose the winner. This supports local contrastive interpretation without implying an additive feature decomposition of the full MCGC score.

The representative-path geometry panel is also interpretation-specific. It explains which original standardized features geometrically compose a graph edge on the deterministic representative path. For the displayed one-edge example, the largest edge-distance shares are carried by `mean texture`, `worst texture`, and `smoothness error`.

## Rank stability across frozen outer splits

Pairwise rank stability was evaluated across all 10 pairs of the five frozen outer splits.

- Mean Spearman rank correlation: **0.731613**
- Median Spearman rank correlation: **0.745495**
- Minimum Spearman rank correlation: **0.528810**
- Maximum Spearman rank correlation: **0.857620**
- Mean top-5 Jaccard overlap: **0.619048**
- Median top-5 Jaccard overlap: **0.666667**
- Mean top-10 Jaccard overlap: **0.492766**
- Median top-10 Jaccard overlap: **0.538462**

Interpretation: the global ranking is moderately to strongly associated across outer splits, but it is not invariant to resampling. Therefore the manuscript should not present exact feature positions as universally fixed. The appropriate claim is that a recurring group of texture, smoothness, concavity, radius, compactness, symmetry, and perimeter-related variables receives large MCGC decision-margin intervention effects across the frozen splits.

## Reference-value sensitivity

The original feature-removal reference is the outer-training mean. A frozen sensitivity check recomputed all feature interventions with the outer-training median while preserving the same dataset, outer splits, fitted model specification, and original winner/runner pair.

Aggregate mean-reference versus median-reference ranking agreement:

- Spearman correlation: **0.955951**
- Top-5 Jaccard overlap: **0.666667**
- Top-10 Jaccard overlap: **0.818182**

Per-split mean-versus-median Spearman correlations ranged from **0.828254** to **0.949722**.

Interpretation: the ranking is highly robust to replacing the training-mean reference by the training median. This substantially reduces the concern that the observed global ranking is an artifact of the specific single-feature baseline value.

## Claim boundary

Supported claims:

- MCGC provides a native, local, signed feature-intervention effect on the actual fitted winner-versus-runner margin.
- MCGC also permits an exact decomposition of standardized representative-path edge geometry into original-feature shares.
- The global intervention ranking is moderately stable across frozen outer splits and highly stable to mean-versus-median reference choice.

Claims that are **not** supported:

- `I_k(q)` is not an additive decomposition of the final class margin; in general `M(q) != sum_k I_k(q)`.
- A large path-geometry share does not mean that the feature contributes the same proportion of the prediction margin.
- The breast-cancer feature ranking is not a biomedical causal ranking and should not be interpreted as one.
- Exact rank positions are not invariant across outer splits.

## Figure role

The three-panel feature-attribution figure is suitable for the main paper if the paper remains XAI-first:

- panel A: local signed decision-feature effects,
- panel B: feature composition of representative-path geometry,
- panel C: five-split global mean absolute decision-feature effect with descriptive uncertainty.

A supplementary validation table or figure should report the outer-split rank stability and the mean-versus-median reference sensitivity values above.

## Frozen conclusion

V1.7 closes the remaining feature-level XAI gap. Together with exact score decomposition, training-example support, representative paths, delete-and-refit faithfulness, and perturbation stability, MCGC now exposes explanation structure at mechanism, example, path, and original-feature levels. The feature-ranking result should be presented as stable in aggregate but not perfectly invariant across resampling.
