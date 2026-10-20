# MCGC v0.5 selective-walker checkpoint

v0.5 tested a direct response to the negative v0.4 multi-seed confirmation: instead of allowing trajectory terms to perturb every query equally, walker evidence was gated by ambiguity in the local contrastive field.

## Selective rule

For local class-evidence margin `m(q)` and a fitted reference scale `m_ref`,

```text
g(q) = exp(-(m(q) / m_ref) ** p)
```

with default `p=1`. The reference scale is the median positive top-two contrastive-field margin across training nodes. No dataset-specific threshold was tuned.

Walker-derived score terms are multiplied by `g(q)`. Clear local decisions therefore receive less trajectory correction, while ambiguous queries receive more.

## Frozen confirmation protocol

The v0.5 benchmark reuses the v0.4 12-seed protocol and compares:

- field-only MCGC,
- unconditional walkers,
- selective walkers,
- distance-weighted k-NN.

No per-seed or per-dataset retuning was performed.

## Result

The gate was numerically active but did not change aggregate predictions relative to unconditional walkers on any of the 36 dataset-seed runs.

| dataset | mean gate | selective - unconditional | selective - field |
|---|---:|---:|---:|
| circles | 0.3836 | 0.0000 (12 ties) | +0.00056 (1 win, 11 ties) |
| moons | 0.4092 | 0.0000 (12 ties) | -0.00056 (1 loss, 11 ties) |
| spirals | 0.5348 | 0.0000 (12 ties) | -0.00667 (3 wins, 2 ties, 7 losses) |

On spirals, selective MCGC remained below distance-weighted k-NN by mean accuracy `-0.0950` and lost on all 12 seeds.

## Interpretation

The uncertainty gate is well-defined, self-scaled, reproducible, and explainable, but this particular score-gating mechanism does not solve the v0.4 performance failure. Multiplying the common walker correction by a query-level gate was insufficient to change class rankings in the frozen protocol.

This checkpoint argues against further post-hoc tuning of walker score weights merely to obtain an accuracy gain. The more informative next question is whether trajectory diagnostics contain **reliability information** even when they do not improve the class decision itself.

The next planned test therefore treats walker signals as uncertainty / abstention evidence rather than as a mandatory class-score correction. Candidate signals include:

- prediction margin,
- walker agreement,
- lost-trail fraction,
- trail coherence,
- disagreement between field-only and full prediction,
- contrast between winner and runner-up trail diagnostics.

The target is not raw accuracy tuning but error detection and selective classification: can MCGC identify which of its own predictions should not be trusted?

No accuracy-improvement claim is supported by v0.5.
