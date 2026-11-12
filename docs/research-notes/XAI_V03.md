# MCGC v0.3: XAI-first design

MCGC v0.3 treats explanation as a first-class research output without changing the v0.2 prediction mechanism.

## Design principle

The explanation layer separates two objects that are easy to confuse:

1. **faithful score decomposition**: the exact additive terms that reconstruct the class score used for prediction;
2. **representative trail**: a deterministic greedy traversal through the fitted class-specific contrastive field, used to inspect and visualize the evidence landscape.

The representative trail is intentionally not presented as the sole causal source of a stochastic multi-walker score.

## Public API

```python
from mcgc import explain_mcgc

predictions, explanations = explain_mcgc(model, X)
```

For each query the explanation contains:

- predicted class and runner-up,
- score margin,
- score for every candidate class,
- exact local-evidence contribution,
- exact mean-trail contribution,
- exact inter-walker agreement contribution,
- exact trail-loss contribution,
- mean stochastic trail coherence,
- a deterministic representative trail for every class,
- the nearest supporting training nodes and their class-specific contrastive evidence.

## Fidelity invariant

For every class `c`, v0.3 checks numerically that

```text
score(c)
= local_component
+ trail_component
+ agreement_component
+ lost_component
```

If this reconstruction fails, the explanation call raises an error instead of returning a misleading explanation.

## Why this is useful

The intended XAI claim is not "the model is fully interpretable". The defensible working language is:

- intrinsically traceable,
- path-explainable,
- interpretable-by-design,
- intrinsically explainable graph classification.

These labels still require empirical validation. In particular, later experiments should measure stability, faithfulness under ablation, usefulness to humans, and whether trail-based explanations reveal information beyond nearest-neighbour prototypes.

## Next XAI experiments

1. explanation stability over random seeds;
2. fidelity under component deletion;
3. contrastive explanations: why class A rather than class B;
4. counterfactual graph perturbations;
5. path visualization on 2-D data;
6. prototype/support-example quality;
7. tests on real tabular datasets where input features remain human-readable.
