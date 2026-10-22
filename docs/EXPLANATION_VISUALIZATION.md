# Explanation visualization

MCGC explanations can be visualized directly from a fitted classifier. The visualization layer uses the same explanation records returned by `explain_mcgc()` and does not define a separate surrogate model.

## Install

Visualization requires Matplotlib. The repository development extras already include it:

```bash
pip install -e ".[dev]"
```

## One-query explanation

```python
from mcgc import plot_mcgc_explanation

fig, axes, explanation = plot_mcgc_explanation(
    model,
    query,
    top_support=5,
)
fig.savefig("mcgc_explanation.pdf", bbox_inches="tight")
```

The default figure contains four views:

1. a local graph view for two-dimensional data, or supporting-example view for higher-dimensional data,
2. fitted class scores and winner-versus-runner margin,
3. winner and runner-up representative evidence traces,
4. the exact contrastive score-component decomposition.

For image-like data, pass the training-sample image shape:

```python
fig, axes, explanation = plot_mcgc_explanation(
    model,
    query,
    top_support=5,
    image_shape=(8, 8),
)
```

This displays the top supporting training examples as image tiles.

## Native paired feature interventions

MCGC also exposes the paired intervention used in the feature-level experiments:

```python
from mcgc import paired_feature_intervention_effects

record = paired_feature_intervention_effects(model, query)
effects = record["effects"]
```

The original query first fixes the winning class `c+` and runner-up `c-`. For feature `k`, that feature is replaced by a reference value and the same class contrast is recomputed:

```text
I_k(q) = M(q) - M(q^(-k)).
```

The default reference is the mean of the fitted training data in the original input space. A custom reference vector can be supplied explicitly:

```python
record = paired_feature_intervention_effects(
    model,
    query,
    reference=my_reference,
)
```

Positive values mean that the replacement decreases the original winner-versus-runner margin, so the original feature value supported the winner relative to the runner-up. Negative values move the fixed class contrast in the opposite direction. These values are paired intervention effects, not an additive decomposition of the prediction.

A generic bar view is available with:

```python
from mcgc import plot_feature_interventions

plot_feature_interventions(
    effects,
    feature_names=feature_names,
)
```

For image data, the effects can instead be reshaped back to the image geometry, as shown in the runnable digits example.

## Interpretation boundary

The visualization preserves the interpretation used by the XAI API:

- score components exactly reconstruct the fitted MCGC score,
- supporting examples are training points in the fitted query neighbourhood,
- representative trails are deterministic summaries of the fitted evidence field,
- representative trails are not claimed to be unique causal paths of the stochastic walker ensemble,
- paired feature interventions are fitted-margin effects and are not assumed to be additive.

## Reproducible demo

```bash
python examples/visualize_mcgc_explanation.py
```

The example uses the scikit-learn digits dataset restricted to classes 3 and 8. It writes:

- `digits_explanation.png/.pdf`,
- `digits_feature_interventions.png/.pdf`,
- `digits_feature_interventions_bar.png`,

under `results/visualization_demo/`.
