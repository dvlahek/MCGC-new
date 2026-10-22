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

## Feature interventions

Paired feature-intervention effects are intentionally kept separate from the core explanation record because they require a user-defined intervention reference. Once the effects are computed, they can be visualized with:

```python
from mcgc import plot_feature_interventions

plot_feature_interventions(
    effects,
    feature_names=feature_names,
)
```

Positive values mean that replacing the feature by the chosen reference decreases the original winner-versus-runner margin. Negative values move the fitted margin in the opposite direction. These values are intervention effects, not an additive decomposition of the prediction.

The same effects can also be passed to `plot_mcgc_explanation(..., feature_effects=...)` to create a separate feature-intervention figure.

## Interpretation boundary

The visualization should preserve the interpretation used by the XAI API:

- score components exactly reconstruct the fitted MCGC score,
- supporting examples are training points in the fitted query neighbourhood,
- representative trails are deterministic summaries of the fitted evidence field,
- representative trails are not claimed to be unique causal paths of the stochastic walker ensemble,
- paired feature interventions are fitted-margin effects and are not assumed to be additive.

## Reproducible demo

```bash
python examples/visualize_mcgc_explanation.py
```

The example uses the scikit-learn digits dataset restricted to classes 3 and 8 and writes PNG/PDF outputs under `results/visualization_demo/`.
