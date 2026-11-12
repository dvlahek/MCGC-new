# Experimental evaluation

The repository contains reproducible scripts for synthetic classification, ablation, reliability and selective-prediction comparisons.

## Predictive ablation

The 12-seed comparison evaluates the effect of walkers relative to the graph-field model on circles, moons and spirals.

| Dataset | Mean full minus field-only accuracy | W / T / L |
| --- | ---: | --- |
| Circles | +0.00056 | 1 / 11 / 0 |
| Moons | -0.00056 | 0 / 11 / 1 |
| Spirals | -0.00667 | 3 / 2 / 7 |

The observed differences do not demonstrate a consistent improvement from walkers.

## Reliability

For the evaluated setting, local graph-field margin error ROC-AUC is 0.9929 on circles, 0.9197 on moons and 0.7375 on spirals. The corresponding trail-disagreement ROC-AUC values are 0.9936, 0.8847 and 0.6083. The circles estimate is based on only six prediction errors.

## Interpretation

The implementation computes class-score and winner-versus-alternative decompositions as well as graph paths. A representative path illustrates the fitted graph structure; it is not a causal explanation or an optimized ranking of training-point influence.

## Scope

These results apply to the datasets and script versions in this repository. They should not be substituted for benchmark numbers or claims in the separate CGEC manuscript. Earlier detailed technical reports are available in `research-notes/`.
