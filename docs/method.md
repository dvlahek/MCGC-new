# Model and API

The classifier standardizes the training features, builds a locally scaled k-nearest-neighbor graph, symmetrizes edge weights and row-normalizes the resulting adjacency matrix.

For class indicator matrix `B`, propagation iterates

```text
S[t+1] = (1 - alpha) B + alpha P S[t]
```

For each class, the fitted contrast subtracts its strongest rival field. Queries are scored from their nearby training examples. The implementation optionally adds class-conditioned stochastic walks with graph connectivity, evidence change and revisit terms.

The stochastic walker component is a separate research option. Set `use_walkers=False` to run the field-only predictor.

## Public entry points

- `MultiAgentChemotacticGraphClassifier` in `mcgc.classifier` provides the scikit-learn-style `fit` and `predict` API.
- `mcgc.xai` exposes class-score and contrastive evidence diagnostics.
- `mcgc.reliability` extracts reliability signals from model diagnostic records.

See the source in `src/mcgc/` for argument defaults and exact formulas. This MCGC implementation should not be cited as identical to the distinct CGEC manuscript formulation.
