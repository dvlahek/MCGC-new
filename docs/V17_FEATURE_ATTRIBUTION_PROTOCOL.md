# v1.7 — Native feature attribution protocol

## Purpose

v1.7 adds feature-level explanations without changing MCGC training, prediction, scoring, graph construction, or the previously frozen benchmark claims. The goal is explanatory: quantify how original input features affect a fixed winner–runner decision and how features compose the geometry of a representative graph trail.

## Dataset

Primary visualization dataset: `breast_cancer` from scikit-learn, because its 30 input variables have domain-readable feature names.

Outer split for seed `s`:

```text
test_size = 0.30
random_state = 7000 + s
stratify = y
```

Global attribution uses seeds `0,1,2,3,4` unless explicitly overridden for a quick preview.

## Frozen classifier

The classifier is unchanged:

```text
n_neighbors=12
n_walkers=12
max_steps=10
diffusion_alpha=0.85
diffusion_steps=40
temperature=0.35
random_state=17
adaptive_walker_gate=True
```

## A. Local decision-feature effect

For query `q`, let the original predicted class be `c+` and runner-up be `c-` and define

```math
M(q) = F_{c+}(q) - F_{c-}(q).
```

For feature `k`, construct `q^(-k)` by replacing only feature `k` with the corresponding training-set mean. Because MCGC standardizes using training statistics, this intervention maps that coordinate to approximately zero in standardized space.

The signed local feature effect is

```math
I_k(q) = M(q) - M(q^{(-k)}),
```

where the same original winner/runner pair `(c+, c-)` is used after intervention.

Interpretation:

- `I_k(q) > 0`: the observed feature value supports the original winner relative to the runner-up.
- `I_k(q) < 0`: the observed feature value supports the runner-up relative to the winner.
- `|I_k(q)|`: magnitude of the paired score intervention.

Each original/intervened query is evaluated as a one-row explanation with the frozen `random_state=17`. This resets the stochastic stream identically for the paired comparison and avoids batch-position RNG differences.

This is a model intervention, not a causal claim about the data-generating process.

## B. Representative-path geometric attribution

For each edge `(i,j)` of the deterministic winner representative trail, using the standardized training coordinates `z`, define

```math
rho_{ij,k} = (z_{ik}-z_{jk})^2 / sum_l (z_{il}-z_{jl})^2.
```

Thus `rho_{ij,k} >= 0` and

```math
sum_k rho_{ij,k} = 1.
```

Because the locally scaled heat-kernel exponent uses the same squared Euclidean feature differences multiplied by one edge-specific denominator, `rho` is also the exact proportional feature decomposition of that edge's distance term in the affinity exponent.

This quantity describes path geometry. It is not itself a decision contribution.

## C. Global decision-feature importance

For each outer-test query, compute `I_k(q)` for its own frozen winner/runner pair. The global importance is

```math
GI_k = mean_q |I_k(q)|.
```

For each seed, first average `|I_k|` over that seed's complete test split. The manuscript visualization reports the mean of the five seed-level values with a descriptive 95% CI across seeds. Also record the fraction of query interventions for which `I_k > 0`.

No feature selection or tuning is performed from these results.

## Representative local query

For the local feature-attribution panel use seed 0 only. Among correctly classified test queries with a nontrivial winner representative trail, choose the query whose winner–runner margin is closest to the median margin of that eligible set. This rule is fixed before inspecting feature effects.

## Planned figure

Three panels:

1. **Local decision-feature effects** — top signed `I_k(q)` values for the representative query.
2. **Feature composition of path geometry** — edge × feature heatmap of `rho_{ij,k}` for the winner representative trail.
3. **Global feature importance** — top `GI_k` values across five outer splits with descriptive seed-level 95% CI.

The figure must distinguish decision-effect attribution from path-geometry attribution explicitly.