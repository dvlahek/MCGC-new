# MCGC v1.5 — frozen XAI stability, multiclass explanation, and path-eligibility protocol

Status: **protocol frozen before science results**

Branch: `feature/v1.5-xai-stability`

Parent checkpoint: v1.4 result commit `98b79cb6c0ac6ae4cf1035e802dbf61068cfaf44`.

## Goal

v1.5 is not another predictive-performance tuning round. It tests whether MCGC explanations behave like stable properties of the fitted graph model, whether the contrastive explanation layer remains exact in multiclass settings, and how often a nontrivial representative path exists.

The three questions are intentionally separated:

1. **Stability:** for prediction-preserving small perturbations of the same query, are support/path explanations more similar to the original explanation than explanations from another query of the same predicted class?
2. **Multiclass exactness:** on real multiclass datasets, does the winner-versus-runner-up decomposition reconstruct the class-score margin to floating-point precision?
3. **Eligibility:** what fraction of test queries admit a nontrivial winner representative path, and how does eligibility relate descriptively to correctness and decision margin?

No result-dependent dataset, perturbation, metric, or threshold changes are allowed after the first full science output.

## Model

Use one fixed MCGC configuration throughout v1.5:

```python
MultiAgentChemotacticGraphClassifier(
    n_neighbors=12,
    n_walkers=12,
    max_steps=10,
    diffusion_alpha=0.85,
    diffusion_steps=40,
    temperature=0.35,
    random_state=17,
    adaptive_walker_gate=True,
)
```

This is the pre-v1.4 canonical configuration. v1.5 deliberately does **not** use per-split selective-risk tuned parameters, because the purpose is to validate the explanation mechanism itself rather than optimize E-AURC.

## Datasets

Frozen suite:

- `digits` — real, 10 classes; primary multiclass explanation dataset
- `wine` — real, 3 classes; secondary multiclass explanation dataset
- `spirals_hard` — nonlinear-manifold stress case
- `breast_cancer` — conventional real binary control

Seeds: `0..4`.

Outer split: stratified 70% train / 30% test with `random_state = 7000 + seed`.

Synthetic generation for `spirals_hard` must reuse the existing frozen project generator rather than introduce a new synthetic distribution.

## A. Explanation stability

### Query set

For each dataset×seed model, evaluate the complete outer-test set for the original explanation.

For computationally bounded perturbation analysis, take at most the first 40 test queries in deterministic test-array order that have a valid prediction and explanation. This cap is fixed before results and is not chosen by explanation quality.

### Perturbations

For each selected original query, generate `8` independent perturbations at each of two fixed scales:

- `epsilon = 0.02`
- `epsilon = 0.05`

For feature `j`:

`x'_j = x_j + epsilon * s_j * z_j`

where `s_j` is the standard deviation of feature `j` in the outer-training data, with zero scales replaced by `1`, and `z_j ~ N(0,1)` from a deterministic RNG seeded by dataset, outer seed, query index, epsilon, and perturbation replicate.

Only perturbations whose predicted class equals the original predicted class are included in the stability comparison. The preservation rate itself is reported and must not be hidden.

### Explanation objects compared

For each retained perturbation compare with the original query:

- top-5 `supporting_neighbors` node-index set
- winner representative-trail node-index set
- winner representative-trail start node
- winner representative-trail terminal node
- dominant winner-vs-runner-up score component

A representative trail is **nontrivial** when its reported `length > 0` and its path contains at least two nodes.

### Metrics

Primary structural metrics:

- support-set Jaccard
- winner-path node-set Jaccard, only where both original and perturbed winner paths are nontrivial

Secondary metrics:

- same start-node indicator
- same terminal-node indicator
- same dominant contrastive component indicator
- absolute change in winner-vs-runner-up margin
- prediction-preservation rate

### Same-class cross-query control

For each original query, select one deterministic control query from the same outer-test split with the same predicted class but a different test index. Choose the cyclic next eligible index within that predicted class.

Compute the same support/path Jaccard metrics between the original query and this same-class control query.

The primary stability contrast is:

`within-query perturbation similarity - same-class cross-query similarity`

Positive values mean the explanation retains query-specific structure beyond merely sharing the predicted class.

No arbitrary pass/fail Jaccard threshold is used. Report distributions, means/medians, paired win/tie/loss counts where defined, and dataset×seed summaries.

## B. Multiclass explanation validation

Primary datasets: `digits`, `wine`.

For every outer-test query:

1. run `explain_contrastive_mcgc` with default positive=`prediction`, negative=`runner_up`;
2. record the exact score margin;
3. record the reconstructed contrastive margin;
4. record absolute reconstruction error;
5. record winner/runner-up labels and representative paths.

Primary fidelity requirement is numerical identity up to existing implementation tolerance. Report maximum and median absolute reconstruction error over all multiclass queries. Any failure of the implementation's existing exactness assertion is a protocol failure, not a result to omit.

Also report the most frequent winner/runner-up class pairs. These frequencies are descriptive only and must not be used to select a favourable quantitative subset.

Any future manuscript example such as “8 rather than 3” must be identified as an illustrative example and must not replace the full quantitative multiclass results.

## C. Path eligibility

For every test query in every dataset×seed run, define winner-path eligibility as:

- winner representative trail `length > 0`, and
- winner path contains at least two nodes.

Report by dataset:

- total queries
- eligible count and rate
- eligibility rate among correct predictions
- eligibility rate among errors
- accuracy among eligible queries
- accuracy among non-eligible queries
- median decision margin among eligible queries
- median decision margin among non-eligible queries

These are descriptive diagnostics. v1.5 does not define a post-hoc abstention rule from eligibility.

## Aggregation and inference

The basic repeated unit is dataset×seed.

For explanation stability, report both query-level descriptive totals and dataset×seed means. Reviewer-facing inference should prefer dataset×seed aggregation to avoid treating many perturbations of the same fitted model as fully independent observations.

For paired within-query versus cross-query similarity, use deterministic paired differences and two-sided sign tests on dataset×seed mean differences where all required metrics are defined.

Do not reinterpret undefined path-Jaccard cases as zero similarity; report the number of evaluable path pairs explicitly.

## Interpretation policy

A strong v1.5 outcome would support the statement that MCGC explanations are locally stable and query-specific under prediction-preserving perturbations, while retaining exact multiclass score faithfulness.

A mixed outcome is acceptable and must delimit the claim. For example, stable support sets with unstable discrete paths would support local-support stability but not path-topology stability.

A negative stability result does not erase v0.9 intervention faithfulness. Faithfulness and stability are separate explanation properties.

No new performance tuning or rescue search is part of v1.5.
