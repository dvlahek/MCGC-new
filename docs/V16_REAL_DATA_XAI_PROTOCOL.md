# v1.6 real-data XAI figure protocol

## Purpose

v1.6 is a visualization/realization stage, not a new predictive-performance search. It produces a real-data contrastive explanation from the frozen MCGC mechanism on `sklearn.datasets.load_digits`.

The scientific goal is to show, for one objectively selected test query, how the same fitted model exposes:

1. the observed query image and winner/runner-up decision;
2. the local query neighbourhood used by the classifier;
3. deterministic representative trails for the winner and runner-up evidence fields;
4. evidence values along those trails; and
5. the exact winner-minus-runner score decomposition.

No classifier parameter is tuned in v1.6.

## Frozen model and split

Use the same frozen MCGC configuration as v1.5:

- `n_neighbors=12`
- `n_walkers=12`
- `max_steps=10`
- `diffusion_alpha=0.85`
- `diffusion_steps=40`
- `temperature=0.35`
- `random_state=17`
- `adaptive_walker_gate=True`

Digits splits use 70% train / 30% test with stratification and `random_state=7000 + seed`, for seeds 0..4, matching v1.5.

## Query selection rule

Selection is deterministic and fixed before viewing the resulting figure.

Candidate queries must be:

- correctly classified;
- nontrivial-path eligible for the winner (`winner trail length > 0` and at least two trail nodes).

Preferred winner/runner pairs are searched in this fixed order:

1. `3 -> 8`
2. `8 -> 3`
3. `9 -> 4`
4. `4 -> 9`

The first preferred pair with at least three eligible candidates pooled across seeds 0..4 is used. If none reaches three candidates, use the most frequent eligible winner/runner pair, with lexical ordering as a deterministic tie-break.

Within the chosen pair, select the query whose winner-minus-runner margin is closest to the median margin of all eligible candidates for that pair. Ties are resolved by smaller seed and then smaller test index.

This rule prevents manual selection of the most visually attractive or most extreme case.

## Figure semantics

### Query panel

Show the actual 8x8 query image, true label, prediction, runner-up, score margin, and path eligibility.

### Local query-neighbourhood panel

Show the five nearest/affinity-ranked training neighbours returned by `explain_mcgc(..., top_support=5)`. These are **local supporting neighbours**, not contribution-ranked class-specific exemplars. Each image must display its training label, affinity, and winner/runner contrastive evidence values.

### Representative trails

Show the actual training images indexed by the deterministic representative trail for the winner and runner-up fields. A representative trail is an illustration/summary of the fitted class-conditioned evidence landscape; it must not be described as the sole stochastic causal trajectory used to make the prediction.

### Evidence trace

Plot the class-specific contrastive evidence values recorded along each representative trail. This makes the trail's evidence ascent visible without claiming monotonicity when the recorded values do not support it.

### Exact contrastive decomposition

Use `explain_contrastive_mcgc` for the selected winner/runner pair. Plot the exact component deltas:

- local component;
- trail component;
- agreement component;
- lost component.

Their sum must reconstruct the actual winner-minus-runner class-score margin to floating-point tolerance.

## Outputs

The script must write:

- a publication-oriented composite PNG;
- a vector PDF;
- a JSON provenance record containing the selected seed/test index, selection candidate counts, labels, margins, local neighbours, trail node indices/evidence traces, and exact contrastive component deltas.

The JSON, not the appearance of the figure, is the canonical provenance for the illustrated example.
