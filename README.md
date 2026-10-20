# MCGC-new

Experimental research prototype for a **Multi-Agent Chemotactic Graph Classifier (MCGC)**.

MCGC represents labelled data as a k-nearest-neighbour graph with class-specific contrastive evidence fields. Class-conditioned stochastic walkers traverse those fields, producing trajectory diagnostics that can be used both for prediction research and for intrinsically traceable explanations.

> Status: experimental v0.6. The repository intentionally retains negative confirmation results. No superiority, novelty-first, or fully-interpretable claim is implied by the code alone.

## Current research position

The project now separates three questions that must not be conflated:

1. **classification** — does walker trajectory evidence improve the graph-field classifier?
2. **explanation** — can the actual class score and class contrast be traced through exact components and graph paths?
3. **reliability** — can internal graph/trajectory diagnostics identify predictions that should not be trusted?

The current evidence is mixed by design:

- the original single-split walker gain on spirals did **not** survive a frozen 12-seed confirmation,
- an uncertainty-gated walker correction did not improve those confirmed predictions,
- exact score and `why A rather than B` decompositions are implemented and tested,
- intrinsic reliability is promising, with local graph-field margin currently stronger than walker-specific diagnostics on the meaningful confirmation datasets.

## Implemented components

- mutualised k-NN graph with locally scaled heat-kernel weights,
- one diffused evidence field per class,
- contrastive class evidence against the strongest rival,
- stochastic class-conditioned chemotactic walkers,
- gradient, geometry, revisit, coherence, agreement and trail-loss diagnostics,
- optional self-scaled uncertainty gate for walker score terms,
- exact XAI score decomposition,
- exact contrastive `why A rather than B` decomposition,
- deterministic representative evidence trails,
- supporting-neighbour provenance,
- intrinsic reliability feature extraction,
- frozen multi-seed, ablation, selective-walker and risk--coverage benchmarks,
- GitHub Actions across Python 3.10, 3.11 and 3.12.

## Install and test

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -e ".[dev]"
pytest
```

## Reproduce the benchmark chain

```bash
python examples/benchmark_synthetic.py
python examples/benchmark_ablation_v02.py
python examples/benchmark_multiseed_v04.py
python examples/benchmark_selective_v05.py
python examples/benchmark_reliability_v06.py
```

## Confirmed classification result

The 12-seed v0.4 confirmation does **not** support a general walker-accuracy gain.

| dataset | mean full MCGC - field only | wins | ties | losses |
|---|---:|---:|---:|---:|
| circles | +0.00056 | 1 | 11 | 0 |
| moons | -0.00056 | 0 | 11 | 1 |
| spirals | -0.00667 | 3 | 2 | 7 |

On spirals, full MCGC also trails distance-weighted k-NN by mean accuracy `-0.0950` across the frozen 12 seeds. This negative result is retained as a constraint on future development rather than tuned away.

## XAI status

`explain_mcgc()` decomposes every class score into the same numerical components used by prediction and verifies exact reconstruction. `explain_contrastive_mcgc()` decomposes the exact winner-versus-alternative score difference. Deterministic representative trails are explicitly treated as illustrations of the fitted evidence landscape, not as fabricated causal paths for the stochastic aggregate.

## Reliability status

v0.6 evaluates intrinsic error detection without fitting a secondary risk model or tuning test-set thresholds.

| dataset | pooled errors | local-margin error AUC | trail-disagreement error AUC |
|---|---:|---:|---:|
| circles | 6 | 0.9929 | 0.9936 |
| moons | 42 | **0.9197** | 0.8847 |
| spirals | 505 | **0.7375** | 0.6083 |

The circles comparison is underpowered because only six errors occurred. On moons and spirals, local graph-field margin is currently the stronger reliability signal. Trail disagreement remains the most promising walker-derived diagnostic but has not yet shown complementary value beyond that baseline.

On spirals, selecting the lowest-risk 50% of predictions by local-margin risk raises retained-set accuracy from `0.7194` to `0.8711` without changing the classifier itself.

## Scientific next step

The next frozen test uses a strict train / calibration / test separation to ask whether walker diagnostics add **incremental reliability information after controlling for local margin**. This is a stronger test than selecting whichever raw diagnostic looks best on the test set.

See:

- [`docs/FORMULATION.md`](docs/FORMULATION.md)
- [`docs/ABLATION_V02.md`](docs/ABLATION_V02.md)
- [`docs/CONFIRMATION_V04.md`](docs/CONFIRMATION_V04.md)
- [`docs/SELECTIVE_V05.md`](docs/SELECTIVE_V05.md)
- [`docs/RELIABILITY_V06_PROTOCOL.md`](docs/RELIABILITY_V06_PROTOCOL.md)
- [`docs/RELIABILITY_V06_RESULTS.md`](docs/RELIABILITY_V06_RESULTS.md)
