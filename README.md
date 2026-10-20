# MCGC-new

Minimal research prototype for a **Multi-Agent Chemotactic Graph Classifier (MCGC)**.

The working idea is to classify a query by asking, for each candidate class, whether multiple class-conditioned walkers can find a geometrically plausible and coherent path through a class-specific contrastive evidence field on a k-nearest-neighbour graph.

> Status: experimental v0.1 research prototype. No novelty or performance claim is implied by the code alone.

## What v0.1 contains

- mutualised k-NN graph with locally scaled heat-kernel weights,
- one diffused evidence field per class,
- contrastive class fields against the strongest rival class,
- multiple stochastic class-conditioned walkers,
- gradient, geometry, revisit, coherence, agreement, and trail-loss terms,
- deterministic inference for a fixed random seed,
- synthetic smoke benchmarks against ordinary k-NN,
- unit tests and GitHub Actions on Python 3.10, 3.11, and 3.12.

The implementation deliberately keeps the diffusion stage and the chemotactic-walker stage separate so that later ablations can test whether the walkers add information beyond the graph field itself.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -e ".[dev]"
```

## Test

```bash
pytest
```

## Reproduce the current smoke benchmark

```bash
python examples/benchmark_synthetic.py
```

With the frozen seeds currently in the example, the local development run produced:

| dataset | k-NN accuracy | MCGC accuracy |
|---|---:|---:|
| circles | 0.9933 | 0.9933 |
| moons | 0.9667 | 0.9600 |

These are **smoke-test results, not a performance claim**. The immediate scientific question is whether trajectory-derived terms improve over the same local/diffused field with the walker contribution removed.

## Important v0.1 design lesson

The first naive implementation rewarded a large uphill trajectory so strongly that a wrong-class walker could outscore a query that was already located inside the correct class basin. The current score therefore treats the query's local contrastive field as the anchor and uses walker evidence only as a refinement. That failure mode is retained in the project history because it is useful for defining the later ablations correctly.

See [`docs/FORMULATION.md`](docs/FORMULATION.md) for the current mathematical/algorithmic decomposition and proposed baselines.
