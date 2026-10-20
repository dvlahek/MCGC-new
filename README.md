# MCGC-new

Minimal research prototype for a **Multi-Agent Chemotactic Graph Classifier (MCGC)**.

The working idea is to classify a query by asking, for each candidate class, whether multiple class-conditioned walkers can find a geometrically plausible and coherent path through a class-specific contrastive evidence field on a k-nearest-neighbour graph.

> Status: experimental v0.2 research prototype. No novelty or performance claim is implied by the code alone.

## Core idea

MCGC deliberately separates two stages:

1. a graph/diffusion stage that constructs one contrastive evidence field per class,
2. a chemotactic stage in which multiple class-conditioned walkers follow local evidence and gradient structure.

The query's local contrastive field is the anchor signal. Walker trajectories refine that signal through trail score, inter-walker agreement and trail-loss evidence.

## What v0.2 contains

- mutualised k-NN graph with locally scaled heat-kernel weights,
- one diffused evidence field per class,
- contrastive class fields against the strongest rival class,
- multiple stochastic class-conditioned walkers,
- gradient, geometry, revisit, coherence, agreement and trail-loss terms,
- explicit ablation controls for the walker stage,
- deterministic inference for a fixed random seed,
- synthetic benchmarks against ordinary and weighted k-NN plus Label Spreading,
- unit tests and GitHub Actions on Python 3.10, 3.11 and 3.12.

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

## Reproduce the smoke benchmark

```bash
python examples/benchmark_synthetic.py
```

## Reproduce the v0.2 ablation benchmark

```bash
python examples/benchmark_ablation_v02.py
```

The frozen single-split v0.2 diagnostic currently gives the following accuracies:

| dataset | field only | one walker | multi no gradient | full MCGC | distance k-NN | Label Spreading |
|---|---:|---:|---:|---:|---:|---:|
| moons | 0.9833 | **0.9889** | 0.9833 | 0.9833 | 0.9778 | 0.9778 |
| circles | 0.9889 | 0.9889 | 0.9889 | 0.9889 | 0.9889 | 0.9889 |
| spirals | 0.7556 | 0.7611 | 0.7667 | **0.7722** | 0.8111 | 0.8111 |

These are **diagnostic results, not a performance claim**. On moons and circles the full walker population has not yet demonstrated an improvement over the same graph field without walkers. On spirals, the current walker stage improves over `field_only`, and the explicit gradient term adds a small further gain, but the prototype still trails distance-weighted k-NN and Label Spreading.

The current agreement and trail-loss terms do not change accuracy in this frozen split. That negative result is retained rather than hidden because the immediate scientific task is to determine which parts of the biological analogy carry measurable predictive information.

## Important design lesson

The first naive implementation rewarded a large uphill trajectory so strongly that a wrong-class walker could outscore a query already located inside the correct class basin. The current score therefore treats local contrastive evidence as the anchor and uses walker evidence only as a refinement. That failure mode is retained in the project history because it defines an important constraint on future scoring designs.

See [`docs/FORMULATION.md`](docs/FORMULATION.md) for the algorithmic decomposition and [`docs/ABLATION_V02.md`](docs/ABLATION_V02.md) for the frozen ablation protocol.
