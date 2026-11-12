# Multi-agent Chemotactic Graph Classifier

Python implementation of a graph-based classifier with class-conditional evidence fields and optional stochastic graph walks.

The package includes prediction, query-level diagnostics, training-example explanations, and reliability experiments. It is a research implementation rather than a production classifier.

**Scope:** This repository contains the MCGC walker-based prototype. It is not the finalized Contrastive Graph-Evidence Classifier (CGEC) implementation associated with the separate manuscript *Graph-Evidence Classification for Intrinsic Explanation and Training-Label Review*.

## Installation

Requires Python 3.10 or later.

```bash
git clone https://github.com/dvlahek/MCGC-new.git
cd MCGC-new
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m pytest
```

## Basic use

```python
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from mcgc import MultiAgentChemotacticGraphClassifier

X, y = make_moons(n_samples=500, noise=0.18, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.3, random_state=0
)

model = MultiAgentChemotacticGraphClassifier(
    n_neighbors=12, diffusion_alpha=0.85,
    diffusion_steps=40, use_walkers=False, random_state=0
)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

The `use_walkers=False` setting runs the field-based classification component. With `use_walkers=True`, class-conditioned walker diagnostics also contribute to the fitted predictor.

## Repository structure

| Path | Contents |
| --- | --- |
| `src/mcgc/` | Estimator, explanation utilities, reliability features |
| `examples/` | Synthetic benchmarks, ablations, selective-prediction experiments |
| `tests/` | Unit tests for classifier, controls, explanations and reliability |
| `docs/method.md` | Model formulation and interpretation |
| `docs/experiments.md` | Evaluation design and results |
| `docs/reproducibility.md` | Environment, execution and scope |
| `docs/research-notes/` | Earlier technical reports retained for provenance |

## Experiments

```bash
python examples/benchmark_synthetic.py
python examples/benchmark_ablation_v02.py
python examples/benchmark_multiseed_v04.py
python examples/benchmark_selective_v05.py
python examples/benchmark_reliability_v06.py
```

The multi-seed comparisons do not establish a general predictive gain from stochastic walkers over the graph-field baseline. See [experimental results](docs/experiments.md) for the reported setting and limitations.

## Development

Changes should include a test for new prediction or explanation behavior. Re-run `python -m pytest` and the affected benchmark script before comparing numerical results. See [reproducibility](docs/reproducibility.md) for evaluation constraints.

No license file has been added. Reuse conditions should be clarified by the repository owner before publishing software under an open-source license.
