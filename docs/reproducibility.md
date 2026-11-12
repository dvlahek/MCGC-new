# Reproducibility

## Environment

Python >= 3.10 with NumPy and scikit-learn as declared in `pyproject.toml`.

```bash
python -m venv .venv
# activate the environment for your platform
python -m pip install -e ".[dev]"
python -m pytest
```

The package uses minimum dependency versions; there is no frozen environment lockfile. Record `python --version` and `python -m pip freeze` when reproducing any published numerical table.

## Execute experiments

Run the scripts in `examples/` from the repository root. Use their hard-coded seed and data generation settings unless explicitly defining a new comparison.

```bash
python examples/benchmark_ablation_v02.py
python examples/benchmark_multiseed_v04.py
python examples/benchmark_selective_v05.py
python examples/benchmark_reliability_v06.py
```

The scripts are older prototype experiments. They are not a complete reproduction pipeline for the later CGEC paper.

## Release checklist

Before claiming a reproducible publication artifact, provide the exact source commit, Python environment, dataset identities, split definitions, raw experimental outputs and commands for regenerating all published figures and tables. A tagged release should reference these immutable files.
