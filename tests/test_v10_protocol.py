from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "examples" / "benchmark_selective_manifold_v10.py"


def load_module():
    spec = importlib.util.spec_from_file_location("benchmark_selective_manifold_v10", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_v10_protocol_is_frozen():
    module = load_module()
    assert module.SEEDS == list(range(12))
    assert module.PRIMARY_DATASETS == ("moons_hard", "circles_hard", "spirals_hard")
    assert module.CONTROL_DATASETS == ("overlap_hd", "breast_cancer")
    assert module.COVERAGES == (0.90, 0.80, 0.70, 0.50)


def test_oracle_aurc_is_best_for_fixed_errors():
    module = load_module()
    error = np.array([0, 1, 0, 1, 0, 0], dtype=int)
    bad_risk = np.array([0.9, 0.1, 0.8, 0.2, 0.7, 0.6])
    observed = module.empirical_aurc(error, bad_risk)
    oracle = module.oracle_aurc(error)
    assert oracle <= observed
    assert np.isclose(module.empirical_aurc(error, error.astype(float)), oracle)


def test_eaurc_zero_for_oracle_ranking():
    module = load_module()
    y = np.array([0, 0, 0, 0, 0, 0])
    pred = np.array([0, 0, 0, 0, 1, 1])
    error = (pred != y).astype(int)
    metrics = module.model_metrics(y, pred, error.astype(float))
    assert np.isclose(metrics["eaurc"], 0.0)


def test_delta_summary_positive_values_are_wins():
    module = load_module()
    summary = module.delta_summary([0.2, 0.1, 0.0, -0.05])
    assert summary["wins"] == 2
    assert summary["ties"] == 1
    assert summary["losses"] == 1
    assert summary["mean"] > 0.0


def test_v10_datasets_available_without_downloads():
    module = load_module()
    for name in module.DATASETS:
        X, y = module.dataset_for(name, 0)
        assert len(X) == len(y)
        assert len(X) > 100
        assert len(np.unique(y)) >= 2
