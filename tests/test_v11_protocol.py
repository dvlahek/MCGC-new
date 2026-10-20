from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "examples" / "benchmark_selective_stronger_v11.py"


def load_module():
    spec = importlib.util.spec_from_file_location("benchmark_selective_stronger_v11", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_v11_protocol_is_frozen():
    module = load_module()
    assert module.SEEDS == list(range(12))
    assert module.PRIMARY_DATASETS == ("moons_hard", "circles_hard", "spirals_hard")
    assert module.CONTROL_DATASETS == ("overlap_hd", "breast_cancer")
    assert module.COVERAGES == (0.90, 0.80, 0.70, 0.50)
    assert module.BASELINES == (
        "weighted_knn_confidence",
        "logistic_confidence",
        "rbf_svm_confidence",
        "random_forest_confidence",
    )


def test_v11_oracle_aurc_is_best_for_fixed_errors():
    module = load_module()
    error = np.array([0, 1, 0, 1, 0, 0], dtype=int)
    bad_risk = np.array([0.9, 0.1, 0.8, 0.2, 0.7, 0.6])
    observed = module.empirical_aurc(error, bad_risk)
    oracle = module.oracle_aurc(error)
    assert oracle <= observed


def test_v11_svm_risk_orientation_prefers_larger_margin():
    module = load_module()

    class DummyBinarySVM:
        def predict(self, X):
            return np.zeros(len(X), dtype=int)

        def decision_function(self, X):
            return np.array([0.2, -1.5, 3.0])

    _, risk = module.svm_prediction_and_risk(DummyBinarySVM(), np.zeros((3, 2)))
    assert np.allclose(risk, np.array([-0.2, -1.5, -3.0]))
    assert risk[2] < risk[1] < risk[0]


def test_v11_delta_summary_positive_values_are_wins():
    module = load_module()
    summary = module.delta_summary([0.2, 0.1, 0.0, -0.05])
    assert summary["wins"] == 2
    assert summary["ties"] == 1
    assert summary["losses"] == 1
    assert summary["mean"] > 0.0


def test_v11_datasets_available_without_downloads():
    module = load_module()
    for name in module.DATASETS:
        X, y = module.dataset_for(name, 0)
        assert len(X) == len(y)
        assert len(X) > 100
        assert len(np.unique(y)) >= 2
