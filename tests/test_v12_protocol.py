from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "examples" / "benchmark_tuned_selective_v12.py"


def load_module():
    spec = importlib.util.spec_from_file_location("benchmark_tuned_selective_v12", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_v12_outer_protocol_is_frozen():
    module = load_module()
    assert module.SEEDS == list(range(12))
    assert module.PRIMARY_DATASETS == ("moons_hard", "circles_hard", "spirals_hard")
    assert module.CONTROL_DATASETS == ("overlap_hd", "breast_cancer")
    assert module.COVERAGES == (0.90, 0.80, 0.70, 0.50)
    assert module.INNER_FOLDS == 3


def test_v12_grids_are_frozen():
    module = load_module()
    assert len(module.KNN_GRID) == 10
    assert len(module.SVM_GRID) == 9
    assert len(module.RF_GRID) == 18
    assert len(module.LOGISTIC_GRID) == 3
    assert {row["n_neighbors"] for row in module.KNN_GRID} == {5, 9, 12, 17, 25}
    assert {row["weights"] for row in module.KNN_GRID} == {"uniform", "distance"}
    assert {row["C"] for row in module.SVM_GRID} == {0.1, 1.0, 10.0}
    assert {row["gamma"] for row in module.SVM_GRID} == {"scale", 0.1, 1.0}


def test_v12_mcgc_remains_frozen():
    module = load_module()
    model = module.mcgc_classifier()
    assert model.n_neighbors == 12
    assert model.n_walkers == 12
    assert model.max_steps == 10
    assert model.diffusion_steps == 40
    assert model.random_state == 17
    assert model.adaptive_walker_gate is True


def test_v12_eaurc_zero_for_oracle_ranking():
    module = load_module()
    y = np.array([0, 0, 0, 0, 0, 0])
    pred = np.array([0, 0, 0, 0, 1, 1])
    error = (pred != y).astype(int)
    metrics = module.model_metrics(y, pred, error.astype(float))
    assert np.isclose(metrics["eaurc"], 0.0)


def test_v12_selection_prefers_lower_eaurc():
    module = load_module()
    X, y = module.dataset_for("moons_hard", 0)
    model, selected, candidates = module.tune_baseline(
        "logistic",
        ({"C": 0.1}, {"C": 1.0}),
        X[:300],
        y[:300],
        seed=0,
    )
    assert model is not None
    assert len(candidates) == 2
    best = min(
        candidates,
        key=lambda row: (
            round(float(row["oof_eaurc"]), 12),
            -round(float(row["oof_accuracy"]), 12),
            row["candidate_key"],
        ),
    )
    assert selected["candidate_key"] == best["candidate_key"]


def test_v12_datasets_available_without_downloads():
    module = load_module()
    for name in module.DATASETS:
        X, y = module.dataset_for(name, 0)
        assert len(X) == len(y)
        assert len(X) > 100
        assert len(np.unique(y)) >= 2
