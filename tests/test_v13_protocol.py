from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "examples" / "benchmark_real_data_v13.py"


def load_module():
    spec = importlib.util.spec_from_file_location("benchmark_real_data_v13", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_v13_protocol_is_frozen_and_real_data_only():
    module = load_module()
    assert module.SEEDS == list(range(12))
    assert module.PRIMARY_DATASETS == ("digits", "wine")
    assert module.CONTROL_DATASETS == ("breast_cancer",)
    assert module.INNER_FOLDS == 3
    assert module.COVERAGES == (0.90, 0.80, 0.70, 0.50)


def test_v13_reuses_v12_baseline_search_spaces_exactly():
    module = load_module()
    assert module.KNN_GRID == module.v12.KNN_GRID
    assert module.LOGISTIC_GRID == module.v12.LOGISTIC_GRID
    assert module.SVM_GRID == module.v12.SVM_GRID
    assert module.RF_GRID == module.v12.RF_GRID
    assert module.BASELINES == module.v12.BASELINES


def test_v13_datasets_are_available_without_downloads():
    module = load_module()
    expected_classes = {"digits": 10, "wine": 3, "breast_cancer": 2}
    for name in module.DATASETS:
        X, y = module.dataset_for(name, 0)
        assert len(X) == len(y)
        assert len(X) > 100
        assert len(np.unique(y)) == expected_classes[name]


def test_v13_mcgc_is_inherited_unchanged_from_v12():
    module = load_module()
    model = module.v12.mcgc_classifier()
    assert model.n_neighbors == 12
    assert model.n_walkers == 12
    assert model.max_steps == 10
    assert model.diffusion_steps == 40
    assert model.random_state == 17
    assert model.adaptive_walker_gate is True


def test_v13_positive_advantage_direction_matches_v12():
    module = load_module()
    summary = module.v12.delta_summary([0.2, 0.1, 0.0, -0.05])
    assert summary["wins"] == 2
    assert summary["ties"] == 1
    assert summary["losses"] == 1
    assert summary["mean"] > 0.0
