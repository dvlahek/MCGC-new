from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "examples" / "benchmark_tuned_mcgc_v14.py"


def load_module():
    spec = importlib.util.spec_from_file_location("benchmark_tuned_mcgc_v14", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_v14_protocol_reuses_v13_real_data_family():
    module = load_module()
    assert module.SEEDS == list(range(12))
    assert module.PRIMARY_DATASETS == ("digits", "wine")
    assert module.CONTROL_DATASETS == ("breast_cancer",)
    assert module.INNER_FOLDS == 3


def test_v14_mcgc_grid_has_frozen_budget_and_contains_old_configuration():
    module = load_module()
    assert len(module.MCGC_GRID) == 18
    assert module.FROZEN_MCGC_PARAMS in module.MCGC_GRID
    assert {row["n_neighbors"] for row in module.MCGC_GRID} == {7, 12, 17}
    assert {row["diffusion_alpha"] for row in module.MCGC_GRID} == {0.75, 0.85, 0.93}
    assert {row["diffusion_steps"] for row in module.MCGC_GRID} == {20, 40}


def test_v14_mcgc_non_tuned_parameters_are_fixed():
    module = load_module()
    model = module.mcgc_classifier(module.FROZEN_MCGC_PARAMS)
    assert model.n_walkers == 12
    assert model.max_steps == 10
    assert model.temperature == 0.35
    assert model.random_state == 17
    assert model.adaptive_walker_gate is True


def test_v14_reuses_v12_baseline_grids_exactly():
    module = load_module()
    assert module.KNN_GRID == module.v12.KNN_GRID
    assert module.LOGISTIC_GRID == module.v12.LOGISTIC_GRID
    assert module.SVM_GRID == module.v12.SVM_GRID
    assert module.RF_GRID == module.v12.RF_GRID
    assert module.BASELINES == module.v12.BASELINES


def test_v14_mcgc_tuning_is_deterministic_on_small_binary_problem():
    module = load_module()
    rng = np.random.default_rng(123)
    X0 = rng.normal(loc=-1.0, scale=0.35, size=(24, 2))
    X1 = rng.normal(loc=1.0, scale=0.35, size=(24, 2))
    X = np.vstack([X0, X1])
    y = np.concatenate([np.zeros(24, dtype=int), np.ones(24, dtype=int)])

    _, selected_a, records_a = module.tune_mcgc(X, y, seed=0)
    _, selected_b, records_b = module.tune_mcgc(X, y, seed=0)

    assert len(records_a) == 18
    assert len(records_b) == 18
    assert selected_a["candidate_key"] == selected_b["candidate_key"]
    assert selected_a["oof_eaurc"] == selected_b["oof_eaurc"]


def test_v14_positive_advantage_means_lower_tuned_mcgc_risk():
    module = load_module()
    # baseline/frozen minus tuned MCGC; positive therefore favors tuned MCGC
    summary = module.v12.delta_summary([0.2, 0.1, 0.0, -0.05])
    assert summary["wins"] == 2
    assert summary["ties"] == 1
    assert summary["losses"] == 1
    assert summary["mean"] > 0.0
