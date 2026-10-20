from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "examples" / "benchmark_xai_stability_v15.py"


def load_module():
    spec = importlib.util.spec_from_file_location("benchmark_xai_stability_v15", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_v15_protocol_constants_are_frozen():
    module = load_module()
    assert module.SEEDS == list(range(5))
    assert module.DATASETS == ("digits", "wine", "spirals_hard", "breast_cancer")
    assert module.MULTICLASS_DATASETS == ("digits", "wine")
    assert module.PERTURB_EPS == (0.02, 0.05)
    assert module.PERTURB_REPS == 8
    assert module.MAX_STABILITY_QUERIES == 40
    assert module.TOP_SUPPORT == 5


def test_v15_mcgc_configuration_is_canonical_and_not_tuned():
    module = load_module()
    model = module.mcgc_classifier()
    assert model.n_neighbors == 12
    assert model.n_walkers == 12
    assert model.max_steps == 10
    assert model.diffusion_alpha == 0.85
    assert model.diffusion_steps == 40
    assert model.temperature == 0.35
    assert model.random_state == 17
    assert model.adaptive_walker_gate is True


def test_v15_datasets_are_local_and_multiclass_counts_are_expected():
    module = load_module()
    expected_classes = {
        "digits": 10,
        "wine": 3,
        "spirals_hard": 2,
        "breast_cancer": 2,
    }
    for name in module.DATASETS:
        X, y = module.dataset_for(name, 0)
        assert len(X) == len(y)
        assert len(X) > 100
        assert len(np.unique(y)) == expected_classes[name]


def test_v15_jaccard_and_nontrivial_path_definitions():
    module = load_module()
    assert module.node_set_jaccard([1, 2, 3], [2, 3, 4]) == 0.5
    assert module.node_set_jaccard([], []) == 1.0
    assert module.nontrivial_trail({"length": 1, "path": [4, 7]}) is True
    assert module.nontrivial_trail({"length": 0, "path": [4]}) is False


def test_v15_same_class_control_is_deterministic_and_excludes_self():
    module = load_module()
    controls = module.next_same_class_controls(np.array([0, 1, 0, 1, 1]))
    assert controls[0] == 2
    assert controls[2] == 0
    assert controls[1] == 3
    assert controls[3] == 4
    assert controls[4] == 1
    assert all(idx != ctrl for idx, ctrl in controls.items())


def test_v15_contrastive_reconstruction_is_numerically_exact_on_small_fit():
    module = load_module()
    X, y = module.dataset_for("wine", 0)
    model = module.mcgc_classifier().fit(X[:140], y[:140])
    pred, explanation = module.explain_one(model, X[140])
    assert pred in set(model.classes_.tolist())
    error = module.contrastive_reconstruction_error(explanation)
    assert error <= 1e-10


def test_v15_perturb_rng_is_reproducible():
    module = load_module()
    a = module.perturb_rng("digits", 2, 17, 1, 6).normal(size=5)
    b = module.perturb_rng("digits", 2, 17, 1, 6).normal(size=5)
    c = module.perturb_rng("digits", 2, 17, 1, 7).normal(size=5)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
