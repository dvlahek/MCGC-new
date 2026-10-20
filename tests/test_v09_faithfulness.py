from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "examples" / "benchmark_xai_faithfulness_v09.py"


def load_module():
    spec = importlib.util.spec_from_file_location("benchmark_xai_faithfulness_v09", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_v09_protocol_is_frozen_to_path_delete_and_refit():
    m = load_module()
    assert m.SEEDS == list(range(5))
    assert m.MAX_ELIGIBLE_QUERIES == 12
    assert m.DATASETS == (
        "moons_hard",
        "spirals_hard",
        "overlap_hd",
        "breast_cancer",
    )


def test_v09_summary_tracks_positive_advantage():
    m = load_module()
    out = m.summarize([0.2, 0.1, 0.0, -0.05])
    assert out["n"] == 4
    assert out["wins"] == 2
    assert out["ties"] == 1
    assert out["losses"] == 1
    assert out["mean"] > 0.0


def test_v09_matched_controls_exclude_path_nodes_and_match_cardinality():
    m = load_module()
    X, y = m.dataset_for("moons_hard", 0)
    model = m.classifier().fit(X[:420], y[:420])
    x = X[421]
    _, explanations = m.explain_mcgc(model, x[None, :], top_support=5)
    winner = explanations[0]["prediction"]
    path = explanations[0]["classes"][winner]["representative_trail"]["path"]
    path = list(dict.fromkeys(int(i) for i in path))
    controls = m.matched_controls(model, x, path)
    assert len(controls) == len(path)
    assert not (set(controls) & set(path))


def test_v09_nearest_controls_exclude_path_nodes():
    m = load_module()
    X, y = m.dataset_for("moons_hard", 1)
    model = m.classifier().fit(X[:420], y[:420])
    x = X[500]
    _, explanations = m.explain_mcgc(model, x[None, :], top_support=5)
    winner = explanations[0]["prediction"]
    path = explanations[0]["classes"][winner]["representative_trail"]["path"]
    path = list(dict.fromkeys(int(i) for i in path))
    controls = m.nearest_off_path_controls(model, x, path)
    assert len(controls) == len(path)
    assert not (set(controls) & set(path))


def test_v09_refit_removes_requested_rows():
    m = load_module()
    X, y = m.dataset_for("overlap_hd", 0)
    X = np.asarray(X[:120])
    y = np.asarray(y[:120])
    refit = m.refit_without(X, y, [1, 3, 5])
    assert len(refit.X_) == 117
