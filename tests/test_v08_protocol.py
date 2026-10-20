from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "examples" / "benchmark_frozen_combined_reliability_v08.py"


def load_module():
    spec = importlib.util.spec_from_file_location("v08_benchmark", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_v08_feature_sets_are_frozen_from_v07():
    module = load_module()
    assert module.FEATURE_SETS == {
        "local_only": ["local_margin_risk"],
        "combined_intrinsic": [
            "local_margin_risk",
            "final_margin_risk",
            "trail_disagreement_risk",
            "coherence_risk",
            "lost_trail_risk",
        ],
    }
    assert module.SEEDS == list(range(8))
    assert module.COVERAGES == (0.9, 0.8, 0.7, 0.5)


def test_v08_delta_summary_tracks_direction_and_sign_test():
    module = load_module()
    summary = module.delta_summary([0.10, 0.05, 0.0, -0.01])
    assert summary["wins"] == 2
    assert summary["ties"] == 1
    assert summary["losses"] == 1
    assert summary["n"] == 4
    assert summary["mean"] > 0
    assert summary["sign_test_p_two_sided"] is not None


def test_v08_dataset_generators_are_available_without_downloads():
    module = load_module()
    for name in (
        "moons_hard",
        "circles_hard",
        "spirals_hard",
        "overlap_hd",
        "breast_cancer",
        "wine",
        "digits_0_4",
    ):
        X, y = module.dataset_for(name, 0)
        assert len(X) == len(y)
        assert len(X) > 100
