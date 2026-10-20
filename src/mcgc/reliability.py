from __future__ import annotations

from typing import Any, Dict, Iterable

import numpy as np


def _top_two(mapping: Dict[Any, float]):
    ordered = sorted(mapping.items(), key=lambda item: item[1], reverse=True)
    if len(ordered) < 2:
        raise ValueError("Reliability diagnostics require at least two classes.")
    return ordered[0], ordered[1]


def _trail_stats(trails: Iterable[Dict[str, Any]]) -> Dict[str, float]:
    trails = list(trails)
    if not trails:
        return {
            "mean_score": 0.0,
            "score_std": 0.0,
            "mean_coherence": 0.0,
            "lost_fraction": 0.0,
            "mean_terminal_evidence": 0.0,
            "mean_length": 0.0,
        }
    return {
        "mean_score": float(np.mean([trail["score"] for trail in trails])),
        "score_std": float(np.std([trail["score"] for trail in trails])),
        "mean_coherence": float(
            np.mean([trail["coherence"] for trail in trails])
        ),
        "lost_fraction": float(np.mean([trail["lost"] for trail in trails])),
        "mean_terminal_evidence": float(
            np.mean([trail["terminal_evidence"] for trail in trails])
        ),
        "mean_length": float(np.mean([trail["length"] for trail in trails])),
    }


def reliability_features_from_diagnostic(diagnostic: Dict[str, Any]) -> Dict[str, Any]:
    """Extract traceable reliability signals from one MCGC diagnostic record.

    The function performs no fitting and does not change the classifier score.
    It simply exposes quantities already produced by MCGC inference in a form
    suitable for error-detection and selective-classification experiments.
    """
    (winner, winner_score), (runner, runner_score) = _top_two(
        diagnostic["class_scores"]
    )
    (_, local_top), (_, local_second) = _top_two(diagnostic["local_evidence"])

    winner_stats = _trail_stats(diagnostic["trails"][winner])
    runner_stats = _trail_stats(diagnostic["trails"][runner])

    final_margin = float(winner_score - runner_score)
    local_margin = float(local_top - local_second)
    trail_advantage = float(
        winner_stats["mean_score"] - runner_stats["mean_score"]
    )
    terminal_advantage = float(
        winner_stats["mean_terminal_evidence"]
        - runner_stats["mean_terminal_evidence"]
    )
    lost_advantage = float(
        runner_stats["lost_fraction"] - winner_stats["lost_fraction"]
    )
    coherence_advantage = float(
        winner_stats["mean_coherence"] - runner_stats["mean_coherence"]
    )

    return {
        "winner": winner,
        "runner_up": runner,
        "walker_gate": float(diagnostic.get("walker_gate", 1.0)),
        "final_margin": final_margin,
        "local_margin": local_margin,
        "winner_mean_trail_score": winner_stats["mean_score"],
        "runner_mean_trail_score": runner_stats["mean_score"],
        "trail_advantage": trail_advantage,
        "winner_trail_score_std": winner_stats["score_std"],
        "winner_mean_coherence": winner_stats["mean_coherence"],
        "winner_lost_fraction": winner_stats["lost_fraction"],
        "runner_lost_fraction": runner_stats["lost_fraction"],
        "lost_advantage": lost_advantage,
        "winner_mean_terminal_evidence": winner_stats["mean_terminal_evidence"],
        "runner_mean_terminal_evidence": runner_stats["mean_terminal_evidence"],
        "terminal_advantage": terminal_advantage,
        "coherence_advantage": coherence_advantage,
        "winner_mean_trail_length": winner_stats["mean_length"],
        "risk_scores": {
            # Larger values consistently mean "more suspicious".
            "local_margin_risk": -local_margin,
            "final_margin_risk": -final_margin,
            "trail_advantage_risk": -trail_advantage,
            "trail_disagreement_risk": winner_stats["score_std"],
            "coherence_risk": 1.0 - winner_stats["mean_coherence"],
            "lost_trail_risk": winner_stats["lost_fraction"],
            "terminal_advantage_risk": -terminal_advantage,
            "coherence_advantage_risk": -coherence_advantage,
            "lost_advantage_risk": -lost_advantage,
        },
    }


def reliability_features(model, X: np.ndarray):
    """Return predictions and intrinsic reliability features for each query."""
    predictions, diagnostics = model.predict_with_diagnostics(X)
    return predictions, [
        reliability_features_from_diagnostic(diagnostic)
        for diagnostic in diagnostics
    ]
