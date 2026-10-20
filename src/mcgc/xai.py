from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
from sklearn.utils.validation import check_is_fitted


def _class_value(model, c: int):
    value = model.classes_[c]
    return value.item() if hasattr(value, "item") else value


def _query_neighbourhood(model, x: np.ndarray):
    distances, starts = model._query_start_nodes(x)
    sigma = max(float(distances[-1]) if len(distances) else 1.0, 1e-12)
    affinity = np.exp(-(distances**2) / (sigma**2))
    affinity /= affinity.sum()
    return distances, starts, affinity


def _score_components(model, class_index: int, starts, affinity, trails):
    local_evidence = float(
        np.sum(affinity * model.contrast_field_[starts, class_index])
    )
    local_component = float(model.local_evidence_weight * local_evidence)

    if not model.use_walkers:
        return {
            "local_evidence": local_evidence,
            "local_component": local_component,
            "mean_trail_score": 0.0,
            "trail_component": 0.0,
            "agreement": 0.0,
            "agreement_component": 0.0,
            "lost_fraction": 0.0,
            "lost_component": 0.0,
            "reconstructed_score": local_component,
        }

    trail_scores = np.asarray([trail.score for trail in trails], dtype=float)
    mean_trail_score = float(np.mean(trail_scores))
    agreement = 1.0 / (1.0 + float(np.std(trail_scores)))
    lost_fraction = float(np.mean([trail.lost for trail in trails]))

    trail_component = float(model.trail_score_weight * mean_trail_score)
    agreement_component = float(model.agreement_weight * agreement)
    lost_component = float(-model.lost_fraction_weight * lost_fraction)
    reconstructed = float(
        local_component + trail_component + agreement_component + lost_component
    )
    return {
        "local_evidence": local_evidence,
        "local_component": local_component,
        "mean_trail_score": mean_trail_score,
        "trail_component": trail_component,
        "agreement": agreement,
        "agreement_component": agreement_component,
        "lost_fraction": lost_fraction,
        "lost_component": lost_component,
        "reconstructed_score": reconstructed,
    }


def _representative_trail(model, starts, affinity, class_index: int) -> Dict[str, Any]:
    """Build a deterministic greedy trail for visual explanation.

    This traversal uses the same field, gradient, geometry and revisit terms as
    the classifier, but takes argmax at each step rather than sampling. It is
    therefore an illustration of the fitted evidence landscape, not a claim
    that this single path generated the stochastic aggregate score.
    """
    local = model.contrast_field_[starts, class_index]
    shifted = local - np.min(local) + 1e-12
    start_score = affinity * shifted
    if float(np.sum(start_score)) <= 0.0:
        current = int(starts[int(np.argmax(affinity))])
    else:
        current = int(starts[int(np.argmax(start_score))])

    visited = {current}
    path = [current]
    evidence_trace = [float(model.contrast_field_[current, class_index])]
    gains: List[float] = []
    lost = False

    if evidence_trace[-1] >= model.target_threshold_[class_index]:
        return {
            "kind": "deterministic_greedy_field_trail",
            "path": path,
            "evidence_trace": evidence_trace,
            "gains": gains,
            "coherence": 1.0,
            "terminal_evidence": evidence_trace[-1],
            "lost": False,
            "length": 0,
        }

    for _ in range(model.max_steps):
        neigh = model.graph_neighbors_[current]
        if len(neigh) == 0:
            lost = True
            break

        current_e = model.contrast_field_[current, class_index]
        next_e = model.contrast_field_[neigh, class_index]
        grad = next_e - current_e
        weights = model.W_[current, neigh]
        revisit = np.array(
            [1.0 if int(node) in visited else 0.0 for node in neigh]
        )
        desirability = (
            model.field_weight * next_e
            + model.gradient_weight * grad
            + model.geometry_weight * np.log(weights + 1e-12)
            - model.revisit_penalty * revisit
        )

        if float(np.max(grad)) <= model.trail_floor:
            lost = bool(current_e <= 0.0)
            break

        pick = int(np.argmax(desirability))
        nxt = int(neigh[pick])
        gains.append(float(grad[pick]))
        current = nxt
        visited.add(current)
        path.append(current)
        evidence_trace.append(float(model.contrast_field_[current, class_index]))
        if evidence_trace[-1] >= model.target_threshold_[class_index]:
            break

    coherence = float(np.mean(np.asarray(gains) > 0.0)) if gains else (
        1.0 if evidence_trace[-1] > 0.0 else 0.0
    )
    return {
        "kind": "deterministic_greedy_field_trail",
        "path": path,
        "evidence_trace": evidence_trace,
        "gains": gains,
        "coherence": coherence,
        "terminal_evidence": evidence_trace[-1],
        "lost": lost,
        "length": len(gains),
    }


def explain_mcgc(model, X: np.ndarray, top_support: int = 5):
    """Explain MCGC predictions without changing the prediction mechanism."""
    check_is_fitted(model, ["class_field_", "contrast_field_", "nn_"])
    if top_support < 1:
        raise ValueError("top_support must be >= 1.")

    Xs = model._scaled(X)
    rng = np.random.default_rng(model.random_state)
    predictions = []
    explanations = []

    for x in Xs:
        scores, trails_by_class = model._score_query(x, rng)
        distances, starts, affinity = _query_neighbourhood(model, x)
        order = np.argsort(scores)[::-1]
        winner = int(order[0])
        runner_up = int(order[1]) if len(order) > 1 else winner
        predictions.append(model.classes_[winner])

        per_class = {}
        for c in range(model.n_classes_):
            trails = trails_by_class[c]
            components = _score_components(model, c, starts, affinity, trails)
            if not np.isclose(
                components["reconstructed_score"], scores[c], rtol=1e-10, atol=1e-12
            ):
                raise RuntimeError(
                    "XAI score decomposition does not reconstruct the classifier score."
                )
            per_class[_class_value(model, c)] = {
                "score": float(scores[c]),
                "score_components": components,
                "mean_stochastic_trail_coherence": (
                    float(np.mean([trail.coherence for trail in trails]))
                    if trails
                    else 0.0
                ),
                "representative_trail": _representative_trail(
                    model, starts, affinity, c
                ),
            }

        support_order = np.argsort(affinity)[::-1][: min(top_support, len(starts))]
        support = []
        for pos in support_order:
            node = int(starts[pos])
            training_label = model.classes_[model.y_index_[node]]
            if hasattr(training_label, "item"):
                training_label = training_label.item()
            support.append(
                {
                    "node_index": node,
                    "distance": float(distances[pos]),
                    "affinity": float(affinity[pos]),
                    "training_label": training_label,
                    "contrastive_evidence": {
                        _class_value(model, c): float(model.contrast_field_[node, c])
                        for c in range(model.n_classes_)
                    },
                }
            )

        explanations.append(
            {
                "prediction": _class_value(model, winner),
                "runner_up": _class_value(model, runner_up),
                "margin": float(scores[winner] - scores[runner_up]),
                "class_scores": {
                    _class_value(model, c): float(scores[c])
                    for c in range(model.n_classes_)
                },
                "classes": per_class,
                "supporting_neighbors": support,
                "fidelity_note": (
                    "Score components exactly reconstruct the class scores. "
                    "Representative trails are deterministic illustrations of "
                    "the fitted evidence field, not replacements for the "
                    "stochastic aggregate used for prediction."
                ),
            }
        )

    return np.asarray(predictions), explanations


def _contrastive_record(explanation: Dict[str, Any], positive, negative):
    if positive not in explanation["classes"]:
        raise ValueError(f"Unknown positive class: {positive!r}")
    if negative not in explanation["classes"]:
        raise ValueError(f"Unknown negative class: {negative!r}")
    if positive == negative:
        raise ValueError("Contrastive classes must be different.")

    pos = explanation["classes"][positive]
    neg = explanation["classes"][negative]
    pc = pos["score_components"]
    nc = neg["score_components"]
    deltas = {
        "local_component_delta": float(pc["local_component"] - nc["local_component"]),
        "trail_component_delta": float(pc["trail_component"] - nc["trail_component"]),
        "agreement_component_delta": float(
            pc["agreement_component"] - nc["agreement_component"]
        ),
        "lost_component_delta": float(pc["lost_component"] - nc["lost_component"]),
    }
    reconstructed_margin = float(sum(deltas.values()))
    score_margin = float(pos["score"] - neg["score"])
    if not np.isclose(reconstructed_margin, score_margin, rtol=1e-10, atol=1e-12):
        raise RuntimeError("Contrastive XAI decomposition does not reconstruct the score margin.")

    ranked = sorted(deltas.items(), key=lambda item: abs(item[1]), reverse=True)
    return {
        "positive_class": positive,
        "negative_class": negative,
        "positive_score": float(pos["score"]),
        "negative_score": float(neg["score"]),
        "score_margin": score_margin,
        "component_deltas": deltas,
        "reconstructed_margin": reconstructed_margin,
        "dominant_component": ranked[0][0],
        "dominant_component_delta": float(ranked[0][1]),
        "positive_representative_trail": pos["representative_trail"],
        "negative_representative_trail": neg["representative_trail"],
        "interpretation_note": (
            "Component deltas exactly reconstruct the requested class-score difference. "
            "Representative trails visualize the corresponding fitted evidence fields."
        ),
    }


def explain_contrastive_mcgc(
    model,
    X: np.ndarray,
    positive_class=None,
    negative_class=None,
    top_support: int = 5,
):
    """Return exact `why positive rather than negative` decompositions.

    By default the positive class is the prediction and the negative class is
    the runner-up, separately for each query. Supplying either class overrides
    that side of the comparison for every query.
    """
    predictions, explanations = explain_mcgc(model, X, top_support=top_support)
    records = []
    for explanation in explanations:
        pos = explanation["prediction"] if positive_class is None else positive_class
        neg = explanation["runner_up"] if negative_class is None else negative_class
        records.append(_contrastive_record(explanation, pos, neg))
    return predictions, records
