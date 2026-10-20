import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

from mcgc import (
    MultiAgentChemotacticGraphClassifier,
    explain_contrastive_mcgc,
    explain_mcgc,
)


def make_model_and_data(adaptive_walker_gate=False):
    X, y = make_moons(n_samples=120, noise=0.14, random_state=9)
    X_train, X_test, y_train, _ = train_test_split(
        X, y, test_size=0.25, random_state=13, stratify=y
    )
    model = MultiAgentChemotacticGraphClassifier(
        n_neighbors=9,
        n_walkers=6,
        max_steps=7,
        diffusion_steps=20,
        random_state=41,
        adaptive_walker_gate=adaptive_walker_gate,
    ).fit(X_train, y_train)
    return model, X_test


def test_explanation_prediction_matches_predict():
    model, X_test = make_model_and_data()
    expected = model.predict(X_test[:5])
    predicted, explanations = explain_mcgc(model, X_test[:5])
    np.testing.assert_array_equal(predicted, expected)
    assert len(explanations) == 5


def test_score_decomposition_is_exact():
    model, X_test = make_model_and_data()
    _, explanations = explain_mcgc(model, X_test[:4])
    for explanation in explanations:
        for class_value, class_info in explanation["classes"].items():
            parts = class_info["score_components"]
            reconstructed = (
                parts["local_component"]
                + parts["trail_component"]
                + parts["agreement_component"]
                + parts["lost_component"]
            )
            assert np.isclose(reconstructed, class_info["score"])
            assert np.isclose(parts["reconstructed_score"], class_info["score"])
            assert class_value in explanation["class_scores"]


def test_representative_trails_are_deterministic():
    model, X_test = make_model_and_data()
    _, a = explain_mcgc(model, X_test[:3])
    _, b = explain_mcgc(model, X_test[:3])
    for ea, eb in zip(a, b):
        assert ea["prediction"] == eb["prediction"]
        assert ea["margin"] == eb["margin"]
        for class_value in ea["classes"]:
            ta = ea["classes"][class_value]["representative_trail"]
            tb = eb["classes"][class_value]["representative_trail"]
            assert ta == tb


def test_explanation_exposes_support_and_contrastive_trails():
    model, X_test = make_model_and_data()
    _, explanations = explain_mcgc(model, X_test[:1], top_support=4)
    explanation = explanations[0]
    assert len(explanation["supporting_neighbors"]) == 4
    assert explanation["prediction"] in explanation["classes"]
    assert explanation["runner_up"] in explanation["classes"]
    assert explanation["margin"] >= 0.0
    for class_info in explanation["classes"].values():
        trail = class_info["representative_trail"]
        assert trail["kind"] == "deterministic_greedy_field_trail"
        assert len(trail["path"]) == len(trail["evidence_trace"])
        assert len(trail["gains"]) == max(0, len(trail["path"]) - 1)


def test_contrastive_explanation_exactly_reconstructs_margin():
    model, X_test = make_model_and_data()
    predicted, contrasts = explain_contrastive_mcgc(model, X_test[:6])
    np.testing.assert_array_equal(predicted, model.predict(X_test[:6]))
    for contrast in contrasts:
        assert contrast["positive_class"] != contrast["negative_class"]
        assert contrast["score_margin"] >= 0.0
        assert np.isclose(
            contrast["score_margin"], contrast["reconstructed_margin"]
        )
        assert np.isclose(
            contrast["score_margin"],
            sum(contrast["component_deltas"].values()),
        )
        assert contrast["dominant_component"] in contrast["component_deltas"]


def test_requested_contrastive_classes_are_respected():
    model, X_test = make_model_and_data()
    _, contrasts = explain_contrastive_mcgc(
        model, X_test[:2], positive_class=1, negative_class=0
    )
    for contrast in contrasts:
        assert contrast["positive_class"] == 1
        assert contrast["negative_class"] == 0


def test_selective_gate_explanation_exactly_matches_prediction():
    model, X_test = make_model_and_data(adaptive_walker_gate=True)
    expected = model.predict(X_test[:7])
    predicted, explanations = explain_mcgc(model, X_test[:7])
    np.testing.assert_array_equal(predicted, expected)

    for explanation in explanations:
        gate = explanation["walker_gate"]
        assert 0.0 <= gate <= 1.0
        assert explanation["walker_margin_scale"] > 0.0
        for class_info in explanation["classes"].values():
            parts = class_info["score_components"]
            assert np.isclose(parts["walker_gate"], gate)
            assert np.isclose(
                parts["trail_component"],
                gate * parts["raw_trail_component"],
            )
            assert np.isclose(
                parts["agreement_component"],
                gate * parts["raw_agreement_component"],
            )
            assert np.isclose(
                parts["lost_component"],
                gate * parts["raw_lost_component"],
            )
            reconstructed = (
                parts["local_component"]
                + parts["trail_component"]
                + parts["agreement_component"]
                + parts["lost_component"]
            )
            assert np.isclose(reconstructed, class_info["score"])


def test_selective_gate_contrastive_margin_is_exact():
    model, X_test = make_model_and_data(adaptive_walker_gate=True)
    predicted, contrasts = explain_contrastive_mcgc(model, X_test[:7])
    np.testing.assert_array_equal(predicted, model.predict(X_test[:7]))
    for contrast in contrasts:
        assert 0.0 <= contrast["walker_gate"] <= 1.0
        assert np.isclose(
            contrast["score_margin"],
            contrast["reconstructed_margin"],
        )
        assert np.isclose(
            contrast["score_margin"],
            sum(contrast["component_deltas"].values()),
        )
