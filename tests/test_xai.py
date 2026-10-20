import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

from mcgc import MultiAgentChemotacticGraphClassifier, explain_mcgc


def make_model_and_data():
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
