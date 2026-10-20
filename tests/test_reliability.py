import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

from mcgc import MultiAgentChemotacticGraphClassifier
from mcgc.reliability import reliability_features


def make_model_and_data():
    X, y = make_moons(n_samples=140, noise=0.18, random_state=12)
    X_train, X_test, y_train, _ = train_test_split(
        X, y, test_size=0.30, random_state=19, stratify=y
    )
    model = MultiAgentChemotacticGraphClassifier(
        n_neighbors=9,
        n_walkers=7,
        max_steps=7,
        diffusion_steps=20,
        random_state=23,
        adaptive_walker_gate=True,
    ).fit(X_train, y_train)
    return model, X_test


def test_reliability_features_match_predictions_and_are_finite():
    model, X_test = make_model_and_data()
    expected = model.predict(X_test[:8])
    predicted, rows = reliability_features(model, X_test[:8])
    np.testing.assert_array_equal(predicted, expected)
    assert len(rows) == 8

    for prediction, row in zip(predicted, rows):
        assert row["winner"] == prediction
        assert row["runner_up"] != row["winner"]
        assert row["final_margin"] >= 0.0
        assert row["local_margin"] >= 0.0
        assert 0.0 <= row["walker_gate"] <= 1.0
        for value in row["risk_scores"].values():
            assert np.isfinite(value)


def test_reliability_algebraic_identities():
    model, X_test = make_model_and_data()
    _, rows = reliability_features(model, X_test[:6])
    for row in rows:
        assert np.isclose(
            row["trail_advantage"],
            row["winner_mean_trail_score"] - row["runner_mean_trail_score"],
        )
        assert np.isclose(
            row["terminal_advantage"],
            row["winner_mean_terminal_evidence"]
            - row["runner_mean_terminal_evidence"],
        )
        assert np.isclose(
            row["lost_advantage"],
            row["runner_lost_fraction"] - row["winner_lost_fraction"],
        )
        assert np.isclose(
            row["risk_scores"]["local_margin_risk"], -row["local_margin"]
        )
        assert np.isclose(
            row["risk_scores"]["final_margin_risk"], -row["final_margin"]
        )
        assert np.isclose(
            row["risk_scores"]["trail_advantage_risk"], -row["trail_advantage"]
        )


def test_reliability_features_are_reproducible_for_fixed_seed():
    model, X_test = make_model_and_data()
    pred_a, rows_a = reliability_features(model, X_test[:7])
    pred_b, rows_b = reliability_features(model, X_test[:7])
    np.testing.assert_array_equal(pred_a, pred_b)
    assert rows_a == rows_b
