import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

from mcgc import MultiAgentChemotacticGraphClassifier


def make_data():
    X, y = make_moons(n_samples=120, noise=0.13, random_state=9)
    return train_test_split(X, y, test_size=0.25, random_state=13, stratify=y)


def test_field_only_is_deterministic_and_has_no_trails():
    X_train, X_test, y_train, _ = make_data()
    clf = MultiAgentChemotacticGraphClassifier(
        n_neighbors=8,
        n_walkers=7,
        max_steps=6,
        diffusion_steps=20,
        random_state=4,
        use_walkers=False,
    ).fit(X_train, y_train)

    a = clf.decision_function(X_test[:10])
    b = clf.decision_function(X_test[:10])
    np.testing.assert_allclose(a, b)

    _, diag = clf.predict_with_diagnostics(X_test[:2])
    for item in diag:
        assert all(len(trails) == 0 for trails in item["trails"].values())


def test_zero_reliability_weights_leave_valid_classifier():
    X_train, X_test, y_train, _ = make_data()
    clf = MultiAgentChemotacticGraphClassifier(
        n_neighbors=8,
        n_walkers=4,
        max_steps=5,
        diffusion_steps=15,
        random_state=2,
        agreement_weight=0.0,
        lost_fraction_weight=0.0,
    ).fit(X_train, y_train)

    pred = clf.predict(X_test[:8])
    assert pred.shape == (8,)
    assert set(np.unique(pred)).issubset(set(np.unique(y_train)))


def test_one_walker_exposes_one_trail_per_class():
    X_train, X_test, y_train, _ = make_data()
    clf = MultiAgentChemotacticGraphClassifier(
        n_neighbors=8,
        n_walkers=1,
        max_steps=5,
        diffusion_steps=15,
        random_state=5,
    ).fit(X_train, y_train)

    _, diag = clf.predict_with_diagnostics(X_test[:1])
    assert all(len(trails) == 1 for trails in diag[0]["trails"].values())
