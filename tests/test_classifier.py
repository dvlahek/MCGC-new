import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

from mcgc import MultiAgentChemotacticGraphClassifier


def make_data():
    X, y = make_moons(n_samples=100, noise=0.12, random_state=7)
    return train_test_split(X, y, test_size=0.25, random_state=11, stratify=y)


def test_fit_predict_shape_and_labels():
    X_train, X_test, y_train, _ = make_data()
    clf = MultiAgentChemotacticGraphClassifier(
        n_neighbors=8, n_walkers=5, max_steps=6, diffusion_steps=20, random_state=3
    )
    pred = clf.fit(X_train, y_train).predict(X_test[:6])
    assert pred.shape == (6,)
    assert set(np.unique(pred)).issubset(set(np.unique(y_train)))


def test_reproducible_for_fixed_seed():
    X_train, X_test, y_train, _ = make_data()
    params = dict(n_neighbors=8, n_walkers=5, max_steps=6, diffusion_steps=20, random_state=42)
    a = MultiAgentChemotacticGraphClassifier(**params).fit(X_train, y_train)
    b = MultiAgentChemotacticGraphClassifier(**params).fit(X_train, y_train)
    np.testing.assert_allclose(a.decision_function(X_test[:8]), b.decision_function(X_test[:8]))
    np.testing.assert_array_equal(a.predict(X_test[:8]), b.predict(X_test[:8]))


def test_diagnostics_expose_trails():
    X_train, X_test, y_train, _ = make_data()
    clf = MultiAgentChemotacticGraphClassifier(
        n_neighbors=8, n_walkers=4, max_steps=5, diffusion_steps=15, random_state=0
    ).fit(X_train, y_train)
    pred, diag = clf.predict_with_diagnostics(X_test[:2])
    assert len(pred) == 2
    assert len(diag) == 2
    assert set(diag[0]["class_scores"]) == set(np.unique(y_train))
    for class_trails in diag[0]["trails"].values():
        assert len(class_trails) == 4
        assert {"score", "coherence", "terminal_evidence", "lost", "length"}.issubset(class_trails[0])
