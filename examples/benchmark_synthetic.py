from __future__ import annotations

import json

from sklearn.datasets import make_circles, make_moons
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from mcgc import MultiAgentChemotacticGraphClassifier


def evaluate(name, X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=17, stratify=y
    )
    baseline = KNeighborsClassifier(n_neighbors=9).fit(X_train, y_train)
    mcgc = MultiAgentChemotacticGraphClassifier(
        n_neighbors=12,
        n_walkers=12,
        max_steps=10,
        diffusion_steps=40,
        random_state=17,
    ).fit(X_train, y_train)

    rows = {}
    for label, model in [("knn", baseline), ("mcgc", mcgc)]:
        pred = model.predict(X_test)
        rows[label] = {
            "accuracy": accuracy_score(y_test, pred),
            "balanced_accuracy": balanced_accuracy_score(y_test, pred),
        }
    return {name: rows}


def main():
    result = {}
    X, y = make_moons(n_samples=500, noise=0.18, random_state=7)
    result.update(evaluate("moons", X, y))
    X, y = make_circles(n_samples=500, noise=0.10, factor=0.45, random_state=7)
    result.update(evaluate("circles", X, y))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
