from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable, Dict

import numpy as np
from sklearn.datasets import make_circles, make_moons
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.semi_supervised import LabelSpreading

from mcgc import MultiAgentChemotacticGraphClassifier


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    maker: Callable[[], tuple[np.ndarray, np.ndarray]]


def make_spirals(n_samples: int = 600, noise: float = 0.16, random_state: int = 7):
    rng = np.random.default_rng(random_state)
    n = n_samples // 2
    theta = np.linspace(0.2, 3.2 * np.pi, n)
    radius = np.linspace(0.15, 1.0, n)
    arm0 = np.column_stack([radius * np.cos(theta), radius * np.sin(theta)])
    arm1 = -arm0
    X = np.vstack([arm0, arm1])
    X += rng.normal(scale=noise, size=X.shape)
    y = np.concatenate([np.zeros(n, dtype=int), np.ones(n, dtype=int)])
    order = rng.permutation(len(X))
    return X[order], y[order]


def metrics(y_true, pred):
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
    }


def fit_label_spreading(X_train, y_train, X_test):
    X_all = np.vstack([X_train, X_test])
    y_all = np.concatenate([y_train, -np.ones(len(X_test), dtype=int)])
    model = LabelSpreading(kernel="knn", n_neighbors=12, alpha=0.2, max_iter=50)
    model.fit(X_all, y_all)
    return model.transduction_[-len(X_test):]


def mcgc_variant(**overrides):
    params = dict(
        n_neighbors=12,
        n_walkers=12,
        max_steps=10,
        diffusion_steps=40,
        random_state=17,
    )
    params.update(overrides)
    return MultiAgentChemotacticGraphClassifier(**params)


def evaluate_dataset(name: str, X: np.ndarray, y: np.ndarray) -> Dict[str, dict]:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=17, stratify=y
    )

    models = {
        "knn_uniform": KNeighborsClassifier(n_neighbors=9, weights="uniform"),
        "knn_distance": KNeighborsClassifier(n_neighbors=9, weights="distance"),
        "field_only": mcgc_variant(use_walkers=False),
        "one_walker": mcgc_variant(n_walkers=1),
        "multi_no_gradient": mcgc_variant(gradient_weight=0.0),
        "mcgc_full": mcgc_variant(),
        "mcgc_no_reliability": mcgc_variant(
            agreement_weight=0.0,
            lost_fraction_weight=0.0,
        ),
    }

    rows: Dict[str, dict] = {}
    for label, model in models.items():
        pred = model.fit(X_train, y_train).predict(X_test)
        rows[label] = metrics(y_test, pred)

    label_spreading_pred = fit_label_spreading(X_train, y_train, X_test)
    rows["label_spreading"] = metrics(y_test, label_spreading_pred)

    full = mcgc_variant().fit(X_train, y_train)
    _, diag = full.predict_with_diagnostics(X_test)
    winner_lost = []
    winner_coherence = []
    winner_length = []
    for d in diag:
        trails = d["trails"][d["winner"]]
        if not trails:
            continue
        winner_lost.append(np.mean([t["lost"] for t in trails]))
        winner_coherence.append(np.mean([t["coherence"] for t in trails]))
        winner_length.append(np.mean([t["length"] for t in trails]))
    rows["mcgc_full"]["mean_winner_lost_fraction"] = float(np.mean(winner_lost))
    rows["mcgc_full"]["mean_winner_trail_coherence"] = float(np.mean(winner_coherence))
    rows["mcgc_full"]["mean_winner_trail_length"] = float(np.mean(winner_length))

    return {name: rows}


def main():
    datasets = [
        DatasetSpec(
            "moons",
            lambda: make_moons(n_samples=600, noise=0.18, random_state=7),
        ),
        DatasetSpec(
            "circles",
            lambda: make_circles(n_samples=600, noise=0.10, factor=0.45, random_state=7),
        ),
        DatasetSpec(
            "spirals",
            lambda: make_spirals(n_samples=600, noise=0.10, random_state=7),
        ),
    ]

    result = {
        "protocol": {
            "version": "0.2",
            "split_random_state": 17,
            "test_fraction": 0.30,
            "mcgc_random_state": 17,
            "note": "Fixed smoke/ablation protocol; not a tuned performance benchmark.",
        },
        "datasets": {},
    }
    for spec in datasets:
        X, y = spec.maker()
        result["datasets"].update(evaluate_dataset(spec.name, X, y))

    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
