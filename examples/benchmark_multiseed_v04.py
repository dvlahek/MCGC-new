from __future__ import annotations

import json

import numpy as np
from sklearn.datasets import make_circles, make_moons
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from mcgc import MultiAgentChemotacticGraphClassifier
from benchmark_ablation_v02 import make_spirals


SEEDS = list(range(12))


def mcgc(**overrides):
    params = dict(
        n_neighbors=12,
        n_walkers=12,
        max_steps=10,
        diffusion_steps=40,
        random_state=17,
    )
    params.update(overrides)
    return MultiAgentChemotacticGraphClassifier(**params)


def paired_summary(values):
    arr = np.asarray(values, dtype=float)
    eps = 1e-12
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "wins": int(np.sum(arr > eps)),
        "ties": int(np.sum(np.abs(arr) <= eps)),
        "losses": int(np.sum(arr < -eps)),
    }


def dataset_for(name, seed):
    if name == "moons":
        return make_moons(n_samples=500, noise=0.18, random_state=seed)
    if name == "circles":
        return make_circles(
            n_samples=500, noise=0.10, factor=0.45, random_state=seed
        )
    if name == "spirals":
        return make_spirals(n_samples=500, noise=0.10, random_state=seed)
    raise ValueError(name)


def evaluate_dataset(name):
    rows = []
    delta_walker = []
    delta_gradient = []
    delta_vs_knn = []

    for seed in SEEDS:
        X, y = dataset_for(name, seed)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.30, random_state=1000 + seed, stratify=y
        )

        field = mcgc(use_walkers=False).fit(X_train, y_train)
        no_gradient = mcgc(gradient_weight=0.0).fit(X_train, y_train)
        full = mcgc().fit(X_train, y_train)
        knn = KNeighborsClassifier(n_neighbors=9, weights="distance").fit(
            X_train, y_train
        )

        acc_field = float(accuracy_score(y_test, field.predict(X_test)))
        acc_no_gradient = float(
            accuracy_score(y_test, no_gradient.predict(X_test))
        )
        acc_full = float(accuracy_score(y_test, full.predict(X_test)))
        acc_knn = float(accuracy_score(y_test, knn.predict(X_test)))

        dw = acc_full - acc_field
        dg = acc_full - acc_no_gradient
        dk = acc_full - acc_knn
        delta_walker.append(dw)
        delta_gradient.append(dg)
        delta_vs_knn.append(dk)
        rows.append(
            {
                "seed": seed,
                "field_only": acc_field,
                "multi_no_gradient": acc_no_gradient,
                "mcgc_full": acc_full,
                "knn_distance": acc_knn,
                "delta_full_minus_field": dw,
                "delta_full_minus_no_gradient": dg,
                "delta_full_minus_knn": dk,
            }
        )

    return {
        "runs": rows,
        "paired": {
            "walker_effect_full_minus_field": paired_summary(delta_walker),
            "gradient_effect_full_minus_no_gradient": paired_summary(delta_gradient),
            "full_minus_knn_distance": paired_summary(delta_vs_knn),
        },
    }


def main():
    result = {
        "protocol": {
            "version": "0.4",
            "seeds": SEEDS,
            "n_seeds": len(SEEDS),
            "test_fraction": 0.30,
            "split_seed": "1000 + dataset_seed",
            "mcgc_random_state": 17,
            "note": (
                "Frozen multi-seed confirmation benchmark. Hyperparameters are not "
                "retuned per seed. Paired deltas are descriptive, not inferential claims."
            ),
        },
        "datasets": {
            name: evaluate_dataset(name) for name in ("moons", "circles", "spirals")
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
