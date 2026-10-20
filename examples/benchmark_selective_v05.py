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
    selective_minus_field = []
    selective_minus_unconditional = []
    selective_minus_knn = []

    for seed in SEEDS:
        X, y = dataset_for(name, seed)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.30, random_state=1000 + seed, stratify=y
        )

        field = mcgc(use_walkers=False).fit(X_train, y_train)
        unconditional = mcgc(adaptive_walker_gate=False).fit(X_train, y_train)
        selective = mcgc(adaptive_walker_gate=True).fit(X_train, y_train)
        knn = KNeighborsClassifier(n_neighbors=9, weights="distance").fit(
            X_train, y_train
        )

        pred_field = field.predict(X_test)
        pred_unconditional = unconditional.predict(X_test)
        pred_selective, diag = selective.predict_with_diagnostics(X_test)
        pred_knn = knn.predict(X_test)

        acc_field = float(accuracy_score(y_test, pred_field))
        acc_unconditional = float(accuracy_score(y_test, pred_unconditional))
        acc_selective = float(accuracy_score(y_test, pred_selective))
        acc_knn = float(accuracy_score(y_test, pred_knn))
        gates = np.asarray([d["walker_gate"] for d in diag], dtype=float)

        dsf = acc_selective - acc_field
        dsu = acc_selective - acc_unconditional
        dsk = acc_selective - acc_knn
        selective_minus_field.append(dsf)
        selective_minus_unconditional.append(dsu)
        selective_minus_knn.append(dsk)
        rows.append(
            {
                "seed": seed,
                "field_only": acc_field,
                "unconditional_walkers": acc_unconditional,
                "selective_walkers": acc_selective,
                "knn_distance": acc_knn,
                "delta_selective_minus_field": dsf,
                "delta_selective_minus_unconditional": dsu,
                "delta_selective_minus_knn": dsk,
                "mean_walker_gate": float(np.mean(gates)),
                "median_walker_gate": float(np.median(gates)),
                "walker_margin_scale": float(selective.walker_margin_scale_),
            }
        )

    return {
        "runs": rows,
        "paired": {
            "selective_minus_field": paired_summary(selective_minus_field),
            "selective_minus_unconditional": paired_summary(
                selective_minus_unconditional
            ),
            "selective_minus_knn_distance": paired_summary(selective_minus_knn),
        },
        "gate": {
            "mean_across_seed_means": float(
                np.mean([row["mean_walker_gate"] for row in rows])
            ),
            "mean_across_seed_medians": float(
                np.mean([row["median_walker_gate"] for row in rows])
            ),
        },
    }


def main():
    result = {
        "protocol": {
            "version": "0.5",
            "seeds": SEEDS,
            "n_seeds": len(SEEDS),
            "test_fraction": 0.30,
            "split_seed": "1000 + dataset_seed",
            "mcgc_random_state": 17,
            "gate": "exp(-(local_margin / median_training_margin) ** 1.0)",
            "note": (
                "Frozen selective-walker confirmation. No per-seed or per-dataset "
                "retuning relative to v0.4. Paired deltas are descriptive."
            ),
        },
        "datasets": {
            name: evaluate_dataset(name) for name in ("moons", "circles", "spirals")
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
