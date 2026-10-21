from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from sklearn.datasets import load_digits
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.semi_supervised import LabelSpreading
from sklearn.svm import SVC

from mcgc import MultiAgentChemotacticGraphClassifier


SMOKE_SEED = 0
SMOKE_NOISE_LEVELS = (0.0, 0.30)
INNER_FOLDS = 2
AUDIT_FOLDS = 3

MCGC_GRID = tuple(
    {"diffusion_alpha": alpha}
    for alpha in (0.0, 0.50, 0.85, 0.95)
)

LABEL_SPREADING_GRID = tuple(
    {"n_neighbors": k, "alpha": alpha}
    for k, alpha in itertools.product((7, 12, 17), (0.20, 0.50, 0.80))
)

KNN_GRID = tuple(
    {"n_neighbors": k, "weights": weights}
    for k, weights in itertools.product((5, 12, 25), ("uniform", "distance"))
)

SVM_GRID = tuple(
    {"C": C, "gamma": gamma}
    for C, gamma in itertools.product((1.0, 10.0), ("scale", 0.1))
)


def corrupt_labels(y_clean: np.ndarray, eta: float, seed: int):
    y_clean = np.asarray(y_clean)
    y_noisy = y_clean.copy()
    classes = np.unique(y_clean)
    n_flip = int(round(float(eta) * len(y_clean)))
    mask = np.zeros(len(y_clean), dtype=bool)
    if n_flip == 0:
        return y_noisy, mask

    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(y_clean), size=n_flip, replace=False)
    mask[chosen] = True
    for idx in chosen:
        alternatives = classes[classes != y_clean[idx]]
        y_noisy[idx] = rng.choice(alternatives)
    return y_noisy, mask


def mcgc_for(params: dict):
    return MultiAgentChemotacticGraphClassifier(
        n_neighbors=12,
        n_walkers=12,
        max_steps=10,
        diffusion_alpha=float(params["diffusion_alpha"]),
        diffusion_steps=40,
        temperature=0.35,
        random_state=17,
        adaptive_walker_gate=True,
    )


def label_spreading_for(params: dict):
    return make_pipeline(
        StandardScaler(),
        LabelSpreading(
            kernel="knn",
            n_neighbors=int(params["n_neighbors"]),
            alpha=float(params["alpha"]),
            max_iter=100,
            tol=1e-3,
        ),
    )


def knn_for(params: dict):
    return make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(
            n_neighbors=int(params["n_neighbors"]),
            weights=str(params["weights"]),
        ),
    )


def svm_for(params: dict):
    return make_pipeline(
        StandardScaler(),
        SVC(C=float(params["C"]), gamma=params["gamma"], kernel="rbf"),
    )


def tune_model(kind: str, grid: tuple[dict, ...], X, y_noisy, seed: int):
    cv = StratifiedKFold(
        n_splits=INNER_FOLDS,
        shuffle=True,
        random_state=19000 + seed,
    )
    records = []
    for params in grid:
        fold_scores = []
        for fit_idx, val_idx in cv.split(X, y_noisy):
            if kind == "mcgc":
                model = mcgc_for(params)
            elif kind == "label_spreading":
                model = label_spreading_for(params)
            elif kind == "knn":
                model = knn_for(params)
            elif kind == "svm":
                model = svm_for(params)
            else:
                raise ValueError(kind)
            model.fit(X[fit_idx], y_noisy[fit_idx])
            pred = model.predict(X[val_idx])
            fold_scores.append(balanced_accuracy_score(y_noisy[val_idx], pred))
        records.append(
            {
                "params": dict(params),
                "mean_noisy_cv_balanced_accuracy": float(np.mean(fold_scores)),
                "fold_scores": [float(v) for v in fold_scores],
            }
        )

    records.sort(
        key=lambda row: (
            -round(row["mean_noisy_cv_balanced_accuracy"], 12),
            json.dumps(row["params"], sort_keys=True),
        )
    )
    return records[0], records


def fit_selected(kind: str, params: dict, X, y):
    if kind == "mcgc":
        model = mcgc_for(params)
    elif kind == "label_spreading":
        model = label_spreading_for(params)
    elif kind == "knn":
        model = knn_for(params)
    elif kind == "svm":
        model = svm_for(params)
    else:
        raise ValueError(kind)
    return model.fit(X, y)


def classification_metrics(y_true, pred):
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
    }


def _lookup(mapping: dict, label):
    if label in mapping:
        return mapping[label]
    py_label = label.item() if hasattr(label, "item") else label
    if py_label in mapping:
        return mapping[py_label]
    raise KeyError(label)


def oof_mcgc_audit(X, y_noisy, corruption_mask, selected_params: dict, seed: int):
    cv = StratifiedKFold(
        n_splits=AUDIT_FOLDS,
        shuffle=True,
        random_state=20000 + seed,
    )
    scores = np.empty(len(y_noisy), dtype=float)
    suggested = np.empty(len(y_noisy), dtype=y_noisy.dtype)
    seen = np.zeros(len(y_noisy), dtype=bool)

    for fit_idx, val_idx in cv.split(X, y_noisy):
        model = mcgc_for(selected_params).fit(X[fit_idx], y_noisy[fit_idx])
        _, diagnostics = model.predict_with_diagnostics(X[val_idx])
        for pos, row in zip(val_idx, diagnostics):
            observed = y_noisy[pos]
            evidence = row["local_evidence"]
            observed_e = float(_lookup(evidence, observed))
            rivals = [
                (label, float(value))
                for label, value in evidence.items()
                if label != (observed.item() if hasattr(observed, "item") else observed)
            ]
            rival_label, rival_e = max(rivals, key=lambda item: item[1])
            scores[pos] = float(rival_e - observed_e)
            suggested[pos] = rival_label
            seen[pos] = True

    if not np.all(seen):
        raise RuntimeError("OOF audit did not score every outer-training sample")

    n_corrupt = int(np.sum(corruption_mask))
    if n_corrupt == 0:
        metrics = {
            "roc_auc": None,
            "pr_auc": None,
            "precision_at_k": None,
            "k": 0,
        }
    else:
        order = np.argsort(-scores, kind="stable")
        top = order[:n_corrupt]
        metrics = {
            "roc_auc": float(roc_auc_score(corruption_mask.astype(int), scores)),
            "pr_auc": float(average_precision_score(corruption_mask.astype(int), scores)),
            "precision_at_k": float(np.mean(corruption_mask[top])),
            "k": n_corrupt,
        }

    return {
        "metrics": metrics,
        "scores": scores.tolist(),
        "suggested_labels": suggested.tolist(),
    }


def evaluate_noise_level(X_train, X_test, y_train_clean, y_test, eta: float, seed: int):
    noise_seed = 18000 + 1000 * seed + int(round(100 * eta))
    y_train_noisy, corruption_mask = corrupt_labels(y_train_clean, eta, noise_seed)

    specs = (
        ("mcgc", MCGC_GRID),
        ("label_spreading", LABEL_SPREADING_GRID),
        ("knn", KNN_GRID),
        ("svm", SVM_GRID),
    )

    models = {}
    tuning = {}
    selected_params = {}
    for kind, grid in specs:
        selected, records = tune_model(kind, grid, X_train, y_train_noisy, seed)
        params = selected["params"]
        fitted = fit_selected(kind, params, X_train, y_train_noisy)
        pred = fitted.predict(X_test)
        models[kind] = classification_metrics(y_test, pred)
        tuning[kind] = {
            "selected": selected,
            "all_candidates": records,
        }
        selected_params[kind] = params

    audit = oof_mcgc_audit(
        X_train,
        y_train_noisy,
        corruption_mask,
        selected_params["mcgc"],
        seed,
    )

    return {
        "eta": float(eta),
        "noise_seed": int(noise_seed),
        "n_train": int(len(X_train)),
        "n_corrupted": int(np.sum(corruption_mask)),
        "corruption_fraction_realized": float(np.mean(corruption_mask)),
        "models": models,
        "tuning": tuning,
        "mcgc_oof_label_audit": audit,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("v18_label_noise_smoke.json"),
    )
    args = parser.parse_args()

    data = load_digits()
    X = np.asarray(data.data)
    y = np.asarray(data.target)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=5000 + SMOKE_SEED,
        stratify=y,
    )

    runs = [
        evaluate_noise_level(
            X_train,
            X_test,
            y_train,
            y_test,
            eta,
            SMOKE_SEED,
        )
        for eta in SMOKE_NOISE_LEVELS
    ]

    result = {
        "protocol": {
            "version": "1.8-smoke",
            "status": "diagnostic smoke only; not confirmatory evidence",
            "dataset": "digits",
            "seed": SMOKE_SEED,
            "noise_levels": list(SMOKE_NOISE_LEVELS),
            "outer_split": "70% train / 30% untouched clean test",
            "inner_tuning": "2-fold stratified CV using noisy outer-training labels only",
            "audit": "3-fold OOF local-evidence conflict; audited sample excluded from fitted field",
            "mcgc_grid": list(MCGC_GRID),
            "label_spreading_grid": list(LABEL_SPREADING_GRID),
            "knn_grid": list(KNN_GRID),
            "svm_grid": list(SVM_GRID),
        },
        "runs": runs,
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")

    print("=== MCGC v1.8 LABEL-NOISE SMOKE ===")
    for run in runs:
        eta = run["eta"]
        m = run["models"]["mcgc"]
        l = run["models"]["label_spreading"]
        k = run["models"]["knn"]
        s = run["models"]["svm"]
        alpha = run["tuning"]["mcgc"]["selected"]["params"]["diffusion_alpha"]
        ls = run["tuning"]["label_spreading"]["selected"]["params"]
        audit = run["mcgc_oof_label_audit"]["metrics"]
        print(
            f"eta={eta:.2f} TEST_ACC MCGC/LS/kNN/SVM="
            f"{m['accuracy']:.4f}/{l['accuracy']:.4f}/{k['accuracy']:.4f}/{s['accuracy']:.4f} "
            f"BA={m['balanced_accuracy']:.4f}/{l['balanced_accuracy']:.4f}/"
            f"{k['balanced_accuracy']:.4f}/{s['balanced_accuracy']:.4f} "
            f"MCGC_ALPHA={alpha} LS={ls}"
        )
        if audit["roc_auc"] is not None:
            print(
                f"eta={eta:.2f} OOF_AUDIT ROC_AUC={audit['roc_auc']:.4f} "
                f"PR_AUC={audit['pr_auc']:.4f} P@K={audit['precision_at_k']:.4f} "
                f"K={audit['k']}"
            )
    print(f"WROTE {args.output}")


if __name__ == "__main__":
    main()
