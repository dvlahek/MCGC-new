from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
from sklearn.datasets import load_breast_cancer, load_digits, load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
V18 = ROOT / "examples" / "smoke_label_noise_v18.py"
V18B = ROOT / "examples" / "smoke_label_audit_baselines_v18b.py"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


v18 = _load_module(V18, "smoke_label_noise_v18")
v18b = _load_module(V18B, "smoke_label_audit_baselines_v18b")

SEED = 0
DATASETS = ("digits", "wine", "breast_cancer")
NOISE_LEVELS = (0.05, 0.10, 0.20, 0.30)
NOISE_TYPES = ("symmetric", "class_dependent", "boundary_targeted")
METHODS = v18b.METHODS


def dataset_for(name: str):
    if name == "digits":
        data = load_digits()
    elif name == "wine":
        data = load_wine()
    elif name == "breast_cancer":
        data = load_breast_cancer()
    else:
        raise ValueError(name)
    return np.asarray(data.data), np.asarray(data.target)


def _n_flip(n: int, eta: float) -> int:
    return int(round(float(eta) * int(n)))


def corrupt_symmetric(y_clean: np.ndarray, eta: float, seed: int):
    return v18.corrupt_labels(y_clean, eta, seed)


def _nearest_centroid_target_map(X: np.ndarray, y: np.ndarray):
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    classes = np.unique(y)
    centroids = {
        c: np.mean(Xs[y == c], axis=0)
        for c in classes
    }
    mapping = {}
    for c in classes:
        candidates = [r for r in classes if r != c]
        mapping[c] = min(
            candidates,
            key=lambda r: float(np.linalg.norm(centroids[c] - centroids[r])),
        )
    return mapping


def corrupt_class_dependent(
    X: np.ndarray,
    y_clean: np.ndarray,
    eta: float,
    seed: int,
):
    y_clean = np.asarray(y_clean)
    y_noisy = y_clean.copy()
    mask = np.zeros(len(y_clean), dtype=bool)
    n_flip = _n_flip(len(y_clean), eta)
    if n_flip == 0:
        return y_noisy, mask, {}

    mapping = _nearest_centroid_target_map(X, y_clean)
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(y_clean), size=n_flip, replace=False)
    mask[chosen] = True
    for idx in chosen:
        y_noisy[idx] = mapping[y_clean[idx]]
    return y_noisy, mask, {
        str(k.item() if hasattr(k, "item") else k): (
            v.item() if hasattr(v, "item") else v
        )
        for k, v in mapping.items()
    }


def _boundary_scores(X: np.ndarray, y: np.ndarray):
    """Smaller score means closer to an opposite-class sample than own-class support.

    The corruption mechanism uses only the outer-training features and their clean
    synthetic-benchmark labels. These clean labels are never exposed to model tuning
    or audit scoring after corruption.
    """
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    n = len(Xs)
    scores = np.empty(n, dtype=float)
    nearest_rival_label = np.empty(n, dtype=y.dtype)

    # Diagnostic datasets are small enough for an explicit distance matrix.
    sq = np.sum(Xs * Xs, axis=1, keepdims=True)
    dist2 = np.maximum(sq + sq.T - 2.0 * (Xs @ Xs.T), 0.0)
    np.fill_diagonal(dist2, np.inf)

    for i in range(n):
        same = y == y[i]
        same[i] = False
        rival = y != y[i]
        d_same = float(np.sqrt(np.min(dist2[i, same]))) if np.any(same) else np.inf
        rival_idx = np.flatnonzero(rival)
        best = int(rival_idx[np.argmin(dist2[i, rival])])
        d_rival = float(np.sqrt(dist2[i, best]))
        # Low/negative values identify geometrically ambiguous boundary samples.
        scores[i] = d_rival - d_same
        nearest_rival_label[i] = y[best]
    return scores, nearest_rival_label


def corrupt_boundary_targeted(
    X: np.ndarray,
    y_clean: np.ndarray,
    eta: float,
    seed: int,
):
    del seed  # deterministic ranking once the outer split is fixed
    y_clean = np.asarray(y_clean)
    y_noisy = y_clean.copy()
    mask = np.zeros(len(y_clean), dtype=bool)
    n_flip = _n_flip(len(y_clean), eta)
    if n_flip == 0:
        return y_noisy, mask, None

    boundary_score, rival_label = _boundary_scores(X, y_clean)
    order = np.argsort(boundary_score, kind="stable")
    chosen = order[:n_flip]
    mask[chosen] = True
    y_noisy[chosen] = rival_label[chosen]
    return y_noisy, mask, {
        "selected_boundary_score_mean": float(np.mean(boundary_score[chosen])),
        "selected_boundary_score_median": float(np.median(boundary_score[chosen])),
        "unselected_boundary_score_mean": float(np.mean(boundary_score[~mask])),
    }


def corrupt(
    noise_type: str,
    X: np.ndarray,
    y_clean: np.ndarray,
    eta: float,
    seed: int,
):
    if noise_type == "symmetric":
        y_noisy, mask = corrupt_symmetric(y_clean, eta, seed)
        return y_noisy, mask, None
    if noise_type == "class_dependent":
        return corrupt_class_dependent(X, y_clean, eta, seed)
    if noise_type == "boundary_targeted":
        return corrupt_boundary_targeted(X, y_clean, eta, seed)
    raise ValueError(noise_type)


def tune_all(X_train, y_train_noisy, seed: int):
    grids = {
        "mcgc": v18.MCGC_GRID,
        "label_spreading": v18.LABEL_SPREADING_GRID,
        "knn": v18.KNN_GRID,
        "svm": v18.SVM_GRID,
    }
    selected_params = {}
    tuning = {}
    for kind in METHODS:
        selected, records = v18.tune_model(
            kind,
            grids[kind],
            X_train,
            y_train_noisy,
            seed,
        )
        selected_params[kind] = selected["params"]
        tuning[kind] = {
            "selected": selected,
            "all_candidates": records,
        }
    return selected_params, tuning


def evaluate_condition(
    dataset: str,
    noise_type: str,
    eta: float,
    seed: int,
):
    X, y = dataset_for(dataset)
    X_train, X_test, y_train_clean, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=5000 + seed,
        stratify=y,
    )

    type_offset = {
        "symmetric": 0,
        "class_dependent": 100000,
        "boundary_targeted": 200000,
    }[noise_type]
    noise_seed = 18000 + type_offset + 1000 * seed + int(round(100 * eta))
    y_train_noisy, corruption_mask, corruption_meta = corrupt(
        noise_type,
        X_train,
        y_train_clean,
        eta,
        noise_seed,
    )

    selected_params, tuning = tune_all(X_train, y_train_noisy, seed)
    audits = v18b.oof_audit_all(
        X_train,
        y_train_noisy,
        corruption_mask,
        selected_params,
        seed,
    )

    test_metrics = {}
    for kind in METHODS:
        fitted = v18.fit_selected(
            kind,
            selected_params[kind],
            X_train,
            y_train_noisy,
        )
        pred = fitted.predict(X_test)
        test_metrics[kind] = v18.classification_metrics(y_test, pred)

    return {
        "dataset": dataset,
        "noise_type": noise_type,
        "eta": float(eta),
        "noise_seed": int(noise_seed),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_corrupted": int(np.sum(corruption_mask)),
        "realized_corruption_fraction": float(np.mean(corruption_mask)),
        "corruption_meta": corruption_meta,
        "selected_params": selected_params,
        "tuning": tuning,
        "test_metrics": test_metrics,
        "audits": audits,
    }


def compact_summary(runs):
    rows = []
    for run in runs:
        row = {
            "dataset": run["dataset"],
            "noise_type": run["noise_type"],
            "eta": run["eta"],
            "n_corrupted": run["n_corrupted"],
        }
        for kind in METHODS:
            m = run["audits"][kind]["metrics"]
            row[f"{kind}_roc_auc"] = m["roc_auc"]
            row[f"{kind}_pr_auc"] = m["pr_auc"]
            row[f"{kind}_precision_at_k"] = m["precision_at_k"]
            row[f"{kind}_test_accuracy"] = run["test_metrics"][kind]["accuracy"]
        rows.append(row)
    return rows


def print_summary(runs):
    print("=== MCGC v1.8c HARD LABEL-NOISE DIAGNOSTIC ===")
    for run in runs:
        print(
            f"{run['dataset']:14s} {run['noise_type']:17s} eta={run['eta']:.2f} "
            f"K={run['n_corrupted']}"
        )
        for kind in METHODS:
            m = run["audits"][kind]["metrics"]
            acc = run["test_metrics"][kind]["accuracy"]
            print(
                f"  {kind:16s} ROC={m['roc_auc']:.4f} PR={m['pr_auc']:.4f} "
                f"P@K={m['precision_at_k']:.4f} TEST_ACC={acc:.4f}"
            )
        print("  SELECTED", json.dumps(run["selected_params"], sort_keys=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("v18c_hard_label_noise_diagnostic.json"),
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        choices=DATASETS,
        default=list(DATASETS),
    )
    parser.add_argument(
        "--noise-types",
        nargs="*",
        choices=NOISE_TYPES,
        default=list(NOISE_TYPES),
    )
    parser.add_argument(
        "--noise-levels",
        nargs="*",
        type=float,
        default=list(NOISE_LEVELS),
    )
    args = parser.parse_args()

    runs = []
    for dataset in args.datasets:
        for noise_type in args.noise_types:
            for eta in args.noise_levels:
                runs.append(evaluate_condition(dataset, noise_type, eta, SEED))

    result = {
        "protocol": {
            "version": "1.8c-diagnostic",
            "status": "hard-noise diagnostic only; not confirmatory evidence",
            "seed": SEED,
            "datasets": list(args.datasets),
            "noise_types": list(args.noise_types),
            "noise_levels": [float(v) for v in args.noise_levels],
            "outer_split": "70% train / 30% untouched clean test",
            "tuning": "2-fold stratified CV on noisy outer-training labels only, same grids as v1.8",
            "audit": "3-fold OOF; audited sample excluded from the fitted audit model",
            "primary_diagnostic_metrics": ["PR-AUC", "precision@k"],
            "secondary_diagnostic_metric": "ROC-AUC",
            "noise_definitions": {
                "symmetric": "random samples flipped uniformly to a different class",
                "class_dependent": "random samples flipped to the nearest other class centroid in standardized outer-training space",
                "boundary_targeted": "samples with the smallest nearest-rival minus nearest-same-class distance margin are flipped to their nearest rival label",
            },
            "boundary_noise_note": "clean outer-training labels are used only to generate the synthetic corruption mechanism; they are not exposed to tuning or audit scoring after corruption",
        },
        "runs": runs,
        "compact_summary": compact_summary(runs),
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print_summary(runs)
    print(f"WROTE {args.output}")


if __name__ == "__main__":
    main()
