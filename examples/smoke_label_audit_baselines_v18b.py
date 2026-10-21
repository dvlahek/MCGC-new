from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
from sklearn.datasets import load_digits
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split


ROOT = Path(__file__).resolve().parents[1]
V18 = ROOT / "examples" / "smoke_label_noise_v18.py"


def _load_v18():
    spec = importlib.util.spec_from_file_location("smoke_label_noise_v18", V18)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


v18 = _load_v18()

SEED = v18.SMOKE_SEED
ETA = 0.30
AUDIT_FOLDS = v18.AUDIT_FOLDS
METHODS = ("mcgc", "label_spreading", "knn", "svm")


def audit_metrics(mask: np.ndarray, scores: np.ndarray):
    mask = np.asarray(mask, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    k = int(mask.sum())
    order = np.argsort(-scores, kind="stable")
    top = order[:k]
    return {
        "roc_auc": float(roc_auc_score(mask.astype(int), scores)),
        "pr_auc": float(average_precision_score(mask.astype(int), scores)),
        "precision_at_k": float(np.mean(mask[top])),
        "k": k,
    }


def probability_conflict(model, X_val, y_observed):
    probs = np.asarray(model.predict_proba(X_val), dtype=float)
    classes = np.asarray(model.classes_)
    out = np.empty(len(X_val), dtype=float)
    suggested = np.empty(len(X_val), dtype=y_observed.dtype)

    for i, observed in enumerate(y_observed):
        obs_idx = int(np.flatnonzero(classes == observed)[0])
        rival_idx = np.arange(len(classes)) != obs_idx
        rival_candidates = np.flatnonzero(rival_idx)
        best_rival_idx = int(rival_candidates[np.argmax(probs[i, rival_idx])])
        out[i] = float(probs[i, best_rival_idx] - probs[i, obs_idx])
        suggested[i] = classes[best_rival_idx]
    return out, suggested


def svm_conflict(model, X_val, y_observed):
    decision = np.asarray(model.decision_function(X_val), dtype=float)
    classes = np.asarray(model.classes_)
    if decision.ndim == 1:
        decision = np.column_stack([-decision, decision])
    out = np.empty(len(X_val), dtype=float)
    suggested = np.empty(len(X_val), dtype=y_observed.dtype)

    for i, observed in enumerate(y_observed):
        obs_idx = int(np.flatnonzero(classes == observed)[0])
        rival_mask = np.arange(len(classes)) != obs_idx
        rival_candidates = np.flatnonzero(rival_mask)
        best_rival_idx = int(rival_candidates[np.argmax(decision[i, rival_mask])])
        out[i] = float(decision[i, best_rival_idx] - decision[i, obs_idx])
        suggested[i] = classes[best_rival_idx]
    return out, suggested


def mcgc_conflict(model, X_val, y_observed):
    _, diagnostics = model.predict_with_diagnostics(X_val)
    out = np.empty(len(X_val), dtype=float)
    suggested = np.empty(len(X_val), dtype=y_observed.dtype)

    for i, (observed, row) in enumerate(zip(y_observed, diagnostics)):
        evidence = row["local_evidence"]
        observed_py = observed.item() if hasattr(observed, "item") else observed
        observed_e = float(v18._lookup(evidence, observed))
        rivals = [
            (label, float(value))
            for label, value in evidence.items()
            if label != observed_py
        ]
        rival_label, rival_e = max(rivals, key=lambda item: item[1])
        out[i] = float(rival_e - observed_e)
        suggested[i] = rival_label
    return out, suggested


def fit_method(kind: str, params: dict, X_fit, y_fit):
    return v18.fit_selected(kind, params, X_fit, y_fit)


def oof_audit_all(X, y_noisy, corruption_mask, selected_params, seed: int):
    cv = StratifiedKFold(
        n_splits=AUDIT_FOLDS,
        shuffle=True,
        random_state=20000 + seed,
    )
    score_map = {name: np.empty(len(y_noisy), dtype=float) for name in METHODS}
    suggestion_map = {
        name: np.empty(len(y_noisy), dtype=y_noisy.dtype) for name in METHODS
    }
    seen = np.zeros(len(y_noisy), dtype=bool)

    for fit_idx, val_idx in cv.split(X, y_noisy):
        for kind in METHODS:
            model = fit_method(
                kind,
                selected_params[kind],
                X[fit_idx],
                y_noisy[fit_idx],
            )
            observed = y_noisy[val_idx]
            if kind == "mcgc":
                scores, suggested = mcgc_conflict(model, X[val_idx], observed)
            elif kind in ("label_spreading", "knn"):
                scores, suggested = probability_conflict(model, X[val_idx], observed)
            elif kind == "svm":
                scores, suggested = svm_conflict(model, X[val_idx], observed)
            else:
                raise ValueError(kind)
            score_map[kind][val_idx] = scores
            suggestion_map[kind][val_idx] = suggested
        seen[val_idx] = True

    if not np.all(seen):
        raise RuntimeError("OOF audit did not cover all training samples")

    return {
        kind: {
            "metrics": audit_metrics(corruption_mask, score_map[kind]),
            "scores": score_map[kind].tolist(),
            "suggested_labels": suggestion_map[kind].tolist(),
        }
        for kind in METHODS
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("v18b_label_audit_baseline_smoke.json"),
    )
    args = parser.parse_args()

    data = load_digits()
    X = np.asarray(data.data)
    y = np.asarray(data.target)
    X_train, _, y_train_clean, _ = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=5000 + SEED,
        stratify=y,
    )

    noise_seed = 18000 + 1000 * SEED + int(round(100 * ETA))
    y_train_noisy, corruption_mask = v18.corrupt_labels(
        y_train_clean,
        ETA,
        noise_seed,
    )

    grids = {
        "mcgc": v18.MCGC_GRID,
        "label_spreading": v18.LABEL_SPREADING_GRID,
        "knn": v18.KNN_GRID,
        "svm": v18.SVM_GRID,
    }
    tuning = {}
    selected_params = {}
    for kind in METHODS:
        selected, records = v18.tune_model(
            kind,
            grids[kind],
            X_train,
            y_train_noisy,
            SEED,
        )
        tuning[kind] = {
            "selected": selected,
            "all_candidates": records,
        }
        selected_params[kind] = selected["params"]

    audits = oof_audit_all(
        X_train,
        y_train_noisy,
        corruption_mask,
        selected_params,
        SEED,
    )

    result = {
        "protocol": {
            "version": "1.8b-smoke",
            "status": "diagnostic audit-baseline smoke only; not confirmatory evidence",
            "dataset": "digits",
            "seed": SEED,
            "eta": ETA,
            "noise_seed": noise_seed,
            "n_train": int(len(X_train)),
            "n_corrupted": int(corruption_mask.sum()),
            "audit_folds": AUDIT_FOLDS,
            "audit_cv_seed": 20000 + SEED,
            "tuning": "same 2-fold noisy-label CV and grids as v1.8 smoke",
            "comparison_rule": "all audit scores are OOF; audited sample excluded from fitted model",
            "score_definitions": {
                "mcgc": "best rival local contrastive evidence minus observed-label local evidence",
                "label_spreading": "best rival predicted probability minus observed-label probability",
                "knn": "best rival predicted probability minus observed-label probability",
                "svm": "best rival OVR decision score minus observed-label decision score",
            },
        },
        "selected_params": selected_params,
        "tuning": tuning,
        "audits": audits,
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")

    print("=== MCGC v1.8b LABEL-AUDIT BASELINE SMOKE ===")
    for kind in METHODS:
        m = audits[kind]["metrics"]
        print(
            f"{kind:16s} ROC_AUC={m['roc_auc']:.4f} "
            f"PR_AUC={m['pr_auc']:.4f} P@K={m['precision_at_k']:.4f} K={m['k']}"
        )
    print("SELECTED", json.dumps(selected_params, sort_keys=True))
    print(f"WROTE {args.output}")


if __name__ == "__main__":
    main()
