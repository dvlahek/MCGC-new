#!/usr/bin/env python3
"""Reviewer audit: strict fold-local preprocessing for label review.

Re-runs the manuscript label-review protocol while ensuring that every
preprocessing transform used by probability-based baselines is fitted inside
its corresponding OOF training fold. CGEC already fits its StandardScaler
inside each fold through model.fit().

This script does not overwrite manuscript results. It writes a separate v3.1
audit JSON that can be compared with the currently reported v2.1 values.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd
from cleanlab.rank import get_label_quality_scores
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.semi_supervised import LabelSpreading

from mcgc import MultiAgentChemotacticGraphClassifier

DATASETS = (
    "Hill_Valley_without_noise",
    "bupa",
    "ecoli",
    "mfeat_fourier",
    "mfeat_karhunen",
    "mfeat_zernike",
    "saheart",
    "schizo",
    "spectf",
    "vehicle",
    "wine_quality_red",
    "yeast",
)
CONDITIONS = (
    ("symmetric", 0.10),
    ("symmetric", 0.20),
    ("symmetric", 0.30),
    ("pairflip", 0.20),
)
SEEDS = tuple(range(5))
BUDGETS = (0.05, 0.10)
AUDIT_FOLDS = 5
AUDIT_K = 12
AUDIT_ALPHA = 0.85
MIN_CLASS = 6

METHODS = (
    "cgec_oof",
    "rf_confidence_oof",
    "lr_confidence_oof",
    "cleanlab_lr_oof",
    "knn_disagreement_oof",
    "label_spreading_oof",
)
RIVAL_METHODS = (
    "cgec_oof",
    "rf_confidence_oof",
    "lr_confidence_oof",
    "lr_rival_margin_oof",
    "knn_disagreement_oof",
    "label_spreading_oof",
)


def load_dataset(data_dir: Path, name: str):
    path = data_dir / f"{name}.tsv.gz"
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset: {path}")
    df = pd.read_csv(path, sep="\t")
    y_raw = df.pop("target").to_numpy()
    X = df.to_numpy(dtype=float)
    vals, counts = np.unique(y_raw, return_counts=True)
    keep = np.isin(y_raw, vals[counts >= MIN_CLASS])
    kept_labels = np.unique(y_raw[keep])
    y = np.searchsorted(kept_labels, y_raw[keep])
    return X[keep], y, {
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "dropped_small_class_rows": int((~keep).sum()),
    }


def corrupt(y, n_classes, kind, eta, seed):
    rng = np.random.default_rng(
        21000 + 100 * seed + int(round(100 * eta))
        + (50 if kind == "pairflip" else 0)
    )
    flip = rng.random(len(y)) < eta
    yn = y.copy()
    for i in np.flatnonzero(flip):
        if kind == "pairflip":
            yn[i] = (y[i] + 1) % n_classes
        else:
            choices = [c for c in range(n_classes) if c != y[i]]
            yn[i] = rng.choice(choices)
    return yn, flip


def make_cv(seed):
    return StratifiedKFold(AUDIT_FOLDS, shuffle=True, random_state=22000 + seed)


def expand_proba(P, estimator_classes, n_classes):
    P = np.asarray(P, dtype=float)
    if P.shape[1] == n_classes and np.array_equal(np.asarray(estimator_classes), np.arange(n_classes)):
        return P
    full = np.zeros((len(P), n_classes), dtype=float)
    full[:, np.asarray(estimator_classes, dtype=int)] = P
    return full


def strict_oof_proba(estimator, X, yn, n_classes, seed):
    """OOF probabilities with all estimator preprocessing inside the fold."""
    cv = make_cv(seed)
    P = np.zeros((len(yn), n_classes), dtype=float)
    seen = np.zeros(len(yn), dtype=bool)
    for fit_idx, val_idx in cv.split(X, yn):
        model = estimator()
        model.fit(X[fit_idx], yn[fit_idx])
        fold_p = np.asarray(model.predict_proba(X[val_idx]), dtype=float)
        P[val_idx] = expand_proba(fold_p, model.classes_, n_classes)
        seen[val_idx] = True
    if not np.all(seen):
        raise RuntimeError("OOF probability scoring did not cover all samples")
    return P


def cgec_oof_score_and_rival(X, yn, seed):
    cv = make_cv(seed)
    scores = np.empty(len(yn), dtype=float)
    rivals = np.empty(len(yn), dtype=int)
    seen = np.zeros(len(yn), dtype=bool)
    for fit_idx, val_idx in cv.split(X, yn):
        model = MultiAgentChemotacticGraphClassifier(
            n_neighbors=AUDIT_K,
            diffusion_alpha=AUDIT_ALPHA,
            diffusion_steps=40,
            use_walkers=False,
            random_state=17,
        ).fit(X[fit_idx], yn[fit_idx])
        _, diagnostics = model.predict_with_diagnostics(X[val_idx])
        for global_idx, row in zip(val_idx, diagnostics):
            observed = int(yn[global_idx])
            evidence = {int(k): float(v) for k, v in row["local_evidence"].items()}
            observed_e = evidence.get(observed, 0.0)
            alternatives = [(c, v) for c, v in evidence.items() if c != observed]
            if alternatives:
                rival, rival_e = max(alternatives, key=lambda z: (z[1], -z[0]))
            else:
                rival, rival_e = observed, observed_e
            scores[global_idx] = float(rival_e - observed_e)
            rivals[global_idx] = int(rival)
            seen[global_idx] = True
    if not np.all(seen):
        raise RuntimeError("CGEC OOF scoring did not cover all samples")
    return scores, rivals


def confidence_deficit(P, y):
    return 1.0 - P[np.arange(len(y)), y]


def rival_margin(P, y):
    alt = P.copy()
    alt[np.arange(len(y)), y] = -np.inf
    rival = alt.argmax(axis=1).astype(int)
    score = alt[np.arange(len(y)), rival] - P[np.arange(len(y)), y]
    return score, rival


def best_alternative(P, y):
    alt = P.copy()
    alt[np.arange(len(y)), y] = -np.inf
    return alt.argmax(axis=1).astype(int)


def cleanlab_score(P, y):
    quality = get_label_quality_scores(
        labels=np.asarray(y, dtype=int),
        pred_probs=np.asarray(P, dtype=float),
        method="normalized_margin",
        adjust_pred_probs=False,
    )
    return 1.0 - np.asarray(quality, dtype=float)


def audit_metrics(flip, score):
    flip = np.asarray(flip, dtype=bool)
    score = np.asarray(score, dtype=float)
    order = np.argsort(-score, kind="stable")
    n = len(flip)
    out = {
        "roc_auc": float(roc_auc_score(flip.astype(int), score)),
        "pr_auc": float(average_precision_score(flip.astype(int), score)),
        "review": {},
    }
    for budget in BUDGETS:
        k = max(1, int(round(budget * n)))
        idx = order[:k]
        out["review"][str(budget)] = {
            "precision": float(np.mean(flip[idx])),
            "reviewed_count": int(k),
            "corrupt_found": int(np.sum(flip[idx])),
        }
    return out


def recovery_counts(clean_y, flip, score, rival):
    clean_y = np.asarray(clean_y, dtype=int)
    flip = np.asarray(flip, dtype=bool)
    score = np.asarray(score, dtype=float)
    rival = np.asarray(rival, dtype=int)
    order = np.argsort(-score, kind="stable")

    def pack(mask):
        idx = np.flatnonzero(mask)
        correct = int(np.sum(rival[idx] == clean_y[idx])) if len(idx) else 0
        return {
            "correct": correct,
            "total": int(len(idx)),
            "accuracy": float(correct / len(idx)) if len(idx) else None,
        }

    out = {"all_corrupt": pack(flip)}
    for budget in BUDGETS:
        k = max(1, int(round(budget * len(flip))))
        sel = np.zeros(len(flip), dtype=bool)
        sel[order[:k]] = True
        out[f"top_{int(round(100*budget))}pct_corrupt"] = pack(flip & sel)
    return out


def sign_p(wins, losses):
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    tail = sum(comb(n, i) for i in range(k + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


def run_unit(X, y, meta, name, kind, eta, seed):
    n_classes = int(y.max()) + 1
    Xtr, _, ytr, _ = train_test_split(
        X, y, test_size=0.30, random_state=20000 + seed, stratify=y
    )
    yn, flip = corrupt(ytr, n_classes, kind, eta, seed)
    if flip.sum() == 0 or flip.all():
        return {"status": "degenerate_noise", "dataset": name, "noise": kind, "eta": eta, "seed": seed}

    cgec_score, cgec_rival = cgec_oof_score_and_rival(Xtr, yn, seed)
    P_rf = strict_oof_proba(
        lambda: RandomForestClassifier(n_estimators=200, random_state=97, n_jobs=2),
        Xtr, yn, n_classes, seed,
    )
    P_lr = strict_oof_proba(
        lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000)),
        Xtr, yn, n_classes, seed,
    )
    P_knn = strict_oof_proba(
        lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(AUDIT_K, weights="distance")),
        Xtr, yn, n_classes, seed,
    )
    P_ls = strict_oof_proba(
        lambda: make_pipeline(
            StandardScaler(),
            LabelSpreading(kernel="knn", n_neighbors=AUDIT_K, alpha=AUDIT_ALPHA, max_iter=1000),
        ),
        Xtr, yn, n_classes, seed,
    )

    lr_rival_score, lr_rival = rival_margin(P_lr, yn)
    scores = {
        "cgec_oof": cgec_score,
        "rf_confidence_oof": confidence_deficit(P_rf, yn),
        "lr_confidence_oof": confidence_deficit(P_lr, yn),
        "cleanlab_lr_oof": cleanlab_score(P_lr, yn),
        "knn_disagreement_oof": confidence_deficit(P_knn, yn),
        "label_spreading_oof": confidence_deficit(P_ls, yn),
    }
    rivals = {
        "cgec_oof": cgec_rival,
        "rf_confidence_oof": best_alternative(P_rf, yn),
        "lr_confidence_oof": best_alternative(P_lr, yn),
        "lr_rival_margin_oof": lr_rival,
        "knn_disagreement_oof": best_alternative(P_knn, yn),
        "label_spreading_oof": best_alternative(P_ls, yn),
    }
    rival_scores = {
        "cgec_oof": cgec_score,
        "rf_confidence_oof": scores["rf_confidence_oof"],
        "lr_confidence_oof": scores["lr_confidence_oof"],
        "lr_rival_margin_oof": lr_rival_score,
        "knn_disagreement_oof": scores["knn_disagreement_oof"],
        "label_spreading_oof": scores["label_spreading_oof"],
    }

    return {
        "status": "ok",
        "dataset": name,
        "noise": kind,
        "eta": eta,
        "seed": seed,
        "n_classes": n_classes,
        "n_train": int(len(yn)),
        "n_corrupt": int(flip.sum()),
        "meta": meta,
        "audit": {m: audit_metrics(flip, s) for m, s in scores.items()},
        "rival_recovery": {
            m: recovery_counts(ytr, flip, rival_scores[m], rivals[m])
            for m in RIVAL_METHODS
        },
    }


def dataset_level(records, names):
    ok = [r for r in records if r.get("status") == "ok"]
    rows = []
    for name in names:
        group = [r for r in ok if r["dataset"] == name]
        if not group:
            continue
        row = {"dataset": name, "n_units": len(group), "methods": {}, "rival": {}}
        for m in METHODS:
            row["methods"][m] = {
                "roc_auc": float(np.mean([r["audit"][m]["roc_auc"] for r in group])),
                "pr_auc": float(np.mean([r["audit"][m]["pr_auc"] for r in group])),
                "p5": float(np.mean([r["audit"][m]["review"]["0.05"]["precision"] for r in group])),
                "p10": float(np.mean([r["audit"][m]["review"]["0.1"]["precision"] for r in group])),
            }
        for m in RIVAL_METHODS:
            row["rival"][m] = {}
            for key in ("all_corrupt", "top_5pct_corrupt", "top_10pct_corrupt"):
                corr = sum(r["rival_recovery"][m][key]["correct"] for r in group)
                total = sum(r["rival_recovery"][m][key]["total"] for r in group)
                row["rival"][m][key] = float(corr / total) if total else None
        rows.append(row)

    overall = {
        m: {
            k: float(np.mean([r["methods"][m][k] for r in rows]))
            for k in ("roc_auc", "pr_auc", "p5", "p10")
        }
        for m in METHODS
    }
    rival_overall = {
        m: {
            key: float(np.mean([r["rival"][m][key] for r in rows if r["rival"][m][key] is not None]))
            for key in ("all_corrupt", "top_5pct_corrupt", "top_10pct_corrupt")
        }
        for m in RIVAL_METHODS
    }

    comparisons = {}
    for metric in ("roc_auc", "pr_auc", "p5", "p10"):
        comparisons[metric] = {}
        for baseline in METHODS[1:]:
            diffs = [r["methods"]["cgec_oof"][metric] - r["methods"][baseline][metric] for r in rows]
            w = sum(d > 1e-12 for d in diffs)
            l = sum(d < -1e-12 for d in diffs)
            comparisons[metric][baseline] = {
                "mean_delta": float(np.mean(diffs)),
                "wins": int(w),
                "ties": int(len(diffs)-w-l),
                "losses": int(l),
                "p": float(sign_p(w, l)),
            }
    return {"dataset_level": rows, "overall": overall, "rival_overall": rival_overall, "comparisons": comparisons}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=Path("paper_reproducibility/results/v31_strict_oof_scaling_audit.json"))
    ap.add_argument("--only", default=None, help="Comma-separated dataset subset for smoke testing")
    ap.add_argument("--seeds", default=None, help="Comma-separated seed subset for smoke testing")
    args = ap.parse_args()

    names = tuple(args.only.split(",")) if args.only else DATASETS
    seeds = tuple(int(s) for s in args.seeds.split(",")) if args.seeds else SEEDS

    records = []
    for name in names:
        X, y, meta = load_dataset(args.data_dir, name)
        for seed in seeds:
            for kind, eta in CONDITIONS:
                print(f"{name:28s} {kind:9s} eta={eta:.2f} seed={seed}", flush=True)
                records.append(run_unit(X, y, meta, name, kind, eta, seed))

    result = {
        "protocol": {
            "version": "v3.1-reviewer-oof-scaling",
            "purpose": "strict fold-local preprocessing audit for label review",
            "datasets": list(names),
            "seeds": list(seeds),
            "conditions": list(CONDITIONS),
            "outer_split": "70/30 stratified, random_state=20000+seed",
            "corruption_order": "outer training split first; corruption applied only to outer-training labels; then five-fold OOF scoring",
            "oof_preprocessing": "all StandardScaler fits occur inside the corresponding OOF training fold",
            "cgec": "K=12, alpha=0.85, diffusion_steps=40, walkers disabled",
            "primary_inference_unit": "dataset",
        },
        "records": records,
        "analysis": dataset_level(records, names),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")

    print("\n=== STRICT OOF SCALING AUDIT: TABLE-4 STYLE MEANS ===")
    for m, vals in result["analysis"]["overall"].items():
        print(f"{m:26s} ROC={vals['roc_auc']:.4f} PR={vals['pr_auc']:.4f} P5={vals['p5']:.4f} P10={vals['p10']:.4f}")
    print("\n=== RIVAL RECOVERY: EQUAL-DATASET MEANS ===")
    for m, vals in result["analysis"]["rival_overall"].items():
        print(f"{m:26s} all={vals['all_corrupt']:.4f} top5={vals['top_5pct_corrupt']:.4f} top10={vals['top_10pct_corrupt']:.4f}")
    print(f"\nFULL_JSON={args.output.resolve()}")


if __name__ == "__main__":
    main()
