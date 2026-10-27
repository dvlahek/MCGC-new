#!/usr/bin/env python3
"""Frozen v2.1 OOF LR rival-margin control for CGEC Table 5.

This companion diagnostic evaluates the fairness control requested for the
rival-class recovery analysis. It uses the exact frozen v2.1 outer split,
corruption process, preprocessing, LR estimator, and strict five-fold OOF
folds, but only on the eight multiclass PMLB datasets.

For each held-out training sample:
  rival = argmax_{c != y_obs} p_c
  score = p_rival - p_y_obs

The score is monotone-equivalent to ranking by low Cleanlab normalized-margin
label quality when adjust_pred_probs=False, while using the same rival class
that is evaluated for recovery.

Outputs:
  - JSONL with one record per dataset x corruption condition x seed (160 units)
  - analysis JSON with equal-dataset recovery means and pooled true-corrupt
    counts entering the top-5% and top-10% review budgets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.preprocessing import StandardScaler

DATASETS = (
    "ecoli",
    "mfeat_fourier",
    "mfeat_karhunen",
    "mfeat_zernike",
    "schizo",
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
MIN_CLASS = 6


def load_dataset(data_dir: Path, name: str):
    path = data_dir / f"{name}.tsv.gz"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, sep="\t")
    y_raw = df.pop("target").to_numpy()
    X = df.to_numpy(dtype=float)

    vals, counts = np.unique(y_raw, return_counts=True)
    keep = np.isin(y_raw, vals[counts >= MIN_CLASS])
    kept_labels = np.unique(y_raw[keep])
    y = np.searchsorted(kept_labels, y_raw[keep])

    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    return X[keep], y, {"sha256": sha, "sha256_16": sha[:16]}


def corrupt(y, n_classes, kind, eta, seed):
    rng = np.random.default_rng(
        21000
        + 100 * seed
        + int(round(100 * eta))
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
    return StratifiedKFold(
        AUDIT_FOLDS,
        shuffle=True,
        random_state=22000 + seed,
    )


def oof_lr_probabilities(Xtr, yn, n_classes, seed):
    # Frozen v2.1 preprocessing: scaler is fitted once on the outer training pool.
    scaler = StandardScaler().fit(Xtr)
    Xs = scaler.transform(Xtr)
    P = np.asarray(
        cross_val_predict(
            LogisticRegression(C=1.0, max_iter=3000),
            Xs,
            yn,
            cv=make_cv(seed),
            method="predict_proba",
        ),
        dtype=float,
    )

    if P.shape[1] != n_classes:
        present = np.unique(yn)
        full = np.zeros((len(yn), n_classes), dtype=float)
        full[:, present] = P
        P = full
    return P


def rival_margin(P, observed):
    S = np.asarray(P, dtype=float).copy()
    S[np.arange(len(observed)), observed] = -np.inf
    rival = S.argmax(axis=1).astype(int)
    p_rival = P[np.arange(len(observed)), rival]
    p_observed = P[np.arange(len(observed)), observed]
    score = p_rival - p_observed
    return score, rival


def recovery(clean_y, flip, score, rival):
    clean_y = np.asarray(clean_y, dtype=int)
    flip = np.asarray(flip, dtype=bool)
    order = np.argsort(-np.asarray(score, dtype=float), kind="stable")

    def pack(mask):
        idx = np.flatnonzero(mask)
        total = int(len(idx))
        correct = int(np.sum(rival[idx] == clean_y[idx])) if total else 0
        return {
            "correct": correct,
            "total": total,
            "accuracy": float(correct / total) if total else None,
        }

    out = {"all_corrupt": pack(flip)}
    n = len(flip)
    for b in BUDGETS:
        k = max(1, int(round(b * n)))
        selected = np.zeros(n, dtype=bool)
        selected[order[:k]] = True
        key = f"top_{int(round(100*b))}pct_corrupt"
        out[key] = pack(flip & selected)
        out[key]["reviewed_count"] = int(k)
    return out


def run_unit(X, y, meta, name, kind, eta, seed):
    n_classes = int(y.max()) + 1
    Xtr, _, ytr, _ = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=20000 + seed,
        stratify=y,
    )
    yn, flip = corrupt(ytr, n_classes, kind, eta, seed)
    P = oof_lr_probabilities(Xtr, yn, n_classes, seed)
    score, rival = rival_margin(P, yn)

    return {
        "dataset": name,
        "noise": kind,
        "eta": eta,
        "seed": seed,
        "n_classes": n_classes,
        "n_train": int(len(yn)),
        "n_corrupt": int(flip.sum()),
        "meta": meta,
        "method": "lr_rival_margin_oof",
        "score_definition": "max_{c!=y} p_c - p_y",
        "rival_definition": "argmax_{c!=y} p_c",
        "rival_recovery": recovery(ytr, flip, score, rival),
        "status": "ok",
    }


KEYS = ("all_corrupt", "top_5pct_corrupt", "top_10pct_corrupt")


def analyze(records):
    rows = []
    for ds in DATASETS:
        group = [r for r in records if r["dataset"] == ds and r.get("status") == "ok"]
        row = {"dataset": ds, "n_units": len(group)}
        for key in KEYS:
            correct = sum(r["rival_recovery"][key]["correct"] for r in group)
            total = sum(r["rival_recovery"][key]["total"] for r in group)
            row[key] = {
                "correct": int(correct),
                "total": int(total),
                "accuracy": float(correct / total) if total else None,
            }
        rows.append(row)

    overall = {}
    for key in KEYS:
        acc = [r[key]["accuracy"] for r in rows if r[key]["accuracy"] is not None]
        overall[key] = {
            "mean_dataset_accuracy": float(np.mean(acc)) if acc else None,
            "median_dataset_accuracy": float(np.median(acc)) if acc else None,
            "pooled_correct": int(sum(r[key]["correct"] for r in rows)),
            "pooled_total": int(sum(r[key]["total"] for r in rows)),
            "n_datasets_with_defined_accuracy": int(len(acc)),
        }

    return {
        "protocol": {
            "version": "v2.1-lr-rival-margin-control",
            "datasets": list(DATASETS),
            "conditions": [{"noise": k, "eta": e} for k, e in CONDITIONS],
            "seeds": list(SEEDS),
            "audit_folds": AUDIT_FOLDS,
            "score": "max_{c!=y} p_c - p_y",
            "rival": "argmax_{c!=y} p_c",
            "note": (
                "Same frozen v2.1 OOF LR probabilities/protocol. "
                "Ranking is monotone-equivalent to Cleanlab normalized-margin "
                "suspiciousness when probability adjustment is disabled."
            ),
        },
        "dataset_level": rows,
        "overall_equal_dataset_means": overall,
    }


def fmt_acc(value):
    return "NA" if value is None else f"{value:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--analysis-out", type=Path, default=None)
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    done = set()
    if args.out.exists():
        for line in args.out.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r["dataset"], r["noise"], r["eta"], r["seed"]))

    for ds in DATASETS:
        X, y, meta = load_dataset(args.data_dir, ds)
        for seed in SEEDS:
            for kind, eta in CONDITIONS:
                key = (ds, kind, eta, seed)
                if key in done:
                    continue
                rec = run_unit(X, y, meta, ds, kind, eta, seed)
                with args.out.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(rec) + "\n")
                rr = rec["rival_recovery"]
                print(
                    f"{ds:24s} {kind:9s} eta={eta:.2f} seed={seed} "
                    f"ALL={fmt_acc(rr['all_corrupt']['accuracy'])} "
                    f"TOP5={fmt_acc(rr['top_5pct_corrupt']['accuracy'])} "
                    f"TOP10={fmt_acc(rr['top_10pct_corrupt']['accuracy'])}",
                    flush=True,
                )

    records = [
        json.loads(line)
        for line in args.out.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    expected = len(DATASETS) * len(SEEDS) * len(CONDITIONS)
    unit_keys = {
        (r["dataset"], r["noise"], r["eta"], r["seed"])
        for r in records
        if r.get("status") == "ok"
    }
    if len(unit_keys) != expected:
        print(f"PARTIAL: {len(unit_keys)}/{expected} units; analysis not written")
        return

    analysis = analyze(records)
    analysis_out = args.analysis_out or args.out.with_name(args.out.stem + "_analysis.json")
    analysis_out.write_text(json.dumps(analysis, indent=2, sort_keys=True), encoding="utf-8")

    print("\n=== LR RIVAL-MARGIN CONTROL, 8 MULTICLASS DATASETS ===")
    for key in KEYS:
        r = analysis["overall_equal_dataset_means"][key]
        print(
            f"{key:18s} mean={fmt_acc(r['mean_dataset_accuracy'])} "
            f"pooled_n={r['pooled_total']} pooled_correct={r['pooled_correct']}"
        )
    print(f"WROTE {analysis_out}")
    print(f"WROTE {args.out}")


if __name__ == "__main__":
    main()
