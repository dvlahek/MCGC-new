#!/usr/bin/env python3
"""CGEC v2.1 strict-OOF rival-class recovery diagnostic.

Purpose
-------
Quantify the explanatory usefulness of the suggested rival class for label review.

For every truly corrupted training example, and separately for truly corrupted
examples that appear in the top 5% / top 10% review lists, measure whether the
method's suggested alternative class recovers the hidden clean class.

Frozen v2.1 protocol
--------------------
- same 12 unseen PMLB datasets
- same 70/30 stratified outer split, random_state = 20000 + seed
- same corruption RNG and four corruption conditions
- same five seeds
- same strict five-fold OOF folds, random_state = 22000 + seed
- CGEC K=12, alpha=0.85, diffusion_steps=40, walkers disabled
- RF: 200 trees, random_state=97
- LR: StandardScaler + LogisticRegression(C=1.0, max_iter=3000)
- kNN: StandardScaler + KNeighborsClassifier(K=12, weights="distance")
- Label Spreading: StandardScaler + KNN kernel, K=12, alpha=0.85
- primary inference unit: dataset

Important
---------
Cleanlab normalized-margin and find_label_issues in the manuscript use the same
strict-OOF LR probability matrix. If a Cleanlab rival class is defined as the
highest-probability alternative class, it is therefore identical to the LR rival
class and is not reported as a redundant extra method here.

Outputs
-------
JSONL: one record per dataset x condition x seed.
Analysis JSON: dataset-level pooled rival-recovery rates, equal-dataset means,
and exact sign tests comparing CGEC with RF and LR.

Usage
-----
python examples/benchmark_label_audit_v21_rival_recovery.py \
    --data-dir data_pmlb \
    --out results/v21_rival_recovery.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import warnings
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.semi_supervised import LabelSpreading

from mcgc import MultiAgentChemotacticGraphClassifier

warnings.filterwarnings("ignore")

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
AUDIT_ALPHA = 0.85
AUDIT_K = 12
MIN_CLASS = 6

METHODS = (
    "cgec_oof",
    "rf_confidence_oof",
    "lr_confidence_oof",
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

    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    return X[keep], y, {
        "sha256": sha,
        "sha256_16": sha[:16],
        "dropped_small_class_rows": int((~keep).sum()),
    }


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


def oof_proba(estimator, X, yn, n_classes, seed):
    P = np.asarray(
        cross_val_predict(
            estimator,
            X,
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


def best_alternative(P, observed_labels):
    S = np.asarray(P, dtype=float).copy()
    S[np.arange(len(observed_labels)), observed_labels] = -np.inf
    return S.argmax(axis=1).astype(int)


def confidence_deficit(P, observed_labels):
    return 1.0 - P[np.arange(len(observed_labels)), observed_labels]


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
            evidence = {
                int(label): float(value)
                for label, value in row["local_evidence"].items()
            }

            observed_e = evidence.get(observed, 0.0)
            alternatives = [
                (label, value)
                for label, value in evidence.items()
                if label != observed
            ]

            if alternatives:
                rival_label, rival_e = max(alternatives, key=lambda z: z[1])
            else:
                rival_label, rival_e = observed, observed_e

            scores[global_idx] = float(rival_e - observed_e)
            rivals[global_idx] = int(rival_label)
            seen[global_idx] = True

    if not np.all(seen):
        raise RuntimeError("CGEC OOF rival recovery did not cover all samples")

    return scores, rivals


def recovery_counts(clean_y, flip, score, rival):
    clean_y = np.asarray(clean_y, dtype=int)
    flip = np.asarray(flip, dtype=bool)
    score = np.asarray(score, dtype=float)
    rival = np.asarray(rival, dtype=int)

    order = np.argsort(-score, kind="stable")

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
    for budget in BUDGETS:
        k = max(1, int(round(budget * n)))
        selected = np.zeros(n, dtype=bool)
        selected[order[:k]] = True
        key = f"top_{int(round(100 * budget))}pct_corrupt"
        out[key] = pack(flip & selected)
        out[key]["reviewed_count"] = int(k)
        out[key]["true_corrupt_found"] = int(np.sum(flip & selected))

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

    rec = {
        "dataset": name,
        "noise": kind,
        "eta": eta,
        "seed": seed,
        "n_classes": n_classes,
        "n_train": int(len(yn)),
        "n_corrupt": int(flip.sum()),
        "meta": meta,
    }

    if flip.sum() == 0 or flip.all():
        rec["status"] = "degenerate_noise"
        return rec

    t0 = time.time()

    scaler = StandardScaler().fit(Xtr)
    Xs = scaler.transform(Xtr)

    s_cgec, r_cgec = cgec_oof_score_and_rival(Xtr, yn, seed)

    P_rf = oof_proba(
        RandomForestClassifier(
            n_estimators=200,
            random_state=97,
            n_jobs=2,
        ),
        Xtr,
        yn,
        n_classes,
        seed,
    )

    P_lr = oof_proba(
        LogisticRegression(
            C=1.0,
            max_iter=3000,
        ),
        Xs,
        yn,
        n_classes,
        seed,
    )

    P_knn = oof_proba(
        KNeighborsClassifier(
            AUDIT_K,
            weights="distance",
        ),
        Xs,
        yn,
        n_classes,
        seed,
    )

    P_ls = oof_proba(
        LabelSpreading(
            kernel="knn",
            n_neighbors=AUDIT_K,
            alpha=AUDIT_ALPHA,
            max_iter=1000,
        ),
        Xs,
        yn,
        n_classes,
        seed,
    )

    scores = {
        "cgec_oof": s_cgec,
        "rf_confidence_oof": confidence_deficit(P_rf, yn),
        "lr_confidence_oof": confidence_deficit(P_lr, yn),
        "knn_disagreement_oof": confidence_deficit(P_knn, yn),
        "label_spreading_oof": confidence_deficit(P_ls, yn),
    }
    rivals = {
        "cgec_oof": r_cgec,
        "rf_confidence_oof": best_alternative(P_rf, yn),
        "lr_confidence_oof": best_alternative(P_lr, yn),
        "knn_disagreement_oof": best_alternative(P_knn, yn),
        "label_spreading_oof": best_alternative(P_ls, yn),
    }

    rec["rival_recovery"] = {
        method: recovery_counts(
            clean_y=ytr,
            flip=flip,
            score=scores[method],
            rival=rivals[method],
        )
        for method in METHODS
    }

    rec["cleanlab_rival_note"] = (
        "Cleanlab uses the same strict-OOF LR probability matrix; "
        "highest-probability alternative class is therefore identical "
        "to lr_confidence_oof and is not duplicated."
    )

    rec["status"] = "ok"
    rec["seconds"] = round(time.time() - t0, 1)
    return rec


def sign_p(wins, losses):
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    tail = sum(comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


RECOVERY_KEYS = (
    "all_corrupt",
    "top_5pct_corrupt",
    "top_10pct_corrupt",
)


def pooled_rate(group, method, key):
    correct = sum(
        int(r["rival_recovery"][method][key]["correct"])
        for r in group
    )
    total = sum(
        int(r["rival_recovery"][method][key]["total"])
        for r in group
    )
    return {
        "correct": int(correct),
        "total": int(total),
        "accuracy": float(correct / total) if total else None,
    }


def dataset_level_analysis(records, names):
    ok = [r for r in records if r.get("status") == "ok"]
    rows = []

    for dataset in names:
        group = [r for r in ok if r["dataset"] == dataset]
        if not group:
            continue

        row = {
            "dataset": dataset,
            "n_units": len(group),
            "methods": {},
        }

        for method in METHODS:
            row["methods"][method] = {
                key: pooled_rate(group, method, key)
                for key in RECOVERY_KEYS
            }

        rows.append(row)

    overall = {}
    for method in METHODS:
        overall[method] = {}
        for key in RECOVERY_KEYS:
            vals = [
                r["methods"][method][key]["accuracy"]
                for r in rows
                if r["methods"][method][key]["accuracy"] is not None
            ]
            overall[method][key] = {
                "mean_dataset_accuracy": float(np.mean(vals)) if vals else None,
                "median_dataset_accuracy": float(np.median(vals)) if vals else None,
                "n_datasets": int(len(vals)),
            }

    comparisons = {}
    for key in RECOVERY_KEYS:
        comparisons[key] = {}
        for baseline in ("rf_confidence_oof", "lr_confidence_oof"):
            diffs = []
            for r in rows:
                a = r["methods"]["cgec_oof"][key]["accuracy"]
                b = r["methods"][baseline][key]["accuracy"]
                if a is not None and b is not None:
                    diffs.append(a - b)

            wins = sum(d > 1e-12 for d in diffs)
            losses = sum(d < -1e-12 for d in diffs)
            ties = len(diffs) - wins - losses

            comparisons[key][baseline] = {
                "mean_delta_cgec_minus_baseline": float(np.mean(diffs)) if diffs else None,
                "median_delta_cgec_minus_baseline": float(np.median(diffs)) if diffs else None,
                "dataset_wins": int(wins),
                "dataset_ties": int(ties),
                "dataset_losses": int(losses),
                "sign_test_p_two_sided": float(sign_p(wins, losses)),
                "n_datasets_compared": int(len(diffs)),
            }

    return {
        "dataset_level": rows,
        "overall_equal_dataset_means": overall,
        "comparisons": comparisons,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--analysis-out", type=Path, default=None)
    ap.add_argument("--only", default=None)
    ap.add_argument("--seeds", default=None)
    args = ap.parse_args()

    names = tuple(args.only.split(",")) if args.only else DATASETS
    seeds = (
        tuple(int(s) for s in args.seeds.split(","))
        if args.seeds
        else SEEDS
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)

    done = set()
    if args.out.exists():
        for line in args.out.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r["dataset"], r["noise"], r["eta"], r["seed"]))

    print("=== CGEC v2.1 STRICT-OOF RIVAL-CLASS RECOVERY ===")
    print(f"DATASETS={len(names)} SEEDS={seeds} CONDITIONS={len(CONDITIONS)}")
    print(f"METHODS={METHODS}")

    for name in names:
        X, y, meta = load_dataset(args.data_dir, name)

        for seed in seeds:
            for kind, eta in CONDITIONS:
                key = (name, kind, eta, seed)
                if key in done:
                    continue

                rec = run_unit(X, y, meta, name, kind, eta, seed)

                with args.out.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(rec) + "\n")

                if rec.get("status") == "ok":
                    c = rec["rival_recovery"]["cgec_oof"]
                    rf = rec["rival_recovery"]["rf_confidence_oof"]
                    lr = rec["rival_recovery"]["lr_confidence_oof"]
                    print(
                        f"{name:28s} {kind:9s} eta={eta:.2f} seed={seed} "
                        f"ALL CGEC={c['all_corrupt']['accuracy']:.3f} "
                        f"RF={rf['all_corrupt']['accuracy']:.3f} "
                        f"LR={lr['all_corrupt']['accuracy']:.3f} "
                        f"TOP5 CGEC={c['top_5pct_corrupt']['accuracy']!s}",
                        flush=True,
                    )

        print(f"DONE {name}", flush=True)

    records = [
        json.loads(line)
        for line in args.out.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    expected = {
        (name, kind, eta, seed)
        for name in names
        for seed in seeds
        for kind, eta in CONDITIONS
    }
    have = {
        (r["dataset"], r["noise"], r["eta"], r["seed"])
        for r in records
    }

    if not expected.issubset(have):
        print("PARTIAL RUN: analysis not written until all requested units are present.")
        print(f"WROTE {args.out}")
        return

    subset = [
        r
        for r in records
        if r["dataset"] in names
        and r["seed"] in seeds
        and (r["noise"], r["eta"]) in CONDITIONS
    ]

    analysis = {
        "protocol": {
            "version": "v2.1-rival-recovery",
            "datasets": list(names),
            "conditions": [
                {"noise": kind, "eta": eta}
                for kind, eta in CONDITIONS
            ],
            "seeds": list(seeds),
            "audit_folds": AUDIT_FOLDS,
            "methods": list(METHODS),
            "primary_inference_unit": "dataset",
            "dataset_aggregation": (
                "within each dataset, correct and total counts are pooled "
                "across its 20 condition-seed units; dataset rates are then "
                "compared as paired inferential units"
            ),
            "top_budget_semantics": (
                "accuracy of rival class among truly corrupted samples that "
                "fall inside that method's top review budget"
            ),
            "cleanlab_note": (
                "Cleanlab normalized-margin/find_label_issues use the same "
                "strict-OOF LR probability matrix; a highest-probability "
                "alternative class is therefore identical to LR."
            ),
        },
        **dataset_level_analysis(subset, names),
    }

    analysis_out = (
        args.analysis_out
        if args.analysis_out is not None
        else args.out.with_name(args.out.stem + "_analysis.json")
    )
    analysis_out.write_text(
        json.dumps(analysis, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print("\n=== EQUAL-DATASET MEAN RIVAL-CLASS RECOVERY ===")
    for method in METHODS:
        row = analysis["overall_equal_dataset_means"][method]
        print(
            f"{method:24s} "
            f"ALL={row['all_corrupt']['mean_dataset_accuracy']:.4f} "
            f"TOP5={row['top_5pct_corrupt']['mean_dataset_accuracy']:.4f} "
            f"TOP10={row['top_10pct_corrupt']['mean_dataset_accuracy']:.4f}"
        )

    print("\n=== CGEC vs RF / LR ===")
    for key in RECOVERY_KEYS:
        for baseline in ("rf_confidence_oof", "lr_confidence_oof"):
            r = analysis["comparisons"][key][baseline]
            print(
                f"{key:18s} vs {baseline:20s} "
                f"delta={r['mean_delta_cgec_minus_baseline']:+.4f} "
                f"W/T/L={r['dataset_wins']}/{r['dataset_ties']}/{r['dataset_losses']} "
                f"p={r['sign_test_p_two_sided']:.6g}"
            )

    print(f"WROTE {analysis_out}")
    print(f"WROTE {args.out}")


if __name__ == "__main__":
    main()
