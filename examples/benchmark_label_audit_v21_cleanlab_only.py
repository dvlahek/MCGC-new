#!/usr/bin/env python3
"""CGEC v2.1 strict-OOF label review + Cleanlab comparator.

This is a minimal extension of the frozen v2.1 unseen-PMLB audit.
It reruns only:
  1) strict-OOF CGEC label-review score,
  2) strict-OOF logistic-regression probabilities,
  3) Cleanlab normalized-margin label-quality score computed from those SAME
     LR OOF probabilities.

Frozen protocol (identical to v2.1):
- 12 previously unused PMLB datasets
- outer 70/30 stratified split, random_state = 20000 + seed
- 4 corruption conditions: symmetric 0.10/0.20/0.30, pairflip 0.20
- corruption RNG: 21000 + 100*seed + round(100*eta) + pairflip offset 50
- five seeds: 0..4
- strict 5-fold OOF scoring, shuffle=True, random_state = 22000 + seed
- CGEC: K=12, alpha=0.85, diffusion_steps=40, walkers disabled
- LR: StandardScaler + LogisticRegression(C=1.0, max_iter=3000)
- Cleanlab: get_label_quality_scores(method="normalized_margin",
  adjust_pred_probs=False)
- primary inference unit: dataset

Higher audit score = more suspicious.

Usage:
  python examples/benchmark_label_audit_v21_cleanlab_only.py \
      --data-dir data_pmlb \
      --out results/v21_cleanlab_only.jsonl

Smoke:
  python examples/benchmark_label_audit_v21_cleanlab_only.py \
      --data-dir data_pmlb \
      --out results/v21_cleanlab_smoke.jsonl \
      --only bupa,ecoli \
      --seeds 0

Install Cleanlab once if needed:
  python -m pip install cleanlab
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.preprocessing import StandardScaler

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
AUDIT_K = 12
AUDIT_ALPHA = 0.85
MIN_CLASS = 6
METHODS = ("mcgc_oof", "lr_confidence_oof", "cleanlab_lr_oof")


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
    return StratifiedKFold(
        AUDIT_FOLDS, shuffle=True, random_state=22000 + seed
    )


def lr_oof_proba(Xs, yn, n_classes, seed):
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


def mcgc_oof_score(X, yn, seed):
    cv = make_cv(seed)
    scores = np.empty(len(yn), dtype=float)
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
            rival_e = max(
                (v for c, v in evidence.items() if c != observed),
                default=observed_e,
            )
            scores[global_idx] = float(rival_e - observed_e)
            seen[global_idx] = True

    if not np.all(seen):
        raise RuntimeError("CGEC OOF scoring did not cover all samples")
    return scores


def lr_confidence_score(P, yn):
    return 1.0 - P[np.arange(len(yn)), yn]


def cleanlab_score(P, yn):
    try:
        from cleanlab.rank import get_label_quality_scores
    except ImportError as exc:
        raise RuntimeError(
            "Cleanlab is required. Run: python -m pip install cleanlab"
        ) from exc

    quality = get_label_quality_scores(
        labels=np.asarray(yn, dtype=int),
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
    total_bad = int(flip.sum())
    noise_fraction = total_bad / n

    review = {}
    for budget in BUDGETS:
        k = max(1, int(round(budget * n)))
        top = order[:k]
        found = int(flip[top].sum())
        precision = found / k
        review[str(budget)] = {
            "reviewed_count": k,
            "corrupt_found": found,
            "precision": float(precision),
            "recall": float(found / total_bad) if total_bad else None,
            "enrichment_over_random": (
                float(precision / noise_fraction) if noise_fraction > 0 else None
            ),
        }

    return {
        "roc_auc": float(roc_auc_score(flip.astype(int), score)),
        "pr_auc": float(average_precision_score(flip.astype(int), score)),
        "review": review,
    }


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
        "K": n_classes,
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

    s_cgec = mcgc_oof_score(Xtr, yn, seed)
    P_lr = lr_oof_proba(Xs, yn, n_classes, seed)
    s_lr = lr_confidence_score(P_lr, yn)
    s_cleanlab = cleanlab_score(P_lr, yn)

    scores = {
        "mcgc_oof": s_cgec,
        "lr_confidence_oof": s_lr,
        "cleanlab_lr_oof": s_cleanlab,
    }
    rec["audit"] = {
        method: audit_metrics(flip, scores[method])
        for method in METHODS
    }
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


def dataset_level_analysis(records, names):
    ok = [r for r in records if r.get("status") == "ok"]
    rows = []

    for dataset in names:
        group = [r for r in ok if r["dataset"] == dataset]
        if not group:
            continue
        row = {"dataset": dataset, "n_units": len(group), "methods": {}}
        for method in METHODS:
            row["methods"][method] = {
                "roc_auc_mean": float(np.mean(
                    [r["audit"][method]["roc_auc"] for r in group]
                )),
                "pr_auc_mean": float(np.mean(
                    [r["audit"][method]["pr_auc"] for r in group]
                )),
                "p5_mean": float(np.mean(
                    [r["audit"][method]["review"]["0.05"]["precision"]
                     for r in group]
                )),
                "p10_mean": float(np.mean(
                    [r["audit"][method]["review"]["0.1"]["precision"]
                     for r in group]
                )),
            }
        rows.append(row)

    overall = {}
    for method in METHODS:
        overall[method] = {
            metric: float(np.mean([r["methods"][method][metric] for r in rows]))
            for metric in ("roc_auc_mean", "pr_auc_mean", "p5_mean", "p10_mean")
        }

    comparisons = {}
    for metric in ("roc_auc_mean", "pr_auc_mean", "p5_mean", "p10_mean"):
        comparisons[metric] = {}
        for baseline in ("lr_confidence_oof", "cleanlab_lr_oof"):
            diffs = [
                r["methods"]["mcgc_oof"][metric]
                - r["methods"][baseline][metric]
                for r in rows
            ]
            wins = sum(d > 1e-12 for d in diffs)
            losses = sum(d < -1e-12 for d in diffs)
            ties = len(diffs) - wins - losses
            comparisons[metric][baseline] = {
                "mean_delta_cgec_minus_baseline": float(np.mean(diffs)),
                "median_delta_cgec_minus_baseline": float(np.median(diffs)),
                "dataset_wins": int(wins),
                "dataset_ties": int(ties),
                "dataset_losses": int(losses),
                "sign_test_p_two_sided": float(sign_p(wins, losses)),
            }

    return {
        "dataset_level": rows,
        "overall_dataset_means": overall,
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
        if args.seeds else SEEDS
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)

    done = set()
    if args.out.exists():
        for line in args.out.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r["dataset"], r["noise"], r["eta"], r["seed"]))

    print("=== CGEC v2.1 + CLEANLAB STRICT-OOF LABEL REVIEW ===")
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
                    print(
                        f"{name:28s} {kind:9s} eta={eta:.2f} seed={seed} "
                        f"CGEC={rec['audit']['mcgc_oof']['roc_auc']:.3f} "
                        f"LR={rec['audit']['lr_confidence_oof']['roc_auc']:.3f} "
                        f"Cleanlab={rec['audit']['cleanlab_lr_oof']['roc_auc']:.3f}",
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

    if expected.issubset(have):
        subset = [
            r for r in records
            if r["dataset"] in names
            and r["seed"] in seeds
            and (r["noise"], r["eta"]) in CONDITIONS
        ]
        analysis = {
            "protocol": {
                "version": "v2.1-cleanlab-only",
                "datasets": list(names),
                "conditions": [
                    {"noise": kind, "eta": eta}
                    for kind, eta in CONDITIONS
                ],
                "seeds": list(seeds),
                "audit_folds": AUDIT_FOLDS,
                "methods": list(METHODS),
                "cleanlab_method": "normalized_margin",
                "cleanlab_adjust_pred_probs": False,
                "cleanlab_base_probabilities": "same strict-OOF LR probabilities",
                "primary_inference_unit": "dataset",
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

        print("\n=== OVERALL DATASET-LEVEL MEANS ===")
        for method, row in analysis["overall_dataset_means"].items():
            print(
                f"{method:22s} "
                f"ROC={row['roc_auc_mean']:.4f} "
                f"PR={row['pr_auc_mean']:.4f} "
                f"P@5={row['p5_mean']:.4f} "
                f"P@10={row['p10_mean']:.4f}"
            )

        print("\n=== CGEC vs CLEANLAB ===")
        for metric, comps in analysis["comparisons"].items():
            row = comps["cleanlab_lr_oof"]
            print(
                f"{metric:12s} "
                f"delta={row['mean_delta_cgec_minus_baseline']:+.4f} "
                f"W/T/L={row['dataset_wins']}/{row['dataset_ties']}/{row['dataset_losses']} "
                f"p={row['sign_test_p_two_sided']:.6g}"
            )
        print(f"WROTE {analysis_out}")

    print(f"WROTE {args.out}")


if __name__ == "__main__":
    main()
