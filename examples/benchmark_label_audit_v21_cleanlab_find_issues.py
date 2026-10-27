#!/usr/bin/env python3
"""CGEC v2.1 strict-OOF label review + Cleanlab find_label_issues.

This reuses the frozen v2.1 protocol and the SAME strict 5-fold OOF
logistic-regression probabilities used by the existing Cleanlab score run.

Cleanlab configuration is frozen a priori:
- filter_by="prune_by_noise_rate"
- frac_noise=1.0
- return_indices_ranked_by="normalized_margin"
- n_jobs=1

Two outputs are evaluated:
1) Native find_label_issues subset: flagged fraction, precision, recall, F1.
2) Comparable full ranking for ROC-AUC / PR-AUC / P@5 / P@10:
   flagged examples first in Cleanlab severity order, then unflagged examples
   ordered by normalized-margin label quality.

Usage:
  python examples/benchmark_label_audit_v21_cleanlab_find_issues.py --data-dir data_pmlb --out results/v21_cleanlab_find_issues.jsonl
"""

from __future__ import annotations

import argparse
import json
import time
from math import comb
from pathlib import Path

import numpy as np
from cleanlab.filter import find_label_issues
from cleanlab.rank import get_label_quality_scores

import benchmark_label_audit_v21_cleanlab_only as base

METHODS = (
    "mcgc_oof",
    "lr_confidence_oof",
    "cleanlab_lr_oof",
    "cleanlab_find_issues_oof",
)


def cleanlab_find_issues(P, yn):
    labels = np.asarray(yn, dtype=int)
    probs = np.asarray(P, dtype=float)

    issue_idx = np.asarray(
        find_label_issues(
            labels=labels,
            pred_probs=probs,
            filter_by="prune_by_noise_rate",
            frac_noise=1.0,
            return_indices_ranked_by="normalized_margin",
            n_jobs=1,
        ),
        dtype=int,
    )

    quality = np.asarray(
        get_label_quality_scores(
            labels=labels,
            pred_probs=probs,
            method="normalized_margin",
            adjust_pred_probs=False,
        ),
        dtype=float,
    )

    flagged = np.zeros(len(labels), dtype=bool)
    flagged[issue_idx] = True

    remaining = np.flatnonzero(~flagged)
    remaining = remaining[np.argsort(quality[remaining], kind="stable")]
    full_order = np.concatenate([issue_idx, remaining])

    if len(full_order) != len(labels) or len(np.unique(full_order)) != len(labels):
        raise RuntimeError("Cleanlab full ranking is not a permutation")

    rank_score = np.empty(len(labels), dtype=float)
    rank_score[full_order] = np.arange(len(labels), 0, -1, dtype=float)
    return rank_score, flagged, issue_idx


def native_issue_metrics(flip, flagged):
    flip = np.asarray(flip, dtype=bool)
    flagged = np.asarray(flagged, dtype=bool)

    n_flagged = int(flagged.sum())
    tp = int(np.logical_and(flagged, flip).sum())
    total_bad = int(flip.sum())

    precision = float(tp / n_flagged) if n_flagged else None
    recall = float(tp / total_bad) if total_bad else None
    if precision is None or recall is None:
        f1 = None
    elif precision + recall == 0:
        f1 = 0.0
    else:
        f1 = float(2 * precision * recall / (precision + recall))

    return {
        "n_flagged": n_flagged,
        "flagged_fraction": float(n_flagged / len(flip)),
        "true_corrupt_flagged": tp,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def run_unit(X, y, meta, name, kind, eta, seed):
    K = int(y.max()) + 1

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    Xtr, _, ytr, _ = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=20000 + seed,
        stratify=y,
    )
    yn, flip = base.corrupt(ytr, K, kind, eta, seed)

    rec = {
        "dataset": name,
        "noise": kind,
        "eta": eta,
        "seed": seed,
        "K": K,
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

    s_cgec = base.mcgc_oof_score(Xtr, yn, seed)
    P_lr = base.lr_oof_proba(Xs, yn, K, seed)
    s_lr = base.lr_confidence_score(P_lr, yn)
    s_cleanlab = base.cleanlab_score(P_lr, yn)
    s_find, flagged, issue_idx = cleanlab_find_issues(P_lr, yn)

    scores = {
        "mcgc_oof": s_cgec,
        "lr_confidence_oof": s_lr,
        "cleanlab_lr_oof": s_cleanlab,
        "cleanlab_find_issues_oof": s_find,
    }

    rec["audit"] = {
        method: base.audit_metrics(flip, scores[method])
        for method in METHODS
    }
    rec["cleanlab_find_label_issues_native"] = native_issue_metrics(flip, flagged)
    rec["cleanlab_find_label_issues_native"]["ranked_issue_count"] = int(len(issue_idx))
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


def dataset_analysis(records, names):
    ok = [r for r in records if r.get("status") == "ok"]
    rows = []

    for dataset in names:
        group = [r for r in ok if r["dataset"] == dataset]
        if not group:
            continue

        row = {"dataset": dataset, "n_units": len(group), "methods": {}}
        for method in METHODS:
            row["methods"][method] = {
                "roc_auc_mean": float(np.mean([r["audit"][method]["roc_auc"] for r in group])),
                "pr_auc_mean": float(np.mean([r["audit"][method]["pr_auc"] for r in group])),
                "p5_mean": float(np.mean([r["audit"][method]["review"]["0.05"]["precision"] for r in group])),
                "p10_mean": float(np.mean([r["audit"][method]["review"]["0.1"]["precision"] for r in group])),
            }

        native = [r["cleanlab_find_label_issues_native"] for r in group]
        row["cleanlab_find_label_issues_native"] = {
            "flagged_fraction_mean": float(np.mean([x["flagged_fraction"] for x in native])),
            "precision_mean": float(np.mean([x["precision"] for x in native if x["precision"] is not None])),
            "recall_mean": float(np.mean([x["recall"] for x in native if x["recall"] is not None])),
            "f1_mean": float(np.mean([x["f1"] for x in native if x["f1"] is not None])),
        }
        rows.append(row)

    overall = {}
    for method in METHODS:
        overall[method] = {
            metric: float(np.mean([r["methods"][method][metric] for r in rows]))
            for metric in ("roc_auc_mean", "pr_auc_mean", "p5_mean", "p10_mean")
        }

    native_overall = {
        metric: float(np.mean([r["cleanlab_find_label_issues_native"][metric] for r in rows]))
        for metric in ("flagged_fraction_mean", "precision_mean", "recall_mean", "f1_mean")
    }

    comparisons = {}
    for metric in ("roc_auc_mean", "pr_auc_mean", "p5_mean", "p10_mean"):
        comparisons[metric] = {}
        for baseline in ("cleanlab_lr_oof", "cleanlab_find_issues_oof"):
            diffs = [
                r["methods"]["mcgc_oof"][metric] - r["methods"][baseline][metric]
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
        "cleanlab_find_label_issues_native_overall": native_overall,
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

    names = tuple(args.only.split(",")) if args.only else base.DATASETS
    seeds = tuple(int(s) for s in args.seeds.split(",")) if args.seeds else base.SEEDS
    args.out.parent.mkdir(parents=True, exist_ok=True)

    done = set()
    if args.out.exists():
        for line in args.out.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r["dataset"], r["noise"], r["eta"], r["seed"]))

    print("=== CGEC v2.1 + CLEANLAB find_label_issues STRICT-OOF ===")
    print(f"DATASETS={len(names)} SEEDS={seeds} CONDITIONS={len(base.CONDITIONS)}")
    print(f"METHODS={METHODS}")

    for name in names:
        X, y, meta = base.load_dataset(args.data_dir, name)
        for seed in seeds:
            for kind, eta in base.CONDITIONS:
                key = (name, kind, eta, seed)
                if key in done:
                    continue
                rec = run_unit(X, y, meta, name, kind, eta, seed)
                with args.out.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(rec) + "\n")

                if rec.get("status") == "ok":
                    nflag = rec["cleanlab_find_label_issues_native"]["n_flagged"]
                    print(
                        f"{name:28s} {kind:9s} eta={eta:.2f} seed={seed} "
                        f"CGEC={rec['audit']['mcgc_oof']['roc_auc']:.3f} "
                        f"FindIssues={rec['audit']['cleanlab_find_issues_oof']['roc_auc']:.3f} "
                        f"flagged={nflag}",
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
        for kind, eta in base.CONDITIONS
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
            and (r["noise"], r["eta"]) in base.CONDITIONS
        ]
        analysis = {
            "protocol": {
                "version": "v2.1-cleanlab-find-label-issues",
                "datasets": list(names),
                "conditions": [{"noise": n, "eta": e} for n, e in base.CONDITIONS],
                "seeds": list(seeds),
                "audit_folds": base.AUDIT_FOLDS,
                "methods": list(METHODS),
                "cleanlab_filter_by": "prune_by_noise_rate",
                "cleanlab_frac_noise": 1.0,
                "cleanlab_rank_by": "normalized_margin",
                "cleanlab_base_probabilities": "same strict-OOF LR probabilities",
                "primary_inference_unit": "dataset",
            },
            **dataset_analysis(subset, names),
        }

        analysis_out = args.analysis_out or args.out.with_name(args.out.stem + "_analysis.json")
        analysis_out.write_text(json.dumps(analysis, indent=2, sort_keys=True), encoding="utf-8")

        print("\n=== OVERALL DATASET-LEVEL MEANS ===")
        for method, row in analysis["overall_dataset_means"].items():
            print(
                f"{method:26s} ROC={row['roc_auc_mean']:.4f} "
                f"PR={row['pr_auc_mean']:.4f} "
                f"P@5={row['p5_mean']:.4f} P@10={row['p10_mean']:.4f}"
            )

        print("\n=== CGEC vs CLEANLAB find_label_issues RANKING ===")
        for metric, comps in analysis["comparisons"].items():
            row = comps["cleanlab_find_issues_oof"]
            print(
                f"{metric:12s} delta={row['mean_delta_cgec_minus_baseline']:+.4f} "
                f"W/T/L={row['dataset_wins']}/{row['dataset_ties']}/{row['dataset_losses']} "
                f"p={row['sign_test_p_two_sided']:.6g}"
            )

        native = analysis["cleanlab_find_label_issues_native_overall"]
        print("\n=== CLEANLAB find_label_issues NATIVE ISSUE SET ===")
        print(
            f"flagged_fraction={native['flagged_fraction_mean']:.4f} "
            f"precision={native['precision_mean']:.4f} "
            f"recall={native['recall_mean']:.4f} "
            f"F1={native['f1_mean']:.4f}"
        )
        print(f"WROTE {analysis_out}")

    print(f"WROTE {args.out}")


if __name__ == "__main__":
    main()
