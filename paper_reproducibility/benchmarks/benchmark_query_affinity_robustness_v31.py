#!/usr/bin/env python3
"""Reviewer audit: query-affinity bandwidth robustness.

Compares the manuscript query-centered affinity
    exp(-d(q,j)^2 / sigma_q^2)
with a pairwise locally scaled alternative
    exp(-d(q,j)^2 / max(sigma_q * sigma_j, eps)).

The fitted training graph and class-evidence field are held fixed. Only the
query-to-training weighting is changed. This isolates the reviewer question
about comparability between training-edge and query affinities.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
FIELD_PATH = ROOT / "paper_reproducibility" / "xai" / "field_only_cgec.py"
V26_PATH = ROOT / "examples" / "benchmark_corrected_predictive_v26.py"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


field_mod = load_module(FIELD_PATH, "field_only_cgec_v31")
v26 = load_module(V26_PATH, "benchmark_corrected_predictive_v26_v31")
v12 = v26.v12

FieldOnlyCGEC = field_mod.FieldOnlyCGEC
CGECConfig = field_mod.CGECConfig
SEEDS = list(v26.SEEDS)
NONLINEAR = tuple(v26.NONLINEAR_PRIMARY)
REAL = tuple(v26.REAL_PRIMARY)
DATASETS = NONLINEAR + REAL


def train_sigma(model):
    dist = cdist(model.Z_, model.Z_)
    np.fill_diagonal(dist, np.inf)
    k = model.config.n_neighbors
    nn = np.argsort(dist, axis=1)[:, :k]
    return np.take_along_axis(dist, nn, axis=1)[:, -1]


def score_query(model, sigma_train, x, mode):
    cfg = model.config
    x = np.asarray(x, dtype=float).reshape(1, -1)
    zq = model.scaler_.transform(x)[0]
    d = np.linalg.norm(model.Z_ - zq, axis=1)
    starts = np.argsort(d)[: cfg.n_neighbors]
    sigma_q = max(float(np.sort(d)[cfg.n_neighbors - 1]), cfg.eps)

    if mode == "query_squared":
        denom = sigma_q ** 2
        raw = np.exp(-(d[starts] ** 2) / denom)
    elif mode == "pairwise_local":
        denom = np.maximum(sigma_q * sigma_train[starts], cfg.eps)
        raw = np.exp(-(d[starts] ** 2) / denom)
    else:
        raise ValueError(mode)

    a = raw / raw.sum()
    L = np.asarray(a @ model.contrast_field_[starts, :], dtype=float)
    order = np.argsort(L)[::-1]
    winner = int(order[0])
    runner = int(order[1]) if len(order) > 1 else winner
    margin = float(L[winner] - L[runner])
    return model.classes_[winner], -margin


def metrics_for_mode(model, X_test, y_test, mode, sigma_train):
    pred = []
    risk = []
    for x in X_test:
        p, r = score_query(model, sigma_train, x, mode)
        pred.append(p)
        risk.append(r)
    return v12.model_metrics(y_test, np.asarray(pred), np.asarray(risk, dtype=float)), np.asarray(pred)


def sign_p(wins, losses):
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


def evaluate(name, seed):
    X, y = v26.dataset_for(name, seed)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=5000 + seed, stratify=y
    )
    model = FieldOnlyCGEC(CGECConfig()).fit(X_train, y_train)
    sigma_train = train_sigma(model)
    base, p_base = metrics_for_mode(model, X_test, y_test, "query_squared", sigma_train)
    alt, p_alt = metrics_for_mode(model, X_test, y_test, "pairwise_local", sigma_train)
    return {
        "seed": int(seed),
        "n_test": int(len(y_test)),
        "query_squared": base,
        "pairwise_local": alt,
        "prediction_agreement": float(np.mean(p_base == p_alt)),
        "delta_alt_minus_base": {
            "accuracy": float(alt["accuracy"] - base["accuracy"]),
            "eaurc": float(alt["eaurc"] - base["eaurc"]),
            "aurc": float(alt["aurc"] - base["aurc"]),
            "error_detection_roc_auc": (
                None if alt["error_detection_roc_auc"] is None or base["error_detection_roc_auc"] is None
                else float(alt["error_detection_roc_auc"] - base["error_detection_roc_auc"])
            ),
        },
    }


def summarize(datasets, names):
    runs = [r for name in names for r in datasets[name]["runs"]]
    out = {
        "n_units": len(runs),
        "mean_prediction_agreement": float(np.mean([r["prediction_agreement"] for r in runs])),
    }
    for metric in ("accuracy", "eaurc", "aurc", "error_detection_roc_auc"):
        vals = [r["delta_alt_minus_base"][metric] for r in runs if r["delta_alt_minus_base"][metric] is not None]
        arr = np.asarray(vals, dtype=float)
        eps = 1e-12
        w = int(np.sum(arr > eps))
        l = int(np.sum(arr < -eps))
        out[metric] = {
            "mean_alt_minus_base": float(np.mean(arr)),
            "median_alt_minus_base": float(np.median(arr)),
            "wins_alt": w,
            "ties": int(len(arr)-w-l),
            "losses_alt": l,
            "sign_test_p": float(sign_p(w, l)),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=Path("paper_reproducibility/results/v31_query_affinity_robustness.json"))
    ap.add_argument("--only", default=None, help="Comma-separated dataset subset")
    ap.add_argument("--seeds", default=None, help="Comma-separated seed subset")
    args = ap.parse_args()

    names = tuple(args.only.split(",")) if args.only else DATASETS
    seeds = tuple(int(s) for s in args.seeds.split(",")) if args.seeds else tuple(SEEDS)

    datasets = {}
    for name in names:
        print(f"DATASET {name}", flush=True)
        runs = []
        for seed in seeds:
            rec = evaluate(name, seed)
            runs.append(rec)
            d = rec["delta_alt_minus_base"]
            print(
                f"  seed={seed:2d} agree={rec['prediction_agreement']:.4f} "
                f"dAcc={d['accuracy']:+.6f} dEAURC={d['eaurc']:+.6f}",
                flush=True,
            )
        datasets[name] = {"runs": runs}

    nonlinear_names = tuple(n for n in names if n in NONLINEAR)
    real_names = tuple(n for n in names if n in REAL)
    result = {
        "protocol": {
            "version": "v3.1-query-affinity-robustness",
            "current": "exp(-d^2 / sigma_q^2)",
            "alternative": "exp(-d^2 / max(sigma_q*sigma_j, eps))",
            "fixed_components": "training graph, diffusion field, K=12, alpha=0.85, nD=40",
            "outer_split": "70/30 stratified, random_state=5000+seed",
            "datasets": list(names),
            "seeds": list(seeds),
        },
        "datasets": datasets,
        "summary": {
            "all": summarize(datasets, names),
            "nonlinear": summarize(datasets, nonlinear_names) if nonlinear_names else None,
            "real": summarize(datasets, real_names) if real_names else None,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")

    print("\n=== QUERY-AFFINITY ROBUSTNESS ===")
    for group, block in result["summary"].items():
        if block is None:
            continue
        print(f"{group}: n={block['n_units']} agreement={block['mean_prediction_agreement']:.4f}")
        print("  dEAURC alt-base:", json.dumps(block["eaurc"], sort_keys=True))
    print(f"\nFULL_JSON={args.output.resolve()}")


if __name__ == "__main__":
    main()
