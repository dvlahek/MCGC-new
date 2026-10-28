from __future__ import annotations

import argparse
import importlib.util
import json
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from mcgc import MultiAgentChemotacticGraphClassifier

ROOT = Path(__file__).resolve().parents[1]
V17_SCRIPT = ROOT / "examples" / "benchmark_feature_attribution_v17.py"

def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module

v17 = load_module(V17_SCRIPT, "benchmark_feature_attribution_v17")
SEEDS = (0, 1, 2, 3, 4)

def field_classifier():
    return MultiAgentChemotacticGraphClassifier(
        n_neighbors=12,
        diffusion_alpha=0.85,
        diffusion_steps=40,
        random_state=17,
        standardize=True,
        use_walkers=False,
    )

def local_evidence_vector(model, x):
    xs = model._scaled(np.asarray(x, dtype=float)[None, :])[0]
    distances, starts = model._query_start_nodes(xs)
    sigma = max(float(distances[-1]) if len(distances) else 1.0, 1e-12)
    affinity = np.exp(-(distances**2) / (sigma**2))
    affinity /= affinity.sum()
    return np.asarray(affinity @ model.contrast_field_[starts, :], dtype=float)

def winner_runner(model, x):
    local = local_evidence_vector(model, x)
    order = np.argsort(local)
    wi = int(order[-1])
    ri = int(order[-2])
    return (
        model.classes_[wi].item()
        if hasattr(model.classes_[wi], "item") else model.classes_[wi],
        model.classes_[ri].item()
        if hasattr(model.classes_[ri], "item") else model.classes_[ri],
        float(local[wi] - local[ri]),
    )

def fixed_pair_margin(model, x, winner, runner):
    local = local_evidence_vector(model, x)
    wi = int(np.flatnonzero(model.classes_ == winner)[0])
    ri = int(np.flatnonzero(model.classes_ == runner)[0])
    return float(local[wi] - local[ri])

def feature_effects(model, x, reference):
    winner, runner, base_margin = winner_runner(model, x)
    effects = np.empty(x.shape[0], dtype=float)
    for k in range(x.shape[0]):
        xp = np.asarray(x, dtype=float).copy()
        xp[k] = reference[k]
        effects[k] = (
            base_margin
            - fixed_pair_margin(model, xp, winner, runner)
        )
    return effects

def topk_set(values, feature_names, k):
    order = np.argsort(values)[::-1][:k]
    return {str(feature_names[i]) for i in order}

def jaccard(a, b):
    union = a | b
    return 1.0 if not union else float(len(a & b) / len(union))

def summary(values):
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }

def evaluate_seed(seed: int):
    feature_names, X_train, X_test, y_train, y_test = v17.split(seed)
    model = field_classifier().fit(X_train, y_train)
    mean_ref = np.mean(X_train, axis=0)
    median_ref = np.median(X_train, axis=0)

    mean_rows = []
    median_rows = []

    for i, x in enumerate(X_test):
        mean_rows.append(np.abs(feature_effects(model, x, mean_ref)))
        median_rows.append(np.abs(feature_effects(model, x, median_ref)))
        if (i + 1) % 25 == 0:
            print(
                f"seed={seed} query {i+1}/{len(X_test)}",
                flush=True,
            )

    mean_vec = np.mean(np.asarray(mean_rows), axis=0)
    median_vec = np.mean(np.asarray(median_rows), axis=0)

    return {
        "seed": int(seed),
        "n_test": int(len(X_test)),
        "feature_names": [str(x) for x in feature_names],
        "mean_reference_mean_abs_effect": {
            str(n): float(v)
            for n, v in zip(feature_names, mean_vec)
        },
        "median_reference_mean_abs_effect": {
            str(n): float(v)
            for n, v in zip(feature_names, median_vec)
        },
        "mean_vs_median": {
            "spearman": float(spearmanr(mean_vec, median_vec).statistic),
            "top5_jaccard": jaccard(
                topk_set(mean_vec, feature_names, 5),
                topk_set(median_vec, feature_names, 5),
            ),
            "top10_jaccard": jaccard(
                topk_set(mean_vec, feature_names, 10),
                topk_set(median_vec, feature_names, 10),
            ),
        },
    }

def mapping_to_vector(mapping, names):
    return np.asarray([float(mapping[str(n)]) for n in names], dtype=float)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "v30_field_only_feature_stability.json",
    )
    args = ap.parse_args()

    runs = []
    for seed in SEEDS:
        print(f"=== FEATURE STABILITY seed={seed} ===", flush=True)
        runs.append(evaluate_seed(seed))

    names = runs[0]["feature_names"]
    mean_vectors = {
        int(r["seed"]): mapping_to_vector(
            r["mean_reference_mean_abs_effect"], names
        )
        for r in runs
    }
    median_vectors = {
        int(r["seed"]): mapping_to_vector(
            r["median_reference_mean_abs_effect"], names
        )
        for r in runs
    }

    pair_rows = []
    for a, b in combinations(sorted(mean_vectors), 2):
        va = mean_vectors[a]
        vb = mean_vectors[b]
        pair_rows.append({
            "seed_a": int(a),
            "seed_b": int(b),
            "spearman": float(spearmanr(va, vb).statistic),
            "top5_jaccard": jaccard(
                topk_set(va, names, 5),
                topk_set(vb, names, 5),
            ),
            "top10_jaccard": jaccard(
                topk_set(va, names, 10),
                topk_set(vb, names, 10),
            ),
        })

    mean_agg = np.mean(
        np.asarray([mean_vectors[s] for s in sorted(mean_vectors)]),
        axis=0,
    )
    median_agg = np.mean(
        np.asarray([median_vectors[s] for s in sorted(median_vectors)]),
        axis=0,
    )

    result = {
        "protocol": {
            "version": "v3.0-feature",
            "status":
                "final field-only feature-intervention stability confirmation",
            "dataset": "breast_cancer",
            "seeds": list(SEEDS),
            "split":
                "70/30 stratified, random_state=7000+seed",
            "field_only_definition":
                "K=12, alpha=0.85, nD=40, use_walkers=False",
            "intervention":
                "replace one raw feature by outer-training reference; "
                "winner and runner fixed from original local-evidence prediction",
            "primary_margin":
                "L_winner(x)-L_runner(x)",
            "selection_from_results": False,
        },
        "feature_names": names,
        "runs": runs,
        "rank_stability_mean_reference": {
            "n_pairs": len(pair_rows),
            "pairs": pair_rows,
            "spearman": summary([r["spearman"] for r in pair_rows]),
            "top5_jaccard": summary(
                [r["top5_jaccard"] for r in pair_rows]
            ),
            "top10_jaccard": summary(
                [r["top10_jaccard"] for r in pair_rows]
            ),
        },
        "reference_sensitivity": {
            "per_seed_spearman": summary([
                r["mean_vs_median"]["spearman"] for r in runs
            ]),
            "per_seed_top5_jaccard": summary([
                r["mean_vs_median"]["top5_jaccard"] for r in runs
            ]),
            "aggregate": {
                "spearman": float(
                    spearmanr(mean_agg, median_agg).statistic
                ),
                "top5_jaccard": jaccard(
                    topk_set(mean_agg, names, 5),
                    topk_set(median_agg, names, 5),
                ),
                "top10_jaccard": jaccard(
                    topk_set(mean_agg, names, 10),
                    topk_set(median_agg, names, 10),
                ),
            },
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    rs = result["rank_stability_mean_reference"]
    ref = result["reference_sensitivity"]["aggregate"]
    print("\n=== FINAL FIELD-ONLY FEATURE STABILITY ===")
    print(
        f"Spearman mean={rs['spearman']['mean']:.6f} "
        f"median={rs['spearman']['median']:.6f}"
    )
    print(
        f"Top5 Jaccard mean={rs['top5_jaccard']['mean']:.6f}"
    )
    print(
        "Mean-vs-median aggregate "
        f"Spearman={ref['spearman']:.6f} "
        f"Top5={ref['top5_jaccard']:.6f}"
    )
    print(f"FULL_JSON={args.output.resolve()}")

if __name__ == "__main__":
    main()
