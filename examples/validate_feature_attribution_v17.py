from __future__ import annotations

import argparse
import importlib.util
import json
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "examples" / "benchmark_feature_attribution_v17.py"


def load_benchmark_module():
    spec = importlib.util.spec_from_file_location("benchmark_feature_attribution_v17", BENCHMARK)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


v17 = load_benchmark_module()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def feature_vector(mapping: dict, feature_names):
    return np.asarray([float(mapping[str(name)]) for name in feature_names], dtype=float)


def topk_set(values: np.ndarray, feature_names, k: int):
    order = np.argsort(values)[::-1][:k]
    return {str(feature_names[i]) for i in order}


def jaccard(a: set[str], b: set[str]):
    union = a | b
    return 1.0 if not union else float(len(a & b) / len(union))


def pairwise_rank_stability(payload, topks=(5, 10)):
    feature_names = payload["feature_names"]
    runs = payload["global_runs"]
    vectors = {
        int(run["seed"]): feature_vector(run["mean_abs_effect"], feature_names)
        for run in runs
    }
    rows = []
    for a, b in combinations(sorted(vectors), 2):
        rho = float(spearmanr(vectors[a], vectors[b]).statistic)
        row = {"seed_a": a, "seed_b": b, "spearman": rho}
        for k in topks:
            row[f"top{k}_jaccard"] = jaccard(
                topk_set(vectors[a], feature_names, k),
                topk_set(vectors[b], feature_names, k),
            )
        rows.append(row)

    def summary(key):
        vals = np.asarray([r[key] for r in rows], dtype=float)
        return {
            "mean": float(np.mean(vals)),
            "median": float(np.median(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
        }

    out = {
        "n_pairs": len(rows),
        "pairs": rows,
        "spearman": summary("spearman"),
    }
    for k in topks:
        out[f"top{k}_jaccard"] = summary(f"top{k}_jaccard")
    return out


def mean_abs_effect_for_reference(seed: int, reference: str):
    feature_names, X_train, X_test, y_train, y_test = v17.split(seed)
    model = v17.classifier().fit(X_train, y_train)
    if reference == "mean":
        ref = np.mean(X_train, axis=0)
    elif reference == "median":
        ref = np.median(X_train, axis=0)
    else:
        raise ValueError(reference)

    rows = []
    for i, x in enumerate(X_test):
        _, exp = v17.explain_one(model, x)
        effects, _ = v17.feature_effects(model, x, ref, exp)
        rows.append(np.abs(effects))
        if (i + 1) % 25 == 0:
            print(f"reference={reference} seed={seed} query {i+1}/{len(X_test)}", flush=True)
    return feature_names, np.mean(np.asarray(rows), axis=0)


def reference_sensitivity(seeds, topks=(5, 10)):
    per_seed = []
    mean_vectors = []
    median_vectors = []
    feature_names = None

    for seed in seeds:
        names, mean_vec = mean_abs_effect_for_reference(seed, "mean")
        names2, median_vec = mean_abs_effect_for_reference(seed, "median")
        if list(names) != list(names2):
            raise RuntimeError("Feature-name mismatch between mean and median runs.")
        if feature_names is None:
            feature_names = list(names)
        mean_vectors.append(mean_vec)
        median_vectors.append(median_vec)

        row = {
            "seed": int(seed),
            "spearman": float(spearmanr(mean_vec, median_vec).statistic),
        }
        for k in topks:
            row[f"top{k}_jaccard"] = jaccard(
                topk_set(mean_vec, feature_names, k),
                topk_set(median_vec, feature_names, k),
            )
        per_seed.append(row)

    mean_agg = np.mean(np.asarray(mean_vectors), axis=0)
    median_agg = np.mean(np.asarray(median_vectors), axis=0)
    aggregate = {
        "spearman": float(spearmanr(mean_agg, median_agg).statistic),
        "mean_reference_ranking": [
            str(feature_names[i]) for i in np.argsort(mean_agg)[::-1]
        ],
        "median_reference_ranking": [
            str(feature_names[i]) for i in np.argsort(median_agg)[::-1]
        ],
    }
    for k in topks:
        aggregate[f"top{k}_jaccard"] = jaccard(
            topk_set(mean_agg, feature_names, k),
            topk_set(median_agg, feature_names, k),
        )

    return {
        "per_seed": per_seed,
        "aggregate": aggregate,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v17-json", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=Path("v17_feature_attribution_validation.json"))
    ap.add_argument("--skip-reference-sensitivity", action="store_true")
    args = ap.parse_args()

    payload = load_json(args.v17_json)
    seeds = [int(s) for s in payload["protocol"]["seeds"]]

    result = {
        "protocol": {
            "version": "1.7-validation",
            "rank_stability": "pairwise Spearman and top-k Jaccard across frozen outer splits",
            "reference_sensitivity": "mean-vs-median single-feature replacement; same dataset, splits, model, and fixed original winner/runner",
            "top_k": [5, 10],
        },
        "rank_stability": pairwise_rank_stability(payload),
    }

    if not args.skip_reference_sensitivity:
        result["reference_sensitivity"] = reference_sensitivity(seeds)

    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")

    rs = result["rank_stability"]
    print("=== V1.7 FEATURE ATTRIBUTION VALIDATION ===")
    print(
        "RANK_STABILITY "
        f"spearman_mean={rs['spearman']['mean']:.6f} "
        f"spearman_min={rs['spearman']['min']:.6f} "
        f"top5_jaccard_mean={rs['top5_jaccard']['mean']:.6f} "
        f"top10_jaccard_mean={rs['top10_jaccard']['mean']:.6f}"
    )
    if "reference_sensitivity" in result:
        agg = result["reference_sensitivity"]["aggregate"]
        print(
            "REFERENCE_SENSITIVITY "
            f"spearman={agg['spearman']:.6f} "
            f"top5_jaccard={agg['top5_jaccard']:.6f} "
            f"top10_jaccard={agg['top10_jaccard']:.6f}"
        )
    print(f"WROTE {args.output}")


if __name__ == "__main__":
    main()
