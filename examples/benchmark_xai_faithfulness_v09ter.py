from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / "examples" / "benchmark_xai_faithfulness_v09bis.py"


def _load_parent():
    spec = importlib.util.spec_from_file_location("benchmark_xai_faithfulness_v09bis", PARENT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


v09b = _load_parent()
EPS = 1e-12


def sign_test_p(wins: int, losses: int):
    n = wins + losses
    if n == 0:
        return None
    k = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return float(min(1.0, 2.0 * tail))


def summarize(values):
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return {"n": 0, "mean": None, "median": None, "std": None, "wins": 0, "ties": 0, "losses": 0, "sign_test_p_two_sided": None}
    wins = int(np.sum(arr > EPS))
    ties = int(np.sum(np.abs(arr) <= EPS))
    losses = int(np.sum(arr < -EPS))
    return {
        "n": int(len(arr)),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "sign_test_p_two_sided": sign_test_p(wins, losses),
    }


def unrestricted_top_local_contribution(model, x, path_nodes, winner, runner):
    xs = model._scaled(np.asarray(x)[None, :])[0]
    distances, starts = model._query_start_nodes(xs)
    sigma = max(float(distances[-1]) if len(distances) else 1.0, 1e-12)
    affinity = np.exp(-(distances**2) / (sigma**2))
    affinity /= affinity.sum()
    wi = int(np.flatnonzero(model.classes_ == winner)[0])
    ri = int(np.flatnonzero(model.classes_ == runner)[0])
    ranked = []
    for pos, node in enumerate(starts):
        node = int(node)
        contrast_gap = float(model.contrast_field_[node, wi] - model.contrast_field_[node, ri])
        contribution = float(affinity[pos] * abs(contrast_gap))
        ranked.append((contribution, node))
    ranked.sort(key=lambda z: (-z[0], z[1]))
    need = len(set(int(i) for i in path_nodes))
    if len(ranked) < need:
        return None
    return [int(node) for _, node in ranked[:need]]


def evaluate_query(model, X_train, y_train, x, query_index):
    pred, explanations = v09b.v09.explain_mcgc(model, np.asarray(x)[None, :], top_support=5)
    explanation = explanations[0]
    winner = explanation["prediction"]
    runner = explanation["runner_up"]
    path_nodes = v09b.unique_path_nodes(explanation)
    if len(path_nodes) < 2:
        return {"query_index": int(query_index), "status": "ineligible_short_path"}

    original_margin = float(explanation["class_scores"][winner] - explanation["class_scores"][runner])
    baseline_nodes = unrestricted_top_local_contribution(model, x, path_nodes, winner, runner)
    if baseline_nodes is None or len(baseline_nodes) != len(path_nodes):
        return {"query_index": int(query_index), "status": "ineligible_equal_size"}

    path_result = v09b.margin_drop_after(model, X_train, y_train, x, winner, runner, path_nodes, original_margin)
    base_result = v09b.margin_drop_after(model, X_train, y_train, x, winner, runner, baseline_nodes, original_margin)

    path_set = set(int(i) for i in path_nodes)
    base_set = set(int(i) for i in baseline_nodes)
    overlap = len(path_set & base_set) / len(path_set | base_set) if (path_set | base_set) else 1.0

    return {
        "query_index": int(query_index),
        "status": "ok",
        "winner": v09b.v09._plain(winner),
        "runner_up": v09b.v09._plain(runner),
        "original_prediction": v09b.v09._plain(pred[0]),
        "original_margin": original_margin,
        "path_nodes": [int(i) for i in path_nodes],
        "path_length": int(len(path_nodes)),
        "baseline_nodes": [int(i) for i in baseline_nodes],
        "path": path_result,
        "baseline": base_result,
        "path_minus_baseline_drop": float(path_result["margin_drop"] - base_result["margin_drop"]),
        "set_jaccard": float(overlap),
        "exact_set_equal": bool(path_set == base_set),
    }


def evaluate_seed(name: str, seed: int):
    X, y = v09b.v09.dataset_for(name, seed)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=4000 + seed, stratify=y
    )
    model = v09b.v09.classifier().fit(X_train, y_train)

    records = []
    eligible = 0
    for i, x in enumerate(X_test):
        rec = evaluate_query(model, X_train, y_train, x, i)
        records.append(rec)
        if rec["status"] == "ok":
            eligible += 1
        if eligible >= v09b.MAX_ELIGIBLE_QUERIES:
            break

    ok = [r for r in records if r["status"] == "ok"]
    return {
        "seed": int(seed),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "eligible_queries": int(len(ok)),
        "inspected_queries": int(len(records)),
        "queries": records,
        "summary": {
            "path_minus_unrestricted_local": summarize([r["path_minus_baseline_drop"] for r in ok]),
            "path_margin_drop": summarize([r["path"]["margin_drop"] for r in ok]),
            "baseline_margin_drop": summarize([r["baseline"]["margin_drop"] for r in ok]),
            "mean_set_jaccard": float(np.mean([r["set_jaccard"] for r in ok])) if ok else None,
            "exact_set_equal_rate": float(np.mean([r["exact_set_equal"] for r in ok])) if ok else None,
            "path_flip_rate": float(np.mean([r["path"]["prediction_flip"] for r in ok])) if ok else None,
            "baseline_flip_rate": float(np.mean([r["baseline"]["prediction_flip"] for r in ok])) if ok else None,
        },
    }


def aggregate_seed_units(datasets):
    units = []
    vals = []
    for dataset_name, data in datasets.items():
        for run in data["runs"]:
            s = run["summary"]["path_minus_unrestricted_local"]
            if s["mean"] is None:
                continue
            vals.append(float(s["mean"]))
            units.append({
                "dataset": dataset_name,
                "seed": int(run["seed"]),
                "mean_contrast": float(s["mean"]),
                "n_queries": int(s["n"]),
                "mean_set_jaccard": run["summary"]["mean_set_jaccard"],
                "exact_set_equal_rate": run["summary"]["exact_set_equal_rate"],
            })
    return {"dataset_seed_units": units, "summary": summarize(vals)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=Path("v09ter_xai_faithfulness_unrestricted.json"))
    args = ap.parse_args()

    datasets = {}
    for name in v09b.DATASETS:
        runs = [evaluate_seed(name, seed) for seed in v09b.SEEDS]
        datasets[name] = {"runs": runs}

    inference = aggregate_seed_units(datasets)
    result = {
        "protocol": {
            "version": "0.9-ter / v2.4",
            "status": "frozen reviewer-response follow-up",
            "parent": "benchmark_xai_faithfulness_v09bis.py",
            "single_change": "top-local-contribution comparator may include representative-path nodes",
            "datasets": list(v09b.DATASETS),
            "seeds": list(v09b.SEEDS),
            "max_eligible_queries_per_seed": int(v09b.MAX_ELIGIBLE_QUERIES),
            "primary_inference_unit": "dataset x seed",
        },
        "datasets": datasets,
        "dataset_seed_inference": inference,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    s = inference["summary"]
    print("=== MCGC v0.9-ter UNRESTRICTED LOCAL-EVIDENCE FAITHFULNESS ===")
    print(f"path_minus_unrestricted_local n={s['n']} mean={s['mean']} median={s['median']} W/T/L={s['wins']}/{s['ties']}/{s['losses']} p={s['sign_test_p_two_sided']}")
    for name, block in datasets.items():
        vals = [r["summary"]["path_minus_unrestricted_local"]["mean"] for r in block["runs"]]
        print(f"{name:16s} mean_unit_contrast={float(np.mean(vals)):.9f}")
    print(f"FULL_JSON={args.output.resolve()}")


if __name__ == "__main__":
    main()
