from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / "examples" / "benchmark_xai_faithfulness_v09.py"


def _load_parent():
    spec = importlib.util.spec_from_file_location("benchmark_xai_faithfulness_v09", PARENT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


v09 = _load_parent()

SEEDS = list(v09.SEEDS)
DATASETS = tuple(v09.DATASETS)
MAX_ELIGIBLE_QUERIES = int(v09.MAX_ELIGIBLE_QUERIES)
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


def unique_path_nodes(explanation):
    winner = explanation["prediction"]
    trail = explanation["classes"][winner]["representative_trail"]
    return [int(i) for i in dict.fromkeys(trail["path"])]


def nearest_label_composition_controls(model, x, path_nodes):
    path_nodes = [int(i) for i in dict.fromkeys(path_nodes)]
    excluded = set(path_nodes)
    xs = model._scaled(np.asarray(x)[None, :])[0]
    distances = np.linalg.norm(model.X_ - xs, axis=1)
    chosen = []
    used = set(excluded)
    for node in path_nodes:
        label_index = int(model.y_index_[node])
        candidates = [
            j for j in range(len(model.X_))
            if j not in used and int(model.y_index_[j]) == label_index
        ]
        if not candidates:
            return None
        pick = min(candidates, key=lambda j: float(distances[j]))
        chosen.append(int(pick))
        used.add(int(pick))
    return chosen


def top_local_contribution_controls(model, x, path_nodes, winner, runner):
    path_nodes = set(int(i) for i in path_nodes)
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
        if node in path_nodes:
            continue
        contrast_gap = float(model.contrast_field_[node, wi] - model.contrast_field_[node, ri])
        contribution = float(affinity[pos] * abs(contrast_gap))
        ranked.append((contribution, node))
    ranked.sort(key=lambda z: (-z[0], z[1]))
    need = len(path_nodes)
    if len(ranked) < need:
        return None
    return [int(node) for _, node in ranked[:need]]


def margin_drop_after(model, X_train, y_train, x, winner, runner, removed, original_margin):
    refit = v09.refit_without(X_train, y_train, removed)
    margin = v09.class_margin(refit, x, winner, runner)
    pred = v09._plain(refit.predict(np.asarray(x)[None, :])[0])
    return {
        "margin_after": float(margin),
        "margin_drop": float(original_margin - margin),
        "prediction_flip": bool(pred != winner),
    }


def evaluate_query(model, X_train, y_train, x, query_index):
    pred, explanations = v09.explain_mcgc(model, np.asarray(x)[None, :], top_support=5)
    explanation = explanations[0]
    winner = explanation["prediction"]
    runner = explanation["runner_up"]
    path_nodes = unique_path_nodes(explanation)
    if len(path_nodes) < 2:
        return {"query_index": int(query_index), "status": "ineligible_short_path"}

    original_margin = float(explanation["class_scores"][winner] - explanation["class_scores"][runner])
    controls = {
        "matched_distance_label": v09.matched_controls(model, x, path_nodes),
        "nearest_any": v09.nearest_off_path_controls(model, x, path_nodes),
        "nearest_label_composition": nearest_label_composition_controls(model, x, path_nodes),
        "top_local_contribution": top_local_contribution_controls(model, x, path_nodes, winner, runner),
    }

    if len(controls["matched_distance_label"]) != len(path_nodes) or len(controls["nearest_any"]) != len(path_nodes):
        return {"query_index": int(query_index), "status": "ineligible_parent_control"}

    path_result = margin_drop_after(model, X_train, y_train, x, winner, runner, path_nodes, original_margin)
    out = {
        "query_index": int(query_index),
        "status": "ok",
        "winner": v09._plain(winner),
        "runner_up": v09._plain(runner),
        "original_prediction": v09._plain(pred[0]),
        "original_margin": original_margin,
        "path_nodes": path_nodes,
        "path_length": int(len(path_nodes)),
        "path": path_result,
        "controls": {},
    }

    for name, nodes in controls.items():
        if nodes is None or len(nodes) != len(path_nodes):
            out["controls"][name] = {"status": "ineligible_equal_size"}
            continue
        result = margin_drop_after(model, X_train, y_train, x, winner, runner, nodes, original_margin)
        out["controls"][name] = {
            "status": "ok",
            "nodes": [int(i) for i in nodes],
            **result,
            "path_minus_control_drop": float(path_result["margin_drop"] - result["margin_drop"]),
        }
    return out


def evaluate_seed(name: str, seed: int):
    X, y = v09.dataset_for(name, seed)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=4000 + seed, stratify=y
    )
    model = v09.classifier().fit(X_train, y_train)
    records = []
    eligible = 0
    for i, x in enumerate(X_test):
        rec = evaluate_query(model, X_train, y_train, x, i)
        records.append(rec)
        if rec["status"] == "ok":
            eligible += 1
        if eligible >= MAX_ELIGIBLE_QUERIES:
            break

    ok = [r for r in records if r["status"] == "ok"]
    control_names = ("matched_distance_label", "nearest_any", "nearest_label_composition", "top_local_contribution")
    summary = {}
    for control in control_names:
        vals = [
            r["controls"][control]["path_minus_control_drop"]
            for r in ok if r["controls"][control].get("status") == "ok"
        ]
        summary[control] = summarize(vals)
    summary["path_margin_drop"] = summarize([r["path"]["margin_drop"] for r in ok])
    summary["path_flip_rate"] = float(np.mean([r["path"]["prediction_flip"] for r in ok])) if ok else None
    return {
        "seed": int(seed),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "eligible_queries": int(len(ok)),
        "inspected_queries": int(len(records)),
        "queries": records,
        "summary": summary,
    }


def aggregate_seed_units(datasets):
    controls = ("matched_distance_label", "nearest_any", "nearest_label_composition", "top_local_contribution")
    result = {}
    for control in controls:
        unit_values = []
        units = []
        for dataset_name, data in datasets.items():
            for run in data["runs"]:
                s = run["summary"][control]
                if s["mean"] is None:
                    continue
                unit_values.append(float(s["mean"]))
                units.append({"dataset": dataset_name, "seed": int(run["seed"]), "mean_contrast": float(s["mean"]), "n_queries": int(s["n"])})
        result[control] = {"dataset_seed_units": units, "summary": summarize(unit_values)}
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=Path("v09bis_xai_faithfulness.json"))
    args = ap.parse_args()

    datasets = {}
    for name in DATASETS:
        runs = [evaluate_seed(name, seed) for seed in SEEDS]
        datasets[name] = {"runs": runs}

    result = {
        "protocol": {
            "version": "0.9-bis / v2.3-A",
            "status": "frozen reviewer-response control audit",
            "parent": "benchmark_xai_faithfulness_v09.py",
            "datasets": DATASETS,
            "seeds": SEEDS,
            "max_eligible_queries_per_seed": MAX_ELIGIBLE_QUERIES,
            "primary_inference_unit": "dataset x seed",
            "primary_controls": ["nearest_label_composition", "top_local_contribution"],
            "historical_controls": ["matched_distance_label", "nearest_any"],
        },
        "datasets": datasets,
        "dataset_seed_inference": aggregate_seed_units(datasets),
    }

    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print("=== MCGC v0.9-bis XAI FAITHFULNESS CONTROL AUDIT ===")
    for control, block in result["dataset_seed_inference"].items():
        s = block["summary"]
        print(f"{control:28s} n={s['n']:2d} mean={s['mean']} median={s['median']} W/T/L={s['wins']}/{s['ties']}/{s['losses']} p={s['sign_test_p_two_sided']}")
    print(f"FULL_JSON={args.output.resolve()}")


if __name__ == "__main__":
    main()
