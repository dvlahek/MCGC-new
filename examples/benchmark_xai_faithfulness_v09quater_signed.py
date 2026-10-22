from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split


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


def signed_winner_support_nodes(model, x, path_nodes, winner, runner):
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
        gap = float(model.contrast_field_[node, wi] - model.contrast_field_[node, ri])
        contribution = float(affinity[pos] * gap)
        if contribution > EPS:
            ranked.append((contribution, node))
    ranked.sort(key=lambda z: (-z[0], z[1]))

    need = len(set(int(i) for i in path_nodes))
    if len(ranked) < need:
        return None, len(ranked)
    return [int(node) for _, node in ranked[:need]], len(ranked)


def evaluate_query(model, X_train, y_train, x, query_index):
    pred, explanations = v09b.v09.explain_mcgc(model, np.asarray(x)[None, :], top_support=5)
    explanation = explanations[0]
    winner = explanation["prediction"]
    runner = explanation["runner_up"]
    path_nodes = v09b.unique_path_nodes(explanation)
    if len(path_nodes) < 2:
        return {"query_index": int(query_index), "status": "ineligible_short_path"}

    original_margin = float(explanation["class_scores"][winner] - explanation["class_scores"][runner])
    path_result = v09b.margin_drop_after(model, X_train, y_train, x, winner, runner, path_nodes, original_margin)

    matched_nodes = v09b.v09.matched_controls(model, x, path_nodes)
    matched = None
    if matched_nodes is not None and len(matched_nodes) == len(path_nodes):
        matched = v09b.margin_drop_after(model, X_train, y_train, x, winner, runner, matched_nodes, original_margin)

    signed_nodes, n_positive = signed_winner_support_nodes(model, x, path_nodes, winner, runner)
    signed = None
    if signed_nodes is not None:
        signed = v09b.margin_drop_after(model, X_train, y_train, x, winner, runner, signed_nodes, original_margin)

    out = {
        "query_index": int(query_index),
        "status": "ok",
        "winner": v09b.v09._plain(winner),
        "runner_up": v09b.v09._plain(runner),
        "original_prediction": v09b.v09._plain(pred[0]),
        "original_margin": original_margin,
        "path_nodes": [int(i) for i in path_nodes],
        "path_length": int(len(path_nodes)),
        "n_positive_signed_candidates": int(n_positive),
        "path": path_result,
        "matched_distance_label": {"status": "ineligible"} if matched is None else {
            "status": "ok",
            "nodes": [int(i) for i in matched_nodes],
            **matched,
            "path_minus_control_drop": float(path_result["margin_drop"] - matched["margin_drop"]),
        },
        "signed_winner_support": {"status": "ineligible_not_enough_positive_nodes"} if signed is None else {
            "status": "ok",
            "nodes": [int(i) for i in signed_nodes],
            **signed,
            "path_minus_control_drop": float(path_result["margin_drop"] - signed["margin_drop"]),
            "set_jaccard": float(len(set(path_nodes) & set(signed_nodes)) / len(set(path_nodes) | set(signed_nodes))),
        },
    }
    return out


def evaluate_seed(name: str, seed: int):
    X, y = v09b.v09.dataset_for(name, seed)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=4000 + seed, stratify=y
    )
    model = v09b.v09.classifier().fit(X_train, y_train)

    records = []
    eligible_paths = 0
    for i, x in enumerate(X_test):
        rec = evaluate_query(model, X_train, y_train, x, i)
        records.append(rec)
        if rec["status"] == "ok":
            eligible_paths += 1
        if eligible_paths >= v09b.MAX_ELIGIBLE_QUERIES:
            break

    ok = [r for r in records if r["status"] == "ok"]
    matched_ok = [r for r in ok if r["matched_distance_label"]["status"] == "ok"]
    signed_ok = [r for r in ok if r["signed_winner_support"]["status"] == "ok"]

    return {
        "seed": int(seed),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "eligible_path_queries": int(len(ok)),
        "signed_evaluable_queries": int(len(signed_ok)),
        "signed_eligibility_rate": float(len(signed_ok) / len(ok)) if ok else None,
        "queries": records,
        "summary": {
            "path_margin_drop_all_paths": summarize([r["path"]["margin_drop"] for r in ok]),
            "matched_path_minus_control": summarize([r["matched_distance_label"]["path_minus_control_drop"] for r in matched_ok]),
            "matched_control_margin_drop": summarize([r["matched_distance_label"]["margin_drop"] for r in matched_ok]),
            "signed_path_minus_control": summarize([r["signed_winner_support"]["path_minus_control_drop"] for r in signed_ok]),
            "signed_path_margin_drop": summarize([r["path"]["margin_drop"] for r in signed_ok]),
            "signed_control_margin_drop": summarize([r["signed_winner_support"]["margin_drop"] for r in signed_ok]),
            "signed_mean_set_jaccard": float(np.mean([r["signed_winner_support"]["set_jaccard"] for r in signed_ok])) if signed_ok else None,
        },
    }


def aggregate(datasets):
    matched_units = []
    signed_units = []
    signed_path_drops = []
    signed_control_drops = []
    eligibility = []
    for dataset_name, block in datasets.items():
        for run in block["runs"]:
            eligibility.append({
                "dataset": dataset_name,
                "seed": int(run["seed"]),
                "rate": run["signed_eligibility_rate"],
                "n": int(run["signed_evaluable_queries"]),
            })
            m = run["summary"]["matched_path_minus_control"]
            if m["mean"] is not None:
                matched_units.append({"dataset": dataset_name, "seed": int(run["seed"]), "value": float(m["mean"]), "n_queries": int(m["n"])})
            s = run["summary"]["signed_path_minus_control"]
            if s["mean"] is not None:
                signed_units.append({"dataset": dataset_name, "seed": int(run["seed"]), "value": float(s["mean"]), "n_queries": int(s["n"])})
                signed_path_drops.append(float(run["summary"]["signed_path_margin_drop"]["mean"]))
                signed_control_drops.append(float(run["summary"]["signed_control_margin_drop"]["mean"]))

    return {
        "matched_distance_label": {
            "dataset_seed_units": matched_units,
            "summary": summarize([u["value"] for u in matched_units]),
        },
        "signed_winner_support": {
            "dataset_seed_units": signed_units,
            "summary": summarize([u["value"] for u in signed_units]),
            "mean_path_margin_drop_across_units": float(np.mean(signed_path_drops)) if signed_path_drops else None,
            "mean_control_margin_drop_across_units": float(np.mean(signed_control_drops)) if signed_control_drops else None,
        },
        "signed_eligibility": eligibility,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=Path("v09quater_signed_faithfulness.json"))
    args = ap.parse_args()

    datasets = {name: {"runs": [evaluate_seed(name, seed) for seed in v09b.SEEDS]} for name in v09b.DATASETS}
    result = {
        "protocol": {
            "version": "0.9-quater / v2.5-A",
            "status": "frozen final reviewer audit",
            "datasets": list(v09b.DATASETS),
            "seeds": list(v09b.SEEDS),
            "primary_inference_unit": "dataset x seed",
            "signed_control": "top positive affinity-weighted winner-minus-runner local contributions; overlap with path allowed",
            "insufficient_positive_policy": "query ineligible; no padding with zero or negative nodes",
            "historical_control_retained": "label+distance matched off-path",
        },
        "datasets": datasets,
        "dataset_seed_inference": aggregate(datasets),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")

    print("=== MCGC v0.9-quater SIGNED WINNER-SUPPORT FAITHFULNESS ===")
    for key in ("matched_distance_label", "signed_winner_support"):
        s = result["dataset_seed_inference"][key]["summary"]
        print(f"{key:24s} n={s['n']} mean={s['mean']} median={s['median']} W/T/L={s['wins']}/{s['ties']}/{s['losses']} p={s['sign_test_p_two_sided']}")
    sw = result["dataset_seed_inference"]["signed_winner_support"]
    print(f"signed mean path drop={sw['mean_path_margin_drop_across_units']} control drop={sw['mean_control_margin_drop_across_units']}")
    print(f"FULL_JSON={args.output.resolve()}")


if __name__ == "__main__":
    main()
