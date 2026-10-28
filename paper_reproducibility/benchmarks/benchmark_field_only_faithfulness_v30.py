from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from mcgc import MultiAgentChemotacticGraphClassifier
from mcgc.xai import explain_mcgc

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / "examples" / "benchmark_xai_faithfulness_v09quater_signed.py"

def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module

q = load_module(PARENT, "benchmark_xai_faithfulness_v09quater_signed")
v09 = q.v09b.v09
EPS = 1e-12

def field_classifier():
    return MultiAgentChemotacticGraphClassifier(
        n_neighbors=12,
        diffusion_alpha=0.85,
        diffusion_steps=40,
        random_state=17,
        standardize=True,
        use_walkers=False,
    )

def py_scalar(value):
    return value.item() if hasattr(value, "item") else value

def local_evidence_vector(model, x):
    xs = model._scaled(np.asarray(x, dtype=float)[None, :])[0]
    distances, starts = model._query_start_nodes(xs)
    sigma = max(float(distances[-1]) if len(distances) else 1.0, 1e-12)
    affinity = np.exp(-(distances**2) / (sigma**2))
    affinity /= affinity.sum()
    return np.asarray(affinity @ model.contrast_field_[starts, :], dtype=float)

def local_pair_margin(model, x, winner, runner):
    local = local_evidence_vector(model, x)
    wi = int(np.flatnonzero(model.classes_ == winner)[0])
    ri = int(np.flatnonzero(model.classes_ == runner)[0])
    return float(local[wi] - local[ri])

def refit_without(X_train, y_train, removed):
    mask = np.ones(len(X_train), dtype=bool)
    mask[np.asarray(sorted(set(int(i) for i in removed)), dtype=int)] = False
    return field_classifier().fit(X_train[mask], y_train[mask])

def field_prediction(model, x):
    local = local_evidence_vector(model, x)
    return py_scalar(model.classes_[int(np.argmax(local))])

def margin_drop_after(model, X_train, y_train, x, winner, runner, removed, original_margin):
    refit = refit_without(X_train, y_train, removed)
    margin = local_pair_margin(refit, x, winner, runner)
    pred = field_prediction(refit, x)
    return {
        "margin_after": float(margin),
        "margin_drop": float(original_margin - margin),
        "prediction_flip": bool(pred != winner),
    }

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
        return {
            "n": 0, "mean": None, "median": None, "std": None,
            "wins": 0, "ties": 0, "losses": 0,
            "sign_test_p_two_sided": None,
        }
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

def signed_winner_support_nodes(model, x, path_nodes, winner, runner):
    xs = model._scaled(np.asarray(x, dtype=float)[None, :])[0]
    distances, starts = model._query_start_nodes(xs)
    sigma = max(float(distances[-1]) if len(distances) else 1.0, 1e-12)
    affinity = np.exp(-(distances**2) / (sigma**2))
    affinity /= affinity.sum()
    wi = int(np.flatnonzero(model.classes_ == winner)[0])
    ri = int(np.flatnonzero(model.classes_ == runner)[0])

    ranked = []
    for pos, node in enumerate(starts):
        node = int(node)
        gap = float(
            model.contrast_field_[node, wi]
            - model.contrast_field_[node, ri]
        )
        contribution = float(affinity[pos] * gap)
        if contribution > EPS:
            ranked.append((contribution, node))
    ranked.sort(key=lambda z: (-z[0], z[1]))

    need = len(set(int(i) for i in path_nodes))
    if len(ranked) < need:
        return None, len(ranked)
    return [int(node) for _, node in ranked[:need]], len(ranked)

def evaluate_query(model, X_train, y_train, x, query_index):
    pred, explanations = explain_mcgc(
        model, np.asarray(x, dtype=float)[None, :], top_support=5
    )
    explanation = explanations[0]
    winner = py_scalar(explanation["prediction"])
    runner = py_scalar(explanation["runner_up"])

    # Hard consistency check: the explanation winner must equal the direct
    # local-evidence winner used by the final manuscript definition.
    direct_winner = field_prediction(model, x)
    if winner != direct_winner:
        raise RuntimeError(
            f"Field-only explanation/prediction mismatch at query {query_index}: "
            f"{winner} vs {direct_winner}"
        )

    path_nodes = unique_path_nodes(explanation)
    if len(path_nodes) < 2:
        return {"query_index": int(query_index), "status": "ineligible_short_path"}

    original_margin = local_pair_margin(model, x, winner, runner)

    path_result = margin_drop_after(
        model, X_train, y_train, x, winner, runner,
        path_nodes, original_margin
    )

    matched_nodes = v09.matched_controls(model, x, path_nodes)
    matched = None
    if matched_nodes is not None and len(matched_nodes) == len(path_nodes):
        matched = margin_drop_after(
            model, X_train, y_train, x, winner, runner,
            matched_nodes, original_margin
        )

    signed_nodes, n_positive = signed_winner_support_nodes(
        model, x, path_nodes, winner, runner
    )
    signed = None
    if signed_nodes is not None:
        signed = margin_drop_after(
            model, X_train, y_train, x, winner, runner,
            signed_nodes, original_margin
        )

    return {
        "query_index": int(query_index),
        "status": "ok",
        "winner": winner,
        "runner_up": runner,
        "original_prediction": py_scalar(pred[0]),
        "original_local_margin": float(original_margin),
        "path_nodes": [int(i) for i in path_nodes],
        "path_length": int(len(path_nodes)),
        "n_positive_signed_candidates": int(n_positive),
        "path": path_result,
        "matched_distance_label":
            {"status": "ineligible"} if matched is None else {
                "status": "ok",
                "nodes": [int(i) for i in matched_nodes],
                **matched,
                "path_minus_control_drop":
                    float(path_result["margin_drop"] - matched["margin_drop"]),
            },
        "signed_winner_support":
            {"status": "ineligible_not_enough_positive_nodes"}
            if signed is None else {
                "status": "ok",
                "nodes": [int(i) for i in signed_nodes],
                **signed,
                "path_minus_control_drop":
                    float(path_result["margin_drop"] - signed["margin_drop"]),
                "set_jaccard": float(
                    len(set(path_nodes) & set(signed_nodes))
                    / len(set(path_nodes) | set(signed_nodes))
                ),
            },
    }

def evaluate_seed(name: str, seed: int):
    X, y = v09.dataset_for(name, seed)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.30,
        random_state=4000 + seed,
        stratify=y,
    )
    model = field_classifier().fit(X_train, y_train)

    records = []
    eligible_paths = 0
    for i, x in enumerate(X_test):
        rec = evaluate_query(model, X_train, y_train, x, i)
        records.append(rec)
        if rec["status"] == "ok":
            eligible_paths += 1
        if eligible_paths >= q.v09b.MAX_ELIGIBLE_QUERIES:
            break

    ok = [r for r in records if r["status"] == "ok"]
    matched_ok = [
        r for r in ok
        if r["matched_distance_label"]["status"] == "ok"
    ]
    signed_ok = [
        r for r in ok
        if r["signed_winner_support"]["status"] == "ok"
    ]

    return {
        "seed": int(seed),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "eligible_path_queries": int(len(ok)),
        "signed_evaluable_queries": int(len(signed_ok)),
        "signed_eligibility_rate":
            float(len(signed_ok) / len(ok)) if ok else None,
        "queries": records,
        "summary": {
            "path_margin_drop_all_paths":
                summarize([r["path"]["margin_drop"] for r in ok]),
            "matched_path_minus_control":
                summarize([
                    r["matched_distance_label"]["path_minus_control_drop"]
                    for r in matched_ok
                ]),
            "matched_control_margin_drop":
                summarize([
                    r["matched_distance_label"]["margin_drop"]
                    for r in matched_ok
                ]),
            "signed_path_minus_control":
                summarize([
                    r["signed_winner_support"]["path_minus_control_drop"]
                    for r in signed_ok
                ]),
            "signed_path_margin_drop":
                summarize([r["path"]["margin_drop"] for r in signed_ok]),
            "signed_control_margin_drop":
                summarize([
                    r["signed_winner_support"]["margin_drop"]
                    for r in signed_ok
                ]),
        },
    }

def aggregate(datasets):
    matched_units = []
    signed_units = []
    signed_path_drops = []
    signed_control_drops = []
    for dataset_name, block in datasets.items():
        for run in block["runs"]:
            m = run["summary"]["matched_path_minus_control"]
            if m["mean"] is not None:
                matched_units.append({
                    "dataset": dataset_name,
                    "seed": int(run["seed"]),
                    "value": float(m["mean"]),
                    "n_queries": int(m["n"]),
                })
            s = run["summary"]["signed_path_minus_control"]
            if s["mean"] is not None:
                signed_units.append({
                    "dataset": dataset_name,
                    "seed": int(run["seed"]),
                    "value": float(s["mean"]),
                    "n_queries": int(s["n"]),
                })
                signed_path_drops.append(
                    float(run["summary"]["signed_path_margin_drop"]["mean"])
                )
                signed_control_drops.append(
                    float(run["summary"]["signed_control_margin_drop"]["mean"])
                )

    return {
        "matched_distance_label": {
            "dataset_seed_units": matched_units,
            "summary": summarize([u["value"] for u in matched_units]),
        },
        "signed_winner_support": {
            "dataset_seed_units": signed_units,
            "summary": summarize([u["value"] for u in signed_units]),
            "mean_path_margin_drop_across_units":
                float(np.mean(signed_path_drops))
                if signed_path_drops else None,
            "mean_control_margin_drop_across_units":
                float(np.mean(signed_control_drops))
                if signed_control_drops else None,
        },
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "v30_field_only_faithfulness.json",
    )
    args = ap.parse_args()

    datasets = {}
    for name in q.v09b.DATASETS:
        print(f"DATASET {name}", flush=True)
        runs = []
        for seed in q.v09b.SEEDS:
            print(f"  seed={seed}", flush=True)
            run = evaluate_seed(name, seed)
            runs.append(run)
            print(
                "    matched="
                f"{run['summary']['matched_path_minus_control']['mean']} "
                "signed="
                f"{run['summary']['signed_path_minus_control']['mean']}",
                flush=True,
            )
        datasets[name] = {"runs": runs}

    result = {
        "protocol": {
            "version": "v3.0-faithfulness",
            "status":
                "final field-only delete-and-refit confirmation",
            "datasets": list(q.v09b.DATASETS),
            "seeds": list(q.v09b.SEEDS),
            "split":
                "70/30 stratified, random_state=4000+seed",
            "predictive_margin":
                "fixed original winner-runner local evidence "
                "L_winner(x)-L_runner(x)",
            "field_only_definition":
                "K=12, alpha=0.85, nD=40, use_walkers=False",
            "representative_path":
                "same deterministic graph-evidence path construction",
            "signed_control":
                "top positive affinity-weighted local winner-minus-runner "
                "contributions; overlap with path allowed",
            "primary_inference_unit": "dataset x seed",
            "selection_from_results": False,
        },
        "datasets": datasets,
        "dataset_seed_inference": aggregate(datasets),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print("\n=== FINAL FIELD-ONLY DELETE-AND-REFIT ===")
    for key, block in result["dataset_seed_inference"].items():
        s = block["summary"]
        print(
            f"{key:24s} n={s['n']} mean={s['mean']} "
            f"W/T/L={s['wins']}/{s['ties']}/{s['losses']} "
            f"p={s['sign_test_p_two_sided']}"
        )
    sw = result["dataset_seed_inference"]["signed_winner_support"]
    print(
        "signed mean path drop="
        f"{sw['mean_path_margin_drop_across_units']} "
        "control drop="
        f"{sw['mean_control_margin_drop_across_units']}"
    )
    print(f"FULL_JSON={args.output.resolve()}")

if __name__ == "__main__":
    main()
