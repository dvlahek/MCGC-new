from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / "examples" / "benchmark_xai_stability_v15bis.py"


def _load_parent():
    spec = importlib.util.spec_from_file_location("benchmark_xai_stability_v15bis", PARENT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


v15b = _load_parent()
v15 = v15b.v15
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
        return {"n": 0, "mean": None, "median": None, "wins": 0, "ties": 0, "losses": 0, "sign_test_p_two_sided": None}
    wins = int(np.sum(arr > EPS))
    ties = int(np.sum(np.abs(arr) <= EPS))
    losses = int(np.sum(arr < -EPS))
    return {
        "n": int(len(arr)),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "sign_test_p_two_sided": sign_test_p(wins, losses),
    }


def trail_path(exp):
    t = v15.winner_trail(exp)
    return [int(i) for i in t["path"]] if v15.nontrivial_trail(t) else None


def tail(path):
    if path is None or len(path) < 2:
        return None
    out = [int(i) for i in path[1:]]
    return out if out else None


def jaccard_or_none(a, b):
    if a is None or b is None:
        return None
    return v15.node_set_jaccard(a, b)


def mean_or_none(xs):
    return None if not xs else float(np.mean(xs))


def evaluate_seed(name: str, seed: int):
    from sklearn.model_selection import train_test_split

    X, y = v15.dataset_for(name, seed)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=7000 + seed, stratify=y
    )
    model = v15.mcgc_classifier().fit(X_train, y_train)

    predictions = []
    explanations = []
    for x in X_test:
        pred, exp = v15.explain_one(model, x)
        predictions.append(pred)
        explanations.append(exp)
    predictions = np.asarray(predictions)
    controls = v15b.nearest_same_prediction_controls(model, X_test, predictions)

    feature_scale = np.std(X_train, axis=0, ddof=0)
    feature_scale = np.where(feature_scale > 0.0, feature_scale, 1.0)
    selected = list(range(min(v15b.MAX_STABILITY_QUERIES, len(X_test))))

    query_records = []
    full_adv = []
    tail_adv = []
    same_start_tail_adv = []

    for query_idx in selected:
        x = X_test[query_idx]
        original_pred = v15.py_scalar(predictions[query_idx])
        original = explanations[query_idx]
        opath = trail_path(original)
        otail = tail(opath)
        if opath is None:
            query_records.append({"query_index": int(query_idx), "status": "ineligible_original_path"})
            continue

        control_idx = controls.get(query_idx)
        cpath = None if control_idx is None else trail_path(explanations[control_idx])
        ctail = tail(cpath)
        cross_full = jaccard_or_none(opath, cpath)
        cross_tail = jaccard_or_none(otail, ctail)
        cross_same_start_tail = None
        if opath is not None and cpath is not None and opath[0] == cpath[0]:
            cross_same_start_tail = cross_tail

        within_full_vals = []
        within_tail_vals = []
        within_same_start_tail_vals = []
        preserved = 0
        attempted = 0
        for eps_idx, eps in enumerate(v15b.PERTURB_EPS):
            for rep in range(v15b.PERTURB_REPS):
                attempted += 1
                rng = v15.perturb_rng(name, seed, query_idx, eps_idx, rep)
                xp = x + eps * feature_scale * rng.normal(size=x.shape)
                pert_pred, pert = v15.explain_one(model, xp)
                if pert_pred != original_pred:
                    continue
                preserved += 1
                ppath = trail_path(pert)
                if ppath is None:
                    continue
                within_full_vals.append(v15.node_set_jaccard(opath, ppath))
                ptail = tail(ppath)
                jt = jaccard_or_none(otail, ptail)
                if jt is not None:
                    within_tail_vals.append(jt)
                    if ppath[0] == opath[0]:
                        within_same_start_tail_vals.append(jt)

        mean_within_full = mean_or_none(within_full_vals)
        mean_within_tail = mean_or_none(within_tail_vals)
        mean_within_same_start_tail = mean_or_none(within_same_start_tail_vals)

        fadv = None if mean_within_full is None or cross_full is None else float(mean_within_full - cross_full)
        tadv = None if mean_within_tail is None or cross_tail is None else float(mean_within_tail - cross_tail)
        ssadv = None if mean_within_same_start_tail is None or cross_same_start_tail is None else float(mean_within_same_start_tail - cross_same_start_tail)

        if fadv is not None:
            full_adv.append(fadv)
        if tadv is not None:
            tail_adv.append(tadv)
        if ssadv is not None:
            same_start_tail_adv.append(ssadv)

        query_records.append({
            "query_index": int(query_idx),
            "status": "ok",
            "prediction": v15.py_scalar(original_pred),
            "control_query_index": None if control_idx is None else int(control_idx),
            "prediction_preservation_rate": float(preserved / attempted),
            "original_start_node": int(opath[0]),
            "control_start_node": None if cpath is None else int(cpath[0]),
            "control_same_start": bool(cpath is not None and opath[0] == cpath[0]),
            "mean_within_full_jaccard": mean_within_full,
            "cross_full_jaccard": cross_full,
            "full_within_minus_cross": fadv,
            "mean_within_tail_jaccard": mean_within_tail,
            "cross_tail_jaccard": cross_tail,
            "tail_within_minus_cross": tadv,
            "same_start_within_tail_jaccard": mean_within_same_start_tail,
            "same_start_cross_tail_jaccard": cross_same_start_tail,
            "same_start_tail_within_minus_cross": ssadv,
            "same_start_preserved_perturbations": int(len(within_same_start_tail_vals)),
        })

    return {
        "seed": int(seed),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "summary": {
            "full_path_advantage": summarize(full_adv),
            "tail_path_advantage": summarize(tail_adv),
            "same_start_tail_advantage": summarize(same_start_tail_adv),
        },
        "queries": query_records,
    }


def aggregate(datasets):
    result = {}
    for metric in ("full_path_advantage", "tail_path_advantage", "same_start_tail_advantage"):
        units = []
        for name, block in datasets.items():
            for run in block["runs"]:
                s = run["summary"][metric]
                if s["mean"] is not None:
                    units.append({"dataset": name, "seed": int(run["seed"]), "value": float(s["mean"]), "n_queries": int(s["n"])})
        result[metric] = {
            "dataset_seed_units": units,
            "summary": summarize([u["value"] for u in units]),
        }
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=Path("v15ter_path_tail_stability.json"))
    args = ap.parse_args()

    datasets = {name: {"runs": [evaluate_seed(name, seed) for seed in v15b.SEEDS]} for name in v15b.DATASETS}
    result = {
        "protocol": {
            "version": "1.5-ter / v2.5-B",
            "status": "frozen final reviewer audit",
            "datasets": list(v15b.DATASETS),
            "seeds": list(v15b.SEEDS),
            "cross_query_control": "nearest same-predicted-class test query in standardized feature space",
            "tail_definition": "representative path after removing first node",
            "same_start_condition": "tail comparison retained only when compared paths share the same first node",
            "primary_inference_unit": "dataset x seed",
        },
        "datasets": datasets,
        "dataset_seed_inference": aggregate(datasets),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")

    print("=== MCGC v1.5-ter PATH-TAIL STABILITY ===")
    for key, block in result["dataset_seed_inference"].items():
        s = block["summary"]
        print(f"{key:28s} n={s['n']} mean={s['mean']} median={s['median']} W/T/L={s['wins']}/{s['ties']}/{s['losses']} p={s['sign_test_p_two_sided']}")
    print(f"FULL_JSON={args.output.resolve()}")


if __name__ == "__main__":
    main()
