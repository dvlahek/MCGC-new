from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / "examples" / "benchmark_xai_stability_v15.py"


def _load_parent():
    spec = importlib.util.spec_from_file_location("benchmark_xai_stability_v15", PARENT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


v15 = _load_parent()
SEEDS = list(v15.SEEDS)
DATASETS = tuple(v15.DATASETS)
TOP_SUPPORT = int(v15.TOP_SUPPORT)
MAX_STABILITY_QUERIES = int(v15.MAX_STABILITY_QUERIES)
PERTURB_EPS = tuple(v15.PERTURB_EPS)
PERTURB_REPS = int(v15.PERTURB_REPS)


def knn_support_nodes(model, x):
    xs = model._scaled(np.asarray(x, dtype=float)[None, :])[0]
    distances, starts = model._query_start_nodes(xs)
    order = np.argsort(distances, kind="stable")
    return [int(starts[i]) for i in order[: min(TOP_SUPPORT, len(starts))]]


def nearest_same_prediction_controls(model, X_test, predictions):
    Xs = model._scaled(np.asarray(X_test, dtype=float))
    controls = {}
    for i in range(len(X_test)):
        same = [j for j in range(len(X_test)) if j != i and v15.py_scalar(predictions[j]) == v15.py_scalar(predictions[i])]
        if not same:
            continue
        d = np.linalg.norm(Xs[same] - Xs[i], axis=1)
        controls[i] = int(same[int(np.argmin(d))])
    return controls


def mean_or_none(values):
    return None if not values else float(np.mean(values))


def evaluate_seed(name: str, seed: int):
    X, y = v15.dataset_for(name, seed)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=7000 + seed, stratify=y
    )
    model = v15.mcgc_classifier().fit(X_train, y_train)

    predictions = []
    explanations = []
    support_knn_identity = []
    for x in X_test:
        pred, exp = v15.explain_one(model, x)
        predictions.append(pred)
        explanations.append(exp)
        support_knn_identity.append(v15.support_nodes(exp) == knn_support_nodes(model, x))
    predictions = np.asarray(predictions)

    controls = nearest_same_prediction_controls(model, X_test, predictions)
    feature_scale = np.std(X_train, axis=0, ddof=0)
    feature_scale = np.where(feature_scale > 0.0, feature_scale, 1.0)
    selected = list(range(min(MAX_STABILITY_QUERIES, len(X_test))))

    query_records = []
    support_advantages = []
    path_advantages = []
    within_support_all = []
    cross_support_all = []
    within_path_all = []
    cross_path_all = []
    perturb_knn_identity = []

    for query_idx in selected:
        x = X_test[query_idx]
        original_pred = v15.py_scalar(predictions[query_idx])
        original = explanations[query_idx]
        original_support = v15.support_nodes(original)
        original_path = v15.winner_trail(original)
        original_path_ok = v15.nontrivial_trail(original_path)

        control_idx = controls.get(query_idx)
        cross_support = None
        cross_path = None
        if control_idx is not None:
            control = explanations[control_idx]
            cross_support = v15.node_set_jaccard(original_support, v15.support_nodes(control))
            control_path = v15.winner_trail(control)
            if original_path_ok and v15.nontrivial_trail(control_path):
                cross_path = v15.node_set_jaccard(original_path["path"], control_path["path"])

        retained_support = []
        retained_path = []
        preserved = 0
        attempted = 0
        for eps_idx, eps in enumerate(PERTURB_EPS):
            for rep in range(PERTURB_REPS):
                attempted += 1
                rng = v15.perturb_rng(name, seed, query_idx, eps_idx, rep)
                xp = x + eps * feature_scale * rng.normal(size=x.shape)
                pert_pred, pert = v15.explain_one(model, xp)
                if pert_pred != original_pred:
                    continue
                preserved += 1
                pert_support = v15.support_nodes(pert)
                perturb_knn_identity.append(pert_support == knn_support_nodes(model, xp))
                retained_support.append(v15.node_set_jaccard(original_support, pert_support))
                pert_path = v15.winner_trail(pert)
                if original_path_ok and v15.nontrivial_trail(pert_path):
                    retained_path.append(v15.node_set_jaccard(original_path["path"], pert_path["path"]))

        mean_within_support = mean_or_none(retained_support)
        mean_within_path = mean_or_none(retained_path)
        support_adv = None if mean_within_support is None or cross_support is None else float(mean_within_support - cross_support)
        path_adv = None if mean_within_path is None or cross_path is None else float(mean_within_path - cross_path)
        if support_adv is not None:
            support_advantages.append(support_adv)
            within_support_all.append(mean_within_support)
            cross_support_all.append(cross_support)
        if path_adv is not None:
            path_advantages.append(path_adv)
            within_path_all.append(mean_within_path)
            cross_path_all.append(cross_path)

        query_records.append({
            "query_index": int(query_idx),
            "prediction": v15.py_scalar(original_pred),
            "true_label": v15.py_scalar(y_test[query_idx]),
            "control_query_index": None if control_idx is None else int(control_idx),
            "prediction_preservation_rate": float(preserved / attempted),
            "support_equals_plain_knn": bool(support_knn_identity[query_idx]),
            "mean_within_support_jaccard": mean_within_support,
            "distance_matched_cross_support_jaccard": cross_support,
            "support_within_minus_cross": support_adv,
            "original_path_eligible": bool(original_path_ok),
            "mean_within_path_jaccard": mean_within_path,
            "distance_matched_cross_path_jaccard": cross_path,
            "path_within_minus_cross": path_adv,
        })

    return {
        "seed": int(seed),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "support_equals_plain_knn_all_test": bool(all(support_knn_identity)),
        "support_equals_plain_knn_rate_all_test": float(np.mean(support_knn_identity)),
        "support_equals_plain_knn_rate_preserved_perturbations": float(np.mean(perturb_knn_identity)) if perturb_knn_identity else None,
        "summary": {
            "n_selected_queries": int(len(selected)),
            "evaluable_support_queries": int(len(support_advantages)),
            "mean_within_support_jaccard": mean_or_none(within_support_all),
            "mean_distance_matched_cross_support_jaccard": mean_or_none(cross_support_all),
            "mean_support_advantage": mean_or_none(support_advantages),
            "evaluable_path_queries": int(len(path_advantages)),
            "mean_within_path_jaccard": mean_or_none(within_path_all),
            "mean_distance_matched_cross_path_jaccard": mean_or_none(cross_path_all),
            "mean_path_advantage": mean_or_none(path_advantages),
        },
        "queries": query_records,
    }


def aggregate(datasets):
    support_units = []
    path_units = []
    identities = []
    for name, data in datasets.items():
        for run in data["runs"]:
            identities.append(run["support_equals_plain_knn_rate_all_test"])
            s = run["summary"]["mean_support_advantage"]
            p = run["summary"]["mean_path_advantage"]
            if s is not None:
                support_units.append({"dataset": name, "seed": run["seed"], "value": float(s)})
            if p is not None:
                path_units.append({"dataset": name, "seed": run["seed"], "value": float(p)})
    return {
        "support_equals_plain_knn_rate_across_seed_units": float(np.mean(identities)) if identities else None,
        "support_advantage_dataset_seed_units": support_units,
        "path_advantage_dataset_seed_units": path_units,
        "support_advantage_mean": float(np.mean([u["value"] for u in support_units])) if support_units else None,
        "path_advantage_mean": float(np.mean([u["value"] for u in path_units])) if path_units else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=Path("v15bis_xai_stability.json"))
    args = ap.parse_args()

    datasets = {name: {"runs": [evaluate_seed(name, seed) for seed in SEEDS]} for name in DATASETS}
    result = {
        "protocol": {
            "version": "1.5-bis / v2.3-B",
            "status": "frozen reviewer-response control audit",
            "parent": "benchmark_xai_stability_v15.py",
            "datasets": DATASETS,
            "seeds": SEEDS,
            "perturb_eps": PERTURB_EPS,
            "perturb_reps": PERTURB_REPS,
            "max_stability_queries": MAX_STABILITY_QUERIES,
            "cross_query_control": "nearest same-predicted-class test query in standardized feature space",
            "support_baseline": "plain standardized kNN top-5",
        },
        "datasets": datasets,
        "aggregate": aggregate(datasets),
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print("=== MCGC v1.5-bis XAI STABILITY CONTROL AUDIT ===")
    print(json.dumps(result["aggregate"], indent=2, sort_keys=True))
    print(f"FULL_JSON={args.output.resolve()}")


if __name__ == "__main__":
    main()
