from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from sklearn.datasets import load_breast_cancer, make_classification, make_moons
from sklearn.model_selection import train_test_split

from mcgc import MultiAgentChemotacticGraphClassifier
from mcgc.xai import explain_contrastive_mcgc, explain_mcgc


SEEDS = list(range(5))
MAX_ELIGIBLE_QUERIES = 12
DATASETS = ("moons_hard", "spirals_hard", "overlap_hd", "breast_cancer")


def make_spirals(n_samples: int = 700, noise: float = 0.16, random_state: int = 7):
    rng = np.random.default_rng(random_state)
    n = n_samples // 2
    theta = np.linspace(0.2, 3.2 * np.pi, n)
    radius = np.linspace(0.15, 1.0, n)
    arm0 = np.column_stack([radius * np.cos(theta), radius * np.sin(theta)])
    arm1 = -arm0
    X = np.vstack([arm0, arm1])
    X += rng.normal(scale=noise, size=X.shape)
    y = np.concatenate([np.zeros(n, dtype=int), np.ones(n, dtype=int)])
    order = rng.permutation(len(X))
    return X[order], y[order]


def dataset_for(name: str, seed: int):
    if name == "moons_hard":
        return make_moons(n_samples=700, noise=0.30, random_state=seed)
    if name == "spirals_hard":
        return make_spirals(n_samples=700, noise=0.16, random_state=seed)
    if name == "overlap_hd":
        return make_classification(
            n_samples=800,
            n_features=8,
            n_informative=4,
            n_redundant=2,
            n_classes=2,
            n_clusters_per_class=2,
            class_sep=0.75,
            flip_y=0.08,
            random_state=seed,
        )
    if name == "breast_cancer":
        data = load_breast_cancer()
        return data.data, data.target
    raise ValueError(name)


def classifier():
    return MultiAgentChemotacticGraphClassifier(
        n_neighbors=12,
        n_walkers=12,
        max_steps=10,
        diffusion_steps=40,
        random_state=17,
        adaptive_walker_gate=True,
    )


def _plain(value):
    return value.item() if hasattr(value, "item") else value


def class_margin(model, x, positive, negative):
    _, explanations = explain_mcgc(model, np.asarray(x)[None, :], top_support=5)
    scores = explanations[0]["class_scores"]
    return float(scores[_plain(positive)] - scores[_plain(negative)])


def matched_controls(model, x, path_nodes):
    path_nodes = [int(i) for i in dict.fromkeys(path_nodes)]
    excluded = set(path_nodes)
    xs = model._scaled(np.asarray(x)[None, :])[0]
    distances = np.linalg.norm(model.X_ - xs, axis=1)
    chosen = []
    used = set(excluded)

    for node in path_nodes:
        label_index = int(model.y_index_[node])
        candidates = [
            j
            for j in range(len(model.X_))
            if j not in used and int(model.y_index_[j]) == label_index
        ]
        if not candidates:
            candidates = [j for j in range(len(model.X_)) if j not in used]
        if not candidates:
            break
        target_distance = float(distances[node])
        pick = min(candidates, key=lambda j: abs(float(distances[j]) - target_distance))
        chosen.append(int(pick))
        used.add(int(pick))

    return chosen


def nearest_off_path_controls(model, x, path_nodes):
    path_nodes = [int(i) for i in dict.fromkeys(path_nodes)]
    excluded = set(path_nodes)
    xs = model._scaled(np.asarray(x)[None, :])[0]
    distances = np.linalg.norm(model.X_ - xs, axis=1)
    order = np.argsort(distances, kind="stable")
    result = [int(j) for j in order if int(j) not in excluded]
    return result[: len(path_nodes)]


def refit_without(X_train, y_train, removed):
    mask = np.ones(len(X_train), dtype=bool)
    mask[np.asarray(sorted(set(removed)), dtype=int)] = False
    return classifier().fit(X_train[mask], y_train[mask])


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
            "n": 0,
            "mean": None,
            "median": None,
            "std": None,
            "min": None,
            "max": None,
            "wins": 0,
            "ties": 0,
            "losses": 0,
            "sign_test_p_two_sided": None,
        }
    eps = 1e-12
    wins = int(np.sum(arr > eps))
    ties = int(np.sum(np.abs(arr) <= eps))
    losses = int(np.sum(arr < -eps))
    return {
        "n": int(len(arr)),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "sign_test_p_two_sided": sign_test_p(wins, losses),
    }


def evaluate_query(model, X_train, y_train, x, query_index):
    pred, explanations = explain_mcgc(model, np.asarray(x)[None, :], top_support=5)
    explanation = explanations[0]
    winner = explanation["prediction"]
    runner = explanation["runner_up"]
    trail = explanation["classes"][winner]["representative_trail"]
    path_nodes = [int(i) for i in dict.fromkeys(trail["path"])]

    if len(path_nodes) < 2:
        return {"query_index": int(query_index), "status": "ineligible_short_path"}

    _, contrastive = explain_contrastive_mcgc(
        model,
        np.asarray(x)[None, :],
        positive_class=winner,
        negative_class=runner,
    )
    exact_error = abs(
        float(contrastive[0]["score_margin"])
        - float(contrastive[0]["reconstructed_margin"])
    )

    original_margin = float(explanation["class_scores"][winner] - explanation["class_scores"][runner])
    matched = matched_controls(model, x, path_nodes)
    nearest = nearest_off_path_controls(model, x, path_nodes)

    if len(matched) != len(path_nodes) or len(nearest) != len(path_nodes):
        return {"query_index": int(query_index), "status": "ineligible_control_match"}

    path_model = refit_without(X_train, y_train, path_nodes)
    matched_model = refit_without(X_train, y_train, matched)
    nearest_model = refit_without(X_train, y_train, nearest)

    margin_path = class_margin(path_model, x, winner, runner)
    margin_matched = class_margin(matched_model, x, winner, runner)
    margin_nearest = class_margin(nearest_model, x, winner, runner)

    drop_path = original_margin - margin_path
    drop_matched = original_margin - margin_matched
    drop_nearest = original_margin - margin_nearest

    path_pred = _plain(path_model.predict(np.asarray(x)[None, :])[0])
    matched_pred = _plain(matched_model.predict(np.asarray(x)[None, :])[0])
    nearest_pred = _plain(nearest_model.predict(np.asarray(x)[None, :])[0])

    return {
        "query_index": int(query_index),
        "status": "ok",
        "winner": _plain(winner),
        "runner_up": _plain(runner),
        "original_prediction": _plain(pred[0]),
        "original_margin": original_margin,
        "path_nodes": path_nodes,
        "path_length": int(len(path_nodes)),
        "matched_control_nodes": matched,
        "nearest_off_path_nodes": nearest,
        "margin_after_path_removal": margin_path,
        "margin_after_matched_removal": margin_matched,
        "margin_after_nearest_removal": margin_nearest,
        "margin_drop_path": drop_path,
        "margin_drop_matched": drop_matched,
        "margin_drop_nearest": drop_nearest,
        "faithfulness_advantage": float(drop_path - drop_matched),
        "path_vs_nearest_advantage": float(drop_path - drop_nearest),
        "path_prediction_flip": bool(path_pred != winner),
        "matched_prediction_flip": bool(matched_pred != winner),
        "nearest_prediction_flip": bool(nearest_pred != winner),
        "contrastive_reconstruction_abs_error": float(exact_error),
    }


def evaluate_seed(name: str, seed: int):
    X, y = dataset_for(name, seed)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=4000 + seed,
        stratify=y,
    )
    model = classifier().fit(X_train, y_train)

    records = []
    inspected = 0
    eligible = 0
    for i, x in enumerate(X_test):
        inspected += 1
        record = evaluate_query(model, X_train, y_train, x, i)
        records.append(record)
        if record["status"] == "ok":
            eligible += 1
        if eligible >= MAX_ELIGIBLE_QUERIES:
            break

    ok = [r for r in records if r["status"] == "ok"]
    return {
        "seed": seed,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "inspected_queries": int(inspected),
        "eligible_queries": int(len(ok)),
        "eligibility_rate": float(len(ok) / inspected) if inspected else 0.0,
        "queries": records,
        "summary": {
            "faithfulness_advantage": summarize([r["faithfulness_advantage"] for r in ok]),
            "path_vs_nearest_advantage": summarize([r["path_vs_nearest_advantage"] for r in ok]),
            "path_flip_rate": float(np.mean([r["path_prediction_flip"] for r in ok])) if ok else None,
            "matched_flip_rate": float(np.mean([r["matched_prediction_flip"] for r in ok])) if ok else None,
            "nearest_flip_rate": float(np.mean([r["nearest_prediction_flip"] for r in ok])) if ok else None,
            "mean_path_length": float(np.mean([r["path_length"] for r in ok])) if ok else None,
            "max_contrastive_reconstruction_abs_error": (
                float(max(r["contrastive_reconstruction_abs_error"] for r in ok)) if ok else None
            ),
        },
    }


def evaluate_dataset(name: str):
    runs = [evaluate_seed(name, seed) for seed in SEEDS]
    ok = [r for run in runs for r in run["queries"] if r["status"] == "ok"]
    inspected = sum(run["inspected_queries"] for run in runs)
    return {
        "runs": runs,
        "pooled": {
            "inspected_queries": int(inspected),
            "eligible_queries": int(len(ok)),
            "eligibility_rate": float(len(ok) / inspected) if inspected else 0.0,
            "faithfulness_advantage": summarize([r["faithfulness_advantage"] for r in ok]),
            "path_vs_nearest_advantage": summarize([r["path_vs_nearest_advantage"] for r in ok]),
            "path_flip_rate": float(np.mean([r["path_prediction_flip"] for r in ok])) if ok else None,
            "matched_flip_rate": float(np.mean([r["matched_prediction_flip"] for r in ok])) if ok else None,
            "nearest_flip_rate": float(np.mean([r["nearest_prediction_flip"] for r in ok])) if ok else None,
            "mean_path_length": float(np.mean([r["path_length"] for r in ok])) if ok else None,
            "max_contrastive_reconstruction_abs_error": (
                float(max(r["contrastive_reconstruction_abs_error"] for r in ok)) if ok else None
            ),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("v09_xai_faithfulness.json"))
    args = parser.parse_args()

    datasets = {name: evaluate_dataset(name) for name in DATASETS}
    all_ok = [
        r
        for dataset in datasets.values()
        for run in dataset["runs"]
        for r in run["queries"]
        if r["status"] == "ok"
    ]
    result = {
        "protocol": {
            "version": "0.9",
            "status": "frozen XAI faithfulness",
            "datasets": DATASETS,
            "seeds": SEEDS,
            "max_eligible_queries_per_seed": MAX_ELIGIBLE_QUERIES,
            "split": "70% train / 30% untouched test",
            "split_seed": "4000 + dataset_seed",
            "primary_contrast": (
                "margin_drop(path_nodes) - margin_drop(label-and-distance-matched off-path nodes)"
            ),
            "secondary_contrast": (
                "margin_drop(path_nodes) - margin_drop(equal-size nearest off-path nodes)"
            ),
            "note": (
                "Classifier and representative-trail construction are unchanged. "
                "Each intervention deletes training nodes and refits the same classifier."
            ),
        },
        "datasets": datasets,
        "pooled": {
            "n": int(len(all_ok)),
            "faithfulness_advantage": summarize([r["faithfulness_advantage"] for r in all_ok]),
            "path_vs_nearest_advantage": summarize([r["path_vs_nearest_advantage"] for r in all_ok]),
            "path_flip_rate": float(np.mean([r["path_prediction_flip"] for r in all_ok])) if all_ok else None,
            "matched_flip_rate": float(np.mean([r["matched_prediction_flip"] for r in all_ok])) if all_ok else None,
            "nearest_flip_rate": float(np.mean([r["nearest_prediction_flip"] for r in all_ok])) if all_ok else None,
        },
    }

    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print("=== MCGC v0.9 XAI FAITHFULNESS ===")
    for name in DATASETS:
        p = datasets[name]["pooled"]
        a = p["faithfulness_advantage"]
        n = p["path_vs_nearest_advantage"]
        print(
            f"{name:16s} eligible={p['eligible_queries']:3d}/{p['inspected_queries']:3d} "
            f"matched_delta={a['mean']!s:>10} W/T/L={a['wins']}/{a['ties']}/{a['losses']} "
            f"nearest_delta={n['mean']!s:>10} W/T/L={n['wins']}/{n['ties']}/{n['losses']}"
        )
    print("=== POOLED ===")
    print(json.dumps(result["pooled"], indent=2, sort_keys=True))
    print(f"FULL_JSON={args.output.resolve()}")


if __name__ == "__main__":
    main()
