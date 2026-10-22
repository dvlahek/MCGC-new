from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split

from mcgc import MultiAgentChemotacticGraphClassifier


ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / "examples" / "benchmark_xai_stability_v15.py"


def _load_parent():
    spec = importlib.util.spec_from_file_location("benchmark_xai_stability_v15", PARENT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


v15 = _load_parent()
DATASETS = tuple(v15.DATASETS)
SEEDS = list(v15.SEEDS)


def make_model(use_walkers: bool):
    return MultiAgentChemotacticGraphClassifier(
        n_neighbors=12,
        n_walkers=12,
        max_steps=10,
        diffusion_alpha=0.85,
        diffusion_steps=40,
        temperature=0.35,
        random_state=17,
        adaptive_walker_gate=True,
        use_walkers=use_walkers,
    )


def top_two_margin(scores):
    ordered = np.sort(np.asarray(scores, dtype=float), axis=1)
    return ordered[:, -1] - ordered[:, -2]


def evaluate_seed(name: str, seed: int):
    X, y = v15.dataset_for(name, seed)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=7000 + seed, stratify=y
    )

    full = make_model(True).fit(X_train, y_train)
    field = make_model(False).fit(X_train, y_train)

    full_scores = full.decision_function(X_test)
    field_scores = field.decision_function(X_test)
    full_pred = full.classes_[np.argmax(full_scores, axis=1)]
    field_pred = field.classes_[np.argmax(field_scores, axis=1)]

    full_margin = top_two_margin(full_scores)
    field_margin = top_two_margin(field_scores)

    return {
        "seed": int(seed),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "full": {
            "accuracy": float(accuracy_score(y_test, full_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, full_pred)),
            "mean_top_two_margin": float(np.mean(full_margin)),
        },
        "field_only": {
            "accuracy": float(accuracy_score(y_test, field_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, field_pred)),
            "mean_top_two_margin": float(np.mean(field_margin)),
        },
        "full_minus_field": {
            "accuracy": float(accuracy_score(y_test, full_pred) - accuracy_score(y_test, field_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, full_pred) - balanced_accuracy_score(y_test, field_pred)),
            "mean_top_two_margin": float(np.mean(full_margin - field_margin)),
        },
        "prediction_disagreement_rate": float(np.mean(full_pred != field_pred)),
        "mean_absolute_margin_change": float(np.mean(np.abs(full_margin - field_margin))),
    }


def summarize_units(units, key):
    vals = np.asarray([u[key] for u in units], dtype=float)
    return {
        "n": int(len(vals)),
        "mean": float(np.mean(vals)),
        "median": float(np.median(vals)),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
        "positive_units": int(np.sum(vals > 1e-12)),
        "zero_units": int(np.sum(np.abs(vals) <= 1e-12)),
        "negative_units": int(np.sum(vals < -1e-12)),
    }


def aggregate(datasets):
    units = []
    for name, block in datasets.items():
        for run in block["runs"]:
            units.append({
                "dataset": name,
                "seed": int(run["seed"]),
                "accuracy_delta": float(run["full_minus_field"]["accuracy"]),
                "balanced_accuracy_delta": float(run["full_minus_field"]["balanced_accuracy"]),
                "mean_margin_delta": float(run["full_minus_field"]["mean_top_two_margin"]),
                "prediction_disagreement_rate": float(run["prediction_disagreement_rate"]),
                "mean_absolute_margin_change": float(run["mean_absolute_margin_change"]),
            })
    return {
        "dataset_seed_units": units,
        "accuracy_delta_full_minus_field": summarize_units(units, "accuracy_delta"),
        "balanced_accuracy_delta_full_minus_field": summarize_units(units, "balanced_accuracy_delta"),
        "mean_margin_delta_full_minus_field": summarize_units(units, "mean_margin_delta"),
        "prediction_disagreement_rate": summarize_units(units, "prediction_disagreement_rate"),
        "mean_absolute_margin_change": summarize_units(units, "mean_absolute_margin_change"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=Path("v25_walker_ablation.json"))
    args = ap.parse_args()

    datasets = {name: {"runs": [evaluate_seed(name, seed) for seed in SEEDS]} for name in DATASETS}
    result = {
        "protocol": {
            "version": "v2.5-C",
            "status": "frozen final reviewer audit",
            "datasets": list(DATASETS),
            "seeds": list(SEEDS),
            "split_seed": "7000 + seed",
            "full": "walkers enabled; adaptive gate enabled",
            "field_only": "walkers disabled; same graph/diffusion and score weights",
            "purpose": "predictive contribution ablation; no retuning",
        },
        "datasets": datasets,
        "aggregate": aggregate(datasets),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")

    print("=== MCGC v2.5 FIELD-ONLY VS FULL-WALKER ABLATION ===")
    for key, value in result["aggregate"].items():
        if key == "dataset_seed_units":
            continue
        print(key, json.dumps(value, sort_keys=True))
    print(f"FULL_JSON={args.output.resolve()}")


if __name__ == "__main__":
    main()
