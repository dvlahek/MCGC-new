from __future__ import annotations

import json

import numpy as np
from sklearn.datasets import make_circles, make_moons
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from mcgc import MultiAgentChemotacticGraphClassifier, reliability_features
from benchmark_ablation_v02 import make_spirals


SEEDS = list(range(12))
COVERAGES = (0.9, 0.8, 0.7, 0.5)
FEATURE_SETS = {
    "local_only": ["local_margin_risk"],
    "local_plus_final": ["local_margin_risk", "final_margin_risk"],
    "local_plus_trail_disagreement": [
        "local_margin_risk",
        "trail_disagreement_risk",
    ],
    "combined_intrinsic": [
        "local_margin_risk",
        "final_margin_risk",
        "trail_disagreement_risk",
        "coherence_risk",
        "lost_trail_risk",
    ],
}


def dataset_for(name, seed):
    if name == "moons":
        return make_moons(n_samples=500, noise=0.18, random_state=seed)
    if name == "circles":
        return make_circles(
            n_samples=500, noise=0.10, factor=0.45, random_state=seed
        )
    if name == "spirals":
        return make_spirals(n_samples=500, noise=0.10, random_state=seed)
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


def feature_matrix(rows, names):
    return np.asarray(
        [[row["risk_scores"][name] for name in names] for row in rows],
        dtype=float,
    )


def selective_accuracy(correct, risk_probability):
    correct = np.asarray(correct, dtype=bool)
    order = np.argsort(np.asarray(risk_probability, dtype=float))
    result = {}
    for coverage in COVERAGES:
        keep = max(1, int(np.ceil(coverage * len(correct))))
        idx = order[:keep]
        result[str(coverage)] = {
            "kept": int(keep),
            "accuracy": float(np.mean(correct[idx])),
            "empirical_coverage": float(keep / len(correct)),
        }
    return result


def risk_model():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            class_weight="balanced",
            C=1.0,
            max_iter=2000,
            random_state=71,
        ),
    )


def evaluate_seed(name, seed):
    X, y = dataset_for(name, seed)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.40,
        random_state=2000 + seed,
        stratify=y,
    )
    X_cal, X_test, y_cal, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=3000 + seed,
        stratify=y_temp,
    )

    fitted = classifier().fit(X_train, y_train)
    pred_cal, rows_cal = reliability_features(fitted, X_cal)
    pred_test, rows_test = reliability_features(fitted, X_test)
    error_cal = np.asarray(pred_cal != y_cal, dtype=int)
    error_test = np.asarray(pred_test != y_test, dtype=int)
    correct_test = error_test == 0

    record = {
        "seed": seed,
        "n_train": int(len(X_train)),
        "n_calibration": int(len(X_cal)),
        "n_test": int(len(X_test)),
        "calibration_errors": int(np.sum(error_cal)),
        "test_errors": int(np.sum(error_test)),
        "test_accuracy": float(np.mean(correct_test)),
        "status": "ok",
        "models": {},
    }

    if len(np.unique(error_cal)) < 2:
        record["status"] = "insufficient_calibration_classes"
        return record

    for model_name, names in FEATURE_SETS.items():
        fitted_risk = risk_model().fit(feature_matrix(rows_cal, names), error_cal)
        probability = fitted_risk.predict_proba(feature_matrix(rows_test, names))[:, 1]
        metrics = {
            "features": names,
            "brier": float(brier_score_loss(error_test, probability)),
            "selective_accuracy": selective_accuracy(correct_test, probability),
        }
        if len(np.unique(error_test)) < 2:
            metrics["roc_auc"] = None
            metrics["average_precision"] = None
        else:
            metrics["roc_auc"] = float(roc_auc_score(error_test, probability))
            metrics["average_precision"] = float(
                average_precision_score(error_test, probability)
            )
        record["models"][model_name] = metrics

    return record


def delta_summary(values):
    arr = np.asarray(values, dtype=float)
    eps = 1e-12
    if len(arr) == 0:
        return {
            "n": 0,
            "mean": None,
            "median": None,
            "std": None,
            "wins": 0,
            "ties": 0,
            "losses": 0,
        }
    return {
        "n": int(len(arr)),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "wins": int(np.sum(arr > eps)),
        "ties": int(np.sum(np.abs(arr) <= eps)),
        "losses": int(np.sum(arr < -eps)),
    }


def evaluate_dataset(name):
    runs = [evaluate_seed(name, seed) for seed in SEEDS]
    primary_delta = []
    final_delta = []
    combined_delta = []

    for run in runs:
        if run["status"] != "ok":
            continue
        local_auc = run["models"]["local_only"]["roc_auc"]
        if local_auc is None:
            continue
        trail_auc = run["models"]["local_plus_trail_disagreement"]["roc_auc"]
        final_auc = run["models"]["local_plus_final"]["roc_auc"]
        combined_auc = run["models"]["combined_intrinsic"]["roc_auc"]
        primary_delta.append(trail_auc - local_auc)
        final_delta.append(final_auc - local_auc)
        combined_delta.append(combined_auc - local_auc)

    model_summaries = {}
    for model_name in FEATURE_SETS:
        aucs = [
            run["models"][model_name]["roc_auc"]
            for run in runs
            if run["status"] == "ok"
            and run["models"][model_name]["roc_auc"] is not None
        ]
        aps = [
            run["models"][model_name]["average_precision"]
            for run in runs
            if run["status"] == "ok"
            and run["models"][model_name]["average_precision"] is not None
        ]
        briers = [
            run["models"][model_name]["brier"]
            for run in runs
            if run["status"] == "ok"
        ]
        model_summaries[model_name] = {
            "evaluable_test_auc_seeds": len(aucs),
            "mean_test_roc_auc": float(np.mean(aucs)) if aucs else None,
            "mean_test_average_precision": float(np.mean(aps)) if aps else None,
            "mean_test_brier": float(np.mean(briers)) if briers else None,
        }

    return {
        "runs": runs,
        "status_counts": {
            status: int(sum(run["status"] == status for run in runs))
            for status in sorted({run["status"] for run in runs})
        },
        "models": model_summaries,
        "paired_auc_deltas_vs_local_only": {
            "local_plus_trail_disagreement": delta_summary(primary_delta),
            "local_plus_final": delta_summary(final_delta),
            "combined_intrinsic": delta_summary(combined_delta),
        },
    }


def main():
    result = {
        "protocol": {
            "version": "0.7",
            "seeds": SEEDS,
            "split": "60% classifier train / 20% risk calibration / 20% untouched test",
            "classifier_train_seed": "2000 + dataset_seed",
            "calibration_test_seed": "3000 + dataset_seed",
            "risk_model": (
                "StandardScaler + LogisticRegression(class_weight='balanced', "
                "C=1.0, max_iter=2000, random_state=71)"
            ),
            "primary_contrast": (
                "test ROC AUC(local_plus_trail_disagreement) - "
                "test ROC AUC(local_only)"
            ),
            "note": (
                "Feature sets and comparisons were frozen before test evaluation. "
                "Seeds without both calibration correctness classes are skipped, "
                "not resplit."
            ),
        },
        "datasets": {
            name: evaluate_dataset(name) for name in ("moons", "circles", "spirals")
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
