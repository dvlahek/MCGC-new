from __future__ import annotations

import json
import math

import numpy as np
from sklearn.datasets import (
    load_breast_cancer,
    load_digits,
    load_wine,
    make_circles,
    make_classification,
    make_moons,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from mcgc import MultiAgentChemotacticGraphClassifier, reliability_features
from benchmark_ablation_v02 import make_spirals


SEEDS = list(range(8))
COVERAGES = (0.9, 0.8, 0.7, 0.5)
FEATURE_SETS = {
    "local_only": ["local_margin_risk"],
    "combined_intrinsic": [
        "local_margin_risk",
        "final_margin_risk",
        "trail_disagreement_risk",
        "coherence_risk",
        "lost_trail_risk",
    ],
}


def dataset_for(name, seed):
    if name == "moons_hard":
        return make_moons(n_samples=700, noise=0.30, random_state=seed)
    if name == "circles_hard":
        return make_circles(
            n_samples=700,
            noise=0.18,
            factor=0.50,
            random_state=seed,
        )
    if name == "spirals_hard":
        return make_spirals(n_samples=700, noise=0.16, random_state=seed)
    if name == "overlap_hd":
        return make_classification(
            n_samples=800,
            n_features=8,
            n_informative=4,
            n_redundant=2,
            n_repeated=0,
            n_classes=2,
            n_clusters_per_class=2,
            class_sep=0.75,
            flip_y=0.08,
            random_state=seed,
        )
    if name == "breast_cancer":
        data = load_breast_cancer()
        return data.data, data.target
    if name == "wine":
        data = load_wine()
        return data.data, data.target
    if name == "digits_0_4":
        data = load_digits()
        mask = data.target <= 4
        return data.data[mask], data.target[mask]
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


def selective_accuracy(correct, risk):
    correct = np.asarray(correct, dtype=bool)
    risk = np.asarray(risk, dtype=float)
    order = np.argsort(risk, kind="stable")
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


def safe_error_metrics(error, risk, *, brier=False):
    error = np.asarray(error, dtype=int)
    risk = np.asarray(risk, dtype=float)
    result = {
        "selective_accuracy": selective_accuracy(error == 0, risk),
        "roc_auc": None,
        "average_precision": None,
    }
    if brier:
        result["brier"] = float(brier_score_loss(error, risk))
    if len(np.unique(error)) >= 2:
        result["roc_auc"] = float(roc_auc_score(error, risk))
        result["average_precision"] = float(average_precision_score(error, risk))
    return result


def external_references(X_train, y_train, X_test, error_test):
    references = {}

    knn = KNeighborsClassifier(n_neighbors=12, weights="distance")
    knn.fit(X_train, y_train)
    knn_prob = knn.predict_proba(X_test)
    knn_pred = knn.classes_[np.argmax(knn_prob, axis=1)]
    knn_error = (knn_pred != error_test["y_true"]).astype(int)
    knn_risk = 1.0 - np.max(knn_prob, axis=1)
    references["weighted_knn_confidence"] = {
        "classifier_accuracy": float(np.mean(knn_error == 0)),
        **safe_error_metrics(knn_error, knn_risk, brier=False),
    }

    logistic = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, max_iter=3000, random_state=73),
    )
    logistic.fit(X_train, y_train)
    logistic_prob = logistic.predict_proba(X_test)
    logistic_pred = logistic.classes_[np.argmax(logistic_prob, axis=1)]
    logistic_error = (logistic_pred != error_test["y_true"]).astype(int)
    logistic_risk = 1.0 - np.max(logistic_prob, axis=1)
    references["logistic_confidence"] = {
        "classifier_accuracy": float(np.mean(logistic_error == 0)),
        **safe_error_metrics(logistic_error, logistic_risk, brier=False),
    }

    return references


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
        "mcgc_test_accuracy": float(np.mean(correct_test)),
        "status": "ok",
        "intrinsic_models": {},
        "external_references": external_references(
            X_train,
            y_train,
            X_test,
            {"y_true": np.asarray(y_test)},
        ),
    }

    if len(np.unique(error_cal)) < 2:
        record["status"] = "insufficient_calibration_classes"
        return record

    for model_name, names in FEATURE_SETS.items():
        fitted_risk = risk_model().fit(feature_matrix(rows_cal, names), error_cal)
        probability = fitted_risk.predict_proba(feature_matrix(rows_test, names))[:, 1]
        metrics = safe_error_metrics(error_test, probability, brier=True)
        metrics["features"] = names
        record["intrinsic_models"][model_name] = metrics

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
            "min": None,
            "max": None,
            "wins": 0,
            "ties": 0,
            "losses": 0,
            "sign_test_p_two_sided": None,
        }

    wins = int(np.sum(arr > eps))
    ties = int(np.sum(np.abs(arr) <= eps))
    losses = int(np.sum(arr < -eps))
    n_non_tied = wins + losses
    sign_p = None
    if n_non_tied:
        k = min(wins, losses)
        tail = sum(math.comb(n_non_tied, i) for i in range(k + 1)) / (2**n_non_tied)
        sign_p = float(min(1.0, 2.0 * tail))

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
        "sign_test_p_two_sided": sign_p,
    }


def model_summary(runs, model_name):
    usable = [run for run in runs if run["status"] == "ok"]
    aucs = [
        run["intrinsic_models"][model_name]["roc_auc"]
        for run in usable
        if run["intrinsic_models"][model_name]["roc_auc"] is not None
    ]
    aps = [
        run["intrinsic_models"][model_name]["average_precision"]
        for run in usable
        if run["intrinsic_models"][model_name]["average_precision"] is not None
    ]
    briers = [run["intrinsic_models"][model_name]["brier"] for run in usable]
    return {
        "evaluable_test_auc_seeds": len(aucs),
        "mean_test_roc_auc": float(np.mean(aucs)) if aucs else None,
        "mean_test_average_precision": float(np.mean(aps)) if aps else None,
        "mean_test_brier": float(np.mean(briers)) if briers else None,
    }


def reference_summary(runs, reference_name):
    records = [run["external_references"][reference_name] for run in runs]
    aucs = [record["roc_auc"] for record in records if record["roc_auc"] is not None]
    aps = [
        record["average_precision"]
        for record in records
        if record["average_precision"] is not None
    ]
    accuracies = [record["classifier_accuracy"] for record in records]
    return {
        "evaluable_test_auc_seeds": len(aucs),
        "mean_classifier_accuracy": float(np.mean(accuracies)),
        "mean_error_detection_roc_auc": float(np.mean(aucs)) if aucs else None,
        "mean_error_detection_average_precision": float(np.mean(aps)) if aps else None,
    }


def evaluate_dataset(name):
    runs = [evaluate_seed(name, seed) for seed in SEEDS]
    paired_delta = []
    for run in runs:
        if run["status"] != "ok":
            continue
        local_auc = run["intrinsic_models"]["local_only"]["roc_auc"]
        combined_auc = run["intrinsic_models"]["combined_intrinsic"]["roc_auc"]
        if local_auc is None or combined_auc is None:
            continue
        paired_delta.append(combined_auc - local_auc)

    return {
        "runs": runs,
        "status_counts": {
            status: int(sum(run["status"] == status for run in runs))
            for status in sorted({run["status"] for run in runs})
        },
        "intrinsic_models": {
            name: model_summary(runs, name) for name in FEATURE_SETS
        },
        "external_references": {
            name: reference_summary(runs, name)
            for name in ("weighted_knn_confidence", "logistic_confidence")
        },
        "paired_auc_delta_combined_minus_local": delta_summary(paired_delta),
    }


def main():
    dataset_names = (
        "moons_hard",
        "circles_hard",
        "spirals_hard",
        "overlap_hd",
        "breast_cancer",
        "wine",
        "digits_0_4",
    )
    datasets = {name: evaluate_dataset(name) for name in dataset_names}

    all_paired = []
    per_family = {"synthetic": [], "real": []}
    for name, result in datasets.items():
        for run in result["runs"]:
            if run["status"] != "ok":
                continue
            local_auc = run["intrinsic_models"]["local_only"]["roc_auc"]
            combined_auc = run["intrinsic_models"]["combined_intrinsic"]["roc_auc"]
            if local_auc is None or combined_auc is None:
                continue
            delta = combined_auc - local_auc
            all_paired.append(delta)
            family = "synthetic" if name in dataset_names[:4] else "real"
            per_family[family].append(delta)

    result = {
        "protocol": {
            "version": "0.8",
            "status": "frozen confirmation",
            "seeds": SEEDS,
            "split": "60% classifier train / 20% risk calibration / 20% untouched test",
            "classifier_train_seed": "2000 + dataset_seed",
            "calibration_test_seed": "3000 + dataset_seed",
            "risk_model": (
                "StandardScaler + LogisticRegression(class_weight='balanced', "
                "C=1.0, max_iter=2000, random_state=71)"
            ),
            "primary_contrast": (
                "test ROC AUC(combined_intrinsic) - test ROC AUC(local_only)"
            ),
            "feature_sets_frozen_from_v07": FEATURE_SETS,
            "note": (
                "No feature selection or dataset-specific retuning is allowed. "
                "Seeds without both calibration correctness classes are skipped, "
                "not resplit. External confidence references are contextual and "
                "are not part of the primary paired MCGC hypothesis."
            ),
        },
        "datasets": datasets,
        "confirmation_summary": {
            "all_evaluable_runs": delta_summary(all_paired),
            "synthetic_evaluable_runs": delta_summary(per_family["synthetic"]),
            "real_evaluable_runs": delta_summary(per_family["real"]),
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
