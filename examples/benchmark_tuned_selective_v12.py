from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np
from sklearn.datasets import load_breast_cancer, make_circles, make_classification, make_moons
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from mcgc import MultiAgentChemotacticGraphClassifier, reliability_features


SEEDS = list(range(12))
PRIMARY_DATASETS = ("moons_hard", "circles_hard", "spirals_hard")
CONTROL_DATASETS = ("overlap_hd", "breast_cancer")
DATASETS = PRIMARY_DATASETS + CONTROL_DATASETS
COVERAGES = (0.90, 0.80, 0.70, 0.50)
INNER_FOLDS = 3

KNN_GRID = tuple(
    {"n_neighbors": n_neighbors, "weights": weights}
    for n_neighbors, weights in itertools.product((5, 9, 12, 17, 25), ("uniform", "distance"))
)
SVM_GRID = tuple(
    {"C": C, "gamma": gamma}
    for C, gamma in itertools.product((0.1, 1.0, 10.0), ("scale", 0.1, 1.0))
)
RF_GRID = tuple(
    {"max_depth": max_depth, "min_samples_leaf": min_samples_leaf, "max_features": max_features}
    for max_depth, min_samples_leaf, max_features in itertools.product(
        (None, 6, 12), (1, 3, 5), ("sqrt", 0.5)
    )
)
LOGISTIC_GRID = tuple({"C": C} for C in (0.1, 1.0, 10.0))

BASELINES = (
    "tuned_knn_confidence",
    "tuned_logistic_confidence",
    "tuned_rbf_svm_confidence",
    "tuned_random_forest_confidence",
)


def make_spirals(n_samples: int = 600, noise: float = 0.16, random_state: int = 7):
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
    if name == "circles_hard":
        return make_circles(n_samples=700, noise=0.18, factor=0.50, random_state=seed)
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
    raise ValueError(name)


def mcgc_classifier():
    return MultiAgentChemotacticGraphClassifier(
        n_neighbors=12,
        n_walkers=12,
        max_steps=10,
        diffusion_steps=40,
        random_state=17,
        adaptive_walker_gate=True,
    )


def empirical_aurc(error, risk):
    error = np.asarray(error, dtype=int)
    risk = np.asarray(risk, dtype=float)
    if len(error) == 0 or error.shape != risk.shape:
        raise ValueError("error and risk must be non-empty arrays of equal shape")
    order = np.argsort(risk, kind="stable")
    ordered_error = error[order]
    selective_risk = np.cumsum(ordered_error) / np.arange(1, len(error) + 1)
    return float(np.mean(selective_risk))


def oracle_aurc(error):
    error = np.asarray(error, dtype=int)
    if len(error) == 0:
        raise ValueError("error must be non-empty")
    return empirical_aurc(error, error.astype(float))


def selective_accuracy(error, risk):
    error = np.asarray(error, dtype=int)
    risk = np.asarray(risk, dtype=float)
    order = np.argsort(risk, kind="stable")
    out = {}
    for coverage in COVERAGES:
        keep = max(1, int(np.ceil(coverage * len(error))))
        idx = order[:keep]
        out[str(coverage)] = {
            "kept": int(keep),
            "empirical_coverage": float(keep / len(error)),
            "accuracy": float(1.0 - np.mean(error[idx])),
        }
    return out


def model_metrics(y_true, prediction, risk):
    y_true = np.asarray(y_true)
    prediction = np.asarray(prediction)
    risk = np.asarray(risk, dtype=float)
    error = (prediction != y_true).astype(int)
    aurc = empirical_aurc(error, risk)
    oracle = oracle_aurc(error)
    auc = None
    if len(np.unique(error)) >= 2:
        auc = float(roc_auc_score(error, risk))
    return {
        "accuracy": float(np.mean(error == 0)),
        "error_rate": float(np.mean(error)),
        "aurc": aurc,
        "oracle_aurc": oracle,
        "eaurc": float(aurc - oracle),
        "error_detection_roc_auc": auc,
        "selective_accuracy": selective_accuracy(error, risk),
    }


def svm_prediction_and_risk(model, X):
    prediction = model.predict(X)
    decision = np.asarray(model.decision_function(X))
    if decision.ndim == 1:
        confidence = np.abs(decision)
    else:
        ordered = np.sort(decision, axis=1)
        confidence = ordered[:, -1] - ordered[:, -2]
    return prediction, -confidence


def classifier_for(kind: str, params: dict):
    if kind == "knn":
        return KNeighborsClassifier(
            n_neighbors=int(params["n_neighbors"]),
            weights=str(params["weights"]),
        )
    if kind == "logistic":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(C=float(params["C"]), max_iter=3000, random_state=73),
        )
    if kind == "svm":
        return make_pipeline(
            StandardScaler(),
            SVC(kernel="rbf", C=float(params["C"]), gamma=params["gamma"], probability=False),
        )
    if kind == "rf":
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=params["max_depth"],
            min_samples_leaf=int(params["min_samples_leaf"]),
            max_features=params["max_features"],
            random_state=97,
            n_jobs=-1,
        )
    raise ValueError(kind)


def prediction_and_risk(kind: str, model, X):
    if kind == "svm":
        return svm_prediction_and_risk(model, X)
    prob = model.predict_proba(X)
    pred = model.classes_[np.argmax(prob, axis=1)]
    risk = 1.0 - np.max(prob, axis=1)
    return pred, risk


def candidate_key(params: dict):
    return json.dumps(params, sort_keys=True, separators=(",", ":"))


def tune_baseline(kind: str, grid, X_train, y_train, seed: int):
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    cv = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=6000 + seed)
    candidate_records = []

    for params in grid:
        oof_pred = np.empty(len(y_train), dtype=y_train.dtype)
        oof_risk = np.empty(len(y_train), dtype=float)
        seen = np.zeros(len(y_train), dtype=bool)

        for fit_idx, val_idx in cv.split(X_train, y_train):
            model = classifier_for(kind, params)
            model.fit(X_train[fit_idx], y_train[fit_idx])
            pred, risk = prediction_and_risk(kind, model, X_train[val_idx])
            oof_pred[val_idx] = pred
            oof_risk[val_idx] = risk
            seen[val_idx] = True

        if not np.all(seen):
            raise RuntimeError("inner CV did not cover every outer-training sample")

        metrics = model_metrics(y_train, oof_pred, oof_risk)
        candidate_records.append(
            {
                "params": dict(params),
                "candidate_key": candidate_key(dict(params)),
                "oof_eaurc": metrics["eaurc"],
                "oof_aurc": metrics["aurc"],
                "oof_accuracy": metrics["accuracy"],
            }
        )

    candidate_records.sort(
        key=lambda row: (
            round(float(row["oof_eaurc"]), 12),
            -round(float(row["oof_accuracy"]), 12),
            row["candidate_key"],
        )
    )
    selected = candidate_records[0]
    final_model = classifier_for(kind, selected["params"])
    final_model.fit(X_train, y_train)
    return final_model, selected, candidate_records


def evaluate_seed(name: str, seed: int):
    X, y = dataset_for(name, seed)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=5000 + seed,
        stratify=y,
    )

    mcgc = mcgc_classifier().fit(X_train, y_train)
    mcgc_pred, mcgc_rows = reliability_features(mcgc, X_test)
    mcgc_risk = np.asarray(
        [row["risk_scores"]["local_margin_risk"] for row in mcgc_rows],
        dtype=float,
    )

    baseline_specs = (
        ("tuned_knn_confidence", "knn", KNN_GRID),
        ("tuned_logistic_confidence", "logistic", LOGISTIC_GRID),
        ("tuned_rbf_svm_confidence", "svm", SVM_GRID),
        ("tuned_random_forest_confidence", "rf", RF_GRID),
    )

    models = {
        "mcgc_local_margin": model_metrics(y_test, mcgc_pred, mcgc_risk),
    }
    tuning = {}

    for output_name, kind, grid in baseline_specs:
        fitted, selected, candidate_records = tune_baseline(kind, grid, X_train, y_train, seed)
        pred, risk = prediction_and_risk(kind, fitted, X_test)
        models[output_name] = model_metrics(y_test, pred, risk)
        tuning[output_name] = {
            "selected": selected,
            "n_candidates": len(candidate_records),
            "all_candidates": candidate_records,
        }

    return {
        "seed": seed,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "models": models,
        "tuning": tuning,
    }


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


def aggregate_model(runs, model_name):
    metrics = [run["models"][model_name] for run in runs]
    out = {}
    for key in ("accuracy", "error_rate", "aurc", "oracle_aurc", "eaurc"):
        vals = [record[key] for record in metrics]
        out[f"mean_{key}"] = float(np.mean(vals))
        out[f"std_{key}"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    aucs = [record["error_detection_roc_auc"] for record in metrics if record["error_detection_roc_auc"] is not None]
    out["mean_error_detection_roc_auc"] = float(np.mean(aucs)) if aucs else None
    out["evaluable_auc_runs"] = len(aucs)
    for coverage in COVERAGES:
        key = str(coverage)
        vals = [record["selective_accuracy"][key]["accuracy"] for record in metrics]
        out[f"mean_selective_accuracy_at_{key}"] = float(np.mean(vals))
    return out


def evaluate_dataset(name: str):
    runs = [evaluate_seed(name, seed) for seed in SEEDS]
    mcgc_e = np.asarray([r["models"]["mcgc_local_margin"]["eaurc"] for r in runs])
    mcgc_a = np.asarray([r["models"]["mcgc_local_margin"]["aurc"] for r in runs])
    return {
        "runs": runs,
        "models": {
            model_name: aggregate_model(runs, model_name)
            for model_name in ("mcgc_local_margin",) + BASELINES
        },
        "paired_eaurc_advantage": {
            f"mcgc_vs_{baseline}": delta_summary(
                np.asarray([r["models"][baseline]["eaurc"] for r in runs]) - mcgc_e
            )
            for baseline in BASELINES
        },
        "paired_aurc_advantage": {
            f"mcgc_vs_{baseline}": delta_summary(
                np.asarray([r["models"][baseline]["aurc"] for r in runs]) - mcgc_a
            )
            for baseline in BASELINES
        },
    }


def pooled_primary(datasets):
    e_deltas = {baseline: [] for baseline in BASELINES}
    a_deltas = {baseline: [] for baseline in BASELINES}
    for name in PRIMARY_DATASETS:
        for run in datasets[name]["runs"]:
            mcgc = run["models"]["mcgc_local_margin"]
            for baseline in BASELINES:
                record = run["models"][baseline]
                e_deltas[baseline].append(record["eaurc"] - mcgc["eaurc"])
                a_deltas[baseline].append(record["aurc"] - mcgc["aurc"])
    return {
        "paired_eaurc_advantage": {
            f"mcgc_vs_{baseline}": delta_summary(e_deltas[baseline])
            for baseline in BASELINES
        },
        "paired_aurc_advantage": {
            f"mcgc_vs_{baseline}": delta_summary(a_deltas[baseline])
            for baseline in BASELINES
        },
    }


def fmt(value):
    return "NA" if value is None else f"{value:.6f}"


def print_summary(result):
    print("=== MCGC v1.2 TUNED SELECTIVE-BASELINE CONFIRMATION ===")
    for name, dataset in result["datasets"].items():
        m = dataset["models"]["mcgc_local_margin"]
        k = dataset["models"]["tuned_knn_confidence"]
        s = dataset["models"]["tuned_rbf_svm_confidence"]
        r = dataset["models"]["tuned_random_forest_confidence"]
        dk = dataset["paired_eaurc_advantage"]["mcgc_vs_tuned_knn_confidence"]
        ds = dataset["paired_eaurc_advantage"]["mcgc_vs_tuned_rbf_svm_confidence"]
        dr = dataset["paired_eaurc_advantage"]["mcgc_vs_tuned_random_forest_confidence"]
        print(
            f"{name:16s} EAURC M/K/S/R={fmt(m['mean_eaurc'])}/{fmt(k['mean_eaurc'])}/"
            f"{fmt(s['mean_eaurc'])}/{fmt(r['mean_eaurc'])} "
            f"dK={fmt(dk['mean'])} {dk['wins']}/{dk['ties']}/{dk['losses']} "
            f"dS={fmt(ds['mean'])} {ds['wins']}/{ds['ties']}/{ds['losses']} "
            f"dR={fmt(dr['mean'])} {dr['wins']}/{dr['ties']}/{dr['losses']}"
        )
    print("=== PRIMARY MANIFOLD FAMILY ===")
    pooled = result["primary_manifold_summary"]
    for baseline in BASELINES:
        key = f"mcgc_vs_{baseline}"
        e = pooled["paired_eaurc_advantage"][key]
        a = pooled["paired_aurc_advantage"][key]
        print(
            f"{key:40s} E-AURC adv mean={fmt(e['mean'])} median={fmt(e['median'])} "
            f"W/T/L={e['wins']}/{e['ties']}/{e['losses']} p={fmt(e['sign_test_p_two_sided'])} | "
            f"AURC adv mean={fmt(a['mean'])} W/T/L={a['wins']}/{a['ties']}/{a['losses']}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("v12_tuned_selective_baselines.json"))
    args = parser.parse_args()

    datasets = {name: evaluate_dataset(name) for name in DATASETS}
    result = {
        "protocol": {
            "version": "1.2",
            "status": "frozen tuned-baseline confirmation",
            "primary_datasets": list(PRIMARY_DATASETS),
            "control_datasets": list(CONTROL_DATASETS),
            "seeds": SEEDS,
            "outer_split": "70% train / 30% untouched test",
            "outer_split_seed": "5000 + dataset_seed",
            "inner_cv": "3-fold stratified CV on outer training only",
            "inner_cv_seed": "6000 + dataset_seed",
            "selection_metric": "lowest OOF E-AURC; accuracy then lexical tie-break",
            "primary_metric": "outer-test E-AURC; lower is better",
            "mcgc_tuned": False,
            "knn_grid": list(KNN_GRID),
            "svm_grid": list(SVM_GRID),
            "rf_grid": list(RF_GRID),
            "logistic_grid": list(LOGISTIC_GRID),
        },
        "datasets": datasets,
        "primary_manifold_summary": pooled_primary(datasets),
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print_summary(result)
    print(f"WROTE {args.output}")


if __name__ == "__main__":
    main()
