from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from sklearn.datasets import load_breast_cancer, make_circles, make_classification, make_moons
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from mcgc import MultiAgentChemotacticGraphClassifier, reliability_features


SEEDS = list(range(12))
PRIMARY_DATASETS = ("moons_hard", "circles_hard", "spirals_hard")
CONTROL_DATASETS = ("overlap_hd", "breast_cancer")
DATASETS = PRIMARY_DATASETS + CONTROL_DATASETS
COVERAGES = (0.90, 0.80, 0.70, 0.50)


def make_spirals(n_samples: int = 600, noise: float = 0.16, random_state: int = 7):
    """Frozen two-arm spiral generator used in earlier MCGC checkpoints."""
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
    """Discrete empirical area under the risk-coverage curve.

    Samples are retained from lowest to highest risk. The curve contains one
    point for every non-empty retained prefix, and AURC is the arithmetic mean
    of the corresponding selective error rates.
    """
    error = np.asarray(error, dtype=int)
    risk = np.asarray(risk, dtype=float)
    if len(error) == 0 or error.shape != risk.shape:
        raise ValueError("error and risk must be non-empty arrays of equal shape")
    order = np.argsort(risk, kind="stable")
    ordered_error = error[order]
    selective_risk = np.cumsum(ordered_error) / np.arange(1, len(error) + 1)
    return float(np.mean(selective_risk))


def oracle_aurc(error):
    """Best possible AURC for the same fixed number of prediction errors."""
    error = np.asarray(error, dtype=int)
    if len(error) == 0:
        raise ValueError("error must be non-empty")
    oracle_risk = error.astype(float)
    return empirical_aurc(error, oracle_risk)


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
    error = (prediction != y_true).astype(int)
    aurc = empirical_aurc(error, risk)
    oracle = oracle_aurc(error)
    auc = None
    if len(np.unique(error)) >= 2:
        auc = float(roc_auc_score(error, np.asarray(risk, dtype=float)))
    return {
        "accuracy": float(np.mean(error == 0)),
        "error_rate": float(np.mean(error)),
        "aurc": aurc,
        "oracle_aurc": oracle,
        "eaurc": float(aurc - oracle),
        "error_detection_roc_auc": auc,
        "selective_accuracy": selective_accuracy(error, risk),
    }


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

    knn = KNeighborsClassifier(n_neighbors=12, weights="distance")
    knn.fit(X_train, y_train)
    knn_prob = knn.predict_proba(X_test)
    knn_pred = knn.classes_[np.argmax(knn_prob, axis=1)]
    knn_risk = 1.0 - np.max(knn_prob, axis=1)

    logistic = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, max_iter=3000, random_state=73),
    )
    logistic.fit(X_train, y_train)
    logistic_prob = logistic.predict_proba(X_test)
    logistic_pred = logistic.classes_[np.argmax(logistic_prob, axis=1)]
    logistic_risk = 1.0 - np.max(logistic_prob, axis=1)

    return {
        "seed": seed,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "models": {
            "mcgc_local_margin": model_metrics(y_test, mcgc_pred, mcgc_risk),
            "weighted_knn_confidence": model_metrics(y_test, knn_pred, knn_risk),
            "logistic_confidence": model_metrics(y_test, logistic_pred, logistic_risk),
        },
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
    result = {}
    for key in ("accuracy", "error_rate", "aurc", "oracle_aurc", "eaurc"):
        vals = [record[key] for record in metrics]
        result[f"mean_{key}"] = float(np.mean(vals))
        result[f"std_{key}"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    aucs = [record["error_detection_roc_auc"] for record in metrics if record["error_detection_roc_auc"] is not None]
    result["mean_error_detection_roc_auc"] = float(np.mean(aucs)) if aucs else None
    result["evaluable_auc_runs"] = len(aucs)
    for coverage in COVERAGES:
        key = str(coverage)
        vals = [record["selective_accuracy"][key]["accuracy"] for record in metrics]
        result[f"mean_selective_accuracy_at_{key}"] = float(np.mean(vals))
    return result


def evaluate_dataset(name: str):
    runs = [evaluate_seed(name, seed) for seed in SEEDS]
    mcgc_e = np.asarray([run["models"]["mcgc_local_margin"]["eaurc"] for run in runs])
    knn_e = np.asarray([run["models"]["weighted_knn_confidence"]["eaurc"] for run in runs])
    log_e = np.asarray([run["models"]["logistic_confidence"]["eaurc"] for run in runs])
    mcgc_a = np.asarray([run["models"]["mcgc_local_margin"]["aurc"] for run in runs])
    knn_a = np.asarray([run["models"]["weighted_knn_confidence"]["aurc"] for run in runs])
    log_a = np.asarray([run["models"]["logistic_confidence"]["aurc"] for run in runs])
    return {
        "runs": runs,
        "models": {
            name_: aggregate_model(runs, name_)
            for name_ in (
                "mcgc_local_margin",
                "weighted_knn_confidence",
                "logistic_confidence",
            )
        },
        "paired_eaurc_advantage": {
            "mcgc_vs_knn": delta_summary(knn_e - mcgc_e),
            "mcgc_vs_logistic": delta_summary(log_e - mcgc_e),
        },
        "paired_aurc_advantage": {
            "mcgc_vs_knn": delta_summary(knn_a - mcgc_a),
            "mcgc_vs_logistic": delta_summary(log_a - mcgc_a),
        },
    }


def pooled_primary(datasets):
    knn_e = []
    log_e = []
    knn_a = []
    log_a = []
    for name in PRIMARY_DATASETS:
        for run in datasets[name]["runs"]:
            mcgc = run["models"]["mcgc_local_margin"]
            knn = run["models"]["weighted_knn_confidence"]
            logistic = run["models"]["logistic_confidence"]
            knn_e.append(knn["eaurc"] - mcgc["eaurc"])
            log_e.append(logistic["eaurc"] - mcgc["eaurc"])
            knn_a.append(knn["aurc"] - mcgc["aurc"])
            log_a.append(logistic["aurc"] - mcgc["aurc"])
    return {
        "paired_eaurc_advantage": {
            "mcgc_vs_knn": delta_summary(knn_e),
            "mcgc_vs_logistic": delta_summary(log_e),
        },
        "paired_aurc_advantage": {
            "mcgc_vs_knn": delta_summary(knn_a),
            "mcgc_vs_logistic": delta_summary(log_a),
        },
    }


def fmt(value):
    return "NA" if value is None else f"{value:.6f}"


def print_summary(result):
    print("=== MCGC v1.0 FROZEN SELECTIVE-MANIFOLD BENCHMARK ===")
    for name, dataset in result["datasets"].items():
        m = dataset["models"]["mcgc_local_margin"]
        k = dataset["models"]["weighted_knn_confidence"]
        l = dataset["models"]["logistic_confidence"]
        dk = dataset["paired_eaurc_advantage"]["mcgc_vs_knn"]
        dl = dataset["paired_eaurc_advantage"]["mcgc_vs_logistic"]
        print(
            f"{name:16s} "
            f"EAURC M/K/L={fmt(m['mean_eaurc'])}/{fmt(k['mean_eaurc'])}/{fmt(l['mean_eaurc'])} "
            f"dK={fmt(dk['mean'])} {dk['wins']}/{dk['ties']}/{dk['losses']} "
            f"dL={fmt(dl['mean'])} {dl['wins']}/{dl['ties']}/{dl['losses']}"
        )
    print("=== PRIMARY MANIFOLD FAMILY ===")
    pooled = result["primary_manifold_summary"]
    for baseline in ("mcgc_vs_knn", "mcgc_vs_logistic"):
        e = pooled["paired_eaurc_advantage"][baseline]
        a = pooled["paired_aurc_advantage"][baseline]
        print(
            f"{baseline:18s} E-AURC advantage mean={fmt(e['mean'])} median={fmt(e['median'])} "
            f"W/T/L={e['wins']}/{e['ties']}/{e['losses']} p={fmt(e['sign_test_p_two_sided'])} | "
            f"AURC advantage mean={fmt(a['mean'])} W/T/L={a['wins']}/{a['ties']}/{a['losses']}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("v10_selective_manifold.json"),
    )
    args = parser.parse_args()

    datasets = {name: evaluate_dataset(name) for name in DATASETS}
    result = {
        "protocol": {
            "version": "1.0",
            "status": "frozen selective-manifold benchmark",
            "primary_datasets": list(PRIMARY_DATASETS),
            "control_datasets": list(CONTROL_DATASETS),
            "seeds": SEEDS,
            "split": "70% train / 30% untouched test",
            "split_seed": "5000 + dataset_seed",
            "primary_metric": "E-AURC; lower is better",
            "primary_contrasts": [
                "E-AURC(weighted kNN) - E-AURC(MCGC local margin)",
                "E-AURC(logistic) - E-AURC(MCGC local margin)",
            ],
            "secondary_metrics": [
                "raw accuracy",
                "ordinary AURC",
                "selective accuracy at 0.90/0.80/0.70/0.50 coverage",
                "error-detection ROC-AUC",
            ],
            "note": "No confidence calibration, feature search, or dataset-specific retuning is used.",
        },
        "datasets": datasets,
        "primary_manifold_summary": pooled_primary(datasets),
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print_summary(result)
    print(f"FULL_JSON={args.output.resolve()}")


if __name__ == "__main__":
    main()
