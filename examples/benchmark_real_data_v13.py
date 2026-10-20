from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
from sklearn.datasets import load_breast_cancer, load_digits, load_wine
from sklearn.model_selection import train_test_split

from mcgc import reliability_features


ROOT = Path(__file__).resolve().parents[1]
V12_SCRIPT = ROOT / "examples" / "benchmark_tuned_selective_v12.py"


def _load_v12():
    spec = importlib.util.spec_from_file_location("benchmark_tuned_selective_v12", V12_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


v12 = _load_v12()

SEEDS = list(range(12))
PRIMARY_DATASETS = ("digits", "wine")
CONTROL_DATASETS = ("breast_cancer",)
DATASETS = PRIMARY_DATASETS + CONTROL_DATASETS
COVERAGES = v12.COVERAGES
INNER_FOLDS = v12.INNER_FOLDS
KNN_GRID = v12.KNN_GRID
LOGISTIC_GRID = v12.LOGISTIC_GRID
SVM_GRID = v12.SVM_GRID
RF_GRID = v12.RF_GRID
BASELINES = v12.BASELINES


def dataset_for(name: str, seed: int):
    del seed  # datasets are fixed; seed controls only the stratified split
    if name == "digits":
        data = load_digits()
        return data.data, data.target
    if name == "wine":
        data = load_wine()
        return data.data, data.target
    if name == "breast_cancer":
        data = load_breast_cancer()
        return data.data, data.target
    raise ValueError(name)


def evaluate_seed(name: str, seed: int):
    X, y = dataset_for(name, seed)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=5000 + seed,
        stratify=y,
    )

    mcgc = v12.mcgc_classifier().fit(X_train, y_train)
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
        "mcgc_local_margin": v12.model_metrics(y_test, mcgc_pred, mcgc_risk),
    }
    tuning = {}

    for output_name, kind, grid in baseline_specs:
        fitted, selected, candidate_records = v12.tune_baseline(kind, grid, X_train, y_train, seed)
        pred, risk = v12.prediction_and_risk(kind, fitted, X_test)
        models[output_name] = v12.model_metrics(y_test, pred, risk)
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


def aggregate_model(runs, model_name):
    return v12.aggregate_model(runs, model_name)


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
            f"mcgc_vs_{baseline}": v12.delta_summary(
                np.asarray([r["models"][baseline]["eaurc"] for r in runs]) - mcgc_e
            )
            for baseline in BASELINES
        },
        "paired_aurc_advantage": {
            f"mcgc_vs_{baseline}": v12.delta_summary(
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
            f"mcgc_vs_{baseline}": v12.delta_summary(values)
            for baseline, values in e_deltas.items()
        },
        "paired_aurc_advantage": {
            f"mcgc_vs_{baseline}": v12.delta_summary(values)
            for baseline, values in a_deltas.items()
        },
    }


def fmt(value):
    return "NA" if value is None else f"{value:.6f}"


def print_summary(result):
    print("=== MCGC v1.3 FROZEN REAL-DATA CONFIRMATION ===")
    for name, dataset in result["datasets"].items():
        m = dataset["models"]["mcgc_local_margin"]
        k = dataset["models"]["tuned_knn_confidence"]
        s = dataset["models"]["tuned_rbf_svm_confidence"]
        r = dataset["models"]["tuned_random_forest_confidence"]
        dk = dataset["paired_eaurc_advantage"]["mcgc_vs_tuned_knn_confidence"]
        ds = dataset["paired_eaurc_advantage"]["mcgc_vs_tuned_rbf_svm_confidence"]
        dr = dataset["paired_eaurc_advantage"]["mcgc_vs_tuned_random_forest_confidence"]
        print(
            f"{name:16s} EAURC M/K/S/R="
            f"{fmt(m['mean_eaurc'])}/{fmt(k['mean_eaurc'])}/{fmt(s['mean_eaurc'])}/{fmt(r['mean_eaurc'])} "
            f"dK={fmt(dk['mean'])} {dk['wins']}/{dk['ties']}/{dk['losses']} "
            f"dS={fmt(ds['mean'])} {ds['wins']}/{ds['ties']}/{ds['losses']} "
            f"dR={fmt(dr['mean'])} {dr['wins']}/{dr['ties']}/{dr['losses']}"
        )

    print("=== PRIMARY NEW REAL-DATA FAMILY ===")
    pooled = result["primary_real_data_summary"]
    for baseline in BASELINES:
        key = f"mcgc_vs_{baseline}"
        e = pooled["paired_eaurc_advantage"][key]
        a = pooled["paired_aurc_advantage"][key]
        print(
            f"{key:42s} E-AURC adv mean={fmt(e['mean'])} median={fmt(e['median'])} "
            f"W/T/L={e['wins']}/{e['ties']}/{e['losses']} p={fmt(e['sign_test_p_two_sided'])} | "
            f"AURC adv mean={fmt(a['mean'])} W/T/L={a['wins']}/{a['ties']}/{a['losses']}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("v13_real_data_confirmation.json"),
    )
    args = parser.parse_args()

    datasets = {name: evaluate_dataset(name) for name in DATASETS}
    result = {
        "protocol": {
            "version": "1.3",
            "status": "frozen real-data confirmation",
            "primary_datasets": list(PRIMARY_DATASETS),
            "control_datasets": list(CONTROL_DATASETS),
            "seeds": SEEDS,
            "outer_split": "70% train / 30% untouched test",
            "outer_split_seed": "5000 + dataset_seed",
            "inner_cv": "3-fold stratified CV on outer training only",
            "inner_cv_seed": "6000 + dataset_seed",
            "primary_metric": "outer-test E-AURC; lower is better",
            "mcgc_tuned": False,
            "mcgc_risk": "local_margin_risk",
            "knn_grid": list(KNN_GRID),
            "logistic_grid": list(LOGISTIC_GRID),
            "svm_grid": list(SVM_GRID),
            "rf_grid": list(RF_GRID),
        },
        "datasets": datasets,
        "primary_real_data_summary": pooled_primary(datasets),
    }

    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print_summary(result)
    print(f"WROTE {args.output}")


if __name__ == "__main__":
    main()
