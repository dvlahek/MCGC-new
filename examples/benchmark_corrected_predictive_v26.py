from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
from pathlib import Path

import numpy as np
from sklearn.datasets import load_breast_cancer, load_digits, load_wine
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.semi_supervised import LabelSpreading

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
NONLINEAR_PRIMARY = ("moons_hard", "circles_hard", "spirals_hard")
CONTROLS = ("overlap_hd", "breast_cancer")
REAL_PRIMARY = ("digits", "wine")
DATASETS = NONLINEAR_PRIMARY + ("overlap_hd", "breast_cancer", "digits", "wine")

SCALED_KNN_GRID = tuple(v12.KNN_GRID)
SVM_GRID = tuple(v12.SVM_GRID)
RF_GRID = tuple(v12.RF_GRID)
LS_GRID = tuple(
    {"n_neighbors": k, "alpha": alpha}
    for k, alpha in itertools.product((5, 9, 12, 17, 25), (0.2, 0.5, 0.8))
)

BASELINES = (
    "scaled_tuned_knn_confidence",
    "tuned_rbf_svm_confidence",
    "tuned_random_forest_confidence",
    "scaled_tuned_label_spreading_confidence",
)


def dataset_for(name: str, seed: int):
    if name in ("moons_hard", "circles_hard", "spirals_hard", "overlap_hd", "breast_cancer"):
        return v12.dataset_for(name, seed)
    if name == "digits":
        data = load_digits()
        return data.data, data.target
    if name == "wine":
        data = load_wine()
        return data.data, data.target
    raise ValueError(name)


def classifier_for(kind: str, params: dict):
    if kind == "scaled_knn":
        return make_pipeline(
            StandardScaler(),
            KNeighborsClassifier(
                n_neighbors=int(params["n_neighbors"]),
                weights=str(params["weights"]),
            ),
        )
    if kind == "svm":
        return v12.classifier_for("svm", params)
    if kind == "rf":
        return v12.classifier_for("rf", params)
    if kind == "label_spreading":
        return make_pipeline(
            StandardScaler(),
            LabelSpreading(
                kernel="knn",
                n_neighbors=int(params["n_neighbors"]),
                alpha=float(params["alpha"]),
                max_iter=100,
            ),
        )
    raise ValueError(kind)


def prediction_and_risk(kind: str, model, X):
    if kind == "svm":
        return v12.svm_prediction_and_risk(model, X)
    prob = np.asarray(model.predict_proba(X), dtype=float)
    classes = np.asarray(model.classes_)
    pred = classes[np.argmax(prob, axis=1)]
    risk = 1.0 - np.max(prob, axis=1)
    return pred, risk


def tune(kind: str, grid, X_train, y_train, seed: int):
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    cv = StratifiedKFold(n_splits=v12.INNER_FOLDS, shuffle=True, random_state=6000 + seed)
    records = []
    for params in grid:
        oof_pred = np.empty(len(y_train), dtype=y_train.dtype)
        oof_risk = np.empty(len(y_train), dtype=float)
        seen = np.zeros(len(y_train), dtype=bool)
        failed = False
        for fit_idx, val_idx in cv.split(X_train, y_train):
            try:
                model = classifier_for(kind, params)
                model.fit(X_train[fit_idx], y_train[fit_idx])
                pred, risk = prediction_and_risk(kind, model, X_train[val_idx])
            except Exception:
                failed = True
                break
            oof_pred[val_idx] = pred
            oof_risk[val_idx] = risk
            seen[val_idx] = True
        if failed or not np.all(seen):
            continue
        metrics = v12.model_metrics(y_train, oof_pred, oof_risk)
        records.append({
            "params": dict(params),
            "candidate_key": v12.candidate_key(dict(params)),
            "oof_eaurc": metrics["eaurc"],
            "oof_aurc": metrics["aurc"],
            "oof_accuracy": metrics["accuracy"],
        })
    if not records:
        raise RuntimeError(f"no valid tuning candidate for {kind}")
    records.sort(key=lambda row: (
        round(float(row["oof_eaurc"]), 12),
        -round(float(row["oof_accuracy"]), 12),
        row["candidate_key"],
    ))
    selected = records[0]
    final = classifier_for(kind, selected["params"])
    final.fit(X_train, y_train)
    return final, selected, records


def evaluate_seed(name: str, seed: int):
    X, y = dataset_for(name, seed)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=5000 + seed, stratify=y
    )

    mcgc = v12.mcgc_classifier().fit(X_train, y_train)
    mcgc_pred, rows = reliability_features(mcgc, X_test)
    mcgc_risk = np.asarray([r["risk_scores"]["local_margin_risk"] for r in rows], dtype=float)

    models = {"mcgc_local_margin": v12.model_metrics(y_test, mcgc_pred, mcgc_risk)}
    tuning = {}
    specs = (
        ("scaled_tuned_knn_confidence", "scaled_knn", SCALED_KNN_GRID),
        ("tuned_rbf_svm_confidence", "svm", SVM_GRID),
        ("tuned_random_forest_confidence", "rf", RF_GRID),
        ("scaled_tuned_label_spreading_confidence", "label_spreading", LS_GRID),
    )
    for out_name, kind, grid in specs:
        fitted, selected, all_records = tune(kind, grid, X_train, y_train, seed)
        pred, risk = prediction_and_risk(kind, fitted, X_test)
        models[out_name] = v12.model_metrics(y_test, pred, risk)
        tuning[out_name] = {
            "selected": selected,
            "n_candidates": len(all_records),
            "all_candidates": all_records,
        }
    return {
        "seed": int(seed),
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
    return {
        "runs": runs,
        "models": {
            m: aggregate_model(runs, m)
            for m in ("mcgc_local_margin",) + BASELINES
        },
        "paired_eaurc_advantage": {
            f"mcgc_vs_{b}": v12.delta_summary(
                np.asarray([r["models"][b]["eaurc"] for r in runs]) - mcgc_e
            )
            for b in BASELINES
        },
    }


def pooled(datasets, names):
    values = {b: [] for b in BASELINES}
    for name in names:
        for run in datasets[name]["runs"]:
            m = run["models"]["mcgc_local_margin"]["eaurc"]
            for b in BASELINES:
                values[b].append(run["models"][b]["eaurc"] - m)
    return {
        f"mcgc_vs_{b}": v12.delta_summary(vals)
        for b, vals in values.items()
    }


def print_summary(result):
    print("=== MCGC v2.6 CORRECTED PREDICTIVE BASELINES ===")
    for name, block in result["datasets"].items():
        m = block["models"]["mcgc_local_margin"]
        line = [f"{name:16s}", f"MCGC_acc={m['mean_accuracy']:.4f}", f"MCGC_EAURC={m['mean_eaurc']:.5f}"]
        for b in BASELINES:
            bm = block["models"][b]
            d = block["paired_eaurc_advantage"][f"mcgc_vs_{b}"]
            line.append(f"{b}:acc={bm['mean_accuracy']:.4f},EA={bm['mean_eaurc']:.5f},W/T/L={d['wins']}/{d['ties']}/{d['losses']},p={d['sign_test_p_two_sided']}")
        print(" | ".join(line))
    print("=== NONLINEAR PRIMARY POOLED ===")
    for k, v in result["nonlinear_primary_pooled_eaurc"].items():
        print(k, json.dumps(v, sort_keys=True))
    print("=== REAL PRIMARY POOLED ===")
    for k, v in result["real_primary_pooled_eaurc"].items():
        print(k, json.dumps(v, sort_keys=True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=Path("v26_corrected_predictive_baselines.json"))
    args = ap.parse_args()

    datasets = {name: evaluate_dataset(name) for name in DATASETS}
    result = {
        "protocol": {
            "version": "2.6",
            "status": "frozen corrected predictive-baseline audit",
            "seeds": SEEDS,
            "datasets": list(DATASETS),
            "nonlinear_primary": list(NONLINEAR_PRIMARY),
            "real_primary": list(REAL_PRIMARY),
            "outer_split": "70/30 stratified, random_state=5000+seed",
            "inner_cv": "3-fold stratified on outer training only, random_state=6000+seed",
            "selection_metric": "lowest OOF E-AURC; higher accuracy then lexical tie-break",
            "scaled_knn_grid": list(SCALED_KNN_GRID),
            "label_spreading_grid": list(LS_GRID),
            "svm_grid": list(SVM_GRID),
            "rf_grid": list(RF_GRID),
        },
        "datasets": datasets,
        "nonlinear_primary_pooled_eaurc": pooled(datasets, NONLINEAR_PRIMARY),
        "real_primary_pooled_eaurc": pooled(datasets, REAL_PRIMARY),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print_summary(result)
    print(f"FULL_JSON={args.output.resolve()}")


if __name__ == "__main__":
    main()
