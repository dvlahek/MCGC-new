from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split

from mcgc import MultiAgentChemotacticGraphClassifier, reliability_features


ROOT = Path(__file__).resolve().parents[1]
V12_SCRIPT = ROOT / "examples" / "benchmark_tuned_selective_v12.py"
V13_SCRIPT = ROOT / "examples" / "benchmark_real_data_v13.py"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


v12 = _load_module(V12_SCRIPT, "benchmark_tuned_selective_v12")
v13 = _load_module(V13_SCRIPT, "benchmark_real_data_v13")

SEEDS = list(range(12))
PRIMARY_DATASETS = ("digits", "wine")
CONTROL_DATASETS = ("breast_cancer",)
DATASETS = PRIMARY_DATASETS + CONTROL_DATASETS
COVERAGES = v12.COVERAGES
INNER_FOLDS = 3

MCGC_GRID = tuple(
    {
        "n_neighbors": n_neighbors,
        "diffusion_alpha": diffusion_alpha,
        "diffusion_steps": diffusion_steps,
    }
    for n_neighbors, diffusion_alpha, diffusion_steps in itertools.product(
        (7, 12, 17),
        (0.75, 0.85, 0.93),
        (20, 40),
    )
)

KNN_GRID = v12.KNN_GRID
LOGISTIC_GRID = v12.LOGISTIC_GRID
SVM_GRID = v12.SVM_GRID
RF_GRID = v12.RF_GRID
BASELINES = v12.BASELINES
FROZEN_MCGC_PARAMS = {
    "n_neighbors": 12,
    "diffusion_alpha": 0.85,
    "diffusion_steps": 40,
}


def dataset_for(name: str, seed: int):
    return v13.dataset_for(name, seed)


def mcgc_classifier(params: dict):
    return MultiAgentChemotacticGraphClassifier(
        n_neighbors=int(params["n_neighbors"]),
        n_walkers=12,
        max_steps=10,
        diffusion_alpha=float(params["diffusion_alpha"]),
        diffusion_steps=int(params["diffusion_steps"]),
        temperature=0.35,
        random_state=17,
        adaptive_walker_gate=True,
    )


def mcgc_prediction_and_risk(model, X):
    prediction, rows = reliability_features(model, X)
    risk = np.asarray(
        [row["risk_scores"]["local_margin_risk"] for row in rows],
        dtype=float,
    )
    return np.asarray(prediction), risk


def candidate_key(params: dict):
    return json.dumps(params, sort_keys=True, separators=(",", ":"))


def tune_mcgc(X_train, y_train, seed: int):
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    cv = StratifiedKFold(
        n_splits=INNER_FOLDS,
        shuffle=True,
        random_state=6000 + seed,
    )
    candidate_records = []

    for params in MCGC_GRID:
        oof_pred = np.empty(len(y_train), dtype=y_train.dtype)
        oof_risk = np.empty(len(y_train), dtype=float)
        seen = np.zeros(len(y_train), dtype=bool)

        for fit_idx, val_idx in cv.split(X_train, y_train):
            model = mcgc_classifier(params).fit(X_train[fit_idx], y_train[fit_idx])
            pred, risk = mcgc_prediction_and_risk(model, X_train[val_idx])
            oof_pred[val_idx] = pred
            oof_risk[val_idx] = risk
            seen[val_idx] = True

        if not np.all(seen):
            raise RuntimeError("inner CV did not cover every outer-training sample")

        metrics = v12.model_metrics(y_train, oof_pred, oof_risk)
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
    final_model = mcgc_classifier(selected["params"]).fit(X_train, y_train)
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

    frozen_mcgc = mcgc_classifier(FROZEN_MCGC_PARAMS).fit(X_train, y_train)
    frozen_pred, frozen_risk = mcgc_prediction_and_risk(frozen_mcgc, X_test)

    tuned_mcgc, mcgc_selected, mcgc_candidates = tune_mcgc(
        X_train,
        y_train,
        seed,
    )
    tuned_pred, tuned_risk = mcgc_prediction_and_risk(tuned_mcgc, X_test)

    models = {
        "frozen_mcgc_local_margin": v12.model_metrics(
            y_test, frozen_pred, frozen_risk
        ),
        "tuned_mcgc_local_margin": v12.model_metrics(
            y_test, tuned_pred, tuned_risk
        ),
    }
    tuning = {
        "tuned_mcgc_local_margin": {
            "selected": mcgc_selected,
            "n_candidates": len(mcgc_candidates),
            "all_candidates": mcgc_candidates,
        }
    }

    baseline_specs = (
        ("tuned_knn_confidence", "knn", KNN_GRID),
        ("tuned_logistic_confidence", "logistic", LOGISTIC_GRID),
        ("tuned_rbf_svm_confidence", "svm", SVM_GRID),
        ("tuned_random_forest_confidence", "rf", RF_GRID),
    )

    for output_name, kind, grid in baseline_specs:
        fitted, selected, candidate_records = v12.tune_baseline(
            kind, grid, X_train, y_train, seed
        )
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
    tuned_e = np.asarray(
        [r["models"]["tuned_mcgc_local_margin"]["eaurc"] for r in runs]
    )
    tuned_a = np.asarray(
        [r["models"]["tuned_mcgc_local_margin"]["aurc"] for r in runs]
    )
    frozen_e = np.asarray(
        [r["models"]["frozen_mcgc_local_margin"]["eaurc"] for r in runs]
    )
    frozen_a = np.asarray(
        [r["models"]["frozen_mcgc_local_margin"]["aurc"] for r in runs]
    )

    model_names = (
        "frozen_mcgc_local_margin",
        "tuned_mcgc_local_margin",
    ) + BASELINES

    return {
        "runs": runs,
        "models": {
            model_name: aggregate_model(runs, model_name)
            for model_name in model_names
        },
        "paired_eaurc_advantage": {
            **{
                f"tuned_mcgc_vs_{baseline}": v12.delta_summary(
                    np.asarray(
                        [r["models"][baseline]["eaurc"] for r in runs]
                    )
                    - tuned_e
                )
                for baseline in BASELINES
            },
            "tuned_mcgc_vs_frozen_mcgc": v12.delta_summary(
                frozen_e - tuned_e
            ),
        },
        "paired_aurc_advantage": {
            **{
                f"tuned_mcgc_vs_{baseline}": v12.delta_summary(
                    np.asarray(
                        [r["models"][baseline]["aurc"] for r in runs]
                    )
                    - tuned_a
                )
                for baseline in BASELINES
            },
            "tuned_mcgc_vs_frozen_mcgc": v12.delta_summary(
                frozen_a - tuned_a
            ),
        },
    }


def pooled_primary(datasets):
    e_deltas = {baseline: [] for baseline in BASELINES}
    a_deltas = {baseline: [] for baseline in BASELINES}
    e_old = []
    a_old = []

    for name in PRIMARY_DATASETS:
        for run in datasets[name]["runs"]:
            tuned = run["models"]["tuned_mcgc_local_margin"]
            frozen = run["models"]["frozen_mcgc_local_margin"]
            e_old.append(frozen["eaurc"] - tuned["eaurc"])
            a_old.append(frozen["aurc"] - tuned["aurc"])
            for baseline in BASELINES:
                record = run["models"][baseline]
                e_deltas[baseline].append(record["eaurc"] - tuned["eaurc"])
                a_deltas[baseline].append(record["aurc"] - tuned["aurc"])

    return {
        "paired_eaurc_advantage": {
            **{
                f"tuned_mcgc_vs_{baseline}": v12.delta_summary(values)
                for baseline, values in e_deltas.items()
            },
            "tuned_mcgc_vs_frozen_mcgc": v12.delta_summary(e_old),
        },
        "paired_aurc_advantage": {
            **{
                f"tuned_mcgc_vs_{baseline}": v12.delta_summary(values)
                for baseline, values in a_deltas.items()
            },
            "tuned_mcgc_vs_frozen_mcgc": v12.delta_summary(a_old),
        },
    }


def fmt(value):
    return "NA" if value is None else f"{value:.6f}"


def print_summary(result):
    print("=== MCGC v1.4 FAIR TUNED-vs-TUNED CONFIRMATION ===")
    for name, dataset in result["datasets"].items():
        f = dataset["models"]["frozen_mcgc_local_margin"]
        m = dataset["models"]["tuned_mcgc_local_margin"]
        k = dataset["models"]["tuned_knn_confidence"]
        s = dataset["models"]["tuned_rbf_svm_confidence"]
        r = dataset["models"]["tuned_random_forest_confidence"]
        d0 = dataset["paired_eaurc_advantage"]["tuned_mcgc_vs_frozen_mcgc"]
        dk = dataset["paired_eaurc_advantage"]["tuned_mcgc_vs_tuned_knn_confidence"]
        ds = dataset["paired_eaurc_advantage"]["tuned_mcgc_vs_tuned_rbf_svm_confidence"]
        dr = dataset["paired_eaurc_advantage"]["tuned_mcgc_vs_tuned_random_forest_confidence"]
        print(
            f"{name:16s} EAURC F/M/K/S/R="
            f"{fmt(f['mean_eaurc'])}/{fmt(m['mean_eaurc'])}/{fmt(k['mean_eaurc'])}/"
            f"{fmt(s['mean_eaurc'])}/{fmt(r['mean_eaurc'])} "
            f"dOld={fmt(d0['mean'])} {d0['wins']}/{d0['ties']}/{d0['losses']} "
            f"dK={fmt(dk['mean'])} {dk['wins']}/{dk['ties']}/{dk['losses']} "
            f"dS={fmt(ds['mean'])} {ds['wins']}/{ds['ties']}/{ds['losses']} "
            f"dR={fmt(dr['mean'])} {dr['wins']}/{dr['ties']}/{dr['losses']}"
        )

    print("=== PRIMARY NEW REAL-DATA FAMILY ===")
    pooled = result["primary_real_data_summary"]
    keys = [
        "tuned_mcgc_vs_frozen_mcgc",
        "tuned_mcgc_vs_tuned_knn_confidence",
        "tuned_mcgc_vs_tuned_logistic_confidence",
        "tuned_mcgc_vs_tuned_rbf_svm_confidence",
        "tuned_mcgc_vs_tuned_random_forest_confidence",
    ]
    for key in keys:
        e = pooled["paired_eaurc_advantage"][key]
        a = pooled["paired_aurc_advantage"][key]
        print(
            f"{key:48s} E-AURC adv mean={fmt(e['mean'])} median={fmt(e['median'])} "
            f"W/T/L={e['wins']}/{e['ties']}/{e['losses']} p={fmt(e['sign_test_p_two_sided'])} | "
            f"AURC adv mean={fmt(a['mean'])} W/T/L={a['wins']}/{a['ties']}/{a['losses']}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("v14_tuned_mcgc.json"),
    )
    args = parser.parse_args()

    datasets = {name: evaluate_dataset(name) for name in DATASETS}
    result = {
        "protocol": {
            "version": "1.4",
            "status": "frozen fair tuned-vs-tuned confirmation",
            "primary_datasets": list(PRIMARY_DATASETS),
            "control_datasets": list(CONTROL_DATASETS),
            "seeds": SEEDS,
            "outer_split": "70% train / 30% untouched test",
            "outer_split_seed": "5000 + dataset_seed",
            "inner_cv": "3-fold stratified CV on outer training only",
            "inner_cv_seed": "6000 + dataset_seed",
            "primary_metric": "outer-test E-AURC; lower is better",
            "inner_selection_metric": "OOF E-AURC; lower is better",
            "mcgc_tuned": True,
            "mcgc_risk": "local_margin_risk",
            "mcgc_grid": list(MCGC_GRID),
            "frozen_mcgc_params": FROZEN_MCGC_PARAMS,
            "knn_grid": list(KNN_GRID),
            "logistic_grid": list(LOGISTIC_GRID),
            "svm_grid": list(SVM_GRID),
            "rf_grid": list(RF_GRID),
        },
        "datasets": datasets,
        "primary_real_data_summary": pooled_primary(datasets),
    }

    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print_summary(result)
    print(f"WROTE {args.output}")


if __name__ == "__main__":
    main()
