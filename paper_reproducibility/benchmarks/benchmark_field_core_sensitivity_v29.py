from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split

from mcgc import MultiAgentChemotacticGraphClassifier, reliability_features


ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / "examples" / "benchmark_tuned_selective_v12.py"

# Frozen one-factor-at-a-time sensitivity protocol.
# No configuration is selected from these results.
CONFIGS = {
    "nominal_K12_a085_nD40": {
        "n_neighbors": 12,
        "diffusion_alpha": 0.85,
        "diffusion_steps": 40,
    },
    "K8": {
        "n_neighbors": 8,
        "diffusion_alpha": 0.85,
        "diffusion_steps": 40,
    },
    "K16": {
        "n_neighbors": 16,
        "diffusion_alpha": 0.85,
        "diffusion_steps": 40,
    },
    "alpha075": {
        "n_neighbors": 12,
        "diffusion_alpha": 0.75,
        "diffusion_steps": 40,
    },
    "alpha095": {
        "n_neighbors": 12,
        "diffusion_alpha": 0.95,
        "diffusion_steps": 40,
    },
    "nD20": {
        "n_neighbors": 12,
        "diffusion_alpha": 0.85,
        "diffusion_steps": 20,
    },
    "nD80": {
        "n_neighbors": 12,
        "diffusion_alpha": 0.85,
        "diffusion_steps": 80,
    },
}

NOMINAL = "nominal_K12_a085_nD40"


def _load_parent():
    spec = importlib.util.spec_from_file_location(
        "benchmark_tuned_selective_v12",
        PARENT,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


v12 = _load_parent()

DATASETS = tuple(v12.PRIMARY_DATASETS)
SEEDS = list(v12.SEEDS)


def make_model(cfg):
    # Field-only predictive core. Walker/scoring parameters are not used.
    return MultiAgentChemotacticGraphClassifier(
        n_neighbors=cfg["n_neighbors"],
        diffusion_alpha=cfg["diffusion_alpha"],
        diffusion_steps=cfg["diffusion_steps"],
        random_state=17,
        standardize=True,
        use_walkers=False,
    )


def local_margin_risk(rows):
    return np.asarray(
        [row["risk_scores"]["local_margin_risk"] for row in rows],
        dtype=float,
    )


def delta_summary(values, eps=1e-12):
    arr = np.asarray(values, dtype=float)

    wins = int(np.sum(arr > eps))
    ties = int(np.sum(np.abs(arr) <= eps))
    losses = int(np.sum(arr < -eps))

    n_non_tied = wins + losses
    sign_p = None

    if n_non_tied:
        k = min(wins, losses)
        tail = (
            sum(math.comb(n_non_tied, i) for i in range(k + 1))
            / (2 ** n_non_tied)
        )
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


def evaluate_model(X_train, X_test, y_train, y_test, cfg):
    model = make_model(cfg).fit(X_train, y_train)

    pred, rows = reliability_features(model, X_test)
    risk = local_margin_risk(rows)

    metrics = v12.model_metrics(y_test, pred, risk)
    metrics["balanced_accuracy"] = float(
        balanced_accuracy_score(y_test, pred)
    )

    return {
        "predictions": np.asarray(pred),
        "risk": risk,
        "metrics": metrics,
    }


def evaluate_seed(dataset_name: str, seed: int):
    X, y = v12.dataset_for(dataset_name, seed)

    # Exact frozen v1.2 outer split.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=5000 + seed,
        stratify=y,
    )

    evaluated = {}

    for name, cfg in CONFIGS.items():
        evaluated[name] = evaluate_model(
            X_train,
            X_test,
            y_train,
            y_test,
            cfg,
        )

    nominal = evaluated[NOMINAL]
    runs = {}

    for name, result in evaluated.items():
        m = result["metrics"]

        entry = {
            "accuracy": float(m["accuracy"]),
            "balanced_accuracy": float(m["balanced_accuracy"]),
            "aurc": float(m["aurc"]),
            "eaurc": float(m["eaurc"]),
            "error_rate": float(m["error_rate"]),
            "oracle_aurc": float(m["oracle_aurc"]),
        }

        if name == NOMINAL:
            entry["paired_vs_nominal"] = {
                "accuracy_delta_config_minus_nominal": 0.0,
                "balanced_accuracy_delta_config_minus_nominal": 0.0,
                "eaurc_delta_config_minus_nominal": 0.0,
                "aurc_delta_config_minus_nominal": 0.0,
                "prediction_disagreement_rate": 0.0,
                "max_abs_local_risk_difference": 0.0,
            }
        else:
            entry["paired_vs_nominal"] = {
                # Positive means the sensitivity configuration has higher accuracy.
                "accuracy_delta_config_minus_nominal": float(
                    m["accuracy"] - nominal["metrics"]["accuracy"]
                ),
                "balanced_accuracy_delta_config_minus_nominal": float(
                    m["balanced_accuracy"]
                    - nominal["metrics"]["balanced_accuracy"]
                ),
                # Positive E-AURC/AURC delta means the sensitivity configuration
                # is worse because lower selective-risk values are better.
                "eaurc_delta_config_minus_nominal": float(
                    m["eaurc"] - nominal["metrics"]["eaurc"]
                ),
                "aurc_delta_config_minus_nominal": float(
                    m["aurc"] - nominal["metrics"]["aurc"]
                ),
                "prediction_disagreement_rate": float(
                    np.mean(
                        result["predictions"]
                        != nominal["predictions"]
                    )
                ),
                "max_abs_local_risk_difference": float(
                    np.max(
                        np.abs(
                            result["risk"]
                            - nominal["risk"]
                        )
                    )
                ),
            }

        runs[name] = entry

    return {
        "seed": int(seed),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "configurations": runs,
    }


def collect_units(datasets, config_name, metric_key):
    values = []

    for dataset_name in DATASETS:
        for run in datasets[dataset_name]["runs"]:
            values.append(
                run["configurations"][config_name][metric_key]
            )

    return np.asarray(values, dtype=float)


def collect_deltas(datasets, config_name, key):
    values = []

    for dataset_name in DATASETS:
        for run in datasets[dataset_name]["runs"]:
            values.append(
                run["configurations"][config_name][
                    "paired_vs_nominal"
                ][key]
            )

    return np.asarray(values, dtype=float)


def summarize_configuration(datasets, config_name):
    accuracy = collect_units(
        datasets, config_name, "accuracy"
    )
    ba = collect_units(
        datasets, config_name, "balanced_accuracy"
    )
    eaurc = collect_units(
        datasets, config_name, "eaurc"
    )
    aurc = collect_units(
        datasets, config_name, "aurc"
    )

    summary = {
        "mean_accuracy": float(np.mean(accuracy)),
        "mean_balanced_accuracy": float(np.mean(ba)),
        "mean_eaurc": float(np.mean(eaurc)),
        "mean_aurc": float(np.mean(aurc)),
    }

    if config_name != NOMINAL:
        summary["paired_vs_nominal"] = {
            "accuracy_delta_config_minus_nominal":
                delta_summary(
                    collect_deltas(
                        datasets,
                        config_name,
                        "accuracy_delta_config_minus_nominal",
                    )
                ),
            "balanced_accuracy_delta_config_minus_nominal":
                delta_summary(
                    collect_deltas(
                        datasets,
                        config_name,
                        "balanced_accuracy_delta_config_minus_nominal",
                    )
                ),
            "eaurc_delta_config_minus_nominal":
                delta_summary(
                    collect_deltas(
                        datasets,
                        config_name,
                        "eaurc_delta_config_minus_nominal",
                    )
                ),
            "aurc_delta_config_minus_nominal":
                delta_summary(
                    collect_deltas(
                        datasets,
                        config_name,
                        "aurc_delta_config_minus_nominal",
                    )
                ),
            "prediction_disagreement_rate":
                delta_summary(
                    collect_deltas(
                        datasets,
                        config_name,
                        "prediction_disagreement_rate",
                    )
                ),
            "max_abs_local_risk_difference":
                delta_summary(
                    collect_deltas(
                        datasets,
                        config_name,
                        "max_abs_local_risk_difference",
                    )
                ),
        }

    return summary


def summarize_dataset(runs):
    out = {}

    for config_name in CONFIGS:
        acc = np.asarray([
            r["configurations"][config_name]["accuracy"]
            for r in runs
        ])
        ba = np.asarray([
            r["configurations"][config_name]["balanced_accuracy"]
            for r in runs
        ])
        e = np.asarray([
            r["configurations"][config_name]["eaurc"]
            for r in runs
        ])
        a = np.asarray([
            r["configurations"][config_name]["aurc"]
            for r in runs
        ])

        out[config_name] = {
            "mean_accuracy": float(np.mean(acc)),
            "mean_balanced_accuracy": float(np.mean(ba)),
            "mean_eaurc": float(np.mean(e)),
            "mean_aurc": float(np.mean(a)),
        }

    return out


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/v29_field_core_parameter_sensitivity.json"
        ),
    )

    args = parser.parse_args()

    datasets = {}

    for dataset_name in DATASETS:
        print(f"DATASET {dataset_name}", flush=True)
        runs = []

        for seed in SEEDS:
            print(f"  seed={seed}", flush=True)

            run = evaluate_seed(dataset_name, seed)
            runs.append(run)

            nominal = run["configurations"][NOMINAL]

            print(
                "    nominal "
                f"acc={nominal['accuracy']:.6f} "
                f"E-AURC={nominal['eaurc']:.6f}",
                flush=True,
            )

            for name in CONFIGS:
                if name == NOMINAL:
                    continue

                x = run["configurations"][name]
                d = x["paired_vs_nominal"]

                print(
                    f"    {name:10s} "
                    f"dAcc={d['accuracy_delta_config_minus_nominal']:+.6f} "
                    f"dE={d['eaurc_delta_config_minus_nominal']:+.6f} "
                    f"disagree={d['prediction_disagreement_rate']:.6f}",
                    flush=True,
                )

        datasets[dataset_name] = {
            "runs": runs,
            "aggregate": summarize_dataset(runs),
        }

    pooled = {
        name: summarize_configuration(datasets, name)
        for name in CONFIGS
    }

    result = {
        "protocol": {
            "version": "v2.9",
            "name": "frozen field-core parameter sensitivity",
            "status": (
                "one-factor-at-a-time sensitivity analysis; "
                "not a tuning or model-selection procedure"
            ),
            "parent_protocol":
                "examples/benchmark_tuned_selective_v12.py",
            "predictive_core":
                "field-only local graph evidence; use_walkers=False",
            "primary_datasets": list(DATASETS),
            "seeds": list(SEEDS),
            "outer_split": {
                "test_size": 0.30,
                "random_state": "5000 + seed",
                "stratified": True,
            },
            "nominal_configuration": NOMINAL,
            "configurations": CONFIGS,
            "selection_from_sensitivity_results": False,
            "interpretation": {
                "accuracy_delta_config_minus_nominal":
                    "positive favors the sensitivity configuration",
                "eaurc_delta_config_minus_nominal":
                    "positive means the sensitivity configuration is worse; lower E-AURC is better",
                "aurc_delta_config_minus_nominal":
                    "positive means the sensitivity configuration is worse; lower AURC is better",
            },
        },
        "datasets": datasets,
        "pooled_primary": pooled,
    }

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.output.write_text(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    print()
    print("=== v2.9 FIELD-CORE PARAMETER SENSITIVITY ===")

    for name, summary in pooled.items():
        print()
        print(
            f"{name:24s} "
            f"acc={summary['mean_accuracy']:.6f} "
            f"BA={summary['mean_balanced_accuracy']:.6f} "
            f"E-AURC={summary['mean_eaurc']:.6f} "
            f"AURC={summary['mean_aurc']:.6f}"
        )

        if name != NOMINAL:
            p = summary["paired_vs_nominal"]

            print(
                "  dAcc(config-nominal): "
                + json.dumps(
                    p[
                        "accuracy_delta_config_minus_nominal"
                    ],
                    sort_keys=True,
                )
            )

            print(
                "  dE(config-nominal):   "
                + json.dumps(
                    p[
                        "eaurc_delta_config_minus_nominal"
                    ],
                    sort_keys=True,
                )
            )

            print(
                "  disagreement:         "
                + json.dumps(
                    p["prediction_disagreement_rate"],
                    sort_keys=True,
                )
            )

    print()
    print(f"FULL_JSON={args.output.resolve()}")


if __name__ == "__main__":
    main()
