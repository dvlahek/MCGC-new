from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from mcgc import MultiAgentChemotacticGraphClassifier, reliability_features

ROOT = Path(__file__).resolve().parents[1]
V26_SCRIPT = ROOT / "examples" / "benchmark_corrected_predictive_v26.py"


def _load_v26():
    spec = importlib.util.spec_from_file_location("benchmark_corrected_predictive_v26", V26_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


v26 = _load_v26()
v12 = v26.v12


def field_only_classifier():
    return MultiAgentChemotacticGraphClassifier(
        n_neighbors=12,
        n_walkers=12,
        max_steps=10,
        diffusion_steps=40,
        random_state=17,
        adaptive_walker_gate=True,
        use_walkers=False,
    )


def run_field_only(name: str, seed: int):
    X, y = v26.dataset_for(name, seed)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=5000 + seed,
        stratify=y,
    )
    model = field_only_classifier().fit(X_train, y_train)
    pred, rows = reliability_features(model, X_test)
    risk = np.asarray(
        [row["risk_scores"]["local_margin_risk"] for row in rows],
        dtype=float,
    )
    return v12.model_metrics(y_test, pred, risk)


def by_seed(block):
    return {int(row["seed"]): row for row in block["runs"]}


def metric_delta_summary(rows, key: str, sign: float = 1.0):
    values = np.asarray(
        [sign * (row["field_only"][key] - row["full_mcgc"][key]) for row in rows],
        dtype=float,
    )
    return v12.delta_summary(values)


def field_advantage_against_baseline(rows, baseline: str, key: str, smaller_better: bool):
    values = []
    for row in rows:
        f = row["field_only"][key]
        b = row["baselines"][baseline][key]
        values.append((b - f) if smaller_better else (f - b))
    return v12.delta_summary(np.asarray(values, dtype=float))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--v26-json",
        type=Path,
        default=Path("results/v26/v26_corrected_predictive_baselines.json"),
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("results/v27/v27_field_only_selective_risk.json"),
    )
    args = ap.parse_args()

    parent = json.loads(args.v26_json.read_text(encoding="utf-8"))
    records = []

    for name in v26.DATASETS:
        parent_runs = by_seed(parent["datasets"][name])
        for seed in v26.SEEDS:
            p = parent_runs[seed]
            field = run_field_only(name, seed)
            records.append(
                {
                    "dataset": name,
                    "seed": int(seed),
                    "field_only": field,
                    "full_mcgc": p["models"]["mcgc_local_margin"],
                    "baselines": {
                        b: p["models"][b] for b in v26.BASELINES
                    },
                }
            )

    nonlinear = [r for r in records if r["dataset"] in v26.NONLINEAR_PRIMARY]
    real = [r for r in records if r["dataset"] in v26.REAL_PRIMARY]

    result = {
        "protocol": {
            "parent": "v2.6 corrected predictive-baseline protocol",
            "change": "MCGC use_walkers=False; all datasets, splits, seeds, graph/diffusion settings, and local-margin risk unchanged",
            "important_note": "MCGC selective risk is negative local-evidence margin in both variants; this audit isolates changes in predicted labels while keeping the ranking signal fixed.",
        },
        "records": records,
        "field_minus_full": {
            "nonlinear": {
                "accuracy": metric_delta_summary(nonlinear, "accuracy", +1.0),
                "aurc": metric_delta_summary(nonlinear, "aurc", -1.0),
                "eaurc": metric_delta_summary(nonlinear, "eaurc", -1.0),
                "error_detection_roc_auc": metric_delta_summary(
                    nonlinear, "error_detection_roc_auc", +1.0
                ),
            },
            "real": {
                "accuracy": metric_delta_summary(real, "accuracy", +1.0),
                "aurc": metric_delta_summary(real, "aurc", -1.0),
                "eaurc": metric_delta_summary(real, "eaurc", -1.0),
                "error_detection_roc_auc": metric_delta_summary(
                    real, "error_detection_roc_auc", +1.0
                ),
            },
        },
        "nonlinear_field_only_vs_baselines": {
            b: {
                "accuracy": field_advantage_against_baseline(
                    nonlinear, b, "accuracy", smaller_better=False
                ),
                "aurc": field_advantage_against_baseline(
                    nonlinear, b, "aurc", smaller_better=True
                ),
                "eaurc": field_advantage_against_baseline(
                    nonlinear, b, "eaurc", smaller_better=True
                ),
                "error_detection_roc_auc": field_advantage_against_baseline(
                    nonlinear, b, "error_detection_roc_auc", smaller_better=False
                ),
            }
            for b in v26.BASELINES
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")

    print("=== FIELD-ONLY VS FULL MCGC: NONLINEAR ===")
    for key, value in result["field_minus_full"]["nonlinear"].items():
        print(key, json.dumps(value, sort_keys=True))
    print("=== FIELD-ONLY VS BASELINES: NONLINEAR E-AURC ===")
    for baseline, metrics in result["nonlinear_field_only_vs_baselines"].items():
        print(baseline, json.dumps(metrics["eaurc"], sort_keys=True))
    print(f"FULL_JSON={args.output.resolve()}")


if __name__ == "__main__":
    main()
