from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from mcgc import MultiAgentChemotacticGraphClassifier, reliability_features

ROOT = Path(__file__).resolve().parents[1]
V26_SCRIPT = ROOT / "examples" / "benchmark_corrected_predictive_v26.py"

def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module

v26 = load_module(V26_SCRIPT, "benchmark_corrected_predictive_v26")
v12 = v26.v12

SEEDS = list(v26.SEEDS)
DATASETS = tuple(v26.DATASETS)
NONLINEAR_PRIMARY = tuple(v26.NONLINEAR_PRIMARY)
REAL_PRIMARY = tuple(v26.REAL_PRIMARY)
BASELINES = tuple(v26.BASELINES)

def field_classifier():
    return MultiAgentChemotacticGraphClassifier(
        n_neighbors=12,
        diffusion_alpha=0.85,
        diffusion_steps=40,
        random_state=17,
        standardize=True,
        use_walkers=False,
    )

def resolve_v26_json(path: Path) -> Path:
    if path.exists():
        return path
    candidates = list(ROOT.rglob("v26_corrected_predictive_baselines.json"))
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(
            "Could not find v26_corrected_predictive_baselines.json. "
            "Pass it explicitly with --v26-json PATH."
        )
    raise RuntimeError(
        "Multiple v26_corrected_predictive_baselines.json files found. "
        "Pass the intended one explicitly with --v26-json PATH."
    )

def delta_summary(values, eps=1e-12):
    arr = np.asarray(values, dtype=float)
    wins = int(np.sum(arr > eps))
    ties = int(np.sum(np.abs(arr) <= eps))
    losses = int(np.sum(arr < -eps))
    n = wins + losses
    p = None
    if n:
        k = min(wins, losses)
        tail = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
        p = float(min(1.0, 2.0 * tail))
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
        "sign_test_p_two_sided": p,
    }

def evaluate_field(name: str, seed: int):
    X, y = v26.dataset_for(name, seed)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.30,
        random_state=5000 + seed,
        stratify=y,
    )
    model = field_classifier().fit(X_train, y_train)
    pred, rows = reliability_features(model, X_test)
    risk = np.asarray(
        [row["risk_scores"]["local_margin_risk"] for row in rows],
        dtype=float,
    )
    metrics = v12.model_metrics(y_test, pred, risk)
    return {
        "seed": int(seed),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "metrics": metrics,
    }

def pooled(result_datasets, baseline_json, names):
    out = {}
    for baseline in BASELINES:
        e_delta = []
        a_delta = []
        auc_delta = []
        acc_delta = []
        for name in names:
            old_runs = {
                int(r["seed"]): r
                for r in baseline_json["datasets"][name]["runs"]
            }
            new_runs = {
                int(r["seed"]): r
                for r in result_datasets[name]["runs"]
            }
            for seed in SEEDS:
                field = new_runs[seed]["metrics"]
                base = old_runs[seed]["models"][baseline]
                # Positive values favour final field-only CGEC.
                e_delta.append(float(base["eaurc"]) - float(field["eaurc"]))
                a_delta.append(float(base["aurc"]) - float(field["aurc"]))
                acc_delta.append(float(field["accuracy"]) - float(base["accuracy"]))
                if (
                    field["error_detection_roc_auc"] is not None
                    and base["error_detection_roc_auc"] is not None
                ):
                    auc_delta.append(
                        float(field["error_detection_roc_auc"])
                        - float(base["error_detection_roc_auc"])
                    )
        out[baseline] = {
            "eaurc_advantage_field_over_baseline": delta_summary(e_delta),
            "aurc_advantage_field_over_baseline": delta_summary(a_delta),
            "error_detection_auc_advantage_field_over_baseline":
                delta_summary(auc_delta) if auc_delta else None,
            "accuracy_advantage_field_over_baseline": delta_summary(acc_delta),
        }
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--v26-json",
        type=Path,
        default=ROOT / "results" / "v26_corrected_predictive_baselines.json",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "v30_field_only_final_predictive.json",
    )
    args = ap.parse_args()

    source = resolve_v26_json(args.v26_json)
    frozen = json.loads(source.read_text(encoding="utf-8"))

    datasets = {}
    for name in DATASETS:
        print(f"DATASET {name}", flush=True)
        runs = []
        for seed in SEEDS:
            print(f"  seed={seed}", flush=True)
            rec = evaluate_field(name, seed)
            runs.append(rec)
            m = rec["metrics"]
            print(
                f"    acc={m['accuracy']:.6f} "
                f"E-AURC={m['eaurc']:.6f} "
                f"AURC={m['aurc']:.6f}",
                flush=True,
            )
        datasets[name] = {"runs": runs}

    result = {
        "protocol": {
            "version": "v3.0",
            "name": "final field-only predictive confirmation",
            "status": "confirmation only; no retraining or retuning of baselines",
            "source_frozen_baselines": str(source),
            "field_only_definition":
                "K=12, alpha=0.85, nD=40, use_walkers=False",
            "outer_split":
                "70/30 stratified, random_state=5000+seed",
            "seeds": SEEDS,
            "datasets": list(DATASETS),
            "nonlinear_primary": list(NONLINEAR_PRIMARY),
            "real_primary": list(REAL_PRIMARY),
            "primary_risk":
                "negative top-two local graph-evidence margin",
            "selection_from_results": False,
        },
        "datasets": datasets,
        "nonlinear_primary": pooled(
            datasets, frozen, NONLINEAR_PRIMARY
        ),
        "real_primary": pooled(
            datasets, frozen, REAL_PRIMARY
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print("\n=== FINAL FIELD-ONLY PREDICTIVE CONFIRMATION ===")
    for family in ("nonlinear_primary", "real_primary"):
        print(f"\n{family.upper()}")
        for baseline, block in result[family].items():
            e = block["eaurc_advantage_field_over_baseline"]
            print(
                f"{baseline:42s} "
                f"Eadv={e['mean']:+.8f} "
                f"W/T/L={e['wins']}/{e['ties']}/{e['losses']} "
                f"p={e['sign_test_p_two_sided']}"
            )
    print(f"\nFULL_JSON={args.output.resolve()}")

if __name__ == "__main__":
    main()
