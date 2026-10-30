from __future__ import annotations

# v3.6b: self-contained medium runtime audit.
# All timing is performed on one workstation under a single-thread policy.
# Runtime scope: all eight real-data datasets x fixed seeds {0,2,5,8,11}.

import os

for _name in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_name] = "1"

import argparse
import importlib.util
import json
import platform
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import sklearn
from sklearn.model_selection import train_test_split


HERE = Path(__file__).resolve().parent
V34B_SCRIPT = HERE / "benchmark_real_data_tuned_cgec_runtime_v34b.py"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


v34b = _load_module(V34B_SCRIPT, "benchmark_real_data_tuned_cgec_runtime_v34b_v36b")
v34 = v34b.v34
v33 = v34.v33
v32 = v34.v32
v26 = v34.v26
v12 = v34.v12

DATASETS = tuple(v34.DATASETS)
BASELINES = tuple(v34.BASELINES)
BASELINE_SPECS = tuple(v34.BASELINE_SPECS)
CGEC_GRID = tuple(v34.CGEC_GRID)
FIXED_CGEC_PARAMS = dict(v34.FIXED_CGEC_PARAMS)
RUNTIME_SEEDS = (0, 2, 5, 8, 11)
TIMING_REPEATS = 3


def read_jsonl(path: Path, key_fields):
    out = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = tuple(row[field] for field in key_fields)
        out[key] = row
    return out


def append_jsonl(path: Path, record: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def candidate_timing_records(
    name: str,
    X_train,
    y_train,
    seed: int,
    candidate_cache: dict,
    candidate_checkpoint: Path,
):
    rows = []
    for idx, params in enumerate(CGEC_GRID, start=1):
        ckey = v33.candidate_key(params)
        key = (name, int(seed), ckey)

        if key in candidate_cache:
            rec = candidate_cache[key]
            row = rec["candidate"]
            elapsed = float(rec["elapsed_seconds"])
            print(
                f"      candidate {idx:02d}/{len(CGEC_GRID)} {ckey} "
                f"SKIP checkpoint E={row['oof_eaurc']:.6f} t={elapsed:.3f}s",
                flush=True,
            )
        else:
            t0 = time.perf_counter()
            row = v33.evaluate_candidate_oof(X_train, y_train, seed, params)
            elapsed = time.perf_counter() - t0
            rec = {
                "dataset": name,
                "seed": int(seed),
                "candidate_key": ckey,
                "candidate": row,
                "elapsed_seconds": float(elapsed),
            }
            append_jsonl(candidate_checkpoint, rec)
            candidate_cache[key] = rec
            print(
                f"      candidate {idx:02d}/{len(CGEC_GRID)} {ckey} "
                f"E={row['oof_eaurc']:.6f} acc={row['oof_accuracy']:.6f} "
                f"t={elapsed:.3f}s",
                flush=True,
            )

        rows.append((row, elapsed))

    rows.sort(key=lambda item: v33.selection_sort_key(item[0]))
    selected = rows[0][0]
    total_seconds = float(sum(elapsed for _, elapsed in rows))
    return selected, rows, total_seconds


def time_fixed_cgec(
    X_train,
    y_train,
    X_test,
    y_test,
    repeats: int,
):
    return v34.time_cgec_final_model(
        FIXED_CGEC_PARAMS,
        X_train,
        y_train,
        X_test,
        y_test,
        repeats,
    )


def time_tuned_cgec(
    name,
    seed,
    X_train,
    y_train,
    X_test,
    y_test,
    repeats,
    candidate_cache,
    candidate_checkpoint,
):
    selected, candidate_rows, candidate_seconds = candidate_timing_records(
        name,
        X_train,
        y_train,
        seed,
        candidate_cache,
        candidate_checkpoint,
    )

    final_timing = v34.time_cgec_final_model(
        selected["params"],
        X_train,
        y_train,
        X_test,
        y_test,
        repeats,
    )

    e2e = (
        candidate_seconds
        + float(final_timing["primary_fit_seconds"])
        + float(final_timing["primary_prediction_seconds"])
    )

    return {
        "selected": selected,
        "n_candidates": int(len(candidate_rows)),
        "candidate_evaluation_seconds_total": float(candidate_seconds),
        "candidate_evaluation_seconds": v34.timing_summary(
            [elapsed for _, elapsed in candidate_rows]
        ),
        "final_model_timing": final_timing,
        "end_to_end_tuned_seconds": float(e2e),
    }


def time_baseline(
    output_name,
    kind,
    grid,
    X_train,
    y_train,
    X_test,
    y_test,
    seed,
    repeats,
):
    t0 = time.perf_counter()
    fitted, selected, all_records = v26.tune(
        kind,
        grid,
        X_train,
        y_train,
        seed,
    )
    tune_seconds = time.perf_counter() - t0

    t1 = time.perf_counter()
    pred, risk = v26.prediction_and_risk(kind, fitted, X_test)
    prediction_seconds = time.perf_counter() - t1
    metrics = v12.model_metrics(y_test, pred, risk)

    selected_final = v34.time_baseline_final_model(
        kind,
        selected["params"],
        X_train,
        y_train,
        X_test,
        y_test,
        repeats,
    )

    return {
        "output_name": output_name,
        "kind": kind,
        "selected": selected,
        "n_candidates": int(len(all_records)),
        "tuning_routine_seconds": float(tune_seconds),
        "outer_prediction_seconds": float(prediction_seconds),
        "end_to_end_tuned_seconds": float(tune_seconds + prediction_seconds),
        "metrics": metrics,
        "selected_final_model_timing": selected_final,
    }


def ensure_component(
    name,
    seed,
    component,
    build_fn,
    component_cache,
    component_checkpoint,
):
    key = (name, int(seed), component)
    if key in component_cache:
        print(f"    {component} SKIP component checkpoint", flush=True)
        return component_cache[key]["payload"]

    payload = build_fn()
    rec = {
        "dataset": name,
        "seed": int(seed),
        "component": component,
        "payload": payload,
    }
    append_jsonl(component_checkpoint, rec)
    component_cache[key] = rec
    return payload


def environment_metadata():
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "logical_cpu_count": os.cpu_count(),
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "thread_environment": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
        "rf_n_jobs": 1,
        "timer": "time.perf_counter wall-clock seconds",
    }


def build_summary(names, seeds, component_cache):
    method_keys = [
        "fixed_cgec",
        "tuned_cgec",
        *[f"baseline::{name}" for name in BASELINES],
    ]

    panel_e2e = {key: [] for key in method_keys}
    panel_final = {key: [] for key in method_keys}
    per_dataset = {}
    predictive_rows = []
    parameter_counter = Counter()

    for name in names:
        ds_e2e = {key: [] for key in method_keys}
        ds_final = {key: [] for key in method_keys}
        fixed_e = []
        tuned_e = []
        baseline_e = {baseline: [] for baseline in BASELINES}

        for seed in seeds:
            fixed = component_cache[(name, int(seed), "fixed_cgec")]["payload"]
            tuned = component_cache[(name, int(seed), "tuned_cgec")]["payload"]

            fixed_final = float(
                fixed["final_model_timing"]["fit_plus_prediction_seconds"]["median"]
            )
            tuned_final = float(
                tuned["final_model_timing"]["fit_plus_prediction_seconds"]["median"]
            )
            tuned_e2e = float(tuned["end_to_end_tuned_seconds"])

            ds_e2e["fixed_cgec"].append(fixed_final)
            ds_e2e["tuned_cgec"].append(tuned_e2e)
            ds_final["fixed_cgec"].append(fixed_final)
            ds_final["tuned_cgec"].append(tuned_final)
            panel_e2e["fixed_cgec"].append(fixed_final)
            panel_e2e["tuned_cgec"].append(tuned_e2e)
            panel_final["fixed_cgec"].append(fixed_final)
            panel_final["tuned_cgec"].append(tuned_final)

            fixed_e.append(float(fixed["final_model_timing"]["metrics"]["eaurc"]))
            tuned_e.append(float(tuned["final_model_timing"]["metrics"]["eaurc"]))
            parameter_counter[tuned["selected"]["candidate_key"]] += 1

            for baseline in BASELINES:
                comp = f"baseline::{baseline}"
                payload = component_cache[(name, int(seed), comp)]["payload"]
                e2e = float(payload["end_to_end_tuned_seconds"])
                final_med = float(
                    payload["selected_final_model_timing"]["fit_plus_prediction_seconds"]["median"]
                )
                ds_e2e[comp].append(e2e)
                ds_final[comp].append(final_med)
                panel_e2e[comp].append(e2e)
                panel_final[comp].append(final_med)
                baseline_e[baseline].append(float(payload["metrics"]["eaurc"]))

        per_dataset[name] = {
            "end_to_end_or_fixed_final_model": {
                key: v34.timing_summary(vals)
                for key, vals in ds_e2e.items()
            },
            "selected_final_model_fit_plus_prediction": {
                key: v34.timing_summary(vals)
                for key, vals in ds_final.items()
            },
        }

        predictive_rows.append({
            "dataset": name,
            "fixed_cgec_mean_eaurc": float(np.mean(fixed_e)),
            "tuned_cgec_mean_eaurc": float(np.mean(tuned_e)),
            "improvement_tuned_over_fixed": float(np.mean(fixed_e) - np.mean(tuned_e)),
            "baseline_mean_eaurc": {
                baseline: float(np.mean(vals))
                for baseline, vals in baseline_e.items()
            },
        })

    tuned_vs_fixed = [
        row["fixed_cgec_mean_eaurc"] - row["tuned_cgec_mean_eaurc"]
        for row in predictive_rows
    ]
    tuned_vs_baselines = {
        baseline: [
            row["baseline_mean_eaurc"][baseline] - row["tuned_cgec_mean_eaurc"]
            for row in predictive_rows
        ]
        for baseline in BASELINES
    }

    return {
        "runtime": {
            "per_dataset": per_dataset,
            "panel_all_dataset_seed_units": {
                "end_to_end_or_fixed_final_model": {
                    key: v34.timing_summary(vals)
                    for key, vals in panel_e2e.items()
                },
                "selected_final_model_fit_plus_prediction": {
                    key: v34.timing_summary(vals)
                    for key, vals in panel_final.items()
                },
            },
            "interpretation": {
                "fixed_cgec": "fit+test prediction only; no tuning",
                "tuned_cgec": "36-candidate inner-CV tuning + final refit + test prediction",
                "baselines": "existing inner-CV tuning + final refit + test prediction",
                "selected_final_model": "median of three fresh fit+prediction repetitions; excludes tuning",
            },
        },
        "predictive_descriptive_on_runtime_seeds": {
            "per_dataset": predictive_rows,
            "dataset_level": {
                "tuned_vs_fixed": v33.delta_summary(tuned_vs_fixed),
                "tuned_vs_baselines": {
                    baseline: v33.delta_summary(values)
                    for baseline, values in tuned_vs_baselines.items()
                },
            },
            "parameter_selection_frequencies": dict(sorted(parameter_counter.items())),
            "note": "descriptive five-seed runtime subset; full predictive matched-tuning evidence is v3.3 with 12 seeds",
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--cache-dir",
        type=Path,
        default=v32.PAPER_ROOT / "source_data" / "pmlb_real_confirmation_v32c",
    )
    ap.add_argument(
        "--candidate-checkpoint",
        type=Path,
        default=v32.PAPER_ROOT / "results" / "v36b_runtime_candidates.jsonl",
    )
    ap.add_argument(
        "--component-checkpoint",
        type=Path,
        default=v32.PAPER_ROOT / "results" / "v36b_runtime_components.jsonl",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=v32.PAPER_ROOT / "results" / "v36b_runtime_summary.json",
    )
    ap.add_argument(
        "--timing-repeats",
        type=int,
        default=TIMING_REPEATS,
    )
    args = ap.parse_args()

    if args.timing_repeats < 1:
        raise ValueError("--timing-repeats must be >= 1")

    names = DATASETS
    seeds = RUNTIME_SEEDS

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    candidate_cache = read_jsonl(
        args.candidate_checkpoint,
        ("dataset", "seed", "candidate_key"),
    )
    component_cache = read_jsonl(
        args.component_checkpoint,
        ("dataset", "seed", "component"),
    )

    loaded = {}
    metadata = {}
    for name in names:
        print(f"LOAD {name}", flush=True)
        X, y, meta = v32.load_pmlb_dataset(name, args.cache_dir)
        loaded[name] = (X, y)
        metadata[name] = meta
        print(
            f"  n={meta['n_samples']} d={meta['n_features']} "
            f"C={meta['n_classes']} sha={meta['content_sha256'][:16]}",
            flush=True,
        )

    for name in names:
        X, y = loaded[name]
        print(f"DATASET {name}", flush=True)
        for seed in seeds:
            print(f"  seed={seed}", flush=True)
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.30,
                random_state=5000 + int(seed),
                stratify=y,
            )

            fixed = ensure_component(
                name,
                seed,
                "fixed_cgec",
                lambda: {
                    "params": dict(FIXED_CGEC_PARAMS),
                    "final_model_timing": time_fixed_cgec(
                        X_train,
                        y_train,
                        X_test,
                        y_test,
                        args.timing_repeats,
                    ),
                },
                component_cache,
                args.component_checkpoint,
            )
            print(
                "    fixed CGEC "
                f"E={fixed['final_model_timing']['metrics']['eaurc']:.6f} "
                f"median={fixed['final_model_timing']['fit_plus_prediction_seconds']['median']:.4f}s",
                flush=True,
            )

            tuned = ensure_component(
                name,
                seed,
                "tuned_cgec",
                lambda: time_tuned_cgec(
                    name,
                    seed,
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    args.timing_repeats,
                    candidate_cache,
                    args.candidate_checkpoint,
                ),
                component_cache,
                args.component_checkpoint,
            )
            print(
                f"    tuned CGEC SELECT {tuned['selected']['candidate_key']} "
                f"E={tuned['final_model_timing']['metrics']['eaurc']:.6f} "
                f"e2e={tuned['end_to_end_tuned_seconds']:.2f}s",
                flush=True,
            )

            for output_name, kind, grid in BASELINE_SPECS:
                component = f"baseline::{output_name}"
                payload = ensure_component(
                    name,
                    seed,
                    component,
                    lambda output_name=output_name, kind=kind, grid=grid: time_baseline(
                        output_name,
                        kind,
                        grid,
                        X_train,
                        y_train,
                        X_test,
                        y_test,
                        seed,
                        args.timing_repeats,
                    ),
                    component_cache,
                    args.component_checkpoint,
                )
                print(
                    f"    {output_name} "
                    f"E={payload['metrics']['eaurc']:.6f} "
                    f"e2e={payload['end_to_end_tuned_seconds']:.2f}s "
                    f"final median={payload['selected_final_model_timing']['fit_plus_prediction_seconds']['median']:.4f}s",
                    flush=True,
                )

    summary = build_summary(names, seeds, component_cache)
    result = {
        "protocol": {
            "version": "v3.6b",
            "name": "self-contained medium descriptive runtime audit",
            "datasets": list(names),
            "seeds": list(seeds),
            "n_dataset_seed_units": int(len(names) * len(seeds)),
            "outer_split": "70/30 stratified, random_state=5000+seed",
            "cgec_fixed": "K=12, alpha=0.85, nD=40",
            "cgec_tuning_grid": {
                "K": list(v33.K_GRID),
                "alpha": list(v33.ALPHA_GRID),
                "nD": list(v33.ND_GRID),
                "n_candidates": int(len(CGEC_GRID)),
            },
            "inner_cv": "3-fold stratified on outer training only, random_state=6000+seed",
            "selection_metric": "lowest OOF E-AURC; higher accuracy then lexical candidate key",
            "timing_repeats_final_model": int(args.timing_repeats),
            "single_thread_policy": True,
            "rf_n_jobs": 1,
            "self_contained": True,
            "predictive_inference": "not primary; full matched-tuning predictive evidence remains v3.3 over 12 seeds",
        },
        "environment": environment_metadata(),
        "dataset_metadata": metadata,
        **summary,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print("=== V3.6B MEDIUM RUNTIME AUDIT COMPLETE ===", flush=True)
    print(json.dumps(result["environment"], indent=2, sort_keys=True), flush=True)
    print("=== RUNTIME PANEL SUMMARY ===", flush=True)
    print(
        json.dumps(
            result["runtime"]["panel_all_dataset_seed_units"],
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    print("=== PREDICTIVE DESCRIPTIVE SUMMARY ===", flush=True)
    print(
        json.dumps(
            result["predictive_descriptive_on_runtime_seeds"]["dataset_level"],
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    print(f"WROTE {args.output}", flush=True)


if __name__ == "__main__":
    main()
