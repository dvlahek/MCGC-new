from __future__ import annotations

# Force single-thread numerical kernels before importing NumPy/scikit-learn.
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
V33_SCRIPT = HERE / "benchmark_real_data_tuned_cgec_v33.py"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


v33 = _load_module(V33_SCRIPT, "benchmark_real_data_tuned_cgec_v33_v34")
v32 = v33.v32
v26 = v32.v26
v12 = v33.v12

DATASETS = tuple(v33.DATASETS)
SEEDS = tuple(v33.SEEDS)
BASELINES = tuple(v33.BASELINES)
CGEC_GRID = tuple(v33.CGEC_GRID)
BASELINE_SPECS = tuple(v32.BASELINE_SPECS)

FIXED_CGEC_PARAMS = {
    "n_neighbors": 12,
    "diffusion_alpha": 0.85,
    "diffusion_steps": 40,
}


def timing_summary(values):
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return None
    return {
        "n": int(len(arr)),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def metric_match(a, b, atol=1e-10):
    if a is None or b is None:
        return a is None and b is None
    return abs(float(a) - float(b)) <= atol


def validate_metrics(label: str, observed: dict, expected: dict):
    for metric in ("accuracy", "aurc", "eaurc", "error_detection_roc_auc"):
        if not metric_match(observed.get(metric), expected.get(metric)):
            raise RuntimeError(
                f"metric mismatch for {label} {metric}: "
                f"observed={observed.get(metric)} expected={expected.get(metric)}"
            )


def append_component(path: Path, record: dict):
    v33.append_jsonl(path, record)


def read_components(path: Path):
    return v33.read_jsonl(path, ("dataset", "seed", "component"))


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
                f"SKIP timing checkpoint E={row['oof_eaurc']:.6f} "
                f"t={elapsed:.3f}s",
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
            v33.append_jsonl(candidate_checkpoint, rec)
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
    total_candidate_seconds = float(sum(elapsed for _, elapsed in rows))
    return selected, rows, total_candidate_seconds


def time_cgec_final_model(params, X_train, y_train, X_test, y_test, repeats: int):
    fit_times = []
    pred_times = []
    total_times = []
    primary_metrics = None

    for rep in range(repeats):
        model = v33.classifier_for(params)

        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        fit_s = time.perf_counter() - t0

        t1 = time.perf_counter()
        pred, risk = v33.prediction_and_risk(model, X_test)
        pred_s = time.perf_counter() - t1

        fit_times.append(float(fit_s))
        pred_times.append(float(pred_s))
        total_times.append(float(fit_s + pred_s))

        if rep == 0:
            primary_metrics = v12.model_metrics(y_test, pred, risk)

    assert primary_metrics is not None
    return {
        "metrics": primary_metrics,
        "fit_seconds": timing_summary(fit_times),
        "prediction_seconds": timing_summary(pred_times),
        "fit_plus_prediction_seconds": timing_summary(total_times),
        "all_repetitions": [
            {
                "repeat": int(i),
                "fit_seconds": fit_times[i],
                "prediction_seconds": pred_times[i],
                "fit_plus_prediction_seconds": total_times[i],
            }
            for i in range(repeats)
        ],
        "primary_fit_seconds": fit_times[0],
        "primary_prediction_seconds": pred_times[0],
    }


def time_baseline_final_model(
    kind: str,
    params: dict,
    X_train,
    y_train,
    X_test,
    y_test,
    repeats: int,
):
    fit_times = []
    pred_times = []
    total_times = []
    primary_metrics = None

    for rep in range(repeats):
        model = v26.classifier_for(kind, params)

        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        fit_s = time.perf_counter() - t0

        t1 = time.perf_counter()
        pred, risk = v26.prediction_and_risk(kind, model, X_test)
        pred_s = time.perf_counter() - t1

        fit_times.append(float(fit_s))
        pred_times.append(float(pred_s))
        total_times.append(float(fit_s + pred_s))

        if rep == 0:
            primary_metrics = v12.model_metrics(y_test, pred, risk)

    assert primary_metrics is not None
    return {
        "metrics": primary_metrics,
        "fit_seconds": timing_summary(fit_times),
        "prediction_seconds": timing_summary(pred_times),
        "fit_plus_prediction_seconds": timing_summary(total_times),
        "all_repetitions": [
            {
                "repeat": int(i),
                "fit_seconds": fit_times[i],
                "prediction_seconds": pred_times[i],
                "fit_plus_prediction_seconds": total_times[i],
            }
            for i in range(repeats)
        ],
    }


def ensure_fixed_cgec_component(
    name,
    seed,
    X_train,
    y_train,
    X_test,
    y_test,
    fixed_run,
    repeats,
    component_cache,
    component_checkpoint,
):
    key = (name, int(seed), "fixed_cgec")
    if key in component_cache:
        print("    fixed CGEC SKIP component checkpoint", flush=True)
        return component_cache[key]["payload"]

    timed = time_cgec_final_model(
        FIXED_CGEC_PARAMS,
        X_train,
        y_train,
        X_test,
        y_test,
        repeats,
    )
    validate_metrics(
        f"{name}/seed={seed}/fixed_cgec",
        timed["metrics"],
        fixed_run["models"]["cgec"],
    )
    payload = {
        "params": dict(FIXED_CGEC_PARAMS),
        "final_model_timing": timed,
    }
    record = {
        "dataset": name,
        "seed": int(seed),
        "component": "fixed_cgec",
        "payload": payload,
    }
    append_component(component_checkpoint, record)
    component_cache[key] = record
    print(
        "    fixed CGEC "
        f"median fit+pred={timed['fit_plus_prediction_seconds']['median']:.4f}s",
        flush=True,
    )
    return payload


def ensure_tuned_cgec_component(
    name,
    seed,
    X_train,
    y_train,
    X_test,
    y_test,
    fixed_run,
    repeats,
    candidate_cache,
    candidate_checkpoint,
    component_cache,
    component_checkpoint,
):
    key = (name, int(seed), "tuned_cgec")
    if key in component_cache:
        print("    tuned CGEC SKIP component checkpoint", flush=True)
        return component_cache[key]["payload"]

    selected, candidate_rows, candidate_seconds = candidate_timing_records(
        name,
        X_train,
        y_train,
        seed,
        candidate_cache,
        candidate_checkpoint,
    )

    final_timing = time_cgec_final_model(
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

    payload = {
        "selected": selected,
        "n_candidates": int(len(candidate_rows)),
        "candidate_evaluation_seconds_total": float(candidate_seconds),
        "candidate_evaluation_seconds": timing_summary(
            [elapsed for _, elapsed in candidate_rows]
        ),
        "final_model_timing": final_timing,
        "end_to_end_tuned_seconds": float(e2e),
        "fixed_reference_eaurc": float(fixed_run["models"]["cgec"]["eaurc"]),
    }
    record = {
        "dataset": name,
        "seed": int(seed),
        "component": "tuned_cgec",
        "payload": payload,
    }
    append_component(component_checkpoint, record)
    component_cache[key] = record

    print(
        f"    tuned CGEC SELECT {selected['candidate_key']} "
        f"outer E={final_timing['metrics']['eaurc']:.6f} "
        f"tune={candidate_seconds:.2f}s "
        f"e2e={e2e:.2f}s",
        flush=True,
    )
    return payload


def ensure_baseline_component(
    name,
    seed,
    output_name,
    kind,
    grid,
    X_train,
    y_train,
    X_test,
    y_test,
    fixed_run,
    repeats,
    component_cache,
    component_checkpoint,
):
    component = f"baseline::{output_name}"
    key = (name, int(seed), component)
    if key in component_cache:
        print(f"    {output_name} SKIP component checkpoint", flush=True)
        return component_cache[key]["payload"]

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

    expected_tuning = fixed_run["tuning"][output_name]["selected"]
    selected_match = (
        str(selected["candidate_key"]) == str(expected_tuning["candidate_key"])
    )
    if not selected_match:
        raise RuntimeError(
            f"baseline selection mismatch for {name}/seed={seed}/{output_name}: "
            f"timed={selected['candidate_key']} "
            f"checkpoint={expected_tuning['candidate_key']}"
        )

    validate_metrics(
        f"{name}/seed={seed}/{output_name}",
        metrics,
        fixed_run["models"][output_name],
    )

    selected_final = time_baseline_final_model(
        kind,
        selected["params"],
        X_train,
        y_train,
        X_test,
        y_test,
        repeats,
    )
    validate_metrics(
        f"{name}/seed={seed}/{output_name}/refit",
        selected_final["metrics"],
        fixed_run["models"][output_name],
    )

    payload = {
        "kind": kind,
        "selected": selected,
        "n_candidates": int(len(all_records)),
        "selection_matches_v32c": bool(selected_match),
        "tuning_routine_seconds": float(tune_seconds),
        "outer_prediction_seconds": float(prediction_seconds),
        "end_to_end_tuned_seconds": float(tune_seconds + prediction_seconds),
        "metrics": metrics,
        "selected_final_model_timing": selected_final,
    }
    record = {
        "dataset": name,
        "seed": int(seed),
        "component": component,
        "payload": payload,
    }
    append_component(component_checkpoint, record)
    component_cache[key] = record

    print(
        f"    {output_name} e2e={payload['end_to_end_tuned_seconds']:.2f}s "
        f"final median={selected_final['fit_plus_prediction_seconds']['median']:.4f}s",
        flush=True,
    )
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
        "timer": "time.perf_counter wall-clock seconds",
    }


def build_summary(names, seeds, fixed_records, component_cache):
    method_keys = [
        "fixed_cgec",
        "tuned_cgec",
        *[f"baseline::{name}" for name in BASELINES],
    ]

    dataset_runtime = {}
    panel_runtime_values = {key: [] for key in method_keys}
    final_model_values = {key: [] for key in method_keys}
    tuned_cgec_tuning_values = []

    parameter_counter = Counter()
    predictive_rows = []

    for name in names:
        ds_values = {key: [] for key in method_keys}
        ds_final_values = {key: [] for key in method_keys}
        tuned_eaurc = []

        for seed in seeds:
            fixed_key = (name, int(seed))
            fixed_run = fixed_records[fixed_key]["run"]

            fixed_payload = component_cache[(name, int(seed), "fixed_cgec")]["payload"]
            tuned_payload = component_cache[(name, int(seed), "tuned_cgec")]["payload"]

            fixed_final = float(
                fixed_payload["final_model_timing"]["fit_plus_prediction_seconds"]["median"]
            )
            tuned_final = float(
                tuned_payload["final_model_timing"]["fit_plus_prediction_seconds"]["median"]
            )
            tuned_e2e = float(tuned_payload["end_to_end_tuned_seconds"])

            ds_values["fixed_cgec"].append(fixed_final)
            ds_values["tuned_cgec"].append(tuned_e2e)
            ds_final_values["fixed_cgec"].append(fixed_final)
            ds_final_values["tuned_cgec"].append(tuned_final)
            panel_runtime_values["fixed_cgec"].append(fixed_final)
            panel_runtime_values["tuned_cgec"].append(tuned_e2e)
            final_model_values["fixed_cgec"].append(fixed_final)
            final_model_values["tuned_cgec"].append(tuned_final)
            tuned_cgec_tuning_values.append(
                float(tuned_payload["candidate_evaluation_seconds_total"])
            )

            parameter_counter[tuned_payload["selected"]["candidate_key"]] += 1
            tuned_eaurc.append(
                float(tuned_payload["final_model_timing"]["metrics"]["eaurc"])
            )

            for baseline in BASELINES:
                comp = f"baseline::{baseline}"
                payload = component_cache[(name, int(seed), comp)]["payload"]
                e2e = float(payload["end_to_end_tuned_seconds"])
                final_med = float(
                    payload["selected_final_model_timing"]["fit_plus_prediction_seconds"]["median"]
                )
                ds_values[comp].append(e2e)
                ds_final_values[comp].append(final_med)
                panel_runtime_values[comp].append(e2e)
                final_model_values[comp].append(final_med)

        dataset_runtime[name] = {
            "end_to_end_or_fixed_final_model": {
                key: timing_summary(vals)
                for key, vals in ds_values.items()
            },
            "selected_final_model_fit_plus_prediction": {
                key: timing_summary(vals)
                for key, vals in ds_final_values.items()
            },
        }

        fixed_mean = float(
            np.mean([
                fixed_records[(name, int(seed))]["run"]["models"]["cgec"]["eaurc"]
                for seed in seeds
            ])
        )
        tuned_mean = float(np.mean(tuned_eaurc))
        baseline_means = {
            baseline: float(
                np.mean([
                    fixed_records[(name, int(seed))]["run"]["models"][baseline]["eaurc"]
                    for seed in seeds
                ])
            )
            for baseline in BASELINES
        }
        predictive_rows.append({
            "dataset": name,
            "fixed_cgec_mean_eaurc": fixed_mean,
            "tuned_cgec_mean_eaurc": tuned_mean,
            "improvement_tuned_over_fixed": fixed_mean - tuned_mean,
            "baseline_mean_eaurc": baseline_means,
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
            "per_dataset": dataset_runtime,
            "panel_all_dataset_seed_units": {
                "end_to_end_or_fixed_final_model": {
                    key: timing_summary(vals)
                    for key, vals in panel_runtime_values.items()
                },
                "selected_final_model_fit_plus_prediction": {
                    key: timing_summary(vals)
                    for key, vals in final_model_values.items()
                },
                "tuned_cgec_candidate_evaluation_total": timing_summary(
                    tuned_cgec_tuning_values
                ),
            },
            "interpretation": {
                "fixed_cgec": "fit+test prediction only; no tuning",
                "tuned_cgec": "36-candidate inner-CV tuning + one final refit + one test prediction",
                "baselines": "their existing inner-CV tuning routine including final refit + one test prediction",
                "selected_final_model": "median of repeated fresh final-model fit+prediction timings; excludes tuning",
            },
        },
        "predictive_secondary_matched_tuning": {
            "per_dataset": predictive_rows,
            "dataset_level_primary": {
                "tuned_vs_fixed": v33.delta_summary(tuned_vs_fixed),
                "tuned_vs_baselines": {
                    baseline: v33.delta_summary(values)
                    for baseline, values in tuned_vs_baselines.items()
                },
            },
            "parameter_selection_frequencies": dict(sorted(parameter_counter.items())),
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
        "--fixed-checkpoint",
        type=Path,
        default=v32.PAPER_ROOT / "results" / "v32c_real_data_confirmation_runs.jsonl",
    )
    ap.add_argument(
        "--candidate-checkpoint",
        type=Path,
        default=v32.PAPER_ROOT / "results" / "v34_timed_cgec_candidates.jsonl",
    )
    ap.add_argument(
        "--component-checkpoint",
        type=Path,
        default=v32.PAPER_ROOT / "results" / "v34_runtime_components.jsonl",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=v32.PAPER_ROOT / "results" / "v34_runtime_matched_tuning_summary.json",
    )
    ap.add_argument(
        "--only",
        default=None,
        help="comma-separated dataset subset for smoke/debug only",
    )
    ap.add_argument(
        "--seeds",
        default=None,
        help="comma-separated seed subset for smoke/debug only",
    )
    ap.add_argument(
        "--timing-repeats",
        type=int,
        default=3,
        help="fresh final-model fit/prediction repetitions; default 3",
    )
    args = ap.parse_args()

    if args.timing_repeats < 1:
        raise ValueError("--timing-repeats must be >= 1")

    names = DATASETS
    if args.only:
        requested = tuple(x.strip() for x in args.only.split(",") if x.strip())
        unknown = [x for x in requested if x not in DATASETS]
        if unknown:
            raise ValueError(f"not in v3.4 dataset panel: {unknown}")
        names = requested

    seeds = SEEDS
    if args.seeds:
        requested = tuple(int(x) for x in args.seeds.split(",") if x.strip())
        unknown = [x for x in requested if x not in SEEDS]
        if unknown:
            raise ValueError(f"not in v3.4 seed set: {unknown}")
        seeds = requested

    fixed_records = v33.load_fixed_checkpoint(args.fixed_checkpoint)
    candidate_cache = v33.read_jsonl(
        args.candidate_checkpoint,
        ("dataset", "seed", "candidate_key"),
    )
    component_cache = read_components(args.component_checkpoint)

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
            fixed_run = fixed_records[(name, int(seed))]["run"]

            ensure_fixed_cgec_component(
                name, seed,
                X_train, y_train, X_test, y_test,
                fixed_run,
                args.timing_repeats,
                component_cache,
                args.component_checkpoint,
            )

            ensure_tuned_cgec_component(
                name, seed,
                X_train, y_train, X_test, y_test,
                fixed_run,
                args.timing_repeats,
                candidate_cache,
                args.candidate_checkpoint,
                component_cache,
                args.component_checkpoint,
            )

            for output_name, kind, grid in BASELINE_SPECS:
                ensure_baseline_component(
                    name, seed,
                    output_name, kind, grid,
                    X_train, y_train, X_test, y_test,
                    fixed_run,
                    args.timing_repeats,
                    component_cache,
                    args.component_checkpoint,
                )

    complete_panel = tuple(names) == DATASETS and tuple(seeds) == SEEDS
    summary = build_summary(names, seeds, fixed_records, component_cache)

    result = {
        "protocol": {
            "version": "v3.4",
            "status": (
                "complete runtime + matched-tuning audit"
                if complete_panel
                else "smoke/debug subset; not manuscript evidence"
            ),
            "primary_fixed_cgec_preserved": True,
            "secondary_analysis": "matched tuning plus descriptive runtime",
            "datasets": list(names),
            "full_dataset_panel": list(DATASETS),
            "seeds": list(seeds),
            "full_seed_set": list(SEEDS),
            "outer_split": "70/30 stratified, random_state=5000+seed",
            "inner_cv": "3-fold stratified on outer training only, random_state=6000+seed",
            "cgec_grid": list(CGEC_GRID),
            "selection_metric": "lowest inner OOF E-AURC; higher accuracy then lexical candidate-key tie-break",
            "timing_repeats_final_model": int(args.timing_repeats),
            "timing_scope": "runtime on tested implementation/hardware; not a scaling-complexity experiment",
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

    print("=== V3.4 DATASET-LEVEL PREDICTIVE SUMMARY ===", flush=True)
    primary = result["predictive_secondary_matched_tuning"]["dataset_level_primary"]
    print("tuned_vs_fixed", json.dumps(primary["tuned_vs_fixed"], sort_keys=True), flush=True)
    for baseline, block in primary["tuned_vs_baselines"].items():
        print(f"tuned_vs_{baseline}", json.dumps(block, sort_keys=True), flush=True)

    print("=== V3.4 PANEL RUNTIME SUMMARY ===", flush=True)
    rt = result["runtime"]["panel_all_dataset_seed_units"]
    for method, block in rt["end_to_end_or_fixed_final_model"].items():
        print(f"runtime {method}", json.dumps(block, sort_keys=True), flush=True)
    print("=== SELECTED FINAL-MODEL COST ===", flush=True)
    for method, block in rt["selected_final_model_fit_plus_prediction"].items():
        print(f"final_model {method}", json.dumps(block, sort_keys=True), flush=True)

    print("=== PARAMETER SELECTION FREQUENCIES ===", flush=True)
    print(
        json.dumps(
            result["predictive_secondary_matched_tuning"]["parameter_selection_frequencies"],
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    print(f"WROTE {args.output}", flush=True)


if __name__ == "__main__":
    main()
