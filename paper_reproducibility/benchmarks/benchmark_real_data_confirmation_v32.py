from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from mcgc import MultiAgentChemotacticGraphClassifier, reliability_features


PAPER_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]
V26_SCRIPT = REPO_ROOT / "examples" / "benchmark_corrected_predictive_v26.py"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


v26 = _load_module(V26_SCRIPT, "benchmark_corrected_predictive_v26")
v12 = v26.v12

# Frozen before result inspection. Do not add/remove datasets after looking at results.
DATASETS = (
    "credit_approval_australia",
    "haberman",
    "heart_disease_cleveland",
    "horse_colic_outcome",
    "ionosphere",
    "penguins",
    "segmentation",
    "sonar",
)

SEEDS = tuple(range(12))
BASELINES = tuple(v26.BASELINES)

BASELINE_SPECS = (
    ("scaled_tuned_knn_confidence", "scaled_knn", v26.SCALED_KNN_GRID),
    ("tuned_rbf_svm_confidence", "svm", v26.SVM_GRID),
    ("tuned_random_forest_confidence", "rf", v26.RF_GRID),
    (
        "scaled_tuned_label_spreading_confidence",
        "label_spreading",
        v26.LS_GRID,
    ),
)

EPS = 1e-12


def sign_test_p(wins: int, losses: int):
    n = wins + losses
    if n == 0:
        return None
    k = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return float(min(1.0, 2.0 * tail))


def delta_summary(values):
    arr = np.asarray(values, dtype=float)
    wins = int(np.sum(arr > EPS))
    ties = int(np.sum(np.abs(arr) <= EPS))
    losses = int(np.sum(arr < -EPS))
    return {
        "n": int(len(arr)),
        "mean": float(np.mean(arr)) if len(arr) else None,
        "median": float(np.median(arr)) if len(arr) else None,
        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "min": float(np.min(arr)) if len(arr) else None,
        "max": float(np.max(arr)) if len(arr) else None,
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "sign_test_p_two_sided": sign_test_p(wins, losses),
    }


def field_classifier():
    return MultiAgentChemotacticGraphClassifier(
        n_neighbors=12,
        diffusion_alpha=0.85,
        diffusion_steps=40,
        random_state=17,
        standardize=True,
        use_walkers=False,
    )


def dataset_fingerprint(X: np.ndarray, y: np.ndarray):
    Xc = np.ascontiguousarray(X, dtype=np.float64)
    yc = np.ascontiguousarray(y, dtype=np.int64)
    h = hashlib.sha256()
    h.update(str(Xc.shape).encode("utf-8"))
    h.update(str(yc.shape).encode("utf-8"))
    h.update(Xc.tobytes(order="C"))
    h.update(yc.tobytes(order="C"))
    return h.hexdigest()


def load_pmlb_dataset(name: str, cache_dir: Path):
    try:
        from pmlb import fetch_data
    except ImportError as exc:
        raise RuntimeError(
            "pmlb is required. Install it with: python -m pip install pmlb"
        ) from exc

    X_raw, y_raw = fetch_data(
        name,
        return_X_y=True,
        local_cache_dir=str(cache_dir),
    )

    X = np.asarray(X_raw, dtype=float)
    y_raw = np.asarray(y_raw)

    if X.ndim != 2 or len(X) != len(y_raw):
        raise RuntimeError(f"invalid X/y shape for {name}")
    if not np.all(np.isfinite(X)):
        raise RuntimeError(f"non-finite feature value in frozen dataset {name}")

    enc = LabelEncoder()
    y = enc.fit_transform(y_raw)

    values, counts = np.unique(y, return_counts=True)
    if len(values) < 2:
        raise RuntimeError(f"dataset {name} has fewer than two classes")
    if int(np.min(counts)) < 20:
        raise RuntimeError(
            f"dataset {name} violates frozen minimum class size 20: {counts.tolist()}"
        )

    meta = {
        "n_samples": int(len(X)),
        "n_features": int(X.shape[1]),
        "n_classes": int(len(values)),
        "class_counts": [int(v) for v in counts],
        "original_classes": [str(v) for v in enc.classes_],
        "content_sha256": dataset_fingerprint(X, y),
    }
    return X, y, meta


def field_metrics(X_train, X_test, y_train, y_test):
    model = field_classifier().fit(X_train, y_train)
    pred, rows = reliability_features(model, X_test)
    risk = np.asarray(
        [row["risk_scores"]["local_margin_risk"] for row in rows],
        dtype=float,
    )
    return v12.model_metrics(y_test, pred, risk)


def baseline_metrics(X_train, X_test, y_train, y_test, seed: int):
    models = {}
    tuning = {}

    for output_name, kind, grid in BASELINE_SPECS:
        fitted, selected, all_records = v26.tune(
            kind,
            grid,
            X_train,
            y_train,
            seed,
        )
        pred, risk = v26.prediction_and_risk(kind, fitted, X_test)
        models[output_name] = v12.model_metrics(y_test, pred, risk)
        tuning[output_name] = {
            "selected": selected,
            "n_candidates": int(len(all_records)),
            "all_candidates": all_records,
        }

    return models, tuning


def evaluate_seed(X, y, seed: int):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=5000 + seed,
        stratify=y,
    )

    models = {
        "cgec": field_metrics(X_train, X_test, y_train, y_test),
    }
    baseline, tuning = baseline_metrics(
        X_train,
        X_test,
        y_train,
        y_test,
        seed,
    )
    models.update(baseline)

    return {
        "seed": int(seed),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "models": models,
        "tuning": tuning,
    }


def metric_mean(runs, method: str, metric: str):
    vals = [
        r["models"][method][metric]
        for r in runs
        if r["models"][method][metric] is not None
    ]
    return float(np.mean(vals)) if vals else None


def summarize_dataset(runs):
    methods = ("cgec",) + BASELINES
    metrics = (
        "accuracy",
        "aurc",
        "eaurc",
        "error_detection_roc_auc",
    )

    means = {
        method: {
            metric: metric_mean(runs, method, metric)
            for metric in metrics
        }
        for method in methods
    }

    paired = {}
    for baseline in BASELINES:
        e_adv = [
            float(r["models"][baseline]["eaurc"])
            - float(r["models"]["cgec"]["eaurc"])
            for r in runs
        ]
        a_adv = [
            float(r["models"][baseline]["aurc"])
            - float(r["models"]["cgec"]["aurc"])
            for r in runs
        ]
        acc_adv = [
            float(r["models"]["cgec"]["accuracy"])
            - float(r["models"][baseline]["accuracy"])
            for r in runs
        ]

        auc_adv = []
        for r in runs:
            c = r["models"]["cgec"]["error_detection_roc_auc"]
            b = r["models"][baseline]["error_detection_roc_auc"]
            if c is not None and b is not None:
                auc_adv.append(float(c) - float(b))

        paired[baseline] = {
            "eaurc_advantage_cgec_over_baseline": delta_summary(e_adv),
            "aurc_advantage_cgec_over_baseline": delta_summary(a_adv),
            "accuracy_advantage_cgec_over_baseline": delta_summary(acc_adv),
            "error_detection_auc_advantage_cgec_over_baseline": (
                delta_summary(auc_adv) if auc_adv else None
            ),
        }

    return {
        "mean_metrics_across_12_seeds": means,
        "paired_seed_level": paired,
    }


def summarize_panel(datasets):
    dataset_units = {}
    seed_units = {}

    for baseline in BASELINES:
        dvals = []
        svals = []
        per_dataset = []

        for name in DATASETS:
            runs = datasets[name]["runs"]
            seed_adv = np.asarray(
                [
                    float(r["models"][baseline]["eaurc"])
                    - float(r["models"]["cgec"]["eaurc"])
                    for r in runs
                ],
                dtype=float,
            )
            dmean = float(np.mean(seed_adv))
            dvals.append(dmean)
            svals.extend(seed_adv.tolist())
            per_dataset.append(
                {
                    "dataset": name,
                    "mean_eaurc_advantage_cgec_over_baseline": dmean,
                }
            )

        dataset_units[baseline] = {
            "per_dataset": per_dataset,
            "summary": delta_summary(dvals),
            "interpretation": (
                "primary confirmation unit; each value is the mean over 12 seeds "
                "for one real dataset"
            ),
        }
        seed_units[baseline] = {
            "summary": delta_summary(svals),
            "interpretation": (
                "descriptive only; 96 dataset-seed runs are not treated as "
                "independent problem domains"
            ),
        }

    return {
        "dataset_level_primary": dataset_units,
        "seed_level_descriptive": seed_units,
    }


def read_checkpoint(path: Path):
    records = {}
    if not path.exists():
        return records
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        records[(row["dataset"], int(row["seed"]))] = row
    return records


def append_checkpoint(path: Path, record: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--cache-dir",
        type=Path,
        default=PAPER_ROOT / "source_data" / "pmlb_real_confirmation_v32",
    )
    ap.add_argument(
        "--checkpoint",
        type=Path,
        default=PAPER_ROOT / "results" / "v32_real_data_confirmation_runs.jsonl",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=PAPER_ROOT / "results" / "v32_real_data_confirmation.json",
    )
    ap.add_argument(
        "--only",
        default=None,
        help="comma-separated frozen dataset subset for smoke/debug only",
    )
    ap.add_argument(
        "--seeds",
        default=None,
        help="comma-separated seed subset for smoke/debug only",
    )
    args = ap.parse_args()

    names = DATASETS
    if args.only:
        requested = tuple(x.strip() for x in args.only.split(",") if x.strip())
        unknown = [x for x in requested if x not in DATASETS]
        if unknown:
            raise ValueError(f"not in frozen dataset panel: {unknown}")
        names = requested

    seeds = SEEDS
    if args.seeds:
        requested = tuple(int(x) for x in args.seeds.split(",") if x.strip())
        unknown = [x for x in requested if x not in SEEDS]
        if unknown:
            raise ValueError(f"not in frozen seed set: {unknown}")
        seeds = requested

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    existing = read_checkpoint(args.checkpoint)

    metadata = {}
    loaded = {}
    for name in names:
        print(f"LOAD {name}", flush=True)
        X, y, meta = load_pmlb_dataset(name, args.cache_dir)
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
            key = (name, int(seed))
            if key in existing:
                print(f"  seed={seed} SKIP checkpoint", flush=True)
                continue

            print(f"  seed={seed}", flush=True)
            run = evaluate_seed(X, y, seed)
            record = {
                "dataset": name,
                "seed": int(seed),
                "run": run,
            }
            append_checkpoint(args.checkpoint, record)
            existing[key] = record

            c = run["models"]["cgec"]
            print(
                f"    CGEC acc={c['accuracy']:.6f} "
                f"E-AURC={c['eaurc']:.6f}",
                flush=True,
            )

    datasets = {}
    for name in names:
        runs = []
        for seed in seeds:
            key = (name, int(seed))
            if key not in existing:
                raise RuntimeError(f"missing run {key}")
            runs.append(existing[key]["run"])
        datasets[name] = {
            "metadata": metadata[name],
            "runs": runs,
            "summary": summarize_dataset(runs),
        }

    complete_frozen_panel = (
        tuple(names) == tuple(DATASETS)
        and tuple(seeds) == tuple(SEEDS)
    )

    result = {
        "protocol": {
            "version": "v3.2",
            "name": "frozen real-data predictive confirmation",
            "status": (
                "complete frozen confirmation"
                if complete_frozen_panel
                else "smoke/debug subset; not manuscript evidence"
            ),
            "dataset_selection_frozen_before_results": True,
            "datasets": list(names),
            "full_frozen_dataset_panel": list(DATASETS),
            "seeds": list(seeds),
            "full_frozen_seed_set": list(SEEDS),
            "cgec": "K=12, alpha=0.85, nD=40, use_walkers=False",
            "outer_split": "70/30 stratified, random_state=5000+seed",
            "baseline_inner_cv": (
                "3-fold stratified on outer training only, "
                "random_state=6000+seed"
            ),
            "baseline_selection": (
                "lowest OOF E-AURC; higher accuracy then lexical tie-break"
            ),
            "primary_metric": "E-AURC; lower is better",
            "primary_confirmation_unit": "dataset mean across 12 seeds",
            "selection_from_results": False,
        },
        "datasets": datasets,
        "panel_summary": summarize_panel(datasets),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print("\n=== v3.2 REAL-DATA CONFIRMATION ===", flush=True)
    print(f"STATUS={result['protocol']['status']}", flush=True)

    for baseline in BASELINES:
        s = result["panel_summary"]["dataset_level_primary"][baseline]["summary"]
        print(
            f"{baseline:42s} "
            f"meanEadv={s['mean']:+.8f} "
            f"dataset W/T/L={s['wins']}/{s['ties']}/{s['losses']} "
            f"p={s['sign_test_p_two_sided']}",
            flush=True,
        )

    print(f"FULL_JSON={args.output.resolve()}", flush=True)
    print(f"CHECKPOINT={args.checkpoint.resolve()}", flush=True)


if __name__ == "__main__":
    main()
