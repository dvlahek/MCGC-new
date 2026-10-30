from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split

from mcgc import MultiAgentChemotacticGraphClassifier, reliability_features


HERE = Path(__file__).resolve().parent
V32C_SCRIPT = HERE / "benchmark_real_data_confirmation_v32c.py"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


v32c = _load_module(V32C_SCRIPT, "benchmark_real_data_confirmation_v32c_v33")
v32 = v32c.v32
v12 = v32.v12

DATASETS = tuple(v32.DATASETS)
SEEDS = tuple(v32.SEEDS)
BASELINES = tuple(v32.BASELINES)

K_GRID = (8, 12, 17, 25)
ALPHA_GRID = (0.50, 0.75, 0.85)
ND_GRID = (20, 40, 80)
CGEC_GRID = tuple(
    {
        "n_neighbors": int(k),
        "diffusion_alpha": float(alpha),
        "diffusion_steps": int(nd),
    }
    for k, alpha, nd in itertools.product(K_GRID, ALPHA_GRID, ND_GRID)
)

EPS = 1e-12


def sign_test_p(wins: int, losses: int):
    n = int(wins + losses)
    if n == 0:
        return None
    k = min(int(wins), int(losses))
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


def candidate_key(params: dict) -> str:
    return (
        f"K={int(params['n_neighbors']):02d}|"
        f"alpha={float(params['diffusion_alpha']):.2f}|"
        f"nD={int(params['diffusion_steps']):03d}"
    )


def classifier_for(params: dict):
    return MultiAgentChemotacticGraphClassifier(
        n_neighbors=int(params["n_neighbors"]),
        diffusion_alpha=float(params["diffusion_alpha"]),
        diffusion_steps=int(params["diffusion_steps"]),
        random_state=17,
        standardize=True,
        use_walkers=False,
    )


def prediction_and_risk(model, X):
    pred, rows = reliability_features(model, X)
    risk = np.asarray(
        [row["risk_scores"]["local_margin_risk"] for row in rows],
        dtype=float,
    )
    return np.asarray(pred), risk


def evaluate_candidate_oof(X_train, y_train, seed: int, params: dict):
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)

    cv = StratifiedKFold(
        n_splits=v12.INNER_FOLDS,
        shuffle=True,
        random_state=6000 + int(seed),
    )

    oof_pred = np.empty(len(y_train), dtype=y_train.dtype)
    oof_risk = np.empty(len(y_train), dtype=float)
    seen = np.zeros(len(y_train), dtype=bool)

    for fit_idx, val_idx in cv.split(X_train, y_train):
        model = classifier_for(params)
        model.fit(X_train[fit_idx], y_train[fit_idx])
        pred, risk = prediction_and_risk(model, X_train[val_idx])
        oof_pred[val_idx] = pred
        oof_risk[val_idx] = risk
        seen[val_idx] = True

    if not np.all(seen):
        raise RuntimeError("inner OOF candidate evaluation left unseen rows")

    metrics = v12.model_metrics(y_train, oof_pred, oof_risk)
    return {
        "params": dict(params),
        "candidate_key": candidate_key(params),
        "oof_eaurc": metrics["eaurc"],
        "oof_aurc": metrics["aurc"],
        "oof_accuracy": metrics["accuracy"],
        "oof_error_detection_roc_auc": metrics["error_detection_roc_auc"],
    }


def selection_sort_key(row: dict):
    e = row["oof_eaurc"]
    a = row["oof_accuracy"]
    e_key = float("inf") if e is None else round(float(e), 12)
    a_key = float("-inf") if a is None else round(float(a), 12)
    return (e_key, -a_key, str(row["candidate_key"]))


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


def load_fixed_checkpoint(path: Path):
    fixed = read_jsonl(path, ("dataset", "seed"))
    missing = [
        (name, seed)
        for name in DATASETS
        for seed in SEEDS
        if (name, seed) not in fixed
    ]
    if missing:
        raise RuntimeError(
            "v32c fixed/baseline checkpoint is incomplete; "
            f"first missing units: {missing[:10]}"
        )
    return fixed


def tune_one_outer_unit(
    name: str,
    X_train,
    y_train,
    seed: int,
    candidate_cache: dict,
    candidate_checkpoint: Path,
):
    records = []

    for idx, params in enumerate(CGEC_GRID, start=1):
        ckey = candidate_key(params)
        cache_key = (name, int(seed), ckey)

        if cache_key in candidate_cache:
            row = candidate_cache[cache_key]["candidate"]
            print(
                f"      candidate {idx:02d}/{len(CGEC_GRID)} {ckey} SKIP checkpoint "
                f"E={row['oof_eaurc']:.6f} acc={row['oof_accuracy']:.6f}",
                flush=True,
            )
        else:
            row = evaluate_candidate_oof(X_train, y_train, seed, params)
            record = {
                "dataset": name,
                "seed": int(seed),
                "candidate_key": ckey,
                "candidate": row,
            }
            append_jsonl(candidate_checkpoint, record)
            candidate_cache[cache_key] = record
            print(
                f"      candidate {idx:02d}/{len(CGEC_GRID)} {ckey} "
                f"E={row['oof_eaurc']:.6f} acc={row['oof_accuracy']:.6f}",
                flush=True,
            )

        records.append(row)

    records.sort(key=selection_sort_key)
    return records[0], records


def evaluate_outer_tuned(
    name: str,
    X,
    y,
    seed: int,
    candidate_cache: dict,
    candidate_checkpoint: Path,
):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=5000 + int(seed),
        stratify=y,
    )

    selected, all_candidates = tune_one_outer_unit(
        name,
        X_train,
        y_train,
        seed,
        candidate_cache,
        candidate_checkpoint,
    )

    final = classifier_for(selected["params"])
    final.fit(X_train, y_train)
    pred, risk = prediction_and_risk(final, X_test)
    test_metrics = v12.model_metrics(y_test, pred, risk)

    return {
        "seed": int(seed),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "selected": selected,
        "n_candidates": int(len(all_candidates)),
        "all_candidates": all_candidates,
        "tuned_cgec": test_metrics,
    }


def metric_mean(runs, method: str, metric: str):
    vals = []
    for r in runs:
        value = r["models"][method][metric]
        if value is not None:
            vals.append(float(value))
    return float(np.mean(vals)) if vals else None


def parameter_selection_summary(tuned_records):
    global_counter = Counter()
    per_dataset = defaultdict(Counter)

    for (name, seed), record in tuned_records.items():
        selected = record["run"]["selected"]["candidate_key"]
        global_counter[selected] += 1
        per_dataset[name][selected] += 1

    return {
        "global": dict(sorted(global_counter.items())),
        "per_dataset": {
            name: dict(sorted(counter.items()))
            for name, counter in sorted(per_dataset.items())
        },
    }


def assemble_combined_runs(names, seeds, fixed_records, tuned_records):
    datasets = {}

    for name in names:
        runs = []
        for seed in seeds:
            fkey = (name, int(seed))
            if fkey not in fixed_records:
                raise RuntimeError(f"missing fixed checkpoint run {fkey}")
            if fkey not in tuned_records:
                raise RuntimeError(f"missing tuned checkpoint run {fkey}")

            fixed_run = fixed_records[fkey]["run"]
            tuned_run = tuned_records[fkey]["run"]

            if int(fixed_run["n_train"]) != int(tuned_run["n_train"]):
                raise RuntimeError(f"n_train mismatch for {fkey}")
            if int(fixed_run["n_test"]) != int(tuned_run["n_test"]):
                raise RuntimeError(f"n_test mismatch for {fkey}")

            models = dict(fixed_run["models"])
            models["cgec_tuned"] = tuned_run["tuned_cgec"]

            runs.append(
                {
                    "seed": int(seed),
                    "n_train": int(fixed_run["n_train"]),
                    "n_test": int(fixed_run["n_test"]),
                    "models": models,
                    "tuned_cgec_selection": tuned_run["selected"],
                }
            )

        methods = ("cgec", "cgec_tuned") + BASELINES
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

        fixed_minus_tuned = [
            float(r["models"]["cgec"]["eaurc"])
            - float(r["models"]["cgec_tuned"]["eaurc"])
            for r in runs
        ]

        tuned_vs_baselines = {}
        for baseline in BASELINES:
            vals = [
                float(r["models"][baseline]["eaurc"])
                - float(r["models"]["cgec_tuned"]["eaurc"])
                for r in runs
            ]
            tuned_vs_baselines[baseline] = delta_summary(vals)

        datasets[name] = {
            "runs": runs,
            "mean_metrics_across_seeds": means,
            "paired_seed_level": {
                "eaurc_improvement_tuned_over_fixed": delta_summary(
                    fixed_minus_tuned
                ),
                "eaurc_advantage_tuned_over_baselines": tuned_vs_baselines,
            },
        }

    return datasets


def panel_summary(datasets, names):
    fixed_vs_tuned = []
    tuned_vs_baselines = {b: [] for b in BASELINES}
    fixed_vs_baselines = {b: [] for b in BASELINES}
    per_dataset = []

    for name in names:
        means = datasets[name]["mean_metrics_across_seeds"]
        fixed = float(means["cgec"]["eaurc"])
        tuned = float(means["cgec_tuned"]["eaurc"])

        row = {
            "dataset": name,
            "fixed_cgec_mean_eaurc": fixed,
            "tuned_cgec_mean_eaurc": tuned,
            "improvement_tuned_over_fixed": fixed - tuned,
            "baselines": {},
        }
        fixed_vs_tuned.append(fixed - tuned)

        for baseline in BASELINES:
            b = float(means[baseline]["eaurc"])
            tuned_adv = b - tuned
            fixed_adv = b - fixed
            tuned_vs_baselines[baseline].append(tuned_adv)
            fixed_vs_baselines[baseline].append(fixed_adv)
            row["baselines"][baseline] = {
                "mean_eaurc": b,
                "advantage_tuned_cgec": tuned_adv,
                "advantage_fixed_cgec": fixed_adv,
            }

        per_dataset.append(row)

    return {
        "per_dataset": per_dataset,
        "dataset_level_primary": {
            "tuned_vs_fixed": delta_summary(fixed_vs_tuned),
            "tuned_vs_baselines": {
                baseline: delta_summary(values)
                for baseline, values in tuned_vs_baselines.items()
            },
            "fixed_vs_baselines_reference": {
                baseline: delta_summary(values)
                for baseline, values in fixed_vs_baselines.items()
            },
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
        default=v32.PAPER_ROOT / "results" / "v33_tuned_cgec_candidates.jsonl",
    )
    ap.add_argument(
        "--run-checkpoint",
        type=Path,
        default=v32.PAPER_ROOT / "results" / "v33_tuned_cgec_runs.jsonl",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=v32.PAPER_ROOT / "results" / "v33_tuned_cgec_summary.json",
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
    args = ap.parse_args()

    names = DATASETS
    if args.only:
        requested = tuple(x.strip() for x in args.only.split(",") if x.strip())
        unknown = [x for x in requested if x not in DATASETS]
        if unknown:
            raise ValueError(f"not in frozen v32c panel: {unknown}")
        names = requested

    seeds = SEEDS
    if args.seeds:
        requested = tuple(int(x) for x in args.seeds.split(",") if x.strip())
        unknown = [x for x in requested if x not in SEEDS]
        if unknown:
            raise ValueError(f"not in frozen seed set: {unknown}")
        seeds = requested

    fixed_records = load_fixed_checkpoint(args.fixed_checkpoint)
    candidate_cache = read_jsonl(
        args.candidate_checkpoint,
        ("dataset", "seed", "candidate_key"),
    )
    tuned_records = read_jsonl(args.run_checkpoint, ("dataset", "seed"))

    args.cache_dir.mkdir(parents=True, exist_ok=True)

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
            rkey = (name, int(seed))
            if rkey in tuned_records:
                selected = tuned_records[rkey]["run"]["selected"]
                test = tuned_records[rkey]["run"]["tuned_cgec"]
                print(
                    f"  seed={seed} SKIP run checkpoint "
                    f"{selected['candidate_key']} "
                    f"E-AURC={test['eaurc']:.6f}",
                    flush=True,
                )
                continue

            print(f"  seed={seed}", flush=True)
            run = evaluate_outer_tuned(
                name,
                X,
                y,
                seed,
                candidate_cache,
                args.candidate_checkpoint,
            )

            record = {
                "dataset": name,
                "seed": int(seed),
                "run": run,
            }
            append_jsonl(args.run_checkpoint, record)
            tuned_records[rkey] = record

            fixed_e = float(fixed_records[rkey]["run"]["models"]["cgec"]["eaurc"])
            tuned_e = float(run["tuned_cgec"]["eaurc"])

            print(
                f"    SELECT {run['selected']['candidate_key']} "
                f"innerE={run['selected']['oof_eaurc']:.6f} "
                f"testE={tuned_e:.6f} "
                f"fixedE={fixed_e:.6f} "
                f"delta={fixed_e - tuned_e:+.6f}",
                flush=True,
            )

    subset_tuned = {
        (name, int(seed)): tuned_records[(name, int(seed))]
        for name in names
        for seed in seeds
        if (name, int(seed)) in tuned_records
    }

    missing = [
        (name, int(seed))
        for name in names
        for seed in seeds
        if (name, int(seed)) not in subset_tuned
    ]
    if missing:
        raise RuntimeError(f"missing tuned runs: {missing[:10]}")

    datasets = assemble_combined_runs(names, seeds, fixed_records, subset_tuned)

    complete_panel = tuple(names) == tuple(DATASETS) and tuple(seeds) == tuple(SEEDS)

    result = {
        "protocol": {
            "version": "v3.3",
            "name": "secondary matched-tuning CGEC audit",
            "status": (
                "complete eight-dataset matched-tuning audit"
                if complete_panel
                else "smoke/debug subset; not manuscript evidence"
            ),
            "primary_result_remains_fixed_cgec": True,
            "dataset_panel": list(names),
            "full_dataset_panel": list(DATASETS),
            "seeds": list(seeds),
            "full_seed_set": list(SEEDS),
            "outer_split": "70/30 stratified, random_state=5000+seed",
            "inner_cv": "3-fold stratified on outer training only, random_state=6000+seed",
            "selection_metric": (
                "lowest inner OOF E-AURC; higher OOF accuracy then lexical candidate key tie-break"
            ),
            "grid": {
                "K": list(K_GRID),
                "alpha": list(ALPHA_GRID),
                "n_D": list(ND_GRID),
                "n_candidates": int(len(CGEC_GRID)),
            },
            "fixed_reference": "K=12, alpha=0.85, n_D=40",
            "fixed_checkpoint": str(args.fixed_checkpoint),
            "candidate_checkpoint": str(args.candidate_checkpoint),
            "run_checkpoint": str(args.run_checkpoint),
            "pmlb_loader": "inherited from v32c pinned-loader protocol",
        },
        "dataset_metadata": metadata,
        "datasets": datasets,
        "panel_summary": panel_summary(datasets, names),
        "parameter_selection": parameter_selection_summary(subset_tuned),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print("\n=== DATASET-LEVEL PRIMARY SUMMARY ===", flush=True)
    primary = result["panel_summary"]["dataset_level_primary"]
    print(
        "tuned_vs_fixed",
        json.dumps(primary["tuned_vs_fixed"], sort_keys=True),
        flush=True,
    )
    for baseline, summary in primary["tuned_vs_baselines"].items():
        print(
            f"tuned_vs_{baseline}",
            json.dumps(summary, sort_keys=True),
            flush=True,
        )

    print("\n=== PARAMETER SELECTION FREQUENCIES ===", flush=True)
    print(
        json.dumps(result["parameter_selection"]["global"], sort_keys=True),
        flush=True,
    )
    print(f"\nWROTE {args.output}", flush=True)


if __name__ == "__main__":
    main()
