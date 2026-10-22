from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
V19 = ROOT / "examples" / "smoke_reliability_weighted_v19.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


v19 = _load(V19, "smoke_reliability_weighted_v19")
v18 = v19.v18

SEEDS = (0, 1, 2, 3, 4)
DATASETS = ("digits", "breast_cancer", "wine")
CONDITIONS = (
    ("boundary_targeted", 0.05),
    ("boundary_targeted", 0.10),
    ("symmetric", 0.30),
)
BUDGETS = (0.01, 0.02, 0.05, 0.10)


def metrics(y_true, pred):
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
    }


def audit_scores(X_train, y_train_noisy, selected_params, seed):
    _, raw, _ = v19.oof_trust_and_consensus(
        X_train,
        y_train_noisy,
        selected_params,
        seed,
    )
    return -np.asarray(raw, dtype=float)  # larger = more suspicious


def review_indices(scores, budget):
    n = len(scores)
    k = max(1, int(round(float(budget) * n)))
    order = np.argsort(-scores, kind="stable")
    return order[:k]


def trapz_review_auc(budgets, values):
    x = np.asarray([0.0] + list(budgets), dtype=float)
    y = np.asarray([0.0] + list(values), dtype=float)
    return float(np.trapezoid(y, x) / x[-1])


def run_condition(dataset, noise_type, eta, seed, with_repair):
    X, y = v19.dataset_for(dataset)

    split_seed = 5000 + seed
    X_train, X_test, y_train_clean, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=split_seed,
        stratify=y,
    )

    y_train_noisy, corruption_mask, noise_seed = v19.make_noise(
        X_train,
        y_train_clean,
        noise_type,
        eta,
        dataset,
    )

    selected_alpha, alpha_records = v18.tune_model(
        "mcgc",
        v18.MCGC_GRID,
        X_train,
        y_train_noisy,
        seed,
    )
    selected_params = selected_alpha["params"]

    scores = audit_scores(
        X_train,
        y_train_noisy,
        selected_params,
        seed,
    )

    total_corrupted = int(np.sum(corruption_mask))
    empirical_noise_fraction = float(total_corrupted / len(y_train_noisy))

    curve = []
    precisions = []
    recalls = []
    enrichments = []

    for budget in BUDGETS:
        idx = review_indices(scores, budget)
        found = int(np.sum(corruption_mask[idx]))
        k = len(idx)

        precision = float(found / k)
        recall = float(found / total_corrupted) if total_corrupted else None
        enrichment = (
            float(precision / empirical_noise_fraction)
            if empirical_noise_fraction > 0
            else None
        )

        row = {
            "budget": float(budget),
            "reviewed_count": int(k),
            "corrupt_found": found,
            "precision_at_budget": precision,
            "corrupt_recall": recall,
            "random_expected_precision": empirical_noise_fraction,
            "precision_enrichment_over_random": enrichment,
        }

        if with_repair:
            repaired = np.asarray(y_train_noisy).copy()
            repaired[idx] = np.asarray(y_train_clean)[idx]

            noisy_model = v18.mcgc_for(selected_params).fit(X_train, y_train_noisy)
            repaired_model = v18.mcgc_for(selected_params).fit(X_train, repaired)

            noisy_m = metrics(y_test, noisy_model.predict(X_test))
            repaired_m = metrics(y_test, repaired_model.predict(X_test))

            row["repair"] = {
                "noisy_balanced_accuracy": noisy_m["balanced_accuracy"],
                "repaired_balanced_accuracy": repaired_m["balanced_accuracy"],
                "balanced_accuracy_gain": float(
                    repaired_m["balanced_accuracy"] - noisy_m["balanced_accuracy"]
                ),
            }

        curve.append(row)
        precisions.append(precision)
        recalls.append(float(recall))
        enrichments.append(float(enrichment))

    return {
        "dataset": dataset,
        "noise_type": noise_type,
        "eta": float(eta),
        "seed": int(seed),
        "split_seed": int(split_seed),
        "noise_seed": int(noise_seed),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_corrupted": total_corrupted,
        "empirical_noise_fraction": empirical_noise_fraction,
        "selected_mcgc_params": dict(selected_params),
        "alpha_tuning_candidates": alpha_records,
        "review_curve": curve,
        "summary": {
            "mean_precision_over_budgets": float(np.mean(precisions)),
            "mean_recall_over_budgets": float(np.mean(recalls)),
            "mean_enrichment_over_budgets": float(np.mean(enrichments)),
            "review_auc_precision": trapz_review_auc(BUDGETS, precisions),
            "review_auc_recall": trapz_review_auc(BUDGETS, recalls),
        },
    }


def aggregate(runs):
    grouped = {}
    for r in runs:
        key = (r["dataset"], r["noise_type"], r["eta"])
        grouped.setdefault(key, []).append(r)

    out = []
    for (dataset, noise_type, eta), group in sorted(grouped.items()):
        budget_rows = []
        for budget in BUDGETS:
            rows = [
                next(x for x in r["review_curve"] if x["budget"] == budget)
                for r in group
            ]
            p = np.asarray([x["precision_at_budget"] for x in rows], dtype=float)
            rec = np.asarray([x["corrupt_recall"] for x in rows], dtype=float)
            enr = np.asarray(
                [x["precision_enrichment_over_random"] for x in rows],
                dtype=float,
            )

            budget_rows.append({
                "budget": float(budget),
                "precision_mean": float(np.mean(p)),
                "precision_sd": float(np.std(p, ddof=1)),
                "precision_min": float(np.min(p)),
                "precision_max": float(np.max(p)),
                "recall_mean": float(np.mean(rec)),
                "recall_sd": float(np.std(rec, ddof=1)),
                "enrichment_mean": float(np.mean(enr)),
                "enrichment_sd": float(np.std(enr, ddof=1)),
                "all_seeds_above_random_precision": bool(
                    all(
                        x["precision_at_budget"] > x["random_expected_precision"]
                        for x in rows
                    )
                ),
            })

        out.append({
            "dataset": dataset,
            "noise_type": noise_type,
            "eta": eta,
            "n_seeds": len(group),
            "budgets": budget_rows,
            "review_auc_precision_mean": float(
                np.mean([r["summary"]["review_auc_precision"] for r in group])
            ),
            "review_auc_precision_sd": float(
                np.std(
                    [r["summary"]["review_auc_precision"] for r in group],
                    ddof=1,
                )
            ),
            "mean_enrichment_over_budgets_mean": float(
                np.mean(
                    [r["summary"]["mean_enrichment_over_budgets"] for r in group]
                )
            ),
        })

    return out


def global_aggregate(condition_aggregates):
    by_budget = []
    for budget in BUDGETS:
        rows = [
            next(x for x in c["budgets"] if x["budget"] == budget)
            for c in condition_aggregates
        ]
        by_budget.append({
            "budget": float(budget),
            "mean_precision_across_conditions": float(
                np.mean([x["precision_mean"] for x in rows])
            ),
            "mean_recall_across_conditions": float(
                np.mean([x["recall_mean"] for x in rows])
            ),
            "mean_enrichment_across_conditions": float(
                np.mean([x["enrichment_mean"] for x in rows])
            ),
            "conditions_all_seeds_above_random": int(
                sum(x["all_seeds_above_random_precision"] for x in rows)
            ),
            "n_conditions": len(rows),
        })
    return by_budget


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("v111_label_review_multiseed_confirmation.json"),
    )
    parser.add_argument(
        "--with-repair",
        action="store_true",
        help="Also refit after each review budget. Slower and secondary.",
    )
    args = parser.parse_args()

    runs = []
    print("=== MCGC v1.11 LABEL-REVIEW MULTI-SEED CONFIRMATION ===")
    print(f"SEEDS={SEEDS} WITH_REPAIR={args.with_repair}")

    for dataset in DATASETS:
        for noise_type, eta in CONDITIONS:
            for seed in SEEDS:
                r = run_condition(
                    dataset,
                    noise_type,
                    eta,
                    seed,
                    args.with_repair,
                )
                runs.append(r)
                c = r["review_curve"]
                print(
                    f"{dataset:14s} {noise_type:17s} eta={eta:.2f} seed={seed} "
                    + " ".join(
                        f"P@{int(100*x['budget'])}%={x['precision_at_budget']:.3f}"
                        for x in c
                    )
                )

    condition_aggregates = aggregate(runs)
    global_by_budget = global_aggregate(condition_aggregates)

    result = {
        "protocol": {
            "version": "1.11-confirmation",
            "status": "frozen multi-seed confirmation",
            "seeds": list(SEEDS),
            "datasets": list(DATASETS),
            "conditions": [
                {"noise_type": n, "eta": e}
                for n, e in CONDITIONS
            ],
            "budgets": list(BUDGETS),
            "outer_split": "70% train / 30% untouched clean test per seed",
            "audit_score": (
                "OOF MCGC label inconsistency A_i = best rival local evidence "
                "- observed-label local evidence"
            ),
            "primary_outcomes": [
                "precision_at_budget",
                "corrupt_recall",
                "precision_enrichment_over_random",
                "review_auc_precision",
            ],
            "random_reference": (
                "analytical expected precision equal to empirical corruption fraction; "
                "no random refit Monte Carlo in primary confirmation"
            ),
            "repair_outcome": (
                "optional secondary outcome enabled only with --with-repair"
            ),
            "important": (
                "clean labels are used only to synthesize corruption and evaluate "
                "audit ranking or optional simulated human repair; never to rank "
                "samples or tune the audit score"
            ),
        },
        "runs": runs,
        "condition_aggregates": condition_aggregates,
        "global_by_budget": global_by_budget,
    }

    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print("\n=== GLOBAL BY BUDGET ===")
    for row in global_by_budget:
        print(
            f"b={row['budget']:.2f} "
            f"P={row['mean_precision_across_conditions']:.3f} "
            f"R={row['mean_recall_across_conditions']:.3f} "
            f"enrich={row['mean_enrichment_across_conditions']:.2f}x "
            f"all-seed-win-conditions="
            f"{row['conditions_all_seeds_above_random']}/{row['n_conditions']}"
        )

    print(f"WROTE {args.output}")


if __name__ == "__main__":
    main()
