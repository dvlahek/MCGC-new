from __future__ import annotations

import argparse
import importlib.util
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.datasets import load_breast_cancer, load_digits, load_wine
from sklearn.model_selection import train_test_split

from mcgc import MultiAgentChemotacticGraphClassifier, explain_mcgc


ROOT = Path(__file__).resolve().parents[1]
V12_SCRIPT = ROOT / "examples" / "benchmark_tuned_selective_v12.py"


def _load_v12():
    spec = importlib.util.spec_from_file_location("benchmark_tuned_selective_v12", V12_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


v12 = _load_v12()

SEEDS = list(range(5))
DATASETS = ("digits", "wine", "spirals_hard", "breast_cancer")
MULTICLASS_DATASETS = ("digits", "wine")
PERTURB_EPS = (0.02, 0.05)
PERTURB_REPS = 8
MAX_STABILITY_QUERIES = 40
TOP_SUPPORT = 5
DATASET_OFFSET = {
    "digits": 1_000_000,
    "wine": 2_000_000,
    "spirals_hard": 3_000_000,
    "breast_cancer": 4_000_000,
}


def dataset_for(name: str, seed: int):
    if name == "digits":
        data = load_digits()
        return data.data, data.target
    if name == "wine":
        data = load_wine()
        return data.data, data.target
    if name == "spirals_hard":
        return v12.make_spirals(n_samples=700, noise=0.16, random_state=seed)
    if name == "breast_cancer":
        data = load_breast_cancer()
        return data.data, data.target
    raise ValueError(name)


def mcgc_classifier():
    return MultiAgentChemotacticGraphClassifier(
        n_neighbors=12,
        n_walkers=12,
        max_steps=10,
        diffusion_alpha=0.85,
        diffusion_steps=40,
        temperature=0.35,
        random_state=17,
        adaptive_walker_gate=True,
    )


def py_scalar(value):
    return value.item() if hasattr(value, "item") else value


def explain_one(model, x):
    pred, rows = explain_mcgc(model, np.asarray(x, dtype=float)[None, :], top_support=TOP_SUPPORT)
    return py_scalar(pred[0]), rows[0]


def winner_trail(explanation):
    winner = explanation["prediction"]
    return explanation["classes"][winner]["representative_trail"]


def nontrivial_trail(trail):
    return bool(trail["length"] > 0 and len(trail["path"]) >= 2)


def node_set_jaccard(a, b):
    a = set(int(x) for x in a)
    b = set(int(x) for x in b)
    union = a | b
    if not union:
        return 1.0
    return float(len(a & b) / len(union))


def support_nodes(explanation):
    return [int(row["node_index"]) for row in explanation["supporting_neighbors"]]


def dominant_component(explanation):
    pos = explanation["prediction"]
    neg = explanation["runner_up"]
    pc = explanation["classes"][pos]["score_components"]
    nc = explanation["classes"][neg]["score_components"]
    deltas = {
        "local_component_delta": float(pc["local_component"] - nc["local_component"]),
        "trail_component_delta": float(pc["trail_component"] - nc["trail_component"]),
        "agreement_component_delta": float(pc["agreement_component"] - nc["agreement_component"]),
        "lost_component_delta": float(pc["lost_component"] - nc["lost_component"]),
    }
    return max(deltas, key=lambda key: abs(deltas[key]))


def contrastive_reconstruction_error(explanation):
    pos = explanation["prediction"]
    neg = explanation["runner_up"]
    pc = explanation["classes"][pos]["score_components"]
    nc = explanation["classes"][neg]["score_components"]
    reconstructed = (
        float(pc["local_component"] - nc["local_component"])
        + float(pc["trail_component"] - nc["trail_component"])
        + float(pc["agreement_component"] - nc["agreement_component"])
        + float(pc["lost_component"] - nc["lost_component"])
    )
    score_margin = float(explanation["class_scores"][pos] - explanation["class_scores"][neg])
    return abs(reconstructed - score_margin)


def perturb_rng(name: str, seed: int, query_idx: int, eps_idx: int, rep: int):
    rng_seed = (
        DATASET_OFFSET[name]
        + seed * 100_000
        + query_idx * 100
        + eps_idx * 10
        + rep
    )
    return np.random.default_rng(rng_seed)


def next_same_class_controls(predictions):
    by_class = defaultdict(list)
    for idx, pred in enumerate(predictions):
        by_class[py_scalar(pred)].append(idx)
    controls = {}
    for indices in by_class.values():
        if len(indices) < 2:
            continue
        for pos, idx in enumerate(indices):
            controls[idx] = indices[(pos + 1) % len(indices)]
    return controls


def mean_or_none(values):
    return None if not values else float(np.mean(values))


def median_or_none(values):
    return None if not values else float(np.median(values))


def rate_or_none(num, den):
    return None if den == 0 else float(num / den)


def evaluate_seed(name: str, seed: int):
    X, y = dataset_for(name, seed)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=7000 + seed,
        stratify=y,
    )
    model = mcgc_classifier().fit(X_train, y_train)

    predictions = []
    explanations = []
    for x in X_test:
        pred, explanation = explain_one(model, x)
        predictions.append(pred)
        explanations.append(explanation)
    predictions = np.asarray(predictions)

    # C. Eligibility over the complete outer-test split.
    eligible = []
    margins = []
    correct = []
    for pred, true, explanation in zip(predictions, y_test, explanations):
        trail = winner_trail(explanation)
        eligible.append(nontrivial_trail(trail))
        margins.append(float(explanation["margin"]))
        correct.append(bool(pred == true))
    eligible = np.asarray(eligible, dtype=bool)
    correct = np.asarray(correct, dtype=bool)
    margins = np.asarray(margins, dtype=float)

    eligibility = {
        "n_test": int(len(y_test)),
        "eligible_count": int(np.sum(eligible)),
        "eligible_rate": float(np.mean(eligible)),
        "eligible_rate_correct": rate_or_none(int(np.sum(eligible & correct)), int(np.sum(correct))),
        "eligible_rate_error": rate_or_none(int(np.sum(eligible & ~correct)), int(np.sum(~correct))),
        "accuracy_eligible": mean_or_none(correct[eligible].astype(float).tolist()),
        "accuracy_noneligible": mean_or_none(correct[~eligible].astype(float).tolist()),
        "median_margin_eligible": median_or_none(margins[eligible].tolist()),
        "median_margin_noneligible": median_or_none(margins[~eligible].tolist()),
    }

    # B. Multiclass exactness over all test queries where applicable.
    multiclass = None
    if name in MULTICLASS_DATASETS:
        errors = [contrastive_reconstruction_error(exp) for exp in explanations]
        pairs = Counter(
            (str(exp["prediction"]), str(exp["runner_up"]))
            for exp in explanations
        )
        multiclass = {
            "n_queries": int(len(explanations)),
            "max_abs_reconstruction_error": float(np.max(errors)),
            "median_abs_reconstruction_error": float(np.median(errors)),
            "mean_abs_reconstruction_error": float(np.mean(errors)),
            "winner_runner_pairs": [
                {"winner": pair[0], "runner_up": pair[1], "count": int(count)}
                for pair, count in pairs.most_common()
            ],
        }

    # A. Stability on a deterministic bounded subset.
    selected = list(range(min(MAX_STABILITY_QUERIES, len(X_test))))
    controls = next_same_class_controls(predictions)
    feature_scale = np.std(X_train, axis=0, ddof=0)
    feature_scale = np.where(feature_scale > 0.0, feature_scale, 1.0)

    query_records = []
    seed_support_diffs = []
    seed_path_diffs = []
    all_within_support = []
    all_cross_support = []
    all_within_path = []
    all_cross_path = []

    for query_idx in selected:
        x = X_test[query_idx]
        original_pred = py_scalar(predictions[query_idx])
        original = explanations[query_idx]
        original_support = support_nodes(original)
        original_trail = winner_trail(original)
        original_path_ok = nontrivial_trail(original_trail)
        original_dom = dominant_component(original)

        control_idx = controls.get(query_idx)
        cross_support = None
        cross_path = None
        if control_idx is not None:
            control = explanations[control_idx]
            cross_support = node_set_jaccard(original_support, support_nodes(control))
            control_trail = winner_trail(control)
            if original_path_ok and nontrivial_trail(control_trail):
                cross_path = node_set_jaccard(original_trail["path"], control_trail["path"])

        perturb_records = []
        retained_support = []
        retained_path = []
        retained_start = []
        retained_terminal = []
        retained_dom = []
        retained_margin_change = []
        attempted = 0
        retained = 0

        for eps_idx, eps in enumerate(PERTURB_EPS):
            for rep in range(PERTURB_REPS):
                attempted += 1
                rng = perturb_rng(name, seed, query_idx, eps_idx, rep)
                xp = x + eps * feature_scale * rng.normal(size=x.shape)
                pert_pred, pert = explain_one(model, xp)
                preserved = bool(pert_pred == original_pred)
                record = {
                    "epsilon": float(eps),
                    "rep": int(rep),
                    "prediction_preserved": preserved,
                }
                if preserved:
                    retained += 1
                    pert_support = support_nodes(pert)
                    support_j = node_set_jaccard(original_support, pert_support)
                    pert_trail = winner_trail(pert)
                    path_j = None
                    if original_path_ok and nontrivial_trail(pert_trail):
                        path_j = node_set_jaccard(original_trail["path"], pert_trail["path"])
                    same_start = bool(
                        len(original_trail["path"]) > 0
                        and len(pert_trail["path"]) > 0
                        and int(original_trail["path"][0]) == int(pert_trail["path"][0])
                    )
                    same_terminal = bool(
                        len(original_trail["path"]) > 0
                        and len(pert_trail["path"]) > 0
                        and int(original_trail["path"][-1]) == int(pert_trail["path"][-1])
                    )
                    same_dom = bool(dominant_component(pert) == original_dom)
                    margin_change = abs(float(pert["margin"]) - float(original["margin"]))
                    retained_support.append(support_j)
                    if path_j is not None:
                        retained_path.append(path_j)
                    retained_start.append(float(same_start))
                    retained_terminal.append(float(same_terminal))
                    retained_dom.append(float(same_dom))
                    retained_margin_change.append(margin_change)
                    record.update(
                        {
                            "support_jaccard": support_j,
                            "path_jaccard": path_j,
                            "same_start": same_start,
                            "same_terminal": same_terminal,
                            "same_dominant_component": same_dom,
                            "abs_margin_change": margin_change,
                        }
                    )
                perturb_records.append(record)

        mean_within_support = mean_or_none(retained_support)
        mean_within_path = mean_or_none(retained_path)
        support_diff = None
        path_diff = None
        if mean_within_support is not None and cross_support is not None:
            support_diff = float(mean_within_support - cross_support)
            seed_support_diffs.append(support_diff)
            all_within_support.append(mean_within_support)
            all_cross_support.append(cross_support)
        if mean_within_path is not None and cross_path is not None:
            path_diff = float(mean_within_path - cross_path)
            seed_path_diffs.append(path_diff)
            all_within_path.append(mean_within_path)
            all_cross_path.append(cross_path)

        query_records.append(
            {
                "query_index": int(query_idx),
                "prediction": py_scalar(original_pred),
                "true_label": py_scalar(y_test[query_idx]),
                "correct": bool(correct[query_idx]),
                "original_margin": float(original["margin"]),
                "original_path_eligible": bool(original_path_ok),
                "control_query_index": None if control_idx is None else int(control_idx),
                "cross_support_jaccard": cross_support,
                "cross_path_jaccard": cross_path,
                "attempted_perturbations": int(attempted),
                "preserved_perturbations": int(retained),
                "preservation_rate": float(retained / attempted),
                "mean_within_support_jaccard": mean_within_support,
                "mean_within_path_jaccard": mean_within_path,
                "within_minus_cross_support": support_diff,
                "within_minus_cross_path": path_diff,
                "same_start_rate": mean_or_none(retained_start),
                "same_terminal_rate": mean_or_none(retained_terminal),
                "same_dominant_component_rate": mean_or_none(retained_dom),
                "mean_abs_margin_change": mean_or_none(retained_margin_change),
                "perturbations": perturb_records,
            }
        )

    attempted_total = sum(r["attempted_perturbations"] for r in query_records)
    preserved_total = sum(r["preserved_perturbations"] for r in query_records)
    stability_summary = {
        "n_selected_queries": int(len(selected)),
        "attempted_perturbations": int(attempted_total),
        "preserved_perturbations": int(preserved_total),
        "prediction_preservation_rate": rate_or_none(preserved_total, attempted_total),
        "evaluable_support_queries": int(len(seed_support_diffs)),
        "mean_within_support_jaccard": mean_or_none(all_within_support),
        "mean_cross_support_jaccard": mean_or_none(all_cross_support),
        "mean_support_advantage": mean_or_none(seed_support_diffs),
        "evaluable_path_queries": int(len(seed_path_diffs)),
        "mean_within_path_jaccard": mean_or_none(all_within_path),
        "mean_cross_path_jaccard": mean_or_none(all_cross_path),
        "mean_path_advantage": mean_or_none(seed_path_diffs),
    }

    return {
        "seed": int(seed),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "accuracy": float(np.mean(predictions == y_test)),
        "eligibility": eligibility,
        "multiclass": multiclass,
        "stability": stability_summary,
        "stability_queries": query_records,
    }


def aggregate_dataset(runs):
    support_adv = [r["stability"]["mean_support_advantage"] for r in runs if r["stability"]["mean_support_advantage"] is not None]
    path_adv = [r["stability"]["mean_path_advantage"] for r in runs if r["stability"]["mean_path_advantage"] is not None]
    return {
        "mean_accuracy": float(np.mean([r["accuracy"] for r in runs])),
        "mean_eligibility_rate": float(np.mean([r["eligibility"]["eligible_rate"] for r in runs])),
        "mean_prediction_preservation_rate": float(np.mean([r["stability"]["prediction_preservation_rate"] for r in runs])),
        "support_advantage": None if not support_adv else v12.delta_summary(support_adv),
        "path_advantage": None if not path_adv else v12.delta_summary(path_adv),
        "multiclass_max_abs_reconstruction_error": None if runs[0]["multiclass"] is None else float(max(r["multiclass"]["max_abs_reconstruction_error"] for r in runs)),
    }


def pooled_stability(datasets):
    support = []
    path = []
    for dataset in datasets.values():
        for run in dataset["runs"]:
            s = run["stability"]["mean_support_advantage"]
            p = run["stability"]["mean_path_advantage"]
            if s is not None:
                support.append(s)
            if p is not None:
                path.append(p)
    return {
        "dataset_seed_support_advantage": None if not support else v12.delta_summary(support),
        "dataset_seed_path_advantage": None if not path else v12.delta_summary(path),
    }


def fmt(value):
    return "NA" if value is None else f"{value:.6f}"


def print_summary(result):
    print("=== MCGC v1.5 FROZEN XAI STABILITY ===")
    for name, dataset in result["datasets"].items():
        a = dataset["aggregate"]
        s = a["support_advantage"]
        p = a["path_advantage"]
        print(
            f"{name:16s} acc={a['mean_accuracy']:.4f} elig={a['mean_eligibility_rate']:.4f} "
            f"preserve={a['mean_prediction_preservation_rate']:.4f} "
            f"support_adv={fmt(None if s is None else s['mean'])} "
            f"path_adv={fmt(None if p is None else p['mean'])} "
            f"multi_err={fmt(a['multiclass_max_abs_reconstruction_error'])}"
        )
    pooled = result["pooled_stability"]
    print("=== DATASETxSEED STABILITY CONTRAST ===")
    for key in ("dataset_seed_support_advantage", "dataset_seed_path_advantage"):
        row = pooled[key]
        if row is None:
            print(f"{key}: NA")
        else:
            print(
                f"{key}: mean={row['mean']:.6f} median={row['median']:.6f} "
                f"W/T/L={row['wins']}/{row['ties']}/{row['losses']} "
                f"p={row['sign_test_p_two_sided']:.6g} n={row['n']}"
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("v15_xai_stability.json"))
    args = parser.parse_args()

    datasets = {}
    for name in DATASETS:
        print(f"START {name}", flush=True)
        runs = []
        for seed in SEEDS:
            print(f"  seed={seed}", flush=True)
            runs.append(evaluate_seed(name, seed))
        datasets[name] = {
            "runs": runs,
            "aggregate": aggregate_dataset(runs),
        }

    result = {
        "protocol": {
            "version": "1.5",
            "status": "frozen XAI stability / multiclass / eligibility confirmation",
            "datasets": list(DATASETS),
            "multiclass_datasets": list(MULTICLASS_DATASETS),
            "seeds": SEEDS,
            "outer_split": "70% train / 30% test",
            "outer_split_seed": "7000 + dataset_seed",
            "perturb_eps": list(PERTURB_EPS),
            "perturb_reps_per_epsilon": PERTURB_REPS,
            "max_stability_queries_per_dataset_seed": MAX_STABILITY_QUERIES,
            "top_support": TOP_SUPPORT,
            "prediction_preserving_perturbations_only_for_similarity": True,
            "cross_query_control": "cyclic next outer-test query with same predicted class",
            "path_eligibility": "winner trail length > 0 and path has >= 2 nodes",
            "model": {
                "n_neighbors": 12,
                "n_walkers": 12,
                "max_steps": 10,
                "diffusion_alpha": 0.85,
                "diffusion_steps": 40,
                "temperature": 0.35,
                "random_state": 17,
                "adaptive_walker_gate": True,
            },
        },
        "datasets": datasets,
        "pooled_stability": pooled_stability(datasets),
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print_summary(result)
    print(f"WROTE {args.output}")


if __name__ == "__main__":
    main()
