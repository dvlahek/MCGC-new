from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from mcgc import MultiAgentChemotacticGraphClassifier, explain_mcgc


DEFAULT_SEEDS = (0, 1, 2, 3, 4)


def classifier():
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


def split(seed: int):
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data,
        data.target,
        test_size=0.30,
        random_state=7000 + seed,
        stratify=data.target,
    )
    return data.feature_names, X_train, X_test, y_train, y_test


def explain_one(model, x):
    pred, rows = explain_mcgc(model, np.asarray(x, dtype=float)[None, :], top_support=5)
    return pred[0].item() if hasattr(pred[0], "item") else pred[0], rows[0]


def nontrivial_winner_trail(exp):
    winner = exp["prediction"]
    trail = exp["classes"][winner]["representative_trail"]
    return bool(trail["length"] > 0 and len(trail["path"]) >= 2)


def fixed_pair_margin(exp, winner, runner):
    return float(exp["class_scores"][winner] - exp["class_scores"][runner])


def feature_effects(model, x, train_mean, baseline_exp):
    winner = baseline_exp["prediction"]
    runner = baseline_exp["runner_up"]
    base_margin = fixed_pair_margin(baseline_exp, winner, runner)
    effects = np.empty(x.shape[0], dtype=float)
    intervened_margins = np.empty(x.shape[0], dtype=float)
    for k in range(x.shape[0]):
        xp = np.asarray(x, dtype=float).copy()
        xp[k] = train_mean[k]
        _, exp_k = explain_one(model, xp)
        m_k = fixed_pair_margin(exp_k, winner, runner)
        intervened_margins[k] = m_k
        effects[k] = base_margin - m_k
    return effects, intervened_margins


def choose_local_query(model, X_test, y_test):
    candidates = []
    stored = {}
    for i, (x, true) in enumerate(zip(X_test, y_test)):
        pred, exp = explain_one(model, x)
        stored[i] = exp
        if pred != true or not nontrivial_winner_trail(exp):
            continue
        candidates.append((i, float(exp["margin"])))
    if not candidates:
        raise RuntimeError("No correctly classified nontrivial-path query found.")
    median_margin = float(np.median([m for _, m in candidates]))
    candidates.sort(key=lambda t: (abs(t[1] - median_margin), t[0]))
    idx = int(candidates[0][0])
    return idx, median_margin, len(candidates), stored[idx]


def path_geometry(model, trail, feature_names):
    path = [int(i) for i in trail["path"]]
    edge_rows = []
    if len(path) < 2:
        return edge_rows
    for step, (a, b) in enumerate(zip(path[:-1], path[1:])):
        diff2 = (model.X_[a] - model.X_[b]) ** 2
        denom = float(np.sum(diff2))
        rho = np.zeros_like(diff2) if denom <= 0 else diff2 / denom
        edge_rows.append(
            {
                "step": int(step),
                "from_node": a,
                "to_node": b,
                "feature_share": {
                    str(name): float(value) for name, value in zip(feature_names, rho)
                },
            }
        )
    return edge_rows


def evaluate_seed(seed: int):
    feature_names, X_train, X_test, y_train, y_test = split(seed)
    model = classifier().fit(X_train, y_train)
    train_mean = np.mean(X_train, axis=0)
    abs_effects = []
    signed_effects = []

    for i, x in enumerate(X_test):
        _, exp = explain_one(model, x)
        effects, _ = feature_effects(model, x, train_mean, exp)
        abs_effects.append(np.abs(effects))
        signed_effects.append(effects)
        if (i + 1) % 25 == 0:
            print(f"seed={seed} global query {i+1}/{len(X_test)}", flush=True)

    abs_effects = np.asarray(abs_effects)
    signed_effects = np.asarray(signed_effects)
    return {
        "seed": int(seed),
        "n_test": int(len(X_test)),
        "mean_abs_effect": {
            str(name): float(v) for name, v in zip(feature_names, np.mean(abs_effects, axis=0))
        },
        "median_abs_effect": {
            str(name): float(v) for name, v in zip(feature_names, np.median(abs_effects, axis=0))
        },
        "winner_support_fraction": {
            str(name): float(v) for name, v in zip(feature_names, np.mean(signed_effects > 0, axis=0))
        },
    }


def aggregate_global(runs, feature_names):
    means = np.asarray([
        [r["mean_abs_effect"][str(name)] for name in feature_names]
        for r in runs
    ])
    mean = np.mean(means, axis=0)
    if len(runs) > 1:
        ci = 1.96 * np.std(means, axis=0, ddof=1) / np.sqrt(len(runs))
    else:
        ci = np.zeros_like(mean)
    support = np.mean(
        np.asarray([
            [r["winner_support_fraction"][str(name)] for name in feature_names]
            for r in runs
        ]),
        axis=0,
    )
    return {
        "mean_abs_effect": {str(n): float(v) for n, v in zip(feature_names, mean)},
        "descriptive_95ci_halfwidth": {str(n): float(v) for n, v in zip(feature_names, ci)},
        "winner_support_fraction": {str(n): float(v) for n, v in zip(feature_names, support)},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=Path("v17_feature_attribution.json"))
    ap.add_argument("--seeds", type=int, nargs="*", default=list(DEFAULT_SEEDS))
    args = ap.parse_args()

    feature_names, X_train, X_test, y_train, y_test = split(0)
    model = classifier().fit(X_train, y_train)
    selected_idx, median_margin, candidate_count, selected_exp = choose_local_query(model, X_test, y_test)
    train_mean = np.mean(X_train, axis=0)
    local_effects, intervened = feature_effects(
        model, X_test[selected_idx], train_mean, selected_exp
    )
    winner = selected_exp["prediction"]
    runner = selected_exp["runner_up"]
    winner_trail = selected_exp["classes"][winner]["representative_trail"]

    local = {
        "seed": 0,
        "test_index": int(selected_idx),
        "true_label": int(y_test[selected_idx]),
        "winner": int(winner),
        "runner_up": int(runner),
        "baseline_margin": float(fixed_pair_margin(selected_exp, winner, runner)),
        "eligible_median_margin": float(median_margin),
        "eligible_candidate_count": int(candidate_count),
        "winner_trail_path": [int(i) for i in winner_trail["path"]],
        "winner_trail_evidence": [float(v) for v in winner_trail["evidence_trace"]],
        "feature_effect": {
            str(name): float(v) for name, v in zip(feature_names, local_effects)
        },
        "intervened_margin": {
            str(name): float(v) for name, v in zip(feature_names, intervened)
        },
        "path_geometry": path_geometry(model, winner_trail, feature_names),
    }

    runs = []
    for seed in args.seeds:
        print(f"=== GLOBAL FEATURE ATTRIBUTION seed={seed} ===", flush=True)
        runs.append(evaluate_seed(int(seed)))

    payload = {
        "protocol": {
            "version": "1.7",
            "dataset": "breast_cancer",
            "seeds": [int(s) for s in args.seeds],
            "reference_intervention": "replace one raw feature by outer-training mean",
            "local_effect": "I_k(q)=M(q)-M(q^(-k)) using fixed original winner/runner",
            "path_geometry": "rho_ijk=(z_ik-z_jk)^2/sum_l(z_il-z_jl)^2",
            "global_importance": "seed-mean of mean_test |I_k|; descriptive 95% CI across seeds",
        },
        "feature_names": [str(x) for x in feature_names],
        "local": local,
        "global_runs": runs,
        "global": aggregate_global(runs, feature_names),
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print("=== MCGC v1.7 FEATURE ATTRIBUTION ===")
    print(f"LOCAL_QUERY seed=0 test_index={selected_idx} winner={winner} runner={runner}")
    print(f"LOCAL_MARGIN={local['baseline_margin']:.6f} eligible_candidates={candidate_count}")
    print(f"WINNER_TRAIL_EDGES={max(0, len(winner_trail['path'])-1)}")
    print(f"WROTE {args.output}")


if __name__ == "__main__":
    main()
