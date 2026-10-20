from __future__ import annotations

import json

import numpy as np
from sklearn.datasets import make_circles, make_moons
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

from mcgc import MultiAgentChemotacticGraphClassifier, reliability_features
from benchmark_ablation_v02 import make_spirals


SEEDS = list(range(12))
COVERAGES = (1.0, 0.9, 0.8, 0.7, 0.5)
RISK_NAMES = (
    "local_margin_risk",
    "final_margin_risk",
    "trail_advantage_risk",
    "trail_disagreement_risk",
    "coherence_risk",
    "lost_trail_risk",
    "terminal_advantage_risk",
    "coherence_advantage_risk",
    "lost_advantage_risk",
)


def dataset_for(name, seed):
    if name == "moons":
        return make_moons(n_samples=500, noise=0.18, random_state=seed)
    if name == "circles":
        return make_circles(
            n_samples=500, noise=0.10, factor=0.45, random_state=seed
        )
    if name == "spirals":
        return make_spirals(n_samples=500, noise=0.10, random_state=seed)
    raise ValueError(name)


def model():
    return MultiAgentChemotacticGraphClassifier(
        n_neighbors=12,
        n_walkers=12,
        max_steps=10,
        diffusion_steps=40,
        random_state=17,
        adaptive_walker_gate=True,
    )


def safe_auc(error, risk):
    error = np.asarray(error, dtype=int)
    if len(np.unique(error)) < 2:
        return None
    return float(roc_auc_score(error, risk))


def selective_curve(correct, risk):
    correct = np.asarray(correct, dtype=bool)
    risk = np.asarray(risk, dtype=float)
    order = np.argsort(risk)
    out = {}
    n = len(correct)
    for coverage in COVERAGES:
        keep = max(1, int(np.ceil(coverage * n)))
        idx = order[:keep]
        out[str(coverage)] = {
            "kept": int(keep),
            "accuracy": float(np.mean(correct[idx])),
            "empirical_coverage": float(keep / n),
        }
    return out


def evaluate_dataset(name):
    pooled_error = []
    pooled_correct = []
    pooled_risk = {risk: [] for risk in RISK_NAMES}
    per_seed_auc = {risk: [] for risk in RISK_NAMES}
    seed_rows = []

    for seed in SEEDS:
        X, y = dataset_for(name, seed)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.30, random_state=1000 + seed, stratify=y
        )
        fitted = model().fit(X_train, y_train)
        pred, rows = reliability_features(fitted, X_test)
        correct = np.asarray(pred == y_test, dtype=bool)
        error = (~correct).astype(int)

        risks = {
            risk: np.asarray([row["risk_scores"][risk] for row in rows], dtype=float)
            for risk in RISK_NAMES
        }
        aucs = {risk: safe_auc(error, values) for risk, values in risks.items()}
        for risk, value in aucs.items():
            if value is not None:
                per_seed_auc[risk].append(value)

        pooled_error.extend(error.tolist())
        pooled_correct.extend(correct.tolist())
        for risk, values in risks.items():
            pooled_risk[risk].extend(values.tolist())

        seed_rows.append(
            {
                "seed": seed,
                "accuracy": float(accuracy_score(y_test, pred)),
                "errors": int(np.sum(error)),
                "auc": aucs,
            }
        )

    pooled_error_arr = np.asarray(pooled_error, dtype=int)
    pooled_correct_arr = np.asarray(pooled_correct, dtype=bool)
    risks_result = {}
    for risk in RISK_NAMES:
        values = np.asarray(pooled_risk[risk], dtype=float)
        valid = per_seed_auc[risk]
        risks_result[risk] = {
            "pooled_error_auc": safe_auc(pooled_error_arr, values),
            "valid_seed_auc_count": len(valid),
            "mean_seed_error_auc": float(np.mean(valid)) if valid else None,
            "std_seed_error_auc": (
                float(np.std(valid, ddof=1)) if len(valid) > 1 else 0.0 if valid else None
            ),
            "selective_curve": selective_curve(pooled_correct_arr, values),
        }

    return {
        "overall_accuracy": float(np.mean(pooled_correct_arr)),
        "pooled_errors": int(np.sum(pooled_error_arr)),
        "pooled_samples": int(len(pooled_error_arr)),
        "seeds": seed_rows,
        "risks": risks_result,
    }


def main():
    result = {
        "protocol": {
            "version": "0.6",
            "seeds": SEEDS,
            "n_seeds": len(SEEDS),
            "test_fraction": 0.30,
            "split_seed": "1000 + dataset_seed",
            "coverages": COVERAGES,
            "risk_orientation": "larger means more suspicious",
            "note": (
                "Frozen intrinsic-reliability benchmark. No risk model is trained and no "
                "threshold is tuned. AUROC tests error ranking; selective curves retain "
                "the lowest-risk samples at predeclared coverage levels."
            ),
        },
        "datasets": {
            name: evaluate_dataset(name) for name in ("moons", "circles", "spirals")
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
