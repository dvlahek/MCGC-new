from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
from sklearn.datasets import load_breast_cancer, load_digits, load_wine
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from mcgc import MultiAgentChemotacticGraphClassifier


ROOT = Path(__file__).resolve().parents[1]
V18 = ROOT / "examples" / "smoke_label_noise_v18.py"
V18C = ROOT / "examples" / "diagnostic_hard_label_noise_v18c.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


v18 = _load(V18, "smoke_label_noise_v18")
v18c = _load(V18C, "diagnostic_hard_label_noise_v18c")

SEED = 0
AUDIT_FOLDS = 3
DATASETS = ("digits", "breast_cancer", "wine")
CONDITIONS = (
    ("boundary_targeted", 0.05),
    ("boundary_targeted", 0.10),
    ("symmetric", 0.30),
)


def dataset_for(name: str):
    if name == "digits":
        d = load_digits()
    elif name == "breast_cancer":
        d = load_breast_cancer()
    elif name == "wine":
        d = load_wine()
    else:
        raise ValueError(name)
    return np.asarray(d.data), np.asarray(d.target)


def _softmax(x: np.ndarray, temperature: float = 0.5):
    z = np.asarray(x, dtype=float) / temperature
    z = z - np.max(z)
    p = np.exp(z)
    return p / p.sum()


def oof_trust_and_consensus(X, y_noisy, selected_params: dict, seed: int):
    cv = StratifiedKFold(
        n_splits=AUDIT_FOLDS,
        shuffle=True,
        random_state=21000 + seed,
    )
    trust_raw = np.empty(len(y_noisy), dtype=float)
    consensus = None
    seen = np.zeros(len(y_noisy), dtype=bool)

    for fit_idx, val_idx in cv.split(X, y_noisy):
        model = v18.mcgc_for(selected_params).fit(X[fit_idx], y_noisy[fit_idx])
        _, diagnostics = model.predict_with_diagnostics(X[val_idx])
        if consensus is None:
            consensus = np.zeros((len(y_noisy), len(model.classes_)), dtype=float)

        class_to_index = {c: j for j, c in enumerate(model.classes_)}
        for pos, row in zip(val_idx, diagnostics):
            observed = y_noisy[pos]
            observed_py = observed.item() if hasattr(observed, "item") else observed
            evidence = row["local_evidence"]
            observed_e = float(v18._lookup(evidence, observed))
            rival_values = [
                float(value)
                for label, value in evidence.items()
                if label != observed_py
            ]
            rival_e = max(rival_values)
            trust_raw[pos] = observed_e - rival_e

            vec = np.asarray([
                float(v18._lookup(evidence, c)) for c in model.classes_
            ])
            probs = _softmax(vec, temperature=0.5)
            for c, j in class_to_index.items():
                consensus[pos, j] = probs[j]
            seen[pos] = True

    if not np.all(seen):
        raise RuntimeError("OOF trust did not score every outer-training sample")
    assert consensus is not None

    scale = float(np.median(np.abs(trust_raw)))
    scale = max(scale, 1e-6)
    z = np.clip(trust_raw / scale, -12.0, 12.0)
    trust = 1.0 / (1.0 + np.exp(-z))
    trust = np.clip(trust, 0.05, 0.95)
    return trust, trust_raw, consensus


def _rebuild_field(model, base: np.ndarray):
    field = np.asarray(base, dtype=float).copy()
    a = float(model.diffusion_alpha)
    for _ in range(model.diffusion_steps):
        field = (1.0 - a) * base + a * (model.P_ @ field)
    model.class_field_ = field

    contrast = np.empty_like(field)
    for c in range(model.n_classes_):
        rivals = np.delete(field, c, axis=1)
        contrast[:, c] = field[:, c] - np.max(rivals, axis=1)
    model.contrast_field_ = contrast

    sorted_contrast = np.sort(contrast, axis=1)
    training_margins = sorted_contrast[:, -1] - sorted_contrast[:, -2]
    positive = training_margins[training_margins > 1e-12]
    model.walker_margin_scale_ = float(np.median(positive) if len(positive) else 1.0)
    model.walker_margin_scale_ = max(model.walker_margin_scale_, 1e-12)

    model.target_threshold_ = np.empty(model.n_classes_, dtype=float)
    for c in range(model.n_classes_):
        own = contrast[model.y_index_ == c, c]
        model.target_threshold_[c] = float(np.quantile(own, 0.50))
    return model


def fit_variant(X, y_noisy, selected_params, trust, consensus, variant: str):
    model = v18.mcgc_for(selected_params).fit(X, y_noisy)
    onehot = np.zeros((len(y_noisy), model.n_classes_), dtype=float)
    onehot[np.arange(len(y_noisy)), model.y_index_] = 1.0

    if variant == "original":
        return model
    if variant == "trust_weighted":
        base = onehot * trust[:, None]
        return _rebuild_field(model, base)
    if variant == "soft_source":
        base = trust[:, None] * onehot + (1.0 - trust[:, None]) * consensus
        return _rebuild_field(model, base)
    raise ValueError(variant)


def metrics(y_true, pred):
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
    }


def make_noise(X_train, y_train_clean, noise_type: str, eta: float, dataset: str):
    seed = 28000 + 100 * int(round(eta * 100)) + sum(ord(ch) for ch in dataset)
    if noise_type == "symmetric":
        return (*v18.corrupt_labels(y_train_clean, eta, seed), seed)
    if noise_type == "boundary_targeted":
        y_noisy, mask, _ = v18c.corrupt_boundary_targeted(
            X_train, y_train_clean, eta, seed
        )
        return y_noisy, mask, seed
    raise ValueError(noise_type)


def run_condition(dataset: str, noise_type: str, eta: float):
    X, y = dataset_for(dataset)
    X_train, X_test, y_train_clean, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=5000 + SEED,
        stratify=y,
    )

    y_train_noisy, corruption_mask, noise_seed = make_noise(
        X_train, y_train_clean, noise_type, eta, dataset
    )

    selected, tuning_records = v18.tune_model(
        "mcgc", v18.MCGC_GRID, X_train, y_train_noisy, SEED
    )
    selected_params = selected["params"]

    trust, trust_raw, consensus = oof_trust_and_consensus(
        X_train, y_train_noisy, selected_params, SEED
    )

    variants = {}
    for variant in ("original", "trust_weighted", "soft_source"):
        model = fit_variant(
            X_train,
            y_train_noisy,
            selected_params,
            trust,
            consensus,
            variant,
        )
        variants[variant] = metrics(y_test, model.predict(X_test))

    clean_mask = ~corruption_mask
    corrupted_mask = corruption_mask
    trust_summary = {
        "mean_all": float(np.mean(trust)),
        "mean_clean": float(np.mean(trust[clean_mask])) if np.any(clean_mask) else None,
        "mean_corrupted": float(np.mean(trust[corrupted_mask])) if np.any(corrupted_mask) else None,
        "median_clean": float(np.median(trust[clean_mask])) if np.any(clean_mask) else None,
        "median_corrupted": float(np.median(trust[corrupted_mask])) if np.any(corrupted_mask) else None,
        "raw_scale_median_abs": float(np.median(np.abs(trust_raw))),
    }

    return {
        "dataset": dataset,
        "noise_type": noise_type,
        "eta": float(eta),
        "noise_seed": int(noise_seed),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_corrupted": int(np.sum(corruption_mask)),
        "selected_params": selected_params,
        "tuning_selected": selected,
        "tuning_candidates": tuning_records,
        "trust_summary": trust_summary,
        "variants": variants,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("v19_reliability_weighted_smoke.json"),
    )
    args = parser.parse_args()

    runs = [
        run_condition(dataset, noise_type, eta)
        for dataset in DATASETS
        for noise_type, eta in CONDITIONS
    ]

    result = {
        "protocol": {
            "version": "1.9-smoke",
            "status": "diagnostic only; no manuscript claim",
            "seed": SEED,
            "datasets": list(DATASETS),
            "conditions": [
                {"noise_type": n, "eta": e} for n, e in CONDITIONS
            ],
            "outer_split": "70% train / 30% untouched clean test",
            "tuning": "same noisy-label-only MCGC alpha grid as v1.8",
            "trust_estimation": "3-fold OOF MCGC local evidence; audited sample excluded",
            "trust_map": "sigmoid((observed evidence - best rival evidence) / median absolute OOF consistency), clipped to [0.05,0.95]",
            "variants": {
                "original": "standard hard one-hot MCGC source",
                "trust_weighted": "one-hot source multiplied by OOF trust",
                "soft_source": "OOF-trust mixture of observed one-hot source and OOF softmax graph consensus",
            },
            "important": "clean training labels are used only to synthesize corruption and evaluate trust separation; they are never used to fit trust or choose the variant",
        },
        "runs": runs,
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")

    print("=== MCGC v1.9 RELIABILITY-WEIGHTED SMOKE ===")
    for run in runs:
        o = run["variants"]["original"]
        w = run["variants"]["trust_weighted"]
        s = run["variants"]["soft_source"]
        ts = run["trust_summary"]
        print(
            f"{run['dataset']:14s} {run['noise_type']:17s} eta={run['eta']:.2f} "
            f"ACC orig/weight/soft={o['accuracy']:.4f}/{w['accuracy']:.4f}/{s['accuracy']:.4f} "
            f"BA={o['balanced_accuracy']:.4f}/{w['balanced_accuracy']:.4f}/{s['balanced_accuracy']:.4f} "
            f"alpha={run['selected_params']['diffusion_alpha']} "
            f"trust clean/corrupt={ts['mean_clean']:.3f}/{ts['mean_corrupted']:.3f}"
        )
    print(f"WROTE {args.output}")


if __name__ == "__main__":
    main()
