from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.datasets import load_breast_cancer, load_digits
from sklearn.model_selection import train_test_split

from field_only_cgec import FieldOnlyCGEC


def make_spirals(n_samples=700, noise=0.16, random_state=0):
    rng = np.random.default_rng(random_state)
    n = n_samples // 2
    theta = np.linspace(0.2, 3.2 * np.pi, n)
    radius = np.linspace(0.15, 1.0, n)
    arm0 = np.column_stack([radius * np.cos(theta), radius * np.sin(theta)])
    arm1 = -arm0
    X = np.vstack([arm0, arm1])
    X += rng.normal(scale=noise, size=X.shape)
    y = np.concatenate([np.zeros(n, dtype=int), np.ones(n, dtype=int)])
    order = rng.permutation(len(X))
    return X[order], y[order]


def dataset_for(name: str, seed: int):
    if name == "digits":
        data = load_digits()
        return data.data, data.target
    if name == "breast_cancer":
        data = load_breast_cancer()
        return data.data, data.target
    if name == "spirals_hard":
        return make_spirals(random_state=seed)
    raise ValueError(name)


def main():
    parser = argparse.ArgumentParser(description="Generate a field-only CGEC query explanation as JSON.")
    parser.add_argument("--dataset", choices=["digits", "breast_cancer", "spirals_hard"], default="digits")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--query-index", type=int, default=122)
    parser.add_argument("--output", type=Path, default=Path("query_explanation.json"))
    args = parser.parse_args()

    X, y = dataset_for(args.dataset, args.seed)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=7000 + args.seed, stratify=y
    )
    if not 0 <= args.query_index < len(X_test):
        raise IndexError(f"query-index must be in [0, {len(X_test)-1}]")

    model = FieldOnlyCGEC().fit(X_train, y_train)
    explanation = model.explain_query(X_test[args.query_index], top_support=5)
    explanation["dataset"] = args.dataset
    explanation["seed"] = args.seed
    explanation["outer_split"] = "stratified 70/30"
    explanation["outer_random_state"] = 7000 + args.seed
    explanation["query_index"] = args.query_index
    explanation["true_label"] = int(y_test[args.query_index])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(explanation, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
