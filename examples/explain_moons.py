from __future__ import annotations

import json

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

from mcgc import MultiAgentChemotacticGraphClassifier, explain_mcgc


def main():
    X, y = make_moons(n_samples=180, noise=0.16, random_state=7)
    X_train, X_test, y_train, _ = train_test_split(
        X, y, test_size=0.25, random_state=17, stratify=y
    )
    model = MultiAgentChemotacticGraphClassifier(
        n_neighbors=10,
        n_walkers=10,
        max_steps=8,
        diffusion_steps=30,
        random_state=17,
    ).fit(X_train, y_train)

    predictions, explanations = explain_mcgc(model, X_test[:2], top_support=4)
    payload = {
        "predictions": predictions.tolist(),
        "explanations": explanations,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
