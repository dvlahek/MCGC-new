from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from mcgc import MultiAgentChemotacticGraphClassifier, plot_mcgc_explanation


OUT = Path("results/visualization_demo")
OUT.mkdir(parents=True, exist_ok=True)


def main():
    digits = load_digits()
    mask = (digits.target == 3) | (digits.target == 8)
    X = digits.data[mask]
    y = digits.target[mask]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=17,
        stratify=y,
    )

    model = MultiAgentChemotacticGraphClassifier(
        n_neighbors=12,
        n_walkers=12,
        max_steps=10,
        diffusion_alpha=0.85,
        diffusion_steps=40,
        temperature=0.35,
        random_state=17,
        adaptive_walker_gate=True,
    ).fit(X_train, y_train)

    # Pick one deterministic held-out query. Users can replace this with any
    # one-dimensional query vector compatible with the fitted model.
    query_index = 0
    query = X_test[query_index]

    fig, _, explanation = plot_mcgc_explanation(
        model,
        query,
        top_support=5,
        image_shape=(8, 8),
    )

    fig.savefig(OUT / "digits_explanation.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUT / "digits_explanation.pdf", bbox_inches="tight")
    plt.close(fig)

    print("true_label", int(y_test[query_index]))
    print("prediction", explanation["prediction"])
    print("runner_up", explanation["runner_up"])
    print("margin", explanation["margin"])
    print("saved", OUT / "digits_explanation.png")


if __name__ == "__main__":
    main()
