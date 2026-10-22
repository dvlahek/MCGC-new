from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from mcgc import (
    MultiAgentChemotacticGraphClassifier,
    paired_feature_intervention_effects,
    plot_feature_interventions,
    plot_mcgc_explanation,
)


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

    intervention = paired_feature_intervention_effects(model, query)
    effects = intervention["effects"]

    # Pixel-space view of the same paired intervention effects.
    fig2, axes = plt.subplots(1, 2, figsize=(8.5, 4.0))
    axes[0].imshow(query.reshape(8, 8), cmap="gray")
    axes[0].set_title(
        f"query: true={int(y_test[query_index])}, pred={explanation['prediction']}"
    )
    axes[0].axis("off")
    im = axes[1].imshow(effects.reshape(8, 8), cmap="coolwarm")
    axes[1].set_title("paired feature-intervention effect")
    axes[1].axis("off")
    fig2.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04, label="margin change")
    fig2.tight_layout()
    fig2.savefig(OUT / "digits_feature_interventions.png", dpi=220, bbox_inches="tight")
    fig2.savefig(OUT / "digits_feature_interventions.pdf", bbox_inches="tight")
    plt.close(fig2)

    # A generic bar view is also available when image geometry is not meaningful.
    fig3, ax3 = plt.subplots(figsize=(10, 4))
    plot_feature_interventions(effects, ax=ax3)
    fig3.tight_layout()
    fig3.savefig(OUT / "digits_feature_interventions_bar.png", dpi=220, bbox_inches="tight")
    plt.close(fig3)

    print("true_label", int(y_test[query_index]))
    print("prediction", explanation["prediction"])
    print("runner_up", explanation["runner_up"])
    print("margin", explanation["margin"])
    print("saved", OUT / "digits_explanation.png")
    print("saved", OUT / "digits_feature_interventions.png")


if __name__ == "__main__":
    main()
