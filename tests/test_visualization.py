import matplotlib

matplotlib.use("Agg")

import numpy as np
from sklearn.datasets import make_moons

from mcgc import (
    MultiAgentChemotacticGraphClassifier,
    paired_feature_intervention_effects,
    plot_feature_interventions,
    plot_mcgc_explanation,
)


def _model():
    X, y = make_moons(n_samples=120, noise=0.12, random_state=3)
    model = MultiAgentChemotacticGraphClassifier(
        n_neighbors=8,
        n_walkers=6,
        max_steps=6,
        random_state=5,
        adaptive_walker_gate=True,
    ).fit(X, y)
    return X, model


def test_plot_mcgc_explanation_2d():
    X, model = _model()
    fig, axes, explanation = plot_mcgc_explanation(model, X[0], top_support=4)
    assert fig is not None
    assert axes.shape == (2, 2)
    assert explanation["prediction"] in model.classes_
    assert np.isclose(
        sum(
            explanation["classes"][explanation["prediction"]]["score_components"][key]
            - explanation["classes"][explanation["runner_up"]]["score_components"][key]
            for key in (
                "local_component",
                "trail_component",
                "agreement_component",
                "lost_component",
            )
        ),
        explanation["margin"],
    )


def test_paired_feature_intervention_effects():
    X, model = _model()
    record = paired_feature_intervention_effects(model, X[0])
    assert record["effects"].shape == X[0].shape
    assert record["reference"].shape == X[0].shape
    assert np.allclose(
        record["effects"],
        record["original_margin"] - record["intervened_margins"],
    )


def test_plot_feature_interventions():
    ax = plot_feature_interventions([0.2, -0.1], feature_names=["a", "b"])
    assert ax.get_title() == "Paired feature interventions"
