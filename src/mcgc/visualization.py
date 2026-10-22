from __future__ import annotations

from typing import Sequence

import numpy as np

from .xai import explain_mcgc


def _plt():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "MCGC visualization requires matplotlib. Install the development "
            "extras with `pip install -e .[dev]` or install matplotlib directly."
        ) from exc
    return plt


def _training_X_original(model) -> np.ndarray:
    X = np.asarray(model.X_, dtype=float)
    if getattr(model, "scaler_", None) is None:
        return X
    return model.scaler_.inverse_transform(X)


def _component_deltas(explanation):
    winner = explanation["prediction"]
    runner = explanation["runner_up"]
    wc = explanation["classes"][winner]["score_components"]
    rc = explanation["classes"][runner]["score_components"]
    return {
        "local": float(wc["local_component"] - rc["local_component"]),
        "trail": float(wc["trail_component"] - rc["trail_component"]),
        "agreement": float(
            wc["agreement_component"] - rc["agreement_component"]
        ),
        "lost": float(wc["lost_component"] - rc["lost_component"]),
    }


def paired_feature_intervention_effects(
    model,
    x: np.ndarray,
    *,
    reference: np.ndarray | None = None,
):
    """Compute native paired feature-intervention effects for one query.

    The original prediction defines the fixed winner ``c+`` and runner-up
    ``c-``. For feature ``k``, the query value is replaced by ``reference[k]``
    and the same class contrast is recomputed:

        I_k(q) = M(q) - M(q^(-k)).

    By default, the reference is the mean of the fitted training data in the
    original input space. Positive values support the original winner relative
    to the original runner-up. Effects are not assumed to be additive.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("x must be a one-dimensional query vector.")

    X_train = _training_X_original(model)
    if reference is None:
        reference = np.mean(X_train, axis=0)
    reference = np.asarray(reference, dtype=float)
    if reference.shape != x.shape:
        raise ValueError("reference must have the same shape as x.")

    _, original_records = explain_mcgc(model, x[None, :], top_support=1)
    original = original_records[0]
    winner = original["prediction"]
    runner = original["runner_up"]
    original_margin = float(
        original["class_scores"][winner] - original["class_scores"][runner]
    )

    effects = np.empty_like(x, dtype=float)
    intervened_margins = np.empty_like(x, dtype=float)
    for k in range(x.size):
        xp = x.copy()
        xp[k] = reference[k]
        _, records = explain_mcgc(model, xp[None, :], top_support=1)
        rec = records[0]
        margin = float(rec["class_scores"][winner] - rec["class_scores"][runner])
        intervened_margins[k] = margin
        effects[k] = original_margin - margin

    return {
        "winner": winner,
        "runner_up": runner,
        "original_margin": original_margin,
        "reference": reference,
        "effects": effects,
        "intervened_margins": intervened_margins,
        "interpretation_note": (
            "Effects are paired changes in the fixed original winner-versus-runner "
            "margin. They are not an additive score decomposition."
        ),
    }


def _draw_support_panel(ax, model, explanation, X_train, image_shape=None):
    support = explanation["supporting_neighbors"]
    if image_shape is None:
        nodes = [int(row["node_index"]) for row in support]
        affinities = [float(row["affinity"]) for row in support]
        labels = [str(row["training_label"]) for row in support]
        y = np.arange(len(nodes))
        ax.barh(y, affinities)
        ax.set_yticks(y, [f"node {n} / y={lab}" for n, lab in zip(nodes, labels)])
        ax.invert_yaxis()
        ax.set_xlabel("Query affinity")
        ax.set_title("Supporting training examples")
        return

    ax.axis("off")
    n = len(support)
    if n == 0:
        ax.text(0.5, 0.5, "No supporting examples", ha="center", va="center")
        return

    tile_w = 0.9 / n
    for j, row in enumerate(support):
        node = int(row["node_index"])
        img = np.asarray(X_train[node], dtype=float).reshape(image_shape)
        left = 0.05 + j * tile_w
        inset = ax.inset_axes([left, 0.18, tile_w * 0.82, 0.70])
        inset.imshow(img, cmap="gray")
        inset.set_title(
            f"y={row['training_label']}\na={row['affinity']:.2f}", fontsize=8
        )
        inset.axis("off")
    ax.set_title("Supporting training examples")


def _draw_trail_panel(ax, explanation):
    winner = explanation["prediction"]
    runner = explanation["runner_up"]
    for label, cls in (("winner", winner), ("runner-up", runner)):
        trail = explanation["classes"][cls]["representative_trail"]
        trace = np.asarray(trail["evidence_trace"], dtype=float)
        ax.plot(np.arange(len(trace)), trace, marker="o", label=f"{label}: {cls}")
    ax.axhline(0.0, linewidth=0.8)
    ax.set_xlabel("Representative-trail step")
    ax.set_ylabel("Contrastive evidence")
    ax.set_title("Representative evidence trails")
    ax.legend(frameon=False)


def _draw_2d_graph_panel(ax, model, x, explanation, X_train):
    y_train = model.classes_[model.y_index_]
    for cls in model.classes_:
        mask = y_train == cls
        ax.scatter(
            X_train[mask, 0],
            X_train[mask, 1],
            s=20,
            alpha=0.45,
            label=f"class {cls}",
        )

    support_nodes = [int(row["node_index"]) for row in explanation["supporting_neighbors"]]
    if support_nodes:
        ax.scatter(
            X_train[support_nodes, 0],
            X_train[support_nodes, 1],
            s=85,
            facecolors="none",
            linewidths=1.5,
            label="support",
        )

    winner = explanation["prediction"]
    trail = explanation["classes"][winner]["representative_trail"]
    path = [int(i) for i in trail["path"]]
    if path:
        pts = X_train[path]
        ax.plot(pts[:, 0], pts[:, 1], marker="o", linewidth=1.6, label="winner trail")

    ax.scatter([x[0]], [x[1]], marker="*", s=180, label="query")
    ax.set_xlabel("feature 1")
    ax.set_ylabel("feature 2")
    ax.set_title("Local graph view")
    ax.legend(frameon=False, fontsize=8)


def plot_feature_interventions(
    effects: Sequence[float],
    *,
    feature_names: Sequence[str] | None = None,
    ax=None,
    title: str = "Paired feature interventions",
):
    """Plot fitted-margin changes from paired feature interventions."""
    plt = _plt()
    effects = np.asarray(effects, dtype=float)
    if effects.ndim != 1:
        raise ValueError("effects must be a one-dimensional sequence.")
    if feature_names is None:
        feature_names = [f"x{i}" for i in range(len(effects))]
    if len(feature_names) != len(effects):
        raise ValueError("feature_names must have the same length as effects.")
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(effects))
    ax.bar(x, effects)
    ax.axhline(0.0, linewidth=0.8)
    ax.set_xticks(x, feature_names, rotation=45, ha="right")
    ax.set_ylabel("Margin change")
    ax.set_title(title)
    return ax


def plot_mcgc_explanation(
    model,
    x: np.ndarray,
    *,
    top_support: int = 5,
    image_shape: tuple[int, int] | None = None,
    feature_effects: Sequence[float] | None = None,
    feature_names: Sequence[str] | None = None,
    figsize: tuple[float, float] = (12.0, 8.0),
):
    """Visualize one fitted MCGC prediction and its intrinsic explanation.

    The four-panel figure contains the same objects returned by
    :func:`explain_mcgc`: support or graph context, fitted class scores,
    representative evidence trails, and the exact contrastive score-component
    decomposition. ``feature_effects`` is optional and produces a separate
    intervention figure without changing the main explanation record.
    """
    plt = _plt()
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("x must be a one-dimensional query vector.")

    prediction, records = explain_mcgc(model, x[None, :], top_support=top_support)
    explanation = records[0]
    X_train = _training_X_original(model)

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    ax00, ax01, ax10, ax11 = axes.ravel()

    if x.size == 2:
        _draw_2d_graph_panel(ax00, model, x, explanation, X_train)
    else:
        _draw_support_panel(ax00, model, explanation, X_train, image_shape=image_shape)

    class_scores = explanation["class_scores"]
    classes = list(class_scores.keys())
    scores = [class_scores[c] for c in classes]
    ax01.bar([str(c) for c in classes], scores)
    ax01.set_ylabel("MCGC class score")
    ax01.set_title(
        f"Prediction {prediction[0]} vs runner-up {explanation['runner_up']}\n"
        f"margin={explanation['margin']:.3f}, gate={explanation['walker_gate']:.3f}"
    )

    _draw_trail_panel(ax10, explanation)

    deltas = _component_deltas(explanation)
    names = list(deltas.keys())
    vals = [deltas[name] for name in names]
    ax11.bar(names, vals)
    ax11.axhline(0.0, linewidth=0.8)
    ax11.set_ylabel("Winner - runner-up component")
    ax11.set_title(
        "Exact contrastive score decomposition\n"
        f"sum={sum(vals):.3f}, margin={explanation['margin']:.3f}"
    )

    fig.suptitle("MCGC intrinsic explanation", fontsize=14)
    fig.tight_layout()

    if feature_effects is not None:
        feature_effects = np.asarray(feature_effects, dtype=float)
        if feature_effects.size != x.size:
            raise ValueError("feature_effects must have one value per query feature.")
        fig2, ax2 = plt.subplots(figsize=(9, 4))
        plot_feature_interventions(
            feature_effects,
            feature_names=feature_names,
            ax=ax2,
        )
        fig2.tight_layout()

    return fig, axes, explanation
