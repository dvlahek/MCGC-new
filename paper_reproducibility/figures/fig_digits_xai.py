from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "xai"))
from field_only_cgec import FieldOnlyCGEC  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    seed = 0
    query_index = 122
    data = load_digits()
    Xtr, Xte, ytr, yte = train_test_split(
        data.data, data.target, test_size=0.30, random_state=7000 + seed, stratify=data.target
    )
    model = FieldOnlyCGEC().fit(Xtr, ytr)
    q = Xte[query_index]
    exp = model.explain_query(q, top_support=5)

    assert int(yte[query_index]) == 3
    assert exp["prediction"] == 3
    assert exp["runner_up"] == 8

    winner = exp["prediction"]
    runner = exp["runner_up"]
    c_plus = model.class_to_index_[winner]
    c_minus = model.class_to_index_[runner]

    base_margin = exp["margin"]
    train_mean = Xtr.mean(axis=0)
    effects = []
    for k in range(Xtr.shape[1]):
        qk = q.copy()
        qk[k] = train_mean[k]
        _, _, _, Lk = model.local_evidence(qk)
        mk = float(Lk[c_plus] - Lk[c_minus])
        effects.append(base_margin - mk)
    effects = np.asarray(effects)
    top_features = np.argsort(-np.abs(effects))[:10]

    fig = plt.figure(figsize=(12.4, 6.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.05], width_ratios=[1.0, 1.55], hspace=0.48, wspace=0.32)

    ax = fig.add_subplot(gs[0, 0])
    ax.axis("off")
    ax.set_title("A  Query and decision", loc="left", fontweight="bold")
    img_ax = ax.inset_axes([0.05, 0.08, 0.48, 0.78])
    img_ax.imshow(q.reshape(8, 8), cmap="gray_r")
    img_ax.axis("off")
    ax.text(
        0.58, 0.67,
        f"True label: {int(yte[query_index])}\n"
        f"Predicted: {winner}\n"
        f"Runner-up: {runner}\n\n"
        f"Local evidence margin: {base_margin:.3f}",
        transform=ax.transAxes, va="top", ha="left", fontsize=10
    )

    ax = fig.add_subplot(gs[0, 1])
    ax.axis("off")
    ax.set_title("B  Local support examples", loc="left", fontweight="bold")
    supports = exp["supporting_neighbors"]
    for i, row in enumerate(supports):
        x0 = 0.015 + i * 0.195
        inset = ax.inset_axes([x0, 0.29, 0.16, 0.52])
        inset.imshow(Xtr[row["node_index"]].reshape(8, 8), cmap="gray_r")
        inset.axis("off")
        ax.text(
            x0, 0.13,
            f"$a_j={row['affinity']:.2f}$\nlabel={row['training_label']}",
            transform=ax.transAxes, fontsize=8, ha="left", va="top"
        )

    ax = fig.add_subplot(gs[1, 0])
    ax.axis("off")
    ax.set_title("C  Representative evidence paths", loc="left", fontweight="bold")
    rows = [(winner, 0.56, "Winner"), (runner, 0.06, "Runner-up")]
    for cls, y0, role in rows:
        pth = exp["paths"][str(cls)]["path"]
        tr = exp["paths"][str(cls)]["evidence_trace"]
        ax.text(0.00, y0 + 0.35, f"{role} class {cls}", transform=ax.transAxes, fontsize=9, fontweight="bold")
        n = len(pth)
        step_w = 0.23
        gap = 0.12
        for j, node in enumerate(pth):
            x0 = 0.04 + j * (step_w + gap)
            inset = ax.inset_axes([x0, y0 + 0.05, step_w, 0.25])
            inset.imshow(Xtr[node].reshape(8, 8), cmap="gray_r")
            inset.axis("off")
            ax.text(
                x0 + step_w / 2, y0,
                f"step {j}\n$D={tr[j]:.2f}$",
                transform=ax.transAxes, fontsize=7.3, ha="center", va="top"
            )
            if j < n - 1:
                ax.annotate(
                    "",
                    xy=(x0 + step_w + gap * 0.85, y0 + 0.18),
                    xytext=(x0 + step_w + gap * 0.15, y0 + 0.18),
                    xycoords=ax.transAxes,
                    arrowprops=dict(arrowstyle="->", lw=1.0),
                )

    ax = fig.add_subplot(gs[1, 1])
    vals = effects[top_features][::-1]
    labels = [f"pixel ({k//8},{k%8})" for k in top_features[::-1]]
    y_pos = np.arange(len(vals))
    ax.barh(y_pos, vals)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.axvline(0, linewidth=0.8)
    ax.set_xlabel("Paired intervention effect $I_k(q)$")
    ax.set_title("D  Feature intervention on the fixed winner--runner margin", loc="left", fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    pdf = args.output_dir / "fig_digits_real_xai.pdf"
    png = args.output_dir / "fig_digits_real_xai.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=400, bbox_inches="tight")
    plt.close(fig)

    provenance = {
        "dataset": "sklearn digits",
        "seed": seed,
        "outer_split": "stratified 70/30",
        "outer_random_state": 7000 + seed,
        "query_test_index": query_index,
        "true_label": int(yte[query_index]),
        "explanation": exp,
        "top_intervention_features": [int(k) for k in top_features],
        "top_intervention_effects": [float(effects[k]) for k in top_features],
    }
    (ROOT / "provenance" / "digits_query.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(pdf)
    print(png)


if __name__ == "__main__":
    main()
