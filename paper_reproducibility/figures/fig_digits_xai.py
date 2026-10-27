from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "xai"))
from field_only_cgec import FieldOnlyCGEC  # noqa: E402


def image_axis(ax, values, title=None):
    ax.imshow(np.asarray(values).reshape(8, 8), cmap="gray_r", vmin=0, vmax=16, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=8, pad=2)


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
    margin = exp["margin"]
    contrib = exp["neighbour_margin_contributions"]
    effects = np.asarray(exp["feature_interventions"]["effects"], dtype=float)
    L = np.asarray(contrib["local_evidence_vector"], dtype=float)
    nodes = np.asarray(contrib["nodes"], dtype=int)
    affinities = np.asarray(contrib["affinities"], dtype=float)
    C = np.asarray(contrib["contributions"], dtype=float)
    assert np.isclose(C.sum(), margin, rtol=1e-11, atol=1e-12)

    fig = plt.figure(figsize=(12.6, 8.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.05], width_ratios=[1.0, 1.55], hspace=0.52, wspace=0.30)

    outer = fig.add_subplot(gs[0, 0])
    outer.axis("off")
    outer.set_title("(a) Query and local class evidence", loc="left", fontweight="bold")
    qax = outer.inset_axes([0.02, 0.12, 0.37, 0.72])
    image_axis(qax, q)
    qax.text(0.5, -0.08, f"query $q$\ntrue label {int(yte[query_index])}", transform=qax.transAxes,
             ha="center", va="top", fontsize=8)
    lax = outer.inset_axes([0.47, 0.10, 0.51, 0.78])
    order = np.argsort(-L)
    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    bar_colors = [cycle[0] if i == model.class_to_index_[winner] else cycle[1] if i == model.class_to_index_[runner] else "0.75" for i in order]
    lax.barh(np.arange(len(order))[::-1], L[order], color=bar_colors, height=0.68)
    lax.set_yticks(np.arange(len(order))[::-1])
    lax.set_yticklabels([str(model.classes_[i]) for i in order], fontsize=7)
    lax.axvline(0, linewidth=0.7)
    lax.set_xlabel(r"local class evidence $L_c(q)$", fontsize=8)
    lax.tick_params(axis="x", labelsize=7)
    lax.spines["top"].set_visible(False)
    lax.spines["right"].set_visible(False)
    lax.text(1.0, 1.02, rf"$\hat y={winner}$, runner-up {runner}" + "\n" + rf"$M(q)={margin:.3f}$",
             transform=lax.transAxes, ha="right", va="bottom", fontsize=8)

    ax = fig.add_subplot(gs[0, 1])
    ax.set_title(r"(b) Exact margin decomposition over neighbours, $M(q)=\sum_j C_j(q)$", loc="left", fontweight="bold")
    affinity_order = np.argsort(-affinities)
    cumulative = 0.0
    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, pos in enumerate(affinity_order):
        c = C[pos]
        ax.bar(i, c, bottom=cumulative, width=0.72, color=cycle[0] if c >= 0 else cycle[1])
        if i < len(affinity_order) - 1:
            ax.plot([i + 0.36, i + 0.64], [cumulative + c, cumulative + c], linewidth=0.6)
        cumulative += c
    ax.bar(len(affinity_order) + 0.4, margin, width=0.72, color="0.25")
    ax.text(len(affinity_order) + 0.4, margin + 0.012, f"{margin:.3f}", ha="center", fontsize=8)
    ax.axhline(0, linewidth=0.7)
    ax.set_ylabel("cumulative winner--runner margin")
    ax.set_xticks(list(range(len(affinity_order))) + [len(affinity_order) + 0.4])
    neighbour_labels = [f"{ytr[nodes[p]]}\n{affinities[p]:.2f}" for p in affinity_order]
    ax.set_xticklabels(neighbour_labels + ["$M(q)$"], fontsize=7)
    ax.set_xlabel("training label and query affinity $a_j(q)$")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    outer = fig.add_subplot(gs[1, 0])
    outer.axis("off")
    outer.set_title(r"(c) Paired feature interventions $I_k(q)$", loc="left", fontweight="bold")
    qax = outer.inset_axes([0.02, 0.13, 0.40, 0.68])
    image_axis(qax, q, "query")
    hax = outer.inset_axes([0.48, 0.13, 0.40, 0.68])
    lim = float(np.max(np.abs(effects)))
    hm = hax.imshow(effects.reshape(8, 8), cmap="coolwarm_r", vmin=-lim, vmax=lim, interpolation="nearest")
    hax.contour(q.reshape(8, 8), levels=[8], linewidths=0.45, alpha=0.45)
    hax.set_xticks([])
    hax.set_yticks([])
    hax.set_title(r"$I_k(q)$ per pixel", fontsize=8, pad=2)
    cax = outer.inset_axes([0.91, 0.16, 0.035, 0.62])
    cb = fig.colorbar(hm, cax=cax)
    cb.ax.tick_params(labelsize=7)
    cb.ax.text(0.5, 1.04, f"favours {winner}", transform=cb.ax.transAxes, fontsize=7, ha="center")
    cb.ax.text(0.5, -0.05, f"favours {runner}", transform=cb.ax.transAxes, fontsize=7, ha="center", va="top")

    ax = fig.add_subplot(gs[1, 1])
    ax.axis("off")
    ax.set_title("(d) Representative evidence paths", loc="left", fontweight="bold")
    rows = [(winner, 0.57, "winner"), (runner, 0.08, "runner-up")]
    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for row_idx, (cls, y0, role) in enumerate(rows):
        p = exp["paths"][str(cls)]["path"]
        trace = exp["paths"][str(cls)]["evidence_trace"]
        threshold = exp["paths"][str(cls)]["threshold"]
        reason = exp["paths"][str(cls)]["termination_reason"]
        color = cycle[row_idx]
        ax.text(0.00, y0 + 0.31, f"{role} class {cls}", transform=ax.transAxes, fontsize=9, fontweight="bold", color=color)
        step_w = 0.14
        gap = 0.055
        for j, node in enumerate(p):
            x0 = 0.03 + j * (step_w + gap)
            iax = ax.inset_axes([x0, y0 + 0.06, step_w, 0.20])
            image_axis(iax, Xtr[node])
            for spine in iax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(1.2)
            reached = trace[j] >= threshold
            ax.text(x0 + step_w / 2, y0, f"step {j}\n$D={trace[j]:.2f}$" + ("  ✓" if reached else ""),
                    transform=ax.transAxes, fontsize=7, ha="center", va="top")
            if j < len(p) - 1:
                ax.add_patch(FancyArrowPatch(
                    (x0 + step_w + 0.004, y0 + 0.16),
                    (x0 + step_w + gap - 0.004, y0 + 0.16),
                    transform=ax.transAxes, arrowstyle="-|>", mutation_scale=8,
                    color=color, linewidth=1.0
                ))
        xe = min(0.97, 0.03 + len(p) * (step_w + gap) + 0.01)
        end_text = rf"$\tau_{{{cls}}}={threshold:.2f}$" + "\n"
        end_text += "threshold reached" if reason == "threshold_reached" else reason.replace("_", " ")
        ax.text(xe, y0 + 0.16, end_text, transform=ax.transAxes, fontsize=7.2, va="center")

    pdf = args.output_dir / "fig_digits_real_xai.pdf"
    png = args.output_dir / "fig_digits_real_xai.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=400, bbox_inches="tight")
    plt.close(fig)

    runtime_provenance = {
        "dataset": "sklearn digits",
        "seed": seed,
        "outer_split": "stratified 70/30",
        "outer_random_state": 7000 + seed,
        "query_test_index": query_index,
        "true_label": int(yte[query_index]),
        "explanation": exp,
    }
    (args.output_dir / "digits_query_runtime.json").write_text(json.dumps(runtime_provenance, indent=2) + "\n")
    print(pdf)
    print(png)


if __name__ == "__main__":
    main()
