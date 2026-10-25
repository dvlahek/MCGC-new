from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "xai"))
from field_only_cgec import FieldOnlyCGEC  # noqa: E402


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    seed = 0
    query_index = 130
    X, y = make_spirals(random_state=seed)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.30, random_state=7000 + seed, stratify=y
    )
    model = FieldOnlyCGEC().fit(Xtr, ytr)
    q = Xte[query_index]
    explanation = model.explain_query(q, top_support=5)

    assert explanation["prediction"] == 0
    assert int(yte[query_index]) == 0

    starts = np.asarray(explanation["query_neighbour_nodes"], dtype=int)
    support = np.asarray(
        [row["node_index"] for row in explanation["supporting_neighbors"]], dtype=int
    )
    support_aff = np.asarray(
        [row["affinity"] for row in explanation["supporting_neighbors"]], dtype=float
    )
    winner_path = explanation["paths"]["0"]["path"]
    runner_path = explanation["paths"]["1"]["path"]
    winner_trace = explanation["paths"]["0"]["evidence_trace"]
    runner_trace = explanation["paths"]["1"]["evidence_trace"]
    tau0 = explanation["paths"]["0"]["threshold"]
    tau1 = explanation["paths"]["1"]["threshold"]
    L0 = explanation["local_evidence"]["0"]
    L1 = explanation["local_evidence"]["1"]

    D0 = model.contrast_field_[:, 0]
    pts = np.vstack([Xtr[winner_path], Xtr[runner_path], Xtr[starts], q[None]])
    ctr = (pts.min(axis=0) + pts.max(axis=0)) / 2
    half = (pts.max(axis=0) - pts.min(axis=0)).max() / 2 + 0.12
    lo, hi = ctr - half, ctr + half
    inside = np.all((Xtr >= lo - 0.1) & (Xtr <= hi + 0.1), axis=1)

    fig = plt.figure(figsize=(13.0, 4.0))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.60, 1.0], wspace=0.42)

    ax = fig.add_subplot(gs[0, 0])
    sc = ax.scatter(
        Xtr[:, 0], Xtr[:, 1], s=8, c=D0, cmap="coolwarm_r", vmin=-1, vmax=1,
        linewidths=0
    )
    ax.add_patch(Rectangle(lo, *(hi - lo), fill=False, linewidth=1.0))
    ax.scatter(q[0], q[1], marker="*", s=85, zorder=5)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(r"A  Class-0 contrast field $D^{(0)}$", loc="left", fontweight="bold", fontsize=11)
    cb = fig.colorbar(sc, ax=ax, orientation="horizontal", fraction=0.06, pad=0.08)
    cb.set_ticks([-1, 0, 1])
    cb.set_ticklabels(["class 1", r"$D^{(0)}=0$", "class 0"])
    cb.ax.tick_params(labelsize=8, length=0)
    cb.outline.set_visible(False)

    ax = fig.add_subplot(gs[0, 1])
    segments = []
    for i in np.flatnonzero(inside):
        for j in model.graph_neighbors_[i]:
            if j > i and inside[j]:
                segments.append([Xtr[i], Xtr[j]])
    if segments:
        ax.add_collection(LineCollection(segments, linewidths=0.35, alpha=0.16))
    ax.scatter(
        Xtr[inside, 0], Xtr[inside, 1], s=22, c=D0[inside], cmap="coolwarm_r",
        vmin=-1, vmax=1, linewidths=0.25, alpha=0.9
    )
    ax.add_collection(
        LineCollection([[q, Xtr[j]] for j in starts], linewidths=0.6, linestyles=":", alpha=0.5)
    )
    ax.scatter(
        Xtr[support, 0], Xtr[support, 1], s=65 + 950 * support_aff,
        facecolors="none", linewidths=1.0, zorder=5
    )

    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    winner_color, runner_color = cycle[0], cycle[1]

    def draw_path(path, color, linestyle, linewidth):
        for u, v in zip(path[:-1], path[1:]):
            ax.add_patch(
                FancyArrowPatch(
                    Xtr[u], Xtr[v], arrowstyle="-|>", mutation_scale=8,
                    linewidth=linewidth, color=color, linestyle=linestyle,
                    shrinkA=2.5, shrinkB=2.5, zorder=7
                )
            )
        ax.scatter(*Xtr[path[0]], s=65, facecolors="white", edgecolors=color, linewidths=1.4, zorder=8)
        ax.scatter(*Xtr[path[-1]], s=58, marker="D", color=color, zorder=8)

    draw_path(winner_path, winner_color, "-", 2.0)
    draw_path(runner_path, runner_color, "--", 1.6)
    ax.scatter(q[0], q[1], marker="*", s=190, zorder=9)
    ax.annotate(r"$q$", q, xytext=(6, -9), textcoords="offset points", fontsize=10, fontweight="bold")
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("B  Query evidence and paths", loc="left", fontweight="bold", fontsize=11)
    ax.text(
        0.015, 0.985,
        rf"$L_0(q)={L0:+.3f}$" + "\n"
        + rf"$L_1(q)={L1:+.3f}$" + "\n"
        + rf"$\hat y=0,\ M(q)={L0-L1:.3f}$",
        transform=ax.transAxes, va="top", fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.3", fill=False, linewidth=0.8)
    )
    handles = [
        Line2D([], [], marker="*", linestyle="", markersize=9, label=r"query $q$"),
        Line2D([], [], marker="o", linestyle="", markersize=7, markerfacecolor="none", label=r"top-5 support ($\propto a_j$)"),
        Line2D([], [], color=winner_color, linewidth=2.0, label="winner path (class 0)"),
        Line2D([], [], color=runner_color, linewidth=1.6, linestyle="--", label="runner-up path (class 1)"),
        Line2D([], [], marker="o", linestyle="", markersize=6, markerfacecolor="white", label=r"entry node $v_0$"),
        Line2D([], [], marker="D", linestyle="", markersize=5, label="path end"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=3,
              fontsize=7, frameon=False, handlelength=1.8, columnspacing=0.8)

    ax = fig.add_subplot(gs[0, 2])
    wline, = ax.plot(np.arange(len(winner_trace)), winner_trace, marker="o", linewidth=1.8, label="class 0 path")
    rline, = ax.plot(np.arange(len(runner_trace)), runner_trace, marker="o", linestyle="--", linewidth=1.5, label="class 1 path")
    ax.axhline(tau0, linestyle=":", linewidth=1.0, color=wline.get_color())
    ax.axhline(tau1, linestyle="-.", linewidth=1.0, color=rline.get_color())
    ax.text(0.98, 0.54, rf"$\tau_0={tau0:.4f}$" + "\n" + rf"$\tau_1={tau1:.4f}$",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8)
    ax.annotate(r"reaches $\tau_0$", (len(winner_trace) - 1, winner_trace[-1]),
                xytext=(7, -4), textcoords="offset points", fontsize=8, va="center")
    ax.annotate(r"terminates below $\tau_1$", (len(runner_trace) - 1, runner_trace[-1]),
                xytext=(-3, -11), textcoords="offset points", fontsize=8, ha="right", va="top")
    ax.axhline(0, linewidth=0.6, alpha=0.55)
    ax.set_xlim(-0.3, max(len(winner_trace), len(runner_trace)) - 0.55)
    ax.set_ylim(-0.02, 0.52)
    ax.set_xlabel(r"path step $r$")
    ax.set_ylabel(r"own-class contrast $D^{(c)}_{v_r}$")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title("C  Evidence along paths", loc="left", fontweight="bold", fontsize=11)
    ax.legend(fontsize=7, frameon=False, loc="lower right")

    fig.subplots_adjust(bottom=0.18)
    pdf = args.output_dir / "fig_spiral_evidence.pdf"
    png = args.output_dir / "fig_spiral_evidence.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=400, bbox_inches="tight")
    plt.close(fig)

    runtime_provenance = {
        "dataset": "noisy spirals (spirals_hard)",
        "seed": seed,
        "outer_split": "stratified 70/30",
        "outer_random_state": 7000 + seed,
        "query_test_index": query_index,
        "true_label": int(yte[query_index]),
        "prediction": explanation["prediction"],
        "runner_up": explanation["runner_up"],
        "local_evidence": explanation["local_evidence"],
        "margin": explanation["margin"],
        "paths": explanation["paths"],
        "supporting_neighbors": explanation["supporting_neighbors"],
    }
    (args.output_dir / "spiral_query_runtime.json").write_text(json.dumps(runtime_provenance, indent=2) + "\n")
    print(pdf)
    print(png)


if __name__ == "__main__":
    main()
