from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "xai"))
from field_only_cgec import FieldOnlyCGEC  # noqa: E402
from fig_spiral_evidence import make_spirals  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    X, y = make_spirals(random_state=0)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.30, random_state=7000, stratify=y
    )
    model = FieldOnlyCGEC().fit(Xtr, ytr)
    q = Xte[130]
    rec = model.margin_contributions(q)

    nodes = np.asarray(rec["nodes"], dtype=int)
    affinities = np.asarray(rec["affinities"], dtype=float)
    C = np.asarray(rec["contributions"], dtype=float)
    margin = float(rec["margin"])
    order = np.argsort(-affinities)
    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    cumulative = 0.0
    for i, pos in enumerate(order):
        lab = int(ytr[nodes[pos]])
        c = float(C[pos])
        ax.bar(
            i, c, bottom=cumulative, width=0.72,
            color=cycle[lab % len(cycle)], hatch=None if c >= 0 else "////"
        )
        if i < len(order) - 1:
            ax.plot([i + 0.36, i + 0.64], [cumulative + c, cumulative + c], linewidth=0.6)
        cumulative += c
    ax.bar(len(order) + 0.4, margin, width=0.72, color="0.25")
    ax.text(len(order) + 0.4, margin + 0.008, f"{margin:.3f}", ha="center", fontsize=9)
    ax.axhline(0, linewidth=0.7)
    ax.set_xticks(list(range(len(order))) + [len(order) + 0.4])
    ax.set_xticklabels([f"{affinities[p]:.2f}" for p in order] + ["$M(q)$"], rotation=90, fontsize=8)
    ax.set_xlabel(r"neighbours ordered by affinity $a_j(q)$")
    ax.set_ylabel("cumulative winner--runner margin")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        handles=[
            Patch(color=cycle[0], label="training label 0"),
            Patch(color=cycle[1], label="training label 1"),
            Patch(facecolor="0.7", hatch="////", label="hatched: lowers $M(q)$"),
        ],
        frameon=False, fontsize=8, loc="upper left"
    )
    ax.set_title(r"Supplementary: exact neighbour decomposition $M(q)=\sum_j C_j(q)$", loc="left", fontweight="bold")

    pdf = args.output_dir / "fig_spiral_waterfall_supplement.pdf"
    png = args.output_dir / "fig_spiral_waterfall_supplement.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(pdf)
    print(png)


if __name__ == "__main__":
    main()
