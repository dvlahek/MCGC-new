from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DATASETS = ("moons_hard", "spirals_hard", "overlap_hd", "breast_cancer")
LABELS = {
    "moons_hard": "Moons",
    "spirals_hard": "Spirals",
    "overlap_hd": "Overlap-HD",
    "breast_cancer": "Breast cancer",
}
WINNER = "tab:blue"
CONTROL = "0.68"
MEAN = "0.12"


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def extract(payload):
    rows = []
    for dataset in DATASETS:
        for run in payload["datasets"][dataset]["runs"]:
            s = run["summary"]
            rows.append(
                {
                    "dataset": dataset,
                    "seed": int(run["seed"]),
                    "matched": float(s["faithfulness_advantage"]["mean"]),
                    "nearest": float(s["path_vs_nearest_advantage"]["mean"]),
                }
            )
    if len(rows) != 20:
        raise RuntimeError(f"Expected 20 dataset×seed units, found {len(rows)}")
    return rows


def dot_panel(ax, rows, key, title):
    y_base = np.arange(len(DATASETS))[::-1].astype(float)
    jitter = np.linspace(-0.13, 0.13, 5)
    ax.axvline(0.0, color="0.25", linestyle="--", linewidth=1.0)

    for yi, dataset in zip(y_base, DATASETS):
        vals = [r[key] for r in rows if r["dataset"] == dataset]
        for j, value in enumerate(vals):
            ax.scatter(value, yi + jitter[j], s=38, color=WINNER, alpha=0.80, zorder=3)
        mean = float(np.mean(vals))
        sem = float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
        ax.errorbar(
            mean,
            yi,
            xerr=1.96 * sem,
            fmt="D",
            color=MEAN,
            markerfacecolor="white",
            markeredgewidth=1.5,
            markersize=7,
            capsize=3,
            linewidth=1.5,
            zorder=4,
        )

    ax.set_yticks(y_base, [LABELS[d] for d in DATASETS])
    ax.set_xlabel("margin-drop advantage of path deletion")
    ax.set_title(title, loc="left", fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.18)


def make_figure(rows, out_png: Path, out_pdf: Path):
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8), sharey=True)
    dot_panel(axes[0], rows, "matched", "a  Path-selected nodes vs matched off-path controls")
    dot_panel(axes[1], rows, "nearest", "b  Path-selected nodes vs nearest off-path controls")

    fig.suptitle(
        "Interventional faithfulness of MCGC path explanations",
        fontsize=16,
        fontweight="bold",
        y=0.99,
    )
    fig.text(
        0.5,
        0.015,
        "Each dot is one dataset × seed unit; diamonds show the dataset mean with descriptive 95% CI. Positive values mean path-node deletion changes the fitted winner–runner margin more strongly.",
        ha="center",
        fontsize=9,
    )
    fig.subplots_adjust(top=0.84, bottom=0.18, left=0.12, right=0.98, wspace=0.16)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v09-json", type=Path, required=True)
    ap.add_argument("--output-prefix", type=Path, default=Path("figures/fig3_faithfulness_publication"))
    args = ap.parse_args()
    rows = extract(load_json(args.v09_json))
    make_figure(rows, args.output_prefix.with_suffix(".png"), args.output_prefix.with_suffix(".pdf"))
    print(f"WROTE {args.output_prefix.with_suffix('.png')}")
    print(f"WROTE {args.output_prefix.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
