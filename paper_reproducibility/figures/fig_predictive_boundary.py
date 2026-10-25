from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

BASELINES = [
    ("scaled_tuned_knn_confidence", "Scaled kNN"),
    ("scaled_tuned_label_spreading_confidence", "Label Spreading"),
    ("tuned_random_forest_confidence", "Random forest"),
    ("tuned_rbf_svm_confidence", "RBF-SVM"),
]
METRICS = [
    ("eaurc_advantage_field_over_baseline", r"E-AURC advantage ($\times 10^{-3}$)", 1000.0),
    ("aurc_advantage_field_over_baseline", r"AURC advantage ($\times 10^{-3}$)", 1000.0),
    ("error_detection_auc_advantage_field_over_baseline", "Error-detection AUC advantage (pp)", 100.0),
]


def draw_group(ax, summary, y_offset, title, metric, scale):
    for i, (key, _) in enumerate(BASELINES):
        s = summary[key][metric]
        y = y_offset - i
        mean = s["mean"] * scale
        ci = 1.96 * s["std"] / np.sqrt(s["n"]) * scale
        ax.errorbar(mean, y, xerr=ci, marker="o", capsize=2)
        star = " *" if s["sign_test_p_two_sided"] < 0.05 else ""
        ax.text(1.01, y, f"{s['wins']}/{s['ties']}/{s['losses']}{star}", transform=ax.get_yaxis_transform(), va="center", fontsize=8)
    ax.text(0.0, y_offset + 0.55, title, transform=ax.get_yaxis_transform(), fontsize=9, fontweight="bold")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = json.loads((ROOT / "results" / "figure_inputs.json").read_text())["predictive"]

    fig, axes = plt.subplots(3, 1, figsize=(8.2, 8.6), sharey=True)
    for ax, (metric, xlabel, scale) in zip(axes, METRICS):
        ax.axvline(0, linewidth=0.8)
        draw_group(ax, data["nonlinear_primary"], 4, "Nonlinear primary (36 units)", metric, scale)
        draw_group(ax, data["real_primary"], -1, "Real data: digits + wine (24 units)", metric, scale)
        ax.set_yticks([4, 3, 2, 1, -1, -2, -3, -4])
        ax.set_yticklabels([b[1] for b in BASELINES] + [b[1] for b in BASELINES])
        ax.set_xlabel(xlabel)
        ax.grid(axis="x", alpha=0.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_ylim(-4.7, 5.1)
    fig.tight_layout()
    pdf = args.output_dir / "fig_predictive_boundary.pdf"
    png = args.output_dir / "fig_predictive_boundary.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(pdf)
    print(png)


if __name__ == "__main__":
    main()
