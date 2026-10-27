from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    inputs = json.loads((ROOT / "results" / "figure_inputs.json").read_text())
    pv = inputs["path_validation"]
    full = pv["full_path_advantage"]
    tail = pv["tail_path_advantage"]
    same = pv["same_start_tail_advantage"]
    matched = pv["matched_distance_label"]
    signed = pv["signed_winner_support"]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), gridspec_kw={"wspace": 0.38})

    ax = axes[0]
    values = [full["mean"], tail["mean"], same["mean"]]
    ys = [2, 1, 0]
    ax.axvline(0, linewidth=1.0)
    for y, v in zip(ys, values):
        ax.plot([0, v], [y, y], linewidth=2.0)
        ax.plot(v, y, marker="o", markersize=7)
    ax.set_yticks(ys)
    ax.set_yticklabels(["Full path", "Tail\n(start omitted)", "Same entry point\n→ same tail"])
    ax.set_xlabel("Within-query overlap advantage (Jaccard)")
    ax.set_title("(a) Query-selected path stability", loc="left", fontweight="bold")
    ax.grid(axis="x", alpha=0.22)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(-0.035, 0.83)
    ax.set_ylim(-0.45, 2.45)
    ax.text(0.56, 2.12, f"+{values[0]:.3f}", fontsize=11, fontweight="bold", ha="left", va="center")
    ax.text(0.56, 1.88, f"17/1 positive/loss\n$p={full['sign_test_p_two_sided']:.2e}$", fontsize=9.5, ha="left", va="center")
    ax.text(0.55, 1.12, f"+{values[1]:.3f}", fontsize=11, fontweight="bold", ha="left", va="center")
    ax.text(0.55, 0.88, f"16/1/1 positive/tie/loss\n$p={tail['sign_test_p_two_sided']:.2e}$", fontsize=9.5, ha="left", va="center")
    ax.text(0.055, 0.10, "0.000", fontsize=11, fontweight="bold", ha="left", va="center")
    ax.text(0.055, -0.14, "14/14 ties", fontsize=9.5, ha="left", va="center")

    ax = axes[1]
    values = [matched["mean"], signed["mean"]]
    ys = [1, 0]
    ax.axvline(0, linewidth=1.0)
    for y, v in zip(ys, values):
        ax.plot([0, v], [y, y], linewidth=2.0)
        ax.plot(v, y, marker="o", markersize=7)
    ax.set_yticks(ys)
    ax.set_yticklabels(["Matched off-path", "Signed local evidence"])
    ax.set_xlabel("Path minus control margin drop")
    ax.set_title("(b) Delete-and-refit path role", loc="left", fontweight="bold")
    ax.grid(axis="x", alpha=0.22)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(-0.105, 0.105)
    ax.set_ylim(-0.38, 1.38)
    ax.text(0.058, 1.12, f"+{values[0]:.4f}", fontsize=11, fontweight="bold", ha="left", va="center")
    ax.text(0.058, 0.88, f"20/0/0\n$p={matched['sign_test_p_two_sided']:.2e}$", fontsize=9.5, ha="left", va="center")
    ax.text(-0.096, 0.14, f"{values[1]:.4f}", fontsize=11, fontweight="bold", ha="left", va="center")
    ax.text(-0.096, -0.10, f"1/0/19\n$p={signed['sign_test_p_two_sided']:.2e}$", fontsize=9.5, ha="left", va="center")

    pdf = args.output_dir / "fig_path_validation.pdf"
    png = args.output_dir / "fig_path_validation.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(pdf)
    print(png)


if __name__ == "__main__":
    main()
