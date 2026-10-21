from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

WINNER = "tab:blue"
RUNNER = "tab:orange"


def load_payload(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def show_digit(ax, image, border="0.6", title=None, title_size=8.5):
    ax.imshow(np.asarray(image), cmap="gray_r", interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(1.4)
        spine.set_edgecolor(border)
    if title:
        ax.set_title(title, fontsize=title_size, pad=3)


def panel_a(ax, p):
    ax.axis("off")
    s = p["selected"]
    ax.text(0.0, 1.0, "A  Query and decision", fontsize=14, fontweight="bold", va="top")
    q = ax.inset_axes([0.00, 0.18, 0.43, 0.68])
    show_digit(q, s["query_image"], border=WINNER)
    lines = [
        (f"True label: {s['true_label']}", "black"),
        (f"Predicted: {s['prediction']}", WINNER),
        (f"Runner-up: {s['runner_up']}", RUNNER),
        (f"Winner-runner margin: {s['margin']:.3f}", "black"),
        (f"Pair median: {s['pair_median_margin']:.3f}", "black"),
        ("Nontrivial winner path: yes", "black"),
    ]
    y = 0.80
    for text, color in lines:
        ax.text(0.49, y, text, fontsize=10.5, color=color, va="top")
        y -= 0.105


def panel_b(ax, p):
    ax.axis("off")
    ax.text(0.0, 1.0, "B  Local query neighbourhood", fontsize=14, fontweight="bold", va="top")
    ax.text(0.0, 0.91, "Affinity-ranked training neighbours", fontsize=9.2, va="top")
    winner = str(p["selected"]["prediction"])
    runner = str(p["selected"]["runner_up"])
    rows = p["local_supporting_neighbors"]
    n = len(rows)
    gap = 0.018
    left0 = 0.0
    width = (1.0 - gap * (n - 1)) / n
    for i, row in enumerate(rows):
        left = left0 + i * (width + gap)
        d = ax.inset_axes([left, 0.12, width, 0.62])
        border = WINNER if int(row["training_label"]) == int(winner) else RUNNER
        show_digit(d, row["image"], border=border)
        ev = row["contrastive_evidence"]
        ax.text(
            left + width / 2,
            0.80,
            f"y={row['training_label']}  aff={row['affinity']:.2f}\nD{winner}={ev[winner]:+.2f}, D{runner}={ev[runner]:+.2f}",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=7.7,
        )


def draw_trail_row(ax, trail, y0, label, cls, color):
    ax.text(0.0, y0 + 0.17, f"{label} class {cls}", color=color, fontsize=11, fontweight="bold", va="center")
    n = min(len(trail["path"]), 5)
    x0 = 0.16
    w = 0.095
    gap = 0.055
    for i in range(n):
        left = x0 + i * (w + gap)
        d = ax.inset_axes([left, y0, w, 0.29])
        show_digit(d, trail["node_images"][i], border=color)
        ax.text(left + w / 2, y0 + 0.31, f"step {i}\ny={trail['node_labels'][i]}, D={trail['evidence_trace'][i]:+.2f}", ha="center", va="bottom", fontsize=8.0, transform=ax.transAxes)
        if i < n - 1:
            ax.annotate("", xy=(left + w + gap*0.75, y0 + 0.14), xytext=(left + w + gap*0.15, y0 + 0.14), xycoords=ax.transAxes, arrowprops={"arrowstyle":"->", "color":color, "lw":1.6})
    if n == 1:
        ax.text(x0 + w + 0.06, y0 + 0.14, "no further positive evidence ascent", color=color, fontsize=9.2, va="center", transform=ax.transAxes)


def panel_c(ax, p):
    ax.axis("off")
    ax.text(0.0, 1.0, "C  Representative evidence trails", fontsize=14, fontweight="bold", va="top")
    draw_trail_row(ax, p["winner_trail"], 0.49, "Winner", p["selected"]["prediction"], WINNER)
    draw_trail_row(ax, p["runner_up_trail"], 0.05, "Runner-up", p["selected"]["runner_up"], RUNNER)


def panel_d(ax, p):
    wt = p["winner_trail"]["evidence_trace"]
    rt = p["runner_up_trail"]["evidence_trace"]
    ax.plot(range(len(wt)), wt, marker="o", color=WINNER, label=f"Winner field: {p['selected']['prediction']}")
    ax.plot(range(len(rt)), rt, marker="s", color=RUNNER, label=f"Runner-up field: {p['selected']['runner_up']}")
    ax.axhline(0.0, linestyle="--", linewidth=0.9, color="0.6")
    ax.set_title("D  Evidence along representative trails", loc="left", fontsize=13, fontweight="bold")
    ax.set_xlabel("Representative-trail step")
    ax.set_ylabel("Contrastive evidence")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="y", alpha=0.18)


def panel_e(ax, p):
    c = p["contrastive"]
    d = c["component_deltas"]
    labels = ["Local", "Trail", "Agreement", "Lost"]
    vals = [d["local_component_delta"], d["trail_component_delta"], d["agreement_component_delta"], d["lost_component_delta"]]
    colors = [WINNER if v >= 0 else RUNNER for v in vals]
    y = np.arange(4)
    ax.barh(y, vals, color=colors, alpha=0.85)
    ax.axvline(0.0, linewidth=0.9, color="0.35")
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_title("E  Exact contrastive score decomposition", loc="left", fontsize=13, fontweight="bold")
    ax.set_xlabel("Contribution to winner - runner score margin")
    for yi, v in zip(y, vals):
        if v >= 0:
            ax.text(v + 0.012, yi, f"{v:+.3f}", va="center", ha="left", fontsize=9)
        else:
            ax.text(v - 0.012, yi, f"{v:+.3f}", va="center", ha="right", fontsize=9)
    ax.text(0.98, 0.05, f"score margin = {c['score_margin']:.3f}\nreconstructed = {c['reconstructed_margin']:.3f}\n|error| = {c['reconstruction_abs_error']:.1e}", transform=ax.transAxes, ha="right", va="bottom", fontsize=9, bbox={"boxstyle":"round,pad=0.3", "facecolor":"white", "edgecolor":"0.8"})
    ax.grid(axis="x", alpha=0.18)


def make_figure(p, out_png: Path, out_pdf: Path):
    fig = plt.figure(figsize=(13.2, 8.5))
    gs = GridSpec(3, 2, figure=fig, height_ratios=(0.98, 0.95, 1.05), width_ratios=(1.0, 1.55), hspace=0.30, wspace=0.30)
    panel_a(fig.add_subplot(gs[0,0]), p)
    panel_b(fig.add_subplot(gs[0,1]), p)
    panel_c(fig.add_subplot(gs[1,:]), p)
    panel_d(fig.add_subplot(gs[2,0]), p)
    panel_e(fig.add_subplot(gs[2,1]), p)
    s = p["selected"]
    fig.suptitle(f"Real-data MCGC explanation on digits: why {s['prediction']} rather than {s['runner_up']}", fontsize=17, fontweight="bold", y=0.988)
    fig.subplots_adjust(top=0.93, bottom=0.10, left=0.06, right=0.985)
    fig.text(0.5, 0.035, "Representative trails summarize the fitted class-conditioned evidence fields; they are not claimed to be the sole stochastic causal trajectory.", ha="center", fontsize=8.8)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path, required=True)
    ap.add_argument("--output-prefix", type=Path, default=Path("figures/fig2_digits_real_xai_publication"))
    args = ap.parse_args()
    p = load_payload(args.json)
    make_figure(p, args.output_prefix.with_suffix(".png"), args.output_prefix.with_suffix(".pdf"))
    print(f"WROTE {args.output_prefix.with_suffix('.png')}")
    print(f"WROTE {args.output_prefix.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
