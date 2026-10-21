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


def top_by_abs(mapping, n=10):
    return sorted(mapping.items(), key=lambda kv: abs(float(kv[1])), reverse=True)[:n]


def panel_a(ax, p):
    local = p["local"]
    rows = top_by_abs(local["feature_effect"], 10)
    names = [k for k, _ in rows][::-1]
    vals = [float(v) for _, v in rows][::-1]
    colors = [WINNER if v >= 0 else RUNNER for v in vals]
    y = np.arange(len(names))
    ax.barh(y, vals, color=colors, alpha=0.88)
    ax.axvline(0.0, color="0.35", linewidth=0.9)
    ax.set_yticks(y, names)
    ax.set_xlabel("change in winner-runner margin after feature removal")
    ax.set_title("a  Local feature effect on one decision", loc="left", fontsize=13, fontweight="bold")
    span = max(max(abs(v) for v in vals), 1e-6)
    ax.set_xlim(min(vals) - 0.15 * span, max(vals) + 0.22 * span)
    for yi, v in zip(y, vals):
        pad = 0.025 * span
        ax.text(v + (pad if v >= 0 else -pad), yi, f"{v:+.3f}", va="center", ha="left" if v >= 0 else "right", fontsize=8.5)
    ax.grid(axis="x", alpha=0.18)
    ax.text(
        0.02, 0.02,
        f"winner={local['winner']}  runner-up={local['runner_up']}  original margin={local['baseline_margin']:.3f}",
        transform=ax.transAxes, ha="left", va="bottom", fontsize=8.3,
    )


def _mean_path_share(p):
    edges = p["local"]["path_geometry"]
    feature_names = p["feature_names"]
    mean_share = {
        name: float(np.mean([e["feature_share"][name] for e in edges]))
        for name in feature_names
    }
    return edges, mean_share


def panel_b(ax, p):
    edges, mean_share = _mean_path_share(p)
    if not edges:
        ax.text(0.5, 0.5, "No nontrivial path geometry", ha="center", va="center")
        ax.axis("off")
        return

    top = [k for k, _ in sorted(mean_share.items(), key=lambda kv: kv[1], reverse=True)[:8]]

    # A one-edge representative path is more legible as a ranked bar plot than
    # as a one-row heat map. Multi-edge paths retain the edge-by-feature heat map.
    if len(edges) == 1:
        vals = np.asarray([float(edges[0]["feature_share"][name]) for name in top][::-1])
        names = top[::-1]
        y = np.arange(len(names))
        ax.barh(y, vals, color=WINNER, alpha=0.84)
        ax.set_yticks(y, names)
        ax.set_xlabel("share of standardized squared edge distance")
        ax.set_title("b  Feature composition of representative-path geometry", loc="left", fontsize=13, fontweight="bold")
        ax.grid(axis="x", alpha=0.18)
        for yi, v in zip(y, vals):
            ax.text(v + 0.006, yi, f"{v:.2f}", va="center", ha="left", fontsize=8.2)
        ax.text(0.98, 0.02, "representative path: 1 edge", transform=ax.transAxes, ha="right", va="bottom", fontsize=8.3)
        return

    matrix = np.asarray([[e["feature_share"][name] for name in top] for e in edges], dtype=float)
    im = ax.imshow(matrix, aspect="auto", cmap="Blues", vmin=0.0, vmax=max(float(matrix.max()), 1e-6))
    ax.set_xticks(np.arange(len(top)), top, rotation=38, ha="right")
    ax.set_yticks(np.arange(len(edges)), [f"edge {e['step']}" for e in edges])
    ax.set_title("b  Feature composition of representative-path geometry", loc="left", fontsize=13, fontweight="bold")
    ax.set_xlabel("feature contribution to standardized squared edge distance")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i, j] >= 0.08:
                ax.text(j, i, f"{matrix[i,j]:.2f}", ha="center", va="center", fontsize=7.5)
    cb = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("feature share")


def panel_c(ax, p):
    g = p["global"]
    rows = sorted(g["mean_abs_effect"].items(), key=lambda kv: float(kv[1]), reverse=True)[:10]
    names = [k for k, _ in rows][::-1]
    vals = np.asarray([float(v) for _, v in rows][::-1])
    ci = np.asarray([float(g["descriptive_95ci_halfwidth"][name]) for name in names])
    y = np.arange(len(names))
    ax.barh(y, vals, xerr=ci, color=WINNER, alpha=0.84, capsize=3)
    ax.set_yticks(y, names)
    ax.set_xlabel("mean absolute winner-runner margin effect")
    ax.set_title("c  Global decision-feature importance", loc="left", fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.18)
    n = len(p["global_runs"])
    note = f"{n} outer split" if n == 1 else f"{n} outer splits"
    ax.text(0.98, 0.02, note, transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5)
    if n == 1:
        ax.text(
            0.02, 0.98,
            "preview only — no across-seed uncertainty",
            transform=ax.transAxes,
            ha="left", va="top", fontsize=8.3,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "0.82"},
        )


def make_figure(p, png: Path, pdf: Path):
    fig = plt.figure(figsize=(16.8, 6.2))
    gs = GridSpec(1, 3, figure=fig, width_ratios=(1.15, 1.05, 1.15), wspace=0.58)
    panel_a(fig.add_subplot(gs[0, 0]), p)
    panel_b(fig.add_subplot(gs[0, 1]), p)
    panel_c(fig.add_subplot(gs[0, 2]), p)
    fig.suptitle("Native feature attribution in MCGC", fontsize=18, fontweight="bold", y=0.985)
    fig.subplots_adjust(top=0.87, bottom=0.14, left=0.10, right=0.985)
    png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path, required=True)
    ap.add_argument("--output-prefix", type=Path, default=Path("figures/fig5_feature_attribution_publication"))
    args = ap.parse_args()
    p = load_payload(args.json)
    png = args.output_prefix.with_suffix(".png")
    pdf = args.output_prefix.with_suffix(".pdf")
    make_figure(p, png, pdf)
    print(f"WROTE {png}")
    print(f"WROTE {pdf}")


if __name__ == "__main__":
    main()
