from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DATASETS = ("digits", "wine", "spirals_hard", "breast_cancer")
LABELS = {
    "digits": "digits",
    "wine": "wine",
    "spirals_hard": "spirals hard",
    "breast_cancer": "breast cancer",
}
PERTURBED = "tab:blue"
CONTROL = "0.65"
ELIGIBLE = "#e6612b"
EPS_COLORS = ("#2c7fb8", "#6a51a3", "#1b9e77", "#e6612b")


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def mean_or_nan(values):
    values = [float(v) for v in values if v is not None]
    return np.nan if not values else float(np.mean(values))


def panel_a(ax, payload):
    rows = []
    for dataset in DATASETS:
        runs = payload["datasets"][dataset]["runs"]
        for metric in ("support", "path"):
            rows.append((dataset, metric, runs))

    y = np.arange(len(rows))[::-1].astype(float)
    for yi, (dataset, metric, runs) in zip(y, rows):
        within_key = f"mean_within_{metric}_jaccard"
        cross_key = f"mean_cross_{metric}_jaccard"
        for run in runs:
            within = run["stability"].get(within_key)
            cross = run["stability"].get(cross_key)
            if within is None or cross is None:
                continue
            ax.plot([cross, within], [yi, yi], color="0.88", linewidth=1.4, zorder=1)
            ax.scatter(cross, yi, s=34, color=CONTROL, zorder=2)
            ax.scatter(within, yi, s=38, color=PERTURBED, zorder=3)

    labels = [f"{LABELS[d]} · {m}" for d, m, _ in rows]
    ax.set_yticks(y, labels)
    ax.set_xlim(-0.02, 1.04)
    ax.set_xlabel("mean Jaccard similarity to the original explanation")
    ax.set_title(
        "a  Explanations remain query-specific\n     (one pair = one dataset × seed)",
        loc="left",
        fontsize=13,
        fontweight="bold",
    )
    ax.grid(axis="x", alpha=0.18)
    ax.scatter([], [], color=PERTURBED, s=38, label="same query, perturbed (prediction kept)")
    ax.scatter([], [], color=CONTROL, s=34, label="other query, same predicted class")
    ax.legend(frameon=False, fontsize=8.8, loc="upper left", bbox_to_anchor=(0.0, -0.15))


def epsilon_support_means(payload, dataset):
    by_eps = {}
    for run in payload["datasets"][dataset]["runs"]:
        for q in run["stability_queries"]:
            for p in q["perturbations"]:
                if not p.get("prediction_preserved", False):
                    continue
                if p.get("support_jaccard") is None:
                    continue
                eps = float(p["epsilon"])
                by_eps.setdefault(eps, []).append(float(p["support_jaccard"]))
    return {eps: float(np.mean(vals)) for eps, vals in by_eps.items()}


def panel_b(ax, payload):
    for color, dataset in zip(EPS_COLORS, DATASETS):
        means = epsilon_support_means(payload, dataset)
        eps = sorted(means)
        vals = [means[e] for e in eps]
        ax.plot(eps, vals, marker="o", linewidth=2.0, color=color, label=LABELS[dataset])
    ax.set_xlim(0.015, 0.055)
    ax.set_ylim(0.0, 1.02)
    ax.set_xticks([0.02, 0.05], ["ε = 0.02 sd", "ε = 0.05 sd"])
    ax.set_ylabel("support-set Jaccard (perturbed vs original)")
    ax.set_title(
        "b  Explanation stability under increasing perturbation",
        loc="left",
        fontsize=13,
        fontweight="bold",
    )
    ax.grid(axis="y", alpha=0.18)
    ax.legend(frameon=False, fontsize=8.8, loc="lower left")


def panel_c(ax, payload):
    x = np.arange(len(DATASETS), dtype=float)
    jitter = np.linspace(-0.12, 0.12, 5)
    eligibility_rates = []

    for xi, dataset in zip(x, DATASETS):
        runs = payload["datasets"][dataset]["runs"]
        rates = []
        for j, run in enumerate(runs):
            e = run["eligibility"]
            eligible = e["median_margin_eligible"]
            trivial = e["median_margin_noneligible"]
            rates.append(float(e["eligible_rate"]))
            if eligible is None or trivial is None:
                continue
            xx = xi + jitter[j]
            ax.plot([xx, xx], [eligible, trivial], color="0.88", linewidth=1.5, zorder=1)
            ax.scatter(xx, trivial, s=36, color=CONTROL, zorder=2)
            ax.scatter(xx, eligible, s=40, color=ELIGIBLE, zorder=3)
        eligibility_rates.append(float(np.mean(rates)))

    ax.set_xticks(x, [LABELS[d].replace(" ", "\n") for d in DATASETS])
    ax.set_ylabel("median decision margin (per dataset × seed)")
    ax.set_title(
        "c  Nontrivial paths appear in less locally obvious decisions",
        loc="left",
        fontsize=13,
        fontweight="bold",
    )
    ax.grid(axis="y", alpha=0.18)
    ax.scatter([], [], s=40, color=ELIGIBLE, label="nontrivial winner path")
    ax.scatter([], [], s=36, color=CONTROL, label="trivial path (start already in basin)")
    ax.legend(frameon=False, fontsize=8.8, loc="upper left")

    ymin, ymax = ax.get_ylim()
    y_text = ymin - 0.10 * (ymax - ymin)
    for xi, rate in zip(x, eligibility_rates):
        ax.text(xi, y_text, f"{100*rate:.0f}% eligible", ha="center", va="top", fontsize=8.5, color="0.35", clip_on=False)


def make_figure(payload, out_png: Path, out_pdf: Path):
    fig, axes = plt.subplots(1, 3, figsize=(15.8, 6.1), gridspec_kw={"width_ratios": [1.28, 0.95, 1.05]})
    panel_a(axes[0], payload)
    panel_b(axes[1], payload)
    panel_c(axes[2], payload)

    fig.suptitle("Stability and scope of intrinsic MCGC explanations", fontsize=16, fontweight="bold", y=0.99)
    fig.subplots_adjust(top=0.83, bottom=0.24, left=0.14, right=0.985, wspace=0.38)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v15-json", type=Path, required=True)
    ap.add_argument("--output-prefix", type=Path, default=Path("figures/fig4_stability_scope_publication"))
    args = ap.parse_args()
    payload = load_json(args.v15_json)
    make_figure(payload, args.output_prefix.with_suffix(".png"), args.output_prefix.with_suffix(".pdf"))
    print(f"WROTE {args.output_prefix.with_suffix('.png')}")
    print(f"WROTE {args.output_prefix.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
