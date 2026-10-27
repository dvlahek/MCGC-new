"""CGEC paper Fig. 3 -- dataset-level strict OOF label review.

The figure shows CGEC minus OOF kNN disagreement for ROC-AUC and PR-AUC.
Rows are ordered once by ROC-AUC difference and the same order is used in
both panels. Positive and negative differences are visually separated with
lollipop stems and different colors.

By default the script uses the frozen compact summary in
``paper_reproducibility/results/fig3_oof_vs_knn_summary.json`` so the paper
figure can be regenerated without the full experiment archive. If the
original strict-OOF audit JSONL is available, pass it through
``--audit-jsonl`` to recompute the dataset averages directly.
"""

from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY = ROOT / "results" / "fig3_oof_vs_knn_summary.json"

INK, INK2, MUTED, GRID = "#111111", "#4a4946", "#8a8984", "#e4e3df"
WIN, LOSS = "#2a78d6", "#d95f4a"

plt.rcParams.update({
    "font.size": 7.5,
    "text.color": INK,
    "axes.labelcolor": INK2,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "axes.edgecolor": MUTED,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "TeX Gyre Termes", "DejaVu Serif"],
    "mathtext.fontset": "stix",
})

NAMES = {
    "Hill_Valley_without_noise": "Hill Valley (no noise)",
    "wine_quality_red": "wine quality (red)",
    "mfeat_fourier": "mfeat Fourier",
    "mfeat_karhunen": "mfeat Karhunen",
    "mfeat_zernike": "mfeat Zernike",
    "saheart": "SAheart",
    "spectf": "SPECTF",
}


def signp(w: int, l: int) -> float:
    n = w + l
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(w, l) + 1)) / 2 ** n)


def load_from_summary(path: Path):
    obj = json.loads(path.read_text(encoding="utf-8"))
    rows = obj["datasets"]
    D = {
        "roc_auc": {r["dataset"]: float(r["roc_auc"]) for r in rows},
        "pr_auc": {r["dataset"]: float(r["pr_auc"]) for r in rows},
    }
    return D


def load_from_audit_jsonl(path: Path):
    records = [
        r
        for r in (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
        if r.get("status") == "ok"
    ]
    datasets = sorted({r["dataset"] for r in records})

    def dm(ds, model, key):
        return float(np.mean([
            r["audit"][model][key]
            for r in records
            if r["dataset"] == ds
        ]))

    return {
        key: {
            ds: dm(ds, "mcgc_oof", key) - dm(ds, "knn_oof", key)
            for ds in datasets
        }
        for key in ("roc_auc", "pr_auc")
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=DEFAULT_SUMMARY,
        help="Frozen compact dataset-level inputs used by the paper figure.",
    )
    parser.add_argument(
        "--audit-jsonl",
        type=Path,
        default=None,
        help="Optional full v2.1 strict-OOF audit JSONL; recomputes dataset averages directly.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs",
    )
    args = parser.parse_args()

    if args.audit_jsonl is not None:
        D = load_from_audit_jsonl(args.audit_jsonl)
    else:
        D = load_from_summary(args.summary_json)

    datasets = sorted(D["roc_auc"])
    order = sorted(datasets, key=lambda d: D["roc_auc"][d])

    args.output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(6.3, 3.6),
        dpi=300,
        sharey=True,
        gridspec_kw=dict(wspace=0.10),
    )

    for ax, key, lab, title in zip(
        axes,
        ("roc_auc", "pr_auc"),
        ("ROC-AUC", "PR-AUC"),
        ("(a)", "(b)"),
    ):
        values = np.array([D[key][d] for d in order])
        y = np.arange(len(order))
        lim = np.abs(values).max() * 1.45
        ax.set_xlim(min(values.min() * 2.3, -lim * 0.25), lim)

        for yy in y:
            ax.axhline(yy, color=GRID, lw=0.5, zorder=0)
        ax.axvline(0, color=INK2, lw=0.7, zorder=1)

        colors = [WIN if x > 0 else LOSS for x in values]
        ax.hlines(y, 0, values, color=colors, lw=1.4, zorder=2)
        ax.scatter(values, y, s=16, color=colors, zorder=3)

        for yy, x in zip(y, values):
            ax.text(
                x + (lim * 0.045 if x > 0 else -lim * 0.045),
                yy,
                f"{x:+.3f}",
                va="center",
                ha="left" if x > 0 else "right",
                fontsize=6.3,
                color=INK2,
            )

        ax.set_title(
            f"{title} {lab} difference",
            loc="center",
            fontsize=8.5,
            fontweight="bold",
            pad=5,
        )
        ax.set_xlabel(
            f"{lab} difference\n(vs. out-of-fold kNN disagreement)",
            fontsize=7,
        )
        ax.tick_params(axis="x", labelsize=6.3, length=2)
        ax.tick_params(axis="y", length=0)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.set_ylim(-0.7, len(order) - 0.3)

    axes[0].set_yticks(np.arange(len(order)))
    axes[0].set_yticklabels(
        [NAMES.get(d, d.replace("_", " ")) for d in order],
        fontsize=7,
    )

    for ext in ("pdf", "png"):
        fig.savefig(
            args.output_dir / f"fig3_oof_vs_knn.{ext}",
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)

    for d in order:
        print(f"{d:28s} ROC {D['roc_auc'][d]:+.4f}  PR {D['pr_auc'][d]:+.4f}")


if __name__ == "__main__":
    main()
