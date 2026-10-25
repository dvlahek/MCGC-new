from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "xai"))
from field_only_cgec import FieldOnlyCGEC  # noqa: E402


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = load_breast_cancer()
    names = np.asarray(data.feature_names)
    malignant_label = 0
    n_splits = 5
    top = 12

    runs = []
    for seed in range(n_splits):
        Xtr, Xte, ytr, yte = train_test_split(
            data.data, data.target, test_size=0.30, random_state=7000 + seed,
            stratify=data.target
        )
        model = FieldOnlyCGEC().fit(Xtr, ytr)
        oriented, abs_effects = [], []
        for x in Xte:
            rec = model.feature_intervention_effects(x)
            I = np.asarray(rec["effects"], dtype=float)
            oriented.append(I if rec["winner"] == malignant_label else -I)
            abs_effects.append(np.abs(I))
        runs.append({
            "oriented": np.asarray(oriented),
            "abs_effects": np.asarray(abs_effects),
            "Xte": Xte,
            "mu": Xtr.mean(axis=0),
            "sd": Xtr.std(axis=0),
        })
        print(f"split {seed} done", flush=True)

    importance = np.asarray([r["abs_effects"].mean(axis=0) for r in runs])
    aggregate_importance = importance.mean(axis=0)
    order = np.argsort(-aggregate_importance)[:top]

    rho = [
        float(spearmanr(importance[i], importance[j]).correlation)
        for i in range(n_splits) for j in range(i + 1, n_splits)
    ]
    top5 = [np.argsort(-importance[i])[:5] for i in range(n_splits)]
    jac = [
        float(jaccard(top5[i], top5[j]))
        for i in range(n_splits) for j in range(i + 1, n_splits)
    ]

    fig, (ax, axb) = plt.subplots(
        1, 2, figsize=(12.0, 6.0), sharey=True,
        gridspec_kw={"width_ratios": [1.75, 1.0], "wspace": 0.08}
    )

    r0 = runs[0]
    rng = np.random.default_rng(0)
    for row, k in enumerate(order):
        y0 = top - 1 - row
        values = r0["oriented"][:, k]
        z = (r0["Xte"][:, k] - r0["mu"][k]) / np.maximum(r0["sd"][k], 1e-12)
        bins = np.linspace(values.min() - 1e-9, values.max() + 1e-9, 40)
        idx = np.digitize(values, bins)
        offsets = np.zeros_like(values)
        for b in np.unique(idx):
            sel = np.flatnonzero(idx == b)
            n = len(sel)
            offsets[sel] = (
                (np.arange(n) - (n - 1) / 2) * 0.045 * (1 if n < 18 else 18 / n)
                + rng.normal(0, 0.008, n)
            )
        ax.axhline(y0, linewidth=0.5, alpha=0.18, zorder=0)
        sc = ax.scatter(
            values, y0 + offsets, c=np.clip(z, -2, 2), cmap="coolwarm",
            vmin=-2, vmax=2, s=9, linewidths=0, zorder=3
        )
    ax.axvline(0, linewidth=0.7)
    ax.set_yticks(range(top))
    ax.set_yticklabels([names[k] for k in order][::-1], fontsize=8)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Class-oriented intervention effect\n(negative: benign direction; positive: malignant direction)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.set_title("A  Per-query intervention effects, split 0", loc="left", fontweight="bold")
    cax = ax.inset_axes([0.02, 0.04, 0.20, 0.025])
    cb = fig.colorbar(sc, cax=cax, orientation="horizontal")
    cb.set_ticks([-2, 2])
    cb.set_ticklabels(["low", "high"])
    cb.ax.tick_params(labelsize=7, length=0)
    cb.ax.set_title("feature value", fontsize=7, pad=2)

    for row, k in enumerate(order):
        y0 = top - 1 - row
        vals = importance[:, k]
        axb.axhline(y0, linewidth=0.5, alpha=0.18, zorder=0)
        axb.hlines(y0, vals.min(), vals.max(), linewidth=1.0, alpha=0.7, zorder=1)
        axb.scatter(vals, np.full(n_splits, y0), s=22, zorder=3, linewidths=0)
    axb.set_xlim(0, importance[:, order].max() * 1.12)
    axb.tick_params(axis="y", length=0)
    axb.set_xlabel("Mean $|I_k(q)|$ over test queries")
    axb.text(
        0.02, -0.14,
        f"5 splits: mean Spearman={np.mean(rho):.2f}; mean top-5 Jaccard={np.mean(jac):.2f}",
        transform=axb.transAxes, ha="left", va="top", fontsize=8
    )
    axb.spines["top"].set_visible(False)
    axb.spines["right"].set_visible(False)
    axb.spines["left"].set_visible(False)
    axb.set_title("B  Global ranking across five splits", loc="left", fontweight="bold")

    pdf = args.output_dir / "fig_bc_interventions.pdf"
    png = args.output_dir / "fig_bc_interventions.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=400, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "dataset": "sklearn breast cancer",
        "splits": list(range(n_splits)),
        "split_protocol": "stratified 70/30 with random_state=7000+seed",
        "feature_selection_for_display": "top 12 by mean absolute intervention effect averaged across all five splits",
        "features": names.tolist(),
        "mean_abs_I_by_split": importance.tolist(),
        "aggregate_mean_abs_I": aggregate_importance.tolist(),
        "display_feature_indices": [int(k) for k in order],
        "display_feature_names": [str(names[k]) for k in order],
        "spearman_pairs": rho,
        "spearman_mean": float(np.mean(rho)),
        "spearman_median": float(np.median(rho)),
        "top5_jaccard_pairs": jac,
        "top5_jaccard_mean": float(np.mean(jac)),
        "top5_jaccard_median": float(np.median(jac)),
    }
    (args.output_dir / "bc_interventions_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(pdf)
    print(png)
    print(f"mean Spearman={np.mean(rho):.6f}; mean top5 Jaccard={np.mean(jac):.6f}")


if __name__ == "__main__":
    main()
