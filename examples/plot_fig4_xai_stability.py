from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DATASET_ORDER = ("digits", "wine", "spirals_hard", "breast_cancer")
LABELS = {
    "digits": "Digits",
    "wine": "Wine",
    "spirals_hard": "Spirals",
    "breast_cancer": "Breast cancer",
}


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def extract_summary(payload):
    rows = []
    for dataset in DATASET_ORDER:
        block = payload["datasets"][dataset]
        agg = block["aggregate"]
        elig_acc = [float(r["eligibility"]["accuracy_eligible"]) for r in block["runs"]]
        nonelig_acc = [float(r["eligibility"]["accuracy_noneligible"]) for r in block["runs"]]
        rows.append(
            {
                "dataset": dataset,
                "support_advantage": float(agg["support_advantage"]["mean"]),
                "path_advantage": float(agg["path_advantage"]["mean"]),
                "prediction_preservation": float(agg["mean_prediction_preservation_rate"]),
                "eligibility_rate": float(agg["mean_eligibility_rate"]),
                "accuracy_eligible": float(np.mean(elig_acc)),
                "accuracy_noneligible": float(np.mean(nonelig_acc)),
                "multiclass_max_error": (
                    None
                    if agg["multiclass_max_abs_reconstruction_error"] is None
                    else float(agg["multiclass_max_abs_reconstruction_error"])
                ),
            }
        )
    return rows


def make_figure(rows, output_prefix: Path):
    labels = [LABELS[r["dataset"]] for r in rows]
    x = np.arange(len(rows), dtype=float)
    width = 0.34

    fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.2))

    ax = axes[0, 0]
    ax.plot(x, [r["support_advantage"] for r in rows], marker="o", label="Support")
    ax.plot(x, [r["path_advantage"] for r in rows], marker="s", label="Representative path")
    ax.axhline(0.0, linewidth=1.0, linestyle="--")
    ax.set_ylabel("Within-query minus same-class control")
    ax.set_title("A  Query-specific explanation stability")
    ax.set_xticks(x, labels, rotation=18, ha="right")
    ax.set_ylim(bottom=0.0)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.18)

    ax = axes[0, 1]
    ax.bar(x - width / 2, [r["prediction_preservation"] for r in rows], width, label="Prediction preserved")
    ax.bar(x + width / 2, [r["eligibility_rate"] for r in rows], width, label="Nontrivial-path eligible")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Rate")
    ax.set_title("B  Perturbation preservation and path eligibility")
    ax.set_xticks(x, labels, rotation=18, ha="right")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.18)

    ax = axes[1, 0]
    ax.bar(x - width / 2, [r["accuracy_eligible"] for r in rows], width, label="Eligible")
    ax.bar(x + width / 2, [r["accuracy_noneligible"] for r in rows], width, label="Non-eligible")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Accuracy")
    ax.set_title("C  Eligible queries are the harder cases")
    ax.set_xticks(x, labels, rotation=18, ha="right")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.18)

    ax = axes[1, 1]
    multiclass = [r for r in rows if r["multiclass_max_error"] is not None]
    mx = np.arange(len(multiclass), dtype=float)
    vals = [max(r["multiclass_max_error"], np.finfo(float).tiny) for r in multiclass]
    ax.scatter(mx, vals, s=70)
    ax.set_yscale("log")
    ax.set_xticks(mx, [LABELS[r["dataset"]] for r in multiclass])
    ax.set_ylabel("Max |score margin − reconstructed margin|")
    ax.set_title("D  Multiclass contrastive reconstruction")
    ax.grid(axis="y", alpha=0.18)
    for i, value in enumerate(vals):
        ax.annotate(f"{value:.1e}", (i, value), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=9)

    fig.suptitle("Stability, scope, and multiclass exactness of MCGC explanations", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v15-json", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, default=Path("figures/fig4_xai_stability"))
    args = parser.parse_args()

    payload = load_json(args.v15_json)
    rows = extract_summary(payload)
    make_figure(rows, args.output_prefix)
    print(f"WROTE {args.output_prefix.with_suffix('.pdf')}")
    print(f"WROTE {args.output_prefix.with_suffix('.png')}")


if __name__ == "__main__":
    main()
