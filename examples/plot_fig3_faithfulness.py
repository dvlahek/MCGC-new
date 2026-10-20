from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DATASET_ORDER = ("moons_hard", "spirals_hard", "overlap_hd", "breast_cancer")
LABELS = {
    "moons_hard": "Moons",
    "spirals_hard": "Spirals",
    "overlap_hd": "Overlap-HD",
    "breast_cancer": "Breast cancer",
}


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def extract_dataset_seed_advantages(payload):
    rows = []
    for dataset in DATASET_ORDER:
        block = payload["datasets"][dataset]
        for run in block["runs"]:
            s = run["summary"]
            rows.append({
                "dataset": dataset,
                "seed": int(run["seed"]),
                "matched": float(s["faithfulness_advantage"]["mean"]),
                "nearest": float(s["path_vs_nearest_advantage"]["mean"]),
            })
    return rows


def _mean_ci(values):
    a = np.asarray(values, dtype=float)
    mean = float(np.mean(a))
    if len(a) <= 1:
        return mean, 0.0
    half = 1.96 * float(np.std(a, ddof=1)) / np.sqrt(len(a))
    return mean, half


def make_panel(rows, key: str, title: str, output_prefix: Path):
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    y_positions = np.arange(len(DATASET_ORDER), dtype=float)
    jitter = np.linspace(-0.13, 0.13, 5)
    ax.axvline(0.0, linewidth=1.0, linestyle="--")

    for yi, dataset in enumerate(DATASET_ORDER):
        vals = [r[key] for r in rows if r["dataset"] == dataset]
        for j, value in enumerate(vals):
            off = jitter[j] if j < len(jitter) else 0.0
            ax.scatter(value, yi + off, s=30, alpha=0.75)
        mean, ci = _mean_ci(vals)
        ax.errorbar(mean, yi, xerr=ci, fmt="D", markersize=6, capsize=3, linewidth=1.5)

    ax.set_yticks(y_positions)
    ax.set_yticklabels([LABELS[d] for d in DATASET_ORDER])
    ax.set_xlabel("Margin-drop advantage")
    ax.set_ylabel("Dataset")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.18)
    fig.tight_layout()

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v09-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    args = parser.parse_args()

    payload = load_json(args.v09_json)
    rows = extract_dataset_seed_advantages(payload)
    if len(rows) != 20:
        raise RuntimeError(f"Expected 20 dataset×seed units, found {len(rows)}")

    make_panel(rows, "matched", "A  Path vs matched off-path", args.output_dir / "fig3a_faithfulness_matched")
    make_panel(rows, "nearest", "B  Path vs nearest off-path", args.output_dir / "fig3b_faithfulness_nearest")
    print(f"WROTE {args.output_dir / 'fig3a_faithfulness_matched.pdf'}")
    print(f"WROTE {args.output_dir / 'fig3b_faithfulness_nearest.pdf'}")


if __name__ == "__main__":
    main()
