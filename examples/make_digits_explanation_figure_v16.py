from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from mcgc import MultiAgentChemotacticGraphClassifier, explain_contrastive_mcgc, explain_mcgc


SEEDS = list(range(5))
TOP_SUPPORT = 5
PREFERRED_PAIRS = ((3, 8), (8, 3), (9, 4), (4, 9))
MIN_PREFERRED_CANDIDATES = 3

WINNER_COLOR = "tab:blue"
RUNNER_COLOR = "tab:orange"
NEUTRAL_COLOR = "0.45"


def mcgc_classifier():
    return MultiAgentChemotacticGraphClassifier(
        n_neighbors=12,
        n_walkers=12,
        max_steps=10,
        diffusion_alpha=0.85,
        diffusion_steps=40,
        temperature=0.35,
        random_state=17,
        adaptive_walker_gate=True,
    )


def py_scalar(value):
    return value.item() if hasattr(value, "item") else value


def nontrivial_trail(trail):
    return bool(trail["length"] > 0 and len(trail["path"]) >= 2)


def load_split(seed: int):
    data = load_digits()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data,
        data.target,
        test_size=0.30,
        random_state=7000 + seed,
        stratify=data.target,
    )
    return X_train, X_test, y_train, y_test


def collect_candidates():
    candidates = []
    fitted = {}

    for seed in SEEDS:
        print(f"SCAN seed={seed}", flush=True)
        X_train, X_test, y_train, y_test = load_split(seed)
        model = mcgc_classifier().fit(X_train, y_train)
        pred, explanations = explain_mcgc(model, X_test, top_support=TOP_SUPPORT)
        fitted[seed] = (model, X_train, X_test, y_train, y_test, pred, explanations)

        for idx, (p, true, exp) in enumerate(zip(pred, y_test, explanations)):
            winner = py_scalar(exp["prediction"])
            runner = py_scalar(exp["runner_up"])
            trail = exp["classes"][winner]["representative_trail"]
            if p != true or not nontrivial_trail(trail):
                continue
            candidates.append(
                {
                    "seed": int(seed),
                    "test_index": int(idx),
                    "winner": int(winner),
                    "runner_up": int(runner),
                    "margin": float(exp["margin"]),
                }
            )

    return candidates, fitted


def choose_pair(candidates):
    pair_counts = Counter((c["winner"], c["runner_up"]) for c in candidates)
    for pair in PREFERRED_PAIRS:
        if pair_counts[pair] >= MIN_PREFERRED_CANDIDATES:
            return pair, pair_counts

    if not pair_counts:
        raise RuntimeError("No correctly classified, path-eligible digits candidates were found.")

    best_count = max(pair_counts.values())
    pair = sorted(pair for pair, count in pair_counts.items() if count == best_count)[0]
    return pair, pair_counts


def choose_candidate(candidates, pair):
    subset = [c for c in candidates if (c["winner"], c["runner_up"]) == pair]
    margins = np.asarray([c["margin"] for c in subset], dtype=float)
    median_margin = float(np.median(margins))
    subset = sorted(
        subset,
        key=lambda c: (abs(c["margin"] - median_margin), c["seed"], c["test_index"]),
    )
    return subset[0], median_margin, len(subset)


def image8(x):
    return np.asarray(x, dtype=float).reshape(8, 8)


def trail_record(trail, X_train, y_train):
    path = [int(i) for i in trail["path"]]
    return {
        "path": path,
        "length": int(trail["length"]),
        "lost": bool(trail["lost"]),
        "coherence": float(trail["coherence"]),
        "terminal_evidence": float(trail["terminal_evidence"]),
        "evidence_trace": [float(v) for v in trail["evidence_trace"]],
        "node_labels": [int(y_train[i]) for i in path],
        "node_images": [image8(X_train[i]).tolist() for i in path],
    }


def support_record(explanation, X_train):
    out = []
    for row in explanation["supporting_neighbors"]:
        node = int(row["node_index"])
        out.append(
            {
                "node_index": node,
                "training_label": int(row["training_label"]),
                "distance": float(row["distance"]),
                "affinity": float(row["affinity"]),
                "contrastive_evidence": {
                    str(k): float(v) for k, v in row["contrastive_evidence"].items()
                },
                "image": image8(X_train[node]).tolist(),
            }
        )
    return out


def selected_payload(selected, median_margin, pair_n, pair_counts, fitted):
    seed = selected["seed"]
    idx = selected["test_index"]
    model, X_train, X_test, y_train, y_test, pred, explanations = fitted[seed]
    exp = explanations[idx]
    winner = py_scalar(exp["prediction"])
    runner = py_scalar(exp["runner_up"])

    _, contrastive = explain_contrastive_mcgc(
        model,
        X_test[idx : idx + 1],
        positive_class=winner,
        negative_class=runner,
        top_support=TOP_SUPPORT,
    )
    contrast = contrastive[0]
    winner_trail = exp["classes"][winner]["representative_trail"]
    runner_trail = exp["classes"][runner]["representative_trail"]

    return {
        "protocol": {
            "version": "1.6",
            "dataset": "digits",
            "seeds_scanned": SEEDS,
            "split_seed": "7000 + seed",
            "preferred_pairs": [list(p) for p in PREFERRED_PAIRS],
            "minimum_preferred_candidates": MIN_PREFERRED_CANDIDATES,
            "selection_rule": "first preferred pair with >=3 candidates; then margin closest to pair median",
        },
        "selected": {
            "seed": int(seed),
            "test_index": int(idx),
            "true_label": int(y_test[idx]),
            "prediction": int(winner),
            "runner_up": int(runner),
            "margin": float(exp["margin"]),
            "pair_median_margin": float(median_margin),
            "pair_candidate_count": int(pair_n),
            "walker_gate": float(exp["walker_gate"]),
            "query_image": image8(X_test[idx]).tolist(),
        },
        "pair_counts": {
            f"{a}->{b}": int(count)
            for (a, b), count in sorted(pair_counts.items())
        },
        "local_supporting_neighbors": support_record(exp, X_train),
        "winner_trail": trail_record(winner_trail, X_train, y_train),
        "runner_up_trail": trail_record(runner_trail, X_train, y_train),
        "contrastive": {
            "positive_class": int(contrast["positive_class"]),
            "negative_class": int(contrast["negative_class"]),
            "positive_score": float(contrast["positive_score"]),
            "negative_score": float(contrast["negative_score"]),
            "score_margin": float(contrast["score_margin"]),
            "reconstructed_margin": float(contrast["reconstructed_margin"]),
            "reconstruction_abs_error": abs(
                float(contrast["score_margin"]) - float(contrast["reconstructed_margin"])
            ),
            "component_deltas": {
                k: float(v) for k, v in contrast["component_deltas"].items()
            },
            "dominant_component": str(contrast["dominant_component"]),
        },
    }


def show_digit(ax, image, title=None, border_color="0.65"):
    ax.imshow(np.asarray(image), cmap="gray_r", interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(1.4)
        spine.set_edgecolor(border_color)
    if title:
        ax.set_title(title, fontsize=9, pad=3)


def panel_query(ax, payload):
    ax.axis("off")
    s = payload["selected"]
    sub = ax.inset_axes([0.02, 0.18, 0.44, 0.70])
    show_digit(sub, s["query_image"], border_color=WINNER_COLOR)
    ax.text(0.02, 0.96, "A  Query and decision", fontsize=13, fontweight="bold", va="top")
    lines = [
        f"True label: {s['true_label']}",
        f"Predicted: {s['prediction']}",
        f"Runner-up: {s['runner_up']}",
        f"Score margin: {s['margin']:.3f}",
        f"Pair-median margin: {s['pair_median_margin']:.3f}",
        "Nontrivial winner path: yes",
    ]
    y = 0.78
    for i, line in enumerate(lines):
        color = "black"
        if i == 1:
            color = WINNER_COLOR
        elif i == 2:
            color = RUNNER_COLOR
        ax.text(0.52, y, line, fontsize=10.5, color=color, va="top")
        y -= 0.105


def panel_support(gs, fig, payload):
    holder = fig.add_subplot(gs)
    holder.axis("off")
    holder.text(0.0, 1.02, "B  Local query neighbourhood", fontsize=13, fontweight="bold", va="bottom")
    holder.text(
        0.0,
        0.93,
        "Affinity-ranked training neighbours used for local evidence (not contribution-ranked exemplars)",
        fontsize=8.5,
        va="bottom",
    )
    inner = GridSpecFromSubplotSpec(1, TOP_SUPPORT, subplot_spec=gs, wspace=0.20)
    winner = payload["selected"]["prediction"]
    runner = payload["selected"]["runner_up"]
    for j, row in enumerate(payload["local_supporting_neighbors"]):
        ax = fig.add_subplot(inner[0, j])
        ev = row["contrastive_evidence"]
        title = (
            f"y={row['training_label']}\n"
            f"aff={row['affinity']:.2f}\n"
            f"D{winner}={ev[str(winner)]:+.2f}\nD{runner}={ev[str(runner)]:+.2f}"
        )
        show_digit(ax, row["image"], title=title)


def panel_paths(gs, fig, payload):
    holder = fig.add_subplot(gs)
    holder.axis("off")
    winner = payload["selected"]["prediction"]
    runner = payload["selected"]["runner_up"]
    holder.text(0.0, 1.02, "C  Representative trails through the fitted evidence fields", fontsize=13, fontweight="bold", va="bottom")

    rows = (("Winner", payload["winner_trail"], WINNER_COLOR), ("Runner-up", payload["runner_up_trail"], RUNNER_COLOR))
    inner = GridSpecFromSubplotSpec(2, 7, subplot_spec=gs, hspace=0.48, wspace=0.18)
    for r, (name, trail, color) in enumerate(rows):
        path = trail["path"][:7]
        for c in range(7):
            ax = fig.add_subplot(inner[r, c])
            if c >= len(path):
                ax.axis("off")
                continue
            image = trail["node_images"][c]
            label = trail["node_labels"][c]
            ev = trail["evidence_trace"][c]
            show_digit(ax, image, title=f"step {c}\ny={label}, D={ev:+.2f}", border_color=color)
            if c == 0:
                ax.set_ylabel(f"{name}\nclass {winner if r == 0 else runner}", color=color, fontsize=9)


def panel_trace(ax, payload):
    winner = payload["selected"]["prediction"]
    runner = payload["selected"]["runner_up"]
    wt = payload["winner_trail"]["evidence_trace"]
    rt = payload["runner_up_trail"]["evidence_trace"]
    ax.plot(np.arange(len(wt)), wt, marker="o", color=WINNER_COLOR, label=f"Winner field: {winner}")
    ax.plot(np.arange(len(rt)), rt, marker="s", color=RUNNER_COLOR, label=f"Runner-up field: {runner}")
    ax.axhline(0.0, linewidth=0.9, linestyle="--", color="0.55")
    ax.set_xlabel("Representative-trail step")
    ax.set_ylabel("Contrastive evidence")
    ax.set_title("D  Evidence along representative trails", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="y", alpha=0.18)


def panel_decomposition(ax, payload):
    c = payload["contrastive"]
    deltas = c["component_deltas"]
    labels = ["Local", "Trail", "Agreement", "Lost"]
    keys = [
        "local_component_delta",
        "trail_component_delta",
        "agreement_component_delta",
        "lost_component_delta",
    ]
    vals = [deltas[k] for k in keys]
    colors = [WINNER_COLOR if v >= 0 else RUNNER_COLOR for v in vals]
    y = np.arange(len(vals))
    ax.barh(y, vals, color=colors, alpha=0.85)
    ax.axvline(0.0, linewidth=0.9, color="0.35")
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Contribution to winner - runner score margin")
    ax.set_title("E  Exact contrastive score decomposition", loc="left", fontweight="bold")
    for yi, v in zip(y, vals):
        ax.text(v, yi, f" {v:+.3f}", va="center", ha="left" if v >= 0 else "right", fontsize=9)
    ax.text(
        0.98,
        0.05,
        f"score margin = {c['score_margin']:.3f}\n"
        f"reconstructed = {c['reconstructed_margin']:.3f}\n"
        f"|error| = {c['reconstruction_abs_error']:.1e}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "0.8"},
    )
    ax.grid(axis="x", alpha=0.18)


def make_figure(payload, out_png: Path, out_pdf: Path):
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(3, 2, figure=fig, height_ratios=(1.05, 1.2, 1.0), width_ratios=(1.0, 1.55), hspace=0.36, wspace=0.28)

    ax_a = fig.add_subplot(gs[0, 0])
    panel_query(ax_a, payload)
    panel_support(gs[0, 1], fig, payload)
    panel_paths(gs[1, :], fig, payload)
    ax_d = fig.add_subplot(gs[2, 0])
    panel_trace(ax_d, payload)
    ax_e = fig.add_subplot(gs[2, 1])
    panel_decomposition(ax_e, payload)

    s = payload["selected"]
    fig.suptitle(
        f"Real-data MCGC explanation on digits: why {s['prediction']} rather than {s['runner_up']}",
        fontsize=17,
        fontweight="bold",
        y=0.995,
    )
    fig.text(
        0.5,
        0.008,
        "Representative trails summarize the fitted class-conditioned evidence fields; they are not claimed to be the sole stochastic causal trajectory.",
        ha="center",
        fontsize=9,
    )
    fig.subplots_adjust(top=0.94, bottom=0.06)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    parser.add_argument("--json", type=Path, default=Path("v16_digits_explanation.json"))
    args = parser.parse_args()

    candidates, fitted = collect_candidates()
    pair, pair_counts = choose_pair(candidates)
    selected, median_margin, pair_n = choose_candidate(candidates, pair)
    payload = selected_payload(selected, median_margin, pair_n, pair_counts, fitted)

    args.json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    stem = f"fig2_digits_real_xai_{pair[0]}_vs_{pair[1]}"
    out_png = args.output_dir / f"{stem}.png"
    out_pdf = args.output_dir / f"{stem}.pdf"
    make_figure(payload, out_png, out_pdf)

    print("=== MCGC v1.6 REAL-DATA XAI ===")
    print(f"PAIR={pair[0]}->{pair[1]} candidates={pair_n}")
    print(
        f"SELECTED seed={selected['seed']} test_index={selected['test_index']} "
        f"margin={selected['margin']:.6f} pair_median={median_margin:.6f}"
    )
    print(f"RECONSTRUCTION_ERROR={payload['contrastive']['reconstruction_abs_error']:.3e}")
    print(f"WROTE {args.json}")
    print(f"WROTE {out_png}")
    print(f"WROTE {out_pdf}")


if __name__ == "__main__":
    main()
