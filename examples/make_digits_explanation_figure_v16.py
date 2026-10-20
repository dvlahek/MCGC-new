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

from mcgc import MultiAgentChemotacticGraphClassifier, explain_mcgc


SEEDS = list(range(5))
TOP_SUPPORT = 5
PREFERRED_PAIRS = ((3, 8), (8, 3), (9, 4), (4, 9))
MIN_PREFERRED_CANDIDATES = 3

WINNER_COLOR = "tab:blue"
RUNNER_COLOR = "tab:orange"


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

        # Important: prediction and XAI are evaluated once as one frozen batch.
        # The stochastic walker RNG advances across the batch, so all later
        # provenance must reuse these exact stored explanations rather than
        # re-explaining a selected row with a freshly reset RNG.
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


def contrastive_from_stored_explanation(exp, winner, runner):
    """Exact contrast using the same stored stochastic realization as selection."""
    pos = exp["classes"][winner]
    neg = exp["classes"][runner]
    pc = pos["score_components"]
    nc = neg["score_components"]
    deltas = {
        "local_component_delta": float(pc["local_component"] - nc["local_component"]),
        "trail_component_delta": float(pc["trail_component"] - nc["trail_component"]),
        "agreement_component_delta": float(
            pc["agreement_component"] - nc["agreement_component"]
        ),
        "lost_component_delta": float(pc["lost_component"] - nc["lost_component"]),
    }
    reconstructed = float(sum(deltas.values()))
    score_margin = float(exp["class_scores"][winner] - exp["class_scores"][runner])
    if not np.isclose(score_margin, float(exp["margin"]), rtol=1e-10, atol=1e-12):
        raise RuntimeError("Stored explanation margin does not match stored class-score difference.")
    if not np.isclose(reconstructed, score_margin, rtol=1e-10, atol=1e-12):
        raise RuntimeError("Stored contrastive decomposition does not reconstruct score margin.")
    ranked = sorted(deltas.items(), key=lambda kv: abs(kv[1]), reverse=True)
    return {
        "positive_class": int(winner),
        "negative_class": int(runner),
        "positive_score": float(pos["score"]),
        "negative_score": float(neg["score"]),
        "score_margin": score_margin,
        "reconstructed_margin": reconstructed,
        "reconstruction_abs_error": abs(score_margin - reconstructed),
        "component_deltas": deltas,
        "dominant_component": ranked[0][0],
    }


def selected_payload(selected, median_margin, pair_n, pair_counts, fitted):
    seed = selected["seed"]
    idx = selected["test_index"]
    model, X_train, X_test, y_train, y_test, pred, explanations = fitted[seed]
    exp = explanations[idx]
    winner = py_scalar(exp["prediction"])
    runner = py_scalar(exp["runner_up"])

    # Reuse the exact batch explanation used for candidate selection.
    contrast = contrastive_from_stored_explanation(exp, winner, runner)
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
            "stochastic_provenance": (
                "Selection, prediction, class scores, and decomposition reuse the exact "
                "stored batch explanation; no selected-row re-explanation with reset RNG."
            ),
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
        "contrastive": contrast,
    }


def show_digit(ax, image, title=None, border_color="0.65"):
    ax.imshow(np.asarray(image), cmap="gray_r", interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(1.4)
        spine.set_edgecolor(border_color)
    if title:
        ax.set_title(title, fontsize=8.5, pad=3)


def panel_query(ax, payload):
    ax.axis("off")
    s = payload["selected"]
    sub = ax.inset_axes([0.02, 0.22, 0.43, 0.64])
    show_digit(sub, s["query_image"], border_color=WINNER_COLOR)
    ax.text(0.02, 0.97, "A  Query and decision", fontsize=13, fontweight="bold", va="top")
    lines = [
        f"True label: {s['true_label']}",
        f"Predicted: {s['prediction']}",
        f"Runner-up: {s['runner_up']}",
        f"Winner-runner score margin: {s['margin']:.3f}",
        f"Pair median: {s['pair_median_margin']:.3f}",
        "Nontrivial winner path: yes",
    ]
    y = 0.78
    for i, line in enumerate(lines):
        color = "black"
        if i == 1:
            color = WINNER_COLOR
        elif i == 2:
            color = RUNNER_COLOR
        ax.text(0.50, y, line, fontsize=10, color=color, va="top")
        y -= 0.105


def panel_support(gs, fig, payload):
    holder = fig.add_subplot(gs)
    holder.axis("off")
    holder.text(0.0, 0.98, "B  Local query neighbourhood", fontsize=13, fontweight="bold", va="top")
    holder.text(
        0.0,
        0.88,
        "Affinity-ranked training neighbours; D = class-conditioned contrastive evidence",
        fontsize=8.5,
        va="top",
    )
    inner = GridSpecFromSubplotSpec(1, TOP_SUPPORT, subplot_spec=gs, wspace=0.22)
    winner = payload["selected"]["prediction"]
    runner = payload["selected"]["runner_up"]
    for j, row in enumerate(payload["local_supporting_neighbors"]):
        ax = fig.add_subplot(inner[0, j])
        ev = row["contrastive_evidence"]
        title = (
            f"y={row['training_label']}  aff={row['affinity']:.2f}\n"
            f"D{winner}={ev[str(winner)]:+.2f}, D{runner}={ev[str(runner)]:+.2f}"
        )
        show_digit(ax, row["image"], title=title)


def panel_paths(gs, fig, payload):
    holder = fig.add_subplot(gs)
    holder.axis("off")
    winner = payload["selected"]["prediction"]
    runner = payload["selected"]["runner_up"]
    holder.text(0.0, 0.99, "C  Representative evidence trails", fontsize=13, fontweight="bold", va="top")

    rows = [
        ("Winner", winner, payload["winner_trail"], WINNER_COLOR, 0.55),
        ("Runner-up", runner, payload["runner_up_trail"], RUNNER_COLOR, 0.08),
    ]
    for name, cls, trail, color, y0 in rows:
        holder.text(0.01, y0 + 0.16, f"{name} class {cls}", color=color, fontsize=10, fontweight="bold", va="center")
        n = min(len(trail["path"]), 5)
        x0 = 0.14
        step_w = 0.12
        gap = 0.035
        for c in range(n):
            left = x0 + c * (step_w + gap)
            ax = holder.inset_axes([left, y0, step_w, 0.28])
            show_digit(
                ax,
                trail["node_images"][c],
                title=f"step {c}\ny={trail['node_labels'][c]}, D={trail['evidence_trace'][c]:+.2f}",
                border_color=color,
            )
            if c < n - 1:
                holder.annotate(
                    "",
                    xy=(left + step_w + gap * 0.75, y0 + 0.14),
                    xytext=(left + step_w + gap * 0.12, y0 + 0.14),
                    xycoords=holder.transAxes,
                    arrowprops={"arrowstyle": "->", "color": color, "lw": 1.5},
                )
        if n == 1:
            holder.text(
                x0 + step_w + 0.04,
                y0 + 0.14,
                "no further positive evidence ascent",
                fontsize=9,
                color=color,
                va="center",
            )


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
        ha = "left" if v >= 0 else "right"
        pad = 0.01 if v >= 0 else -0.01
        ax.text(v + pad, yi, f"{v:+.3f}", va="center", ha=ha, fontsize=9)
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
    fig = plt.figure(figsize=(13.5, 8.3))
    gs = GridSpec(
        3,
        2,
        figure=fig,
        height_ratios=(1.0, 0.82, 1.05),
        width_ratios=(1.0, 1.55),
        hspace=0.34,
        wspace=0.28,
    )

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
        fontsize=8.8,
    )
    fig.subplots_adjust(top=0.94, bottom=0.065)
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
    print(f"SCORE_MARGIN={payload['contrastive']['score_margin']:.6f}")
    print(f"RECONSTRUCTION_ERROR={payload['contrastive']['reconstruction_abs_error']:.3e}")
    print(f"WROTE {args.json}")
    print(f"WROTE {out_png}")
    print(f"WROTE {out_pdf}")


if __name__ == "__main__":
    main()
