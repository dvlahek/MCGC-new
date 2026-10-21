from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from sklearn.model_selection import train_test_split

from mcgc import MultiAgentChemotacticGraphClassifier, explain_mcgc


WINNER = "tab:blue"
RUNNER = "tab:orange"
NEUTRAL = "0.68"
EDGE = "0.84"
SUPPORT = "0.10"


def make_spirals(n_samples: int = 700, noise: float = 0.16, random_state: int = 0):
    """Exact spirals_hard generator used by the frozen benchmark family."""
    rng = np.random.default_rng(random_state)
    n = n_samples // 2
    theta = np.linspace(0.2, 3.2 * np.pi, n)
    radius = np.linspace(0.15, 1.0, n)
    arm0 = np.column_stack([radius * np.cos(theta), radius * np.sin(theta)])
    arm1 = -arm0
    X = np.vstack([arm0, arm1])
    X += rng.normal(scale=noise, size=X.shape)
    y = np.concatenate([np.zeros(n, dtype=int), np.ones(n, dtype=int)])
    order = rng.permutation(len(X))
    return X[order], y[order]


def classifier():
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


def class_index(model, label):
    return int(np.flatnonzero(model.classes_ == label)[0])


def stored_contrastive(exp, winner, runner):
    pc = exp["classes"][winner]["score_components"]
    nc = exp["classes"][runner]["score_components"]
    deltas = {
        "local": float(pc["local_component"] - nc["local_component"]),
        "trail": float(pc["trail_component"] - nc["trail_component"]),
        "agreement": float(pc["agreement_component"] - nc["agreement_component"]),
        "lost": float(pc["lost_component"] - nc["lost_component"]),
    }
    margin = float(exp["class_scores"][winner] - exp["class_scores"][runner])
    reconstructed = float(sum(deltas.values()))
    if not np.isclose(margin, reconstructed, rtol=1e-10, atol=1e-12):
        raise RuntimeError("Contrastive decomposition does not reconstruct the stored score margin.")
    return deltas, margin, reconstructed


def query_neighbours(model, x_raw):
    xs = model._scaled(np.asarray(x_raw, dtype=float)[None, :])[0]
    distances, starts = model._query_start_nodes(xs)
    sigma = max(float(distances[-1]) if len(distances) else 1.0, 1e-12)
    affinity = np.exp(-(distances**2) / (sigma**2))
    affinity /= affinity.sum()
    return distances, starts, affinity


def local_nodes(model, starts, winner_path, runner_path, query_scaled, max_nodes=150):
    nodes = set(int(i) for i in starts)
    nodes.update(int(i) for i in winner_path)
    nodes.update(int(i) for i in runner_path)
    frontier = list(nodes)
    for node in frontier:
        nodes.update(int(j) for j in model.graph_neighbors_[node])
    nodes = np.asarray(sorted(nodes), dtype=int)
    if len(nodes) <= max_nodes:
        return nodes
    dist = np.linalg.norm(model.X_[nodes] - query_scaled[None, :], axis=1)
    keep = np.argsort(dist)[:max_nodes]
    return nodes[keep]


def local_edges(model, nodes):
    keep = set(int(i) for i in nodes)
    seen = set()
    out = []
    for i in nodes:
        for j in model.graph_neighbors_[int(i)]:
            j = int(j)
            if j not in keep:
                continue
            a, b = sorted((int(i), j))
            if a == b or (a, b) in seen:
                continue
            seen.add((a, b))
            out.append((a, b))
    return out


def draw_path(ax, X_train, path, color, linestyle="-", linewidth=2.7, alpha=1.0):
    path = [int(i) for i in path]
    if not path:
        return
    xy = X_train[path]
    ax.plot(xy[:, 0], xy[:, 1], color=color, linestyle=linestyle, linewidth=linewidth, alpha=alpha, zorder=7)
    ax.scatter(xy[:, 0], xy[:, 1], s=38, facecolors="white", edgecolors=color, linewidths=1.7, zorder=8)
    for a, b in zip(xy[:-1], xy[1:]):
        ax.annotate(
            "",
            xy=b,
            xytext=a,
            arrowprops={"arrowstyle": "->", "color": color, "lw": linewidth, "shrinkA": 4, "shrinkB": 4},
            zorder=9,
        )


def panel_a(ax, X_train, model, query, winner, runner, starts):
    wi = class_index(model, winner)
    values = model.contrast_field_[:, wi]
    vmax = max(float(np.max(np.abs(values))), 1e-6)
    sc = ax.scatter(
        X_train[:, 0], X_train[:, 1], c=values, cmap="coolwarm_r", vmin=-vmax, vmax=vmax,
        s=18, alpha=0.78, linewidths=0.0,
    )
    ax.scatter(query[0], query[1], marker="*", s=180, c="black", edgecolors="white", linewidths=0.8, zorder=10)

    pts = np.vstack([query[None, :], X_train[np.asarray(starts, dtype=int)]])
    lo = pts.min(axis=0)
    hi = pts.max(axis=0)
    span = np.maximum(hi - lo, 0.15)
    pad = 0.45 * span
    lo -= pad
    hi += pad
    rect = Rectangle(lo, *(hi - lo), fill=False, linestyle="--", linewidth=1.5, edgecolor="0.20")
    ax.add_patch(rect)

    ax.set_title("a  Class-contrast field", loc="left", fontsize=13, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    cb = plt.colorbar(sc, ax=ax, orientation="horizontal", fraction=0.055, pad=0.075, aspect=30)
    cb.set_ticks([-vmax, 0.0, vmax])
    cb.set_ticklabels([f"runner-up {runner}", "tie", f"winner {winner}"])
    cb.outline.set_visible(False)
    return lo, hi


def panel_b(ax, X_train, y_train, model, query, starts, exp, winner, runner):
    query_scaled = model._scaled(query[None, :])[0]
    wtrail = exp["classes"][winner]["representative_trail"]
    rtrail = exp["classes"][runner]["representative_trail"]
    nodes = local_nodes(model, starts, wtrail["path"], rtrail["path"], query_scaled)
    edges = local_edges(model, nodes)

    for i, j in edges:
        ax.plot(
            [X_train[i, 0], X_train[j, 0]], [X_train[i, 1], X_train[j, 1]],
            color=EDGE, linewidth=0.55, alpha=0.58, zorder=1,
        )

    wi = class_index(model, winner)
    values = model.contrast_field_[nodes, wi]
    vmax = max(float(np.max(np.abs(values))), 1e-6)
    ax.scatter(
        X_train[nodes, 0], X_train[nodes, 1], c=values, cmap="coolwarm_r", vmin=-vmax, vmax=vmax,
        s=28, alpha=0.72, edgecolors="white", linewidths=0.30, zorder=2,
    )

    starts = np.asarray(starts, dtype=int)
    ax.scatter(
        X_train[starts, 0], X_train[starts, 1], s=78, facecolors="none", edgecolors="0.30",
        linewidths=1.1, zorder=5,
    )

    support = [int(r["node_index"]) for r in exp["supporting_neighbors"]]
    ax.scatter(
        X_train[support, 0], X_train[support, 1], s=115, facecolors="none", edgecolors=SUPPORT,
        linewidths=2.0, zorder=6,
    )

    draw_path(ax, X_train, wtrail["path"], WINNER, linestyle="-", linewidth=2.8)
    draw_path(ax, X_train, rtrail["path"], RUNNER, linestyle="--", linewidth=2.5)
    ax.scatter(query[0], query[1], marker="*", s=220, c="black", edgecolors="white", linewidths=0.8, zorder=10)

    nearest = int(starts[0])
    top_labels = [int(r["training_label"]) for r in exp["supporting_neighbors"]]
    winner_support = sum(lbl == int(winner) for lbl in top_labels)
    runner_support = sum(lbl == int(runner) for lbl in top_labels)
    text = (
        f"nearest neighbour: class {int(y_train[nearest])}\n"
        f"top-5 support: {winner_support} winner / {runner_support} runner-up\n"
        f"prediction: class {winner}"
    )
    ax.text(
        0.02, 0.98, text, transform=ax.transAxes, ha="left", va="top", fontsize=8.8,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "0.82", "alpha": 0.93},
        zorder=20,
    )

    all_xy = np.vstack([
        query[None, :], X_train[nodes],
    ])
    lo = all_xy.min(axis=0)
    hi = all_xy.max(axis=0)
    span = np.maximum(hi - lo, 0.2)
    pad = 0.06 * span
    ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0])
    ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
    ax.set_title("b  Local graph explanation", loc="left", fontsize=13, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linestyle("--")
        s.set_linewidth(1.0)
        s.set_edgecolor("0.25")


def panel_c(ax, model, exp, winner, runner):
    wt = exp["classes"][winner]["representative_trail"]
    rt = exp["classes"][runner]["representative_trail"]
    xw = np.arange(len(wt["evidence_trace"]))
    xr = np.arange(len(rt["evidence_trace"]))
    ax.plot(xw, wt["evidence_trace"], marker="o", color=WINNER, linewidth=2.2, label=f"winner {winner}")
    ax.plot(xr, rt["evidence_trace"], marker="s", color=RUNNER, linewidth=2.0, linestyle="--", label=f"runner-up {runner}")

    wi = class_index(model, winner)
    threshold = float(model.target_threshold_[wi])
    ax.axhline(threshold, color="0.40", linestyle=":", linewidth=1.2)
    ax.text(0.04, threshold + 0.015, "winner basin threshold", fontsize=8.0, color="0.35")
    ax.set_title("c  Evidence along representative trails", loc="left", fontsize=12.5, fontweight="bold")
    ax.set_xlabel("representative-trail step")
    ax.set_ylabel("contrastive evidence")
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    ax.grid(axis="y", alpha=0.18)


def panel_d(ax, exp, winner, runner):
    deltas, margin, reconstructed = stored_contrastive(exp, winner, runner)
    labels = ["local field", "trail score", "walker agreement", "lost trails"]
    vals = [deltas["local"], deltas["trail"], deltas["agreement"], deltas["lost"]]
    colors = [WINNER if v >= 0 else RUNNER for v in vals]
    y = np.arange(len(vals))
    ax.barh(y, vals, color=colors, alpha=0.88)
    ax.axvline(0.0, color="0.30", linewidth=0.9)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_title(f"d  Exact 'why {winner} rather than {runner}'", loc="left", fontsize=12.5, fontweight="bold")
    ax.set_xlabel(f"score(class {winner}) - score(class {runner})")
    span = max(abs(v) for v in vals) if vals else 1.0
    for yi, v in zip(y, vals):
        pad = 0.025 * max(span, 0.1)
        ax.text(v + (pad if v >= 0 else -pad), yi, f"{v:+.3f}", va="center", ha="left" if v >= 0 else "right", fontsize=8.7)
    ax.text(
        0.98, 0.07,
        f"margin = {margin:.3f}\nreconstructed = {reconstructed:.3f}\n|error| = {abs(margin-reconstructed):.1e}",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "0.82"},
    )
    ax.grid(axis="x", alpha=0.18)


def make_figure(seed: int, query_index: int, output_prefix: Path, json_path: Path | None):
    X, y = make_spirals(random_state=seed)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=7000 + seed, stratify=y,
    )
    model = classifier().fit(X_train, y_train)

    # Evaluate the complete test split once. This preserves the same frozen RNG
    # progression for prediction, explanation, and score decomposition.
    pred, explanations = explain_mcgc(model, X_test, top_support=5)
    if not (0 <= query_index < len(X_test)):
        raise ValueError(f"query_index must be in [0, {len(X_test)-1}]")

    exp = explanations[query_index]
    query = X_test[query_index]
    winner = int(exp["prediction"])
    runner = int(exp["runner_up"])
    distances, starts, affinity = query_neighbours(model, query)
    deltas, margin, reconstructed = stored_contrastive(exp, winner, runner)

    fig = plt.figure(figsize=(14.5, 8.3))
    gs = GridSpec(
        2, 3, figure=fig,
        width_ratios=(1.02, 1.72, 1.12),
        height_ratios=(1.0, 0.72),
        wspace=0.28, hspace=0.34,
    )
    ax_a = fig.add_subplot(gs[:, 0])
    ax_b = fig.add_subplot(gs[:, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    ax_d = fig.add_subplot(gs[1, 2])

    panel_a(ax_a, X_train, model, query, winner, runner, starts)
    panel_b(ax_b, X_train, y_train, model, query, starts, exp, winner, runner)
    panel_c(ax_c, model, exp, winner, runner)
    panel_d(ax_d, exp, winner, runner)

    fig.suptitle(
        "MCGC explanation anatomy on a nonlinear manifold",
        fontsize=17, fontweight="bold", y=0.985,
    )
    fig.text(
        0.5, 0.018,
        f"spirals_hard · seed {seed} · test query #{query_index} · true {int(y_test[query_index])} · predicted {winner} · runner-up {runner}",
        ha="center", fontsize=9.2, color="0.30",
    )
    fig.subplots_adjust(top=0.92, bottom=0.09, left=0.05, right=0.985)

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    png = output_prefix.with_suffix(".png")
    pdf = output_prefix.with_suffix(".pdf")
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)

    payload = {
        "dataset": "spirals_hard",
        "seed": int(seed),
        "query_index": int(query_index),
        "true_label": int(y_test[query_index]),
        "prediction": int(winner),
        "runner_up": int(runner),
        "correct": bool(int(pred[query_index]) == int(y_test[query_index])),
        "score_margin": float(margin),
        "reconstructed_margin": float(reconstructed),
        "reconstruction_abs_error": float(abs(margin - reconstructed)),
        "component_deltas": deltas,
        "walker_gate": float(exp["walker_gate"]),
        "query_neighbours": [int(i) for i in starts],
        "query_neighbour_affinity": [float(v) for v in affinity],
        "support_nodes": [int(r["node_index"]) for r in exp["supporting_neighbors"]],
        "winner_trail": exp["classes"][winner]["representative_trail"],
        "runner_up_trail": exp["classes"][runner]["representative_trail"],
        "note": "Representative trails are deterministic summaries of the fitted class-conditioned evidence fields; they are not asserted to be the sole stochastic causal trajectory.",
    }
    if json_path is not None:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    print("=== MCGC FIGURE 1: SPIRAL XAI ANATOMY ===")
    print(f"seed={seed} query_index={query_index} true={int(y_test[query_index])} predicted={winner} runner_up={runner}")
    print(f"margin={margin:.6f} reconstructed={reconstructed:.6f} error={abs(margin-reconstructed):.3e}")
    print(f"winner_trail_length={exp['classes'][winner]['representative_trail']['length']}")
    print(f"runner_trail_length={exp['classes'][runner]['representative_trail']['length']}")
    print(f"WROTE {png}")
    print(f"WROTE {pdf}")
    if json_path is not None:
        print(f"WROTE {json_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--query-index", type=int, default=130)
    ap.add_argument("--output-prefix", type=Path, default=Path("figures/fig1_spiral_xai_publication"))
    ap.add_argument("--json", type=Path, default=Path("fig1_spiral_xai_publication.json"))
    args = ap.parse_args()
    make_figure(args.seed, args.query_index, args.output_prefix, args.json)


if __name__ == "__main__":
    main()
