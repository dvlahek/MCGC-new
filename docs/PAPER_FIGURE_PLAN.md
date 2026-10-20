# MCGC paper figure plan

This document freezes the visual communication plan for the manuscript after v1.5. The purpose is to make the figures tell one coherent scientific story rather than reproduce every benchmark table.

## Visual grammar

Use the same graphical language across all figures.

- Winner class: one saturated color.
- Runner-up class: second saturated color.
- Other classes / context points: neutral gray.
- Query: large star or diamond.
- Supporting training nodes: circles with emphasized outline.
- Winner representative path: solid line with arrows.
- Runner-up path: thinner or dashed line.
- Off-path / control samples: gray or hollow markers.
- Zero / no-effect reference lines: neutral dark gray.
- Avoid rainbow maps, unnecessary 3D, excessive legends, dashboard-like styling, and more than 2–3 accent colors per panel.
- All main-paper plots should remain legible in grayscale and when reduced to journal-column width.

## Figure 1 — Method overview

### Purpose

Explain what MCGC does and make clear that prediction, reliability and explanation emerge from the same fitted graph model.

### Layout

Four left-to-right panels.

**A. Sample graph construction**
- training samples in feature space
- local k-NN edges
- class labels shown by point fill

**B. Class-conditioned evidence fields**
- same graph
- schematic evidence intensity for two example classes
- emphasize contrastive field rather than generic diffusion alone

**C. Query scoring / graph traversal**
- inserted query
- nearest graph anchors
- example stochastic walkers faintly
- deterministic representative path highlighted

**D. Model outputs**
- predicted class and runner-up
- class scores
- local graph-field margin / risk
- top support samples
- exact contrastive score decomposition
- representative winner and runner-up paths

### Main caption message

MCGC is a sample-graph classifier in which class-conditioned evidence fields, graph traversal and exact score decomposition provide prediction, selective confidence and intrinsic explanation without fitting a separate post-hoc explainer.

## Figure 2 — One decision fully explained

### Purpose

Give the reader one immediately understandable example of the complete MCGC explanation object.

### Recommended dataset

Use a 2D nonlinear dataset, preferably `spirals_hard`; fall back to `moons_hard` if the spiral example is visually too crowded.

### Panel A — Geometry and evidence

Plot:
- all training points
- k-NN edges only locally around the query to avoid clutter
- winner-vs-runner contrast field as a subtle continuous background
- query as large star
- support samples with larger outlined markers
- winner representative path as solid arrowed line
- runner-up path as dashed line if nontrivial

### Panel B — Evidence along path

x-axis: path step

y-axis: contrastive evidence

Show winner and runner-up representative traces. Mark target threshold if meaningful.

### Panel C — Exact margin decomposition

Horizontal contribution bars for:
- local component delta
- trail component delta
- agreement component delta
- lost-trail component delta

Add a final total bar equal to the class-score margin. The component sum must reconstruct the margin to floating-point precision.

### Panel D — Compact explanation record

Small table / annotation block:
- true class
- predicted class
- runner-up
- total margin
- dominant component
- top support node IDs or labels
- path length
- eligibility flag

### Selection rule

Choose the example by a predeclared visualization rule, not by manually choosing the prettiest success. Preferred rule: among correctly classified eligible queries in one frozen seed, select the query with margin nearest the median eligible-query margin. This yields a representative rather than extreme example.

## Figure 3 — Interventionally validated path faithfulness

### Purpose

Present the v0.9 result as the main empirical XAI validation.

### Main panel

Dataset-by-seed dot/forest plot of paired intervention advantage.

Two contrasts:

1. path-node deletion minus label-and-distance-matched off-path deletion
2. path-node deletion minus equal-size nearest off-path deletion

x-axis: additional winner-vs-runner margin drop caused by deleting path-selected nodes

Vertical reference: 0

Positive values favor the MCGC explanation.

### Preferred plotting unit

Dataset × seed summary, not 240 query-level observations as independent units.

### Annotation

Show:
- matched-control contrast: 20/20 positive dataset×seed units
- nearest-control contrast: 20/20 positive dataset×seed units
- sign-test p approximately 1.91e-06 for both

A light inset may show the 240-query paired distribution, but the dataset×seed aggregation is the reviewer-facing primary visualization.

### Caption message

For queries with nontrivial representative paths, path-selected training samples are more functionally linked to the fitted decision than matched off-path or proximity-only alternatives under delete-and-refit intervention.

## Figure 4 — Explanation stability, multiclass exactness and path eligibility

Three-panel summary of v1.5.

### Panel A — Query-specific stability

Grouped dot plot by dataset.

Rows / metrics:
- support similarity advantage
- representative-path similarity advantage

x-axis: within-query perturbation similarity minus same-class different-query similarity

Reference line: 0

Dataset-level means:
- digits: support ~0.9425, path ~0.9414
- wine: support ~0.8408, path ~0.8127
- spirals_hard: support ~0.7797, path ~0.7570
- breast_cancer: support ~0.9103, path ~0.8904

Primary pooled dataset×seed annotations:
- support: 20/20 positive, mean ~0.8683, p ~1.91e-06
- path: 17/17 evaluable positive, mean ~0.8527, p ~1.53e-05

Do not treat non-evaluable path units as failures; state the denominator explicitly.

### Panel B — Multiclass exactness

Use either a compact log-scale error plot or a two-row table-style panel.

- digits: maximum absolute winner-vs-runner reconstruction error ~8.88e-16
- wine: ~4.44e-16

Main visual message: reconstruction error is at floating-point precision.

### Panel C — Eligibility and difficulty

Left side: eligibility rate bars
- digits ~0.240
- wine ~0.248
- spirals_hard ~0.229
- breast_cancer ~0.282

Right side: eligible vs non-eligible difficulty, using paired dataset/seed summaries of:
- accuracy
- median decision margin

Expected interpretation to test and report, not assume: eligible queries tend to be lower-margin and harder. Avoid implying causality.

### Caption message

Small prediction-preserving input perturbations leave support and representative-path explanations substantially more similar to the original query than explanations for another query of the same predicted class. Multiclass contrastive decomposition is exact to floating-point precision, while nontrivial paths occur for only a subset of queries and are concentrated in more difficult regions.

## Figure 5 — Selective-risk performance and its boundary

### Purpose

Show the secondary performance contribution without overstating it.

### Panel A — Frozen nonlinear-manifold benchmark

Forest/dot plot of paired E-AURC advantage:

baseline E-AURC minus MCGC E-AURC

Compare:
- tuned kNN
- tuned RBF-SVM
- tuned Random Forest

Show per-dataset rows for moons, circles and spirals plus pooled summary.

Positive = MCGC better.

Main message:
- advantage survives against tuned kNN and tuned RF in the predefined manifold family
- tuned RBF-SVM is competitive and defines a clear boundary

### Panel B — Real-data transfer boundary

Use the same x-axis and visual convention for digits and wine; optionally breast_cancer as a control.

Show tuned MCGC vs tuned competitors. This panel should deliberately reveal the negative/mixed transfer result rather than hide it.

### Panel C — Risk–coverage curves

If space allows, include one representative manifold dataset only in the main figure; put the full risk–coverage set in Supplementary Figure S1.

### Caption message

The MCGC local graph-field margin provides a geometry-dependent selective-risk advantage on the predefined nonlinear-manifold family, but this advantage does not transfer universally to conventional real-data benchmarks.

## Supplementary figures

### Figure S1 — Full risk–coverage curves

Datasets:
- moons_hard
- circles_hard
- spirals_hard
- digits
- wine
- breast_cancer

Models:
- MCGC / tuned MCGC as appropriate
- tuned kNN
- tuned RBF-SVM
- tuned RF

### Figure S2 — Multiclass explanation gallery

Include representative frozen examples such as:
- digits: winner 8 vs runner-up 3, if present under the deterministic selection rule
- digits: another common confusion pair
- wine: one representative 3-class explanation

For each: support samples, winner/runner-up path, score decomposition.

### Figure S3 — Stability gallery

For 3–4 representative queries:
- original query
- several prediction-preserving perturbations
- support overlap
- path overlap
- margin change

### Figure S4 — Eligibility distributions

Show per-seed:
- eligible fraction
- error rate among eligible/non-eligible
- margin distributions

### Figure S5 — Negative-result / claim-boundary summary

Compact visual or table showing frozen rejected hypotheses:
- walker terms do not robustly improve raw accuracy
- walker gating does not fix accuracy
- trail disagreement does not add reliable held-out information beyond local margin
- five-feature combined reliability fails broader confirmation
- real-data selective superiority is not universal

This is supplementary transparency, not a headline figure.

## Implementation order

1. Figure 2 — representative full explanation. This fixes the manuscript visual grammar.
2. Figure 3 — v0.9 intervention faithfulness.
3. Figure 4 — v1.5 stability / multiclass / eligibility.
4. Figure 5 — selective-risk result and transfer boundary.
5. Figure 1 — polished conceptual overview after the concrete figures establish terminology and styling.
6. Supplementary figures.

## Reproducibility rules for figure generation

- Every quantitative figure must be generated from frozen JSON/result artifacts, never copied manually from prose.
- Figure-generation scripts should live under `figures/` or `examples/figures/` and be deterministic.
- Store the exact query-selection rule for illustrative explanation panels.
- Do not choose examples after visually inspecting many alternatives without recording the selection criterion.
- Main text should report the same aggregation level shown in the figure.
- Use dataset×seed units for primary stability and faithfulness inference when repeated queries are nested within a fitted model.
- Preserve negative and boundary results in the visual story.

## Tentative main-paper narrative order

1. Figure 1: how MCGC works.
2. Figure 2: what an individual explanation looks like.
3. Figure 3: why the path explanation is functionally faithful.
4. Figure 4: why the explanation is stable, exact in multiclass, and when paths are available.
5. Figure 5: where the intrinsic graph-field confidence helps and where it does not.

This order makes intrinsic XAI the primary contribution and selective prediction a secondary, explicitly bounded property.
