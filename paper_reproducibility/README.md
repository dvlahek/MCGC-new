# CGEC paper reproducibility and explainability

This directory isolates the **field-only CGEC definition used by the final manuscript** from legacy development code in the repository. Prediction in this folder is determined only by affinity-weighted local class evidence. The representative chemotactic path is constructed after prediction from the same fitted graph-evidence field and does not enter the class score.

## Contents

- `xai/field_only_cgec.py` — paper-locked field-only graph, diffusion, local evidence, exact neighbour margin decomposition, paired feature interventions, support-example, and deterministic representative-path implementation.
- `xai/explain_query.py` — CLI that exports one query explanation as JSON for `digits`, `breast_cancer`, or `spirals_hard`.
- `figures/fig_spiral_evidence.py` — regenerates the noisy-spiral graph-evidence/path figure from a frozen held-out query.
- `figures/fig_digits_xai.py` — regenerates the handwritten-digits decision decomposition: class evidence, exact neighbour contributions, paired feature interventions, and winner/runner paths.
- `figures/fig_path_validation.py` — regenerates the path-stability and delete-and-refit validation figure from final v3.0 aggregate inputs.
- `figures/fig_bc_interventions.py` — regenerates the breast-cancer feature-intervention/stability figure. Displayed features are selected by the aggregate five-split ranking, not a single split.
- `figures/fig_predictive_boundary.py` — regenerates the predictive selective-risk figure from final v3.0 aggregate inputs.
- `figures/fig3_oof_vs_knn.py` — regenerates the strict out-of-fold label-review comparison against OOF kNN disagreement. Rows are sorted once by ROC-AUC difference and reused in both panels; positive and negative differences use distinct lollipop styling.
- `figures/fig_spiral_waterfall.py` — supplementary exact neighbour decomposition for the frozen spiral query.
- `results/figure_inputs.json` — compact exact aggregate inputs used to regenerate the predictive and path-validation figures. The full benchmark JSONs remain part of the frozen experiment archive.
- `results/fig3_oof_vs_knn_summary.json` — compact frozen dataset-level ROC-AUC and PR-AUC differences used to regenerate Fig. 3 without the full strict-OOF archive.
- `provenance/` — frozen query-level reference records for the manuscript examples.
- `outputs/` — locally regenerated PDF/PNG files plus runtime provenance and summaries; ignored by Git.

## Frozen CGEC configuration

- `K = 12`
- `alpha = 0.85`
- `n_D = 40`
- representative path weights `(beta1, beta2, beta3, beta4) = (1.0, 1.5, 0.35, 0.75)`
- `s_max = 10`
- `delta_floor = 1e-8`
- numerical safeguard `epsilon = 1e-12`

## Explanation quantities exposed by the public paper API

For a query `q`, `FieldOnlyCGEC.explain_query()` returns:

- prediction and runner-up class,
- local class evidence `L_c(q)` and the winner-runner margin `M(q)`,
- affinity-ranked support examples,
- exact neighbour margin contributions

  `C_j(q) = a_j(q) [D_j^(c+) - D_j^(c-)]`, with `sum_j C_j(q) = M(q)`,

- paired feature intervention effects `I_k(q)` using the training-mean reference and the original winner/runner classes,
- class-specific entry nodes, representative paths, evidence traces, stopping thresholds, and termination reasons.

## Regenerate the paper figures

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r paper_reproducibility/requirements.txt
python paper_reproducibility/regenerate_explainability_figures.py
```

Generated files are written to `paper_reproducibility/outputs/`:

- `fig_spiral_evidence.pdf/png`
- `fig_digits_real_xai.pdf/png`
- `fig_path_validation.pdf/png`
- `fig_bc_interventions.pdf/png`
- `fig_predictive_boundary.pdf/png`
- `fig3_oof_vs_knn.pdf/png`
- `fig_spiral_waterfall_supplement.pdf/png`

The default Fig. 3 regeneration uses `results/fig3_oof_vs_knn_summary.json`. If the original v2.1 strict-OOF audit archive is available, the same figure can be recomputed directly from it:

```bash
python paper_reproducibility/figures/fig3_oof_vs_knn.py \
  --audit-jsonl /path/to/v21_oof_audit.jsonl \
  --output-dir paper_reproducibility/outputs
```

## Query-level explanation

Frozen digits example:

```bash
cd paper_reproducibility/xai
python explain_query.py \
  --dataset digits \
  --seed 0 \
  --query-index 122 \
  --output ../outputs/digits_query_explanation.json
```

Frozen spiral example:

```bash
python explain_query.py \
  --dataset spirals_hard \
  --seed 0 \
  --query-index 130 \
  --output ../outputs/spiral_query_explanation.json
```

## Frozen query checks

The expected digits query is true label 3, predicted class 3, runner-up class 8, with local evidence margin approximately `0.5597656`.

The noisy-spiral query is true label 0 and predicted class 0, with `L0 ≈ +0.048763`, `L1 ≈ -0.048763`, and margin `≈ 0.097526`. Its fitted thresholds are class specific: `tau0 ≈ 0.281323` and `tau1 ≈ 0.281982`.

## Breast-cancer intervention figure

The global feature view uses five independent stratified 70/30 splits with `random_state = 7000 + seed`, `seed = 0,...,4`. The 12 displayed features are selected by mean `|I_k(q)|` averaged across **all five splits**. Panel A then shows class-oriented query-level effects for split 0, while Panel B shows the five split-level mean absolute effects for the same aggregate-selected features. This avoids selecting displayed features from one split and then presenting them as a five-split ranking.

## Scope

This folder is intentionally separate from the historical `src/mcgc/xai.py`, which contains earlier development machinery. The manuscript and its figures should reproduce from this folder until the package-level API is refactored to the final field-only definition.
