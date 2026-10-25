# CGEC paper reproducibility and explainability

This directory isolates the **field-only CGEC definition used by the final manuscript** from legacy development code in the repository. Prediction in this folder is determined only by affinity-weighted local class evidence. The representative chemotactic path is constructed after prediction from the same fitted graph-evidence field and does not enter the class score.

## Contents

- `xai/field_only_cgec.py` — paper-locked field-only graph, diffusion, local evidence, prediction, support-example, and deterministic representative-path implementation.
- `xai/explain_query.py` — small CLI that exports one query explanation as JSON.
- `figures/fig_spiral_evidence.py` — regenerates the noisy-spiral query-level evidence/path figure from a frozen held-out query.
- `figures/fig_digits_xai.py` — regenerates the handwritten-digits query explanation with support examples, winner/runner paths, and paired feature interventions.
- `figures/fig_path_validation.py` — regenerates the path-stability and delete-and-refit validation figure from the final v3.0 summaries.
- `figures/fig_predictive_boundary.py` — regenerates the final predictive-boundary figure from the final v3.0 predictive summaries.
- `results/figure_inputs.json` — compact, exact aggregate inputs used to regenerate the predictive and path-validation figures. The full benchmark JSONs remain part of the frozen experiment archive.
- `provenance/` — frozen query-level provenance for the paper examples.
- `outputs/` — generated PDF/PNG files (created locally and intentionally not versioned).

## Frozen CGEC configuration

- `K = 12`
- `alpha = 0.85`
- `n_D = 40`
- representative path weights `(beta1, beta2, beta3, beta4) = (1.0, 1.5, 0.35, 0.75)`
- `s_max = 10`
- `delta_floor = 1e-8`
- numerical safeguard `epsilon = 1e-12`

## Regenerate the paper figures

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r paper_reproducibility/requirements.txt
python paper_reproducibility/regenerate_explainability_figures.py
```

Generated files are written to `paper_reproducibility/outputs/`.

## Query-level explanation

For the frozen digits example:

```bash
cd paper_reproducibility/xai
python explain_query.py --dataset digits --seed 0 --query-index 122 --output ../outputs/digits_query_explanation.json
```

The JSON contains the prediction, runner-up class, local class evidence, winner-runner margin, affinity-ranked support examples, class-specific entry nodes, representative paths, evidence traces, class-specific thresholds, and termination reasons.

## Frozen query checks

The scripts contain assertions for the paper examples. The expected digits query is true label 3, predicted class 3, runner-up class 8, with local evidence margin approximately `0.5597656`. The noisy-spiral query is true label 0 and predicted class 0, with `L0 ≈ +0.048763`, `L1 ≈ -0.048763`, and margin `≈ 0.097526`. Its fitted thresholds are class specific: `tau0 ≈ 0.281323` and `tau1 ≈ 0.281982`.

## Scope

This folder is intentionally separate from the historical `src/mcgc/xai.py`, which contains earlier development machinery. The manuscript and its figures should reproduce from this folder until the package-level API is refactored to the final field-only definition.
