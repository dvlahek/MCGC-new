# Reviewer audit notes v3.1

## Query-affinity robustness

The manuscript formulation remains the query-centered weighting

`exp(-d(q,j)^2 / sigma_q^2)`.

A reviewer-audit alternative using pairwise local scaling

`exp(-d(q,j)^2 / max(sigma_q*sigma_j, eps))`

was evaluated while keeping the fitted training graph and class-evidence field fixed. Across 60 dataset-seed units, mean prediction agreement was 0.9939638448 and the mean alternative-minus-current E-AURC difference was -0.0001934777. The audit is retained as robustness evidence and does not redefine the manuscript method.

Files:
- `v31_query_affinity_smoke.json`
- `v31_query_affinity_robustness_summary.json`

## Strict OOF preprocessing audit

The reviewer audit moves StandardScaler fitting for LR, kNN and Label Spreading inside each OOF training fold. CGEC already fits preprocessing inside each fold through model fitting. The audit protocol applies the outer 70/30 split first, corrupts only outer-training labels, then performs five-fold OOF scoring.

The two-dataset seed-0 smoke test completed successfully and is archived in `v31_strict_oof_scaling_smoke_summary.json`.

The full strict-OOF run is pending and should be archived separately when complete. Manuscript Table 4/5 values should not be replaced until that full run is inspected.
