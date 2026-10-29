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

The reviewer audit moves StandardScaler fitting for LR, kNN and Label Spreading inside each OOF training fold. CGEC already fits preprocessing inside each fold through model fitting. The protocol applies the outer 70/30 split first, corrupts only outer-training labels, then performs five-fold OOF scoring.

The full run completed on all 12 PMLB datasets, four corruption conditions and five seeds. Numerical changes relative to the manuscript values are small and the main conclusions are unchanged.

### Table 4 summary

Strict-OOF four-decimal values are:

- CGEC: ROC-AUC 0.8178, PR-AUC 0.6010, P@5 0.6918, P@10 0.6340.
- RF confidence: 0.8186, 0.6146, 0.7032, 0.6490.
- LR confidence: 0.8353, 0.6220, 0.7282, 0.6746.
- Cleanlab OOF LR normalized margin: 0.8292, 0.5932, 0.7003, 0.6365.
- kNN disagreement: 0.8040, 0.5522, 0.6191, 0.5902.
- Label Spreading: 0.8080, 0.5846, 0.6643, 0.6183.

CGEC still exceeds kNN on both ROC-AUC and PR-AUC in 10 of 12 datasets, with exact two-sided sign-test p=0.03857421875 for both metrics.

### Table 5 summary

For the eight multiclass datasets, equal-dataset means are:

- CGEC: all corrupted 0.7612, top 5% 0.8931, top 10% 0.8686.
- RF confidence: 0.7712, 0.7617, 0.7585.
- LR confidence: 0.7544, 0.7428, 0.7455.
- LR rival margin: 0.7544, 0.8732, 0.8416.
- kNN disagreement: 0.7531, 0.7538, 0.7612.
- Label Spreading: 0.7421, 0.7308, 0.7354.

The primary top-budget significance pattern is unchanged. CGEC beats RF confidence and LR confidence on all 8 datasets at both top-5% and top-10%, with Holm-adjusted p=0.046875. Against the matched LR rival-margin control, CGEC is 6/2 at both budgets and the Holm-adjusted p-value remains 0.578125.

### Decision

Use the strict fold-local preprocessing protocol in the manuscript. Replace Table 4 and Table 5 values with the v3.1 strict-OOF values so the reported tables correspond to one internally consistent protocol. The scientific conclusions do not change.

Archived compact summary:
- `v31_strict_oof_scaling_audit_summary.json`

The original full local audit JSON used to generate this summary is `v31_strict_oof_scaling_audit.json`, SHA-256 `530f9599d945dd895cc2d155b81020e7488a58dcd7369c36c8d362fb7cbc692e`, size 1,469,272 bytes.
