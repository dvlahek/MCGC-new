# CGEC manuscript reproducibility lock

This file defines the paper-facing source of truth for **Chemotactic Graph Evidence for Prediction Explanation and Training-Label Review**.

## Method scope

The manuscript method is **field-only CGEC**. Stochastic walkers are historical development machinery and are not part of the paper prediction rule, confidence score, representative path, feature intervention, or label-review score.

Prediction follows the field-only chain

`training graph -> class evidence S^(c) -> contrast D^(c) -> local query evidence L_c(q) -> argmax_c L_c(q)`.

The deterministic representative path is a post-prediction structural inspection of the fitted graph-evidence representation.

## Frozen operating configuration

- `K = 12`
- `alpha = 0.85`
- `n_D = 40`
- path weights `(1.0, 1.5, 0.35, 0.75)`
- `s_max = 10`
- `delta_floor = 1e-8`
- `epsilon = 1e-12`
- strict label-review folds `J = 5`

## Manuscript-to-code map

| Manuscript item | Canonical code / frozen input |
|---|---|
| Table 1 | configuration above |
| Table 2 | `benchmarks/benchmark_field_core_sensitivity_v29.py` |
| Table 3 / Fig. 2 | `benchmarks/benchmark_field_only_final_predictive_v30.py`, `results/figure_inputs.json` |
| Fig. 3 | `figures/fig3_oof_vs_knn.py`, `results/fig3_oof_vs_knn_summary.json` |
| Table 4 | `examples/benchmark_label_audit_v21_cleanlab_only.py`, `examples/benchmark_label_audit_v21_cleanlab_find_issues.py` |
| Table 5 | `examples/benchmark_label_audit_v21_rival_recovery.py`, `examples/benchmark_label_audit_v21_lr_rival_margin_control.py`, `results/lr_rival_margin_control_analysis.json` |
| Fig. 4 | `figures/fig_digits_xai.py`, `provenance/digits_query.json` |
| Fig. 5 | `figures/fig_spiral_evidence.py`, `provenance/spiral_query.json` |
| Fig. 6(a) | `benchmarks/benchmark_field_only_path_stability_v30.py`, `figures/fig_path_validation.py` |
| Fig. 6(b) | `benchmarks/benchmark_field_only_faithfulness_v30.py`, `figures/fig_path_validation.py` |
| Fig. 7 | `benchmarks/benchmark_field_only_feature_stability_v30.py`, `figures/fig_bc_interventions.py`, `results/bc_interventions_summary.json` |
| Fig. 8 | strict-OOF CGEC review object from the field-only definition |

## Numerical lock

`results/final_manuscript_results.json` is the compact paper-level numerical lock. It records the final values used for the predictive selective-risk comparison, path stability, delete-and-refit controls, feature-intervention stability, Table 4 label review, and Table 5 rival-class recovery.

Important interpretation locks:

1. Lower E-AURC is better. The nonlinear selective-risk result is not a claim of universal accuracy superiority.
2. The representative path is structural multi-step fitted evidence, not an optimal influence ranking.
3. The signed local-evidence control is stronger than the representative path when the objective is direct influence selection.
4. A high label-review score indicates conflict with fitted evidence and is not proof that the observed label is wrong.
5. Rival-class recovery on binary datasets is trivial, so Table 5 uses only the eight multiclass datasets.
6. Top-budget rival recovery jointly reflects review prioritization and rival-class quality.
7. The LR rival-margin control uses `max_{c!=y} p_c - p_y` with rival `argmax_{c!=y} p_c`; it is the directly matched probability-based control for the CGEC review-margin / rival pairing.
8. CGEC substantially exceeds standard RF/LR confidence-ranked rival recovery at the top review budgets, but the difference versus the directly matched LR rival-margin control is small and non-significant.

## Legacy code boundary

Historical walker-based code under `src/mcgc/` and older benchmark scripts remains in the repository for provenance. It must not be substituted for `paper_reproducibility/xai/field_only_cgec.py` when reproducing manuscript explanations or paper-facing claims.
