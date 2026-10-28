# CGEC paper reproducibility and explainability

This directory is the canonical reproducibility bundle for the manuscript **“Chemotactic Graph Evidence for Prediction Explanation and Training-Label Review.”** It isolates the **field-only CGEC** definition used by the paper from historical development code elsewhere in the repository.

Prediction in this folder is determined only by affinity-weighted local class evidence. Stochastic walkers are not part of the paper classifier or its explanations. The representative chemotactic path is constructed after prediction from the same fitted graph-evidence field and does not enter the class score.

## Canonical paper files

- `xai/field_only_cgec.py` — paper-locked field-only graph, diffusion, local evidence, exact neighbour margin decomposition, paired feature interventions, support examples, and deterministic representative paths.
- `xai/explain_query.py` — query-level explanation CLI.
- `figures/` — generators for the manuscript figures and supplementary spiral waterfall.
- `benchmarks/` — frozen v2.9/v3.0 confirmation scripts used for the final predictive, path, delete-and-refit, and feature-stability claims.
- `results/final_manuscript_results.json` — compact numerical lock for the reported manuscript values.
- `results/lr_rival_margin_control_analysis.json` — full eight-dataset analysis for the fair LR rival-margin control in Table 5.
- `results/figure_inputs.json` — compact exact aggregate inputs used to regenerate the predictive and path-validation figures.
- `results/fig3_oof_vs_knn_summary.json` — frozen dataset-level inputs used to regenerate Fig. 3.
- `results/bc_interventions_summary.json` — frozen breast-cancer intervention summary.
- `provenance/` — frozen query-level reference records for the manuscript examples.
- `REPRODUCIBILITY_LOCK.md` — manuscript-to-code/result map and scope lock.

## Frozen CGEC configuration

- `K = 12`
- `alpha = 0.85`
- `n_D = 40`
- representative path weights `(beta1, beta2, beta3, beta4) = (1.0, 1.5, 0.35, 0.75)`
- `s_max = 10`
- `delta_floor = 1e-8`
- numerical safeguard `epsilon = 1e-12`
- strict label-review folds `J = 5`

## Installation

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -U pip
pip install -e ".[dev]"
pip install -r paper_reproducibility/requirements.txt
```

The paper bundle uses NumPy, SciPy, scikit-learn, matplotlib, pandas, and Cleanlab. The PMLB label-review datasets can be downloaded in the flat format expected by the frozen v2.1 scripts with:

```bash
python examples/download_v21_pmlb_flat.py --out-dir data_pmlb
```

## Final benchmark chain

### Parameter sensitivity — Table 2

```bash
python paper_reproducibility/benchmarks/benchmark_field_core_sensitivity_v29.py \
  --output results/v29_field_core_parameter_sensitivity.json
```

This is a one-factor-at-a-time sensitivity analysis only. No configuration is selected from these results.

### Predictive selective-risk confirmation — Table 3 / Fig. 2

```bash
python paper_reproducibility/benchmarks/benchmark_field_only_final_predictive_v30.py \
  --v26-json /path/to/v26_corrected_predictive_baselines.json \
  --output results/v30_field_only_final_predictive.json
```

The v3.0 confirmation reuses the previously frozen tuned-baseline results and reruns only the final field-only CGEC definition.

### Representative-path stability — Fig. 6(a)

```bash
python paper_reproducibility/benchmarks/benchmark_field_only_path_stability_v30.py \
  --output results/v30_field_only_path_stability.json
```

### Delete-and-refit path controls — Fig. 6(b)

```bash
python paper_reproducibility/benchmarks/benchmark_field_only_faithfulness_v30.py \
  --output results/v30_field_only_faithfulness.json
```

The reported controls are (i) equal-size label- and distance-matched off-path nodes and (ii) the stronger signed local-evidence control based on positive exact neighbour margin contributions.

### Feature-intervention stability — Fig. 7

```bash
python paper_reproducibility/benchmarks/benchmark_field_only_feature_stability_v30.py \
  --output results/v30_field_only_feature_stability.json
```

## Strict out-of-fold label review — Tables 4 and 5 / Fig. 3

The label-review scripts remain in `examples/` because they share the frozen v2.1 audit protocol and PMLB loader:

```bash
python examples/benchmark_label_audit_v21_cleanlab_only.py \
  --data-dir data_pmlb \
  --out results/v21_cleanlab_only.jsonl

python examples/benchmark_label_audit_v21_cleanlab_find_issues.py \
  --data-dir data_pmlb \
  --out results/v21_cleanlab_find_issues.jsonl

python examples/benchmark_label_audit_v21_rival_recovery.py \
  --data-dir data_pmlb \
  --out results/v21_rival_recovery.jsonl

python examples/benchmark_label_audit_v21_lr_rival_margin_control.py \
  --data-dir data_pmlb \
  --out results/v21_lr_rival_margin_control.jsonl
```

The LR rival-margin control uses

`max_{c != y} p_c - p_y`

with rival class `argmax_{c != y} p_c`, using the same strict-OOF LR probabilities as the Cleanlab normalized-margin comparison. This is the directly matched probabilistic control for the CGEC review-score / rival-class pairing.

### Frozen Table 4 means

| Method | ROC-AUC | PR-AUC | P@5% | P@10% |
|---|---:|---:|---:|---:|
| CGEC | 0.8178 | 0.6010 | 0.6914 | 0.6343 |
| RF confidence | 0.8186 | 0.6146 | 0.7032 | 0.6490 |
| LR confidence | 0.8353 | 0.6224 | 0.7282 | 0.6739 |
| Cleanlab (OOF LR, normalized margin) | 0.8292 | 0.5936 | 0.6999 | 0.6356 |
| kNN disagreement | 0.8042 | 0.5518 | 0.6170 | 0.5907 |
| Label Spreading | 0.8082 | 0.5850 | 0.6631 | 0.6187 |

CGEC exceeds OOF kNN disagreement on ROC-AUC and PR-AUC in 10 of 12 datasets (`p = 0.0386` for both exact two-sided sign tests).

### Frozen Table 5 means

Rival-class recovery is restricted to the eight multiclass datasets because the binary case has only one alternative class.

| Method | All corrupted | Top 5% | n_corr,5 | Top 10% | n_corr,10 |
|---|---:|---:|---:|---:|---:|
| CGEC | 0.7613 | 0.8934 | 6168 | 0.8687 | 11550 |
| RF confidence | 0.7709 | 0.7614 | 6458 | 0.7581 | 12145 |
| LR confidence | 0.7546 | 0.7430 | 6192 | 0.7460 | 11797 |
| LR rival margin | 0.7546 | 0.8735 | 5865 | 0.8413 | 10930 |
| kNN disagreement | 0.7525 | 0.7499 | 5681 | 0.7621 | 10963 |
| Label Spreading | 0.7417 | 0.7317 | 6065 | 0.7352 | 11567 |

Against the standard RF/LR confidence rankings, CGEC is higher on all eight multiclass datasets at both top-budget levels; the four comparisons remain below `0.05` after Holm correction across the six top-budget comparisons (`p_Holm = 0.046875`). Against the directly matched LR rival-margin control, CGEC is higher on six of eight datasets at both budgets but the paired differences are not significant (`6/0/2`, raw `p = 0.2890625`, `p_Holm = 0.578125`).

Top-budget recovery is a joint property of review prioritization and rival-class quality. The pooled corrupted-sample denominators are reported explicitly and recovery should not be interpreted as an isolated explanation-accuracy metric.

## Regenerate manuscript figures

```bash
python paper_reproducibility/regenerate_explainability_figures.py
```

Generated files are written to `paper_reproducibility/outputs/`, which is ignored by Git.

The default Fig. 3 regeneration uses `results/fig3_oof_vs_knn_summary.json`. If the original v2.1 strict-OOF audit JSONL is available, the figure can also be recomputed directly from it.

## Query-level explanation

Frozen digits example:

```bash
cd paper_reproducibility/xai
python explain_query.py --dataset digits --seed 0 --query-index 122 \
  --output ../outputs/digits_query_explanation.json
```

Expected: true label `3`, predicted class `3`, runner-up `8`, local evidence margin approximately `0.5597656`.

Frozen noisy-spiral example:

```bash
python explain_query.py --dataset spirals_hard --seed 0 --query-index 130 \
  --output ../outputs/spiral_query_explanation.json
```

Expected: true label `0`, predicted class `0`, `L0 ≈ +0.048763`, `L1 ≈ -0.048763`, margin `≈ 0.097526`, with class-specific thresholds `tau0 ≈ 0.281323` and `tau1 ≈ 0.281982`.

## Public explanation quantities

For a query `q`, `FieldOnlyCGEC.explain_query()` exposes:

- prediction and runner-up class;
- local class evidence `L_c(q)` and top-two margin `M(q)`;
- affinity-ranked support examples;
- exact neighbour margin contributions `C_j(q) = a_j(q)[D_j^(c+) - D_j^(c-)]`, with `sum_j C_j(q) = M(q)`;
- paired feature intervention effects `I_k(q)` using the training-mean reference and fixed original winner/runner classes;
- class-specific entry nodes, representative paths, evidence traces, thresholds, and termination reasons.

## Scope lock

The manuscript and its figures must reproduce from this folder and the explicitly referenced frozen v2.1 label-review scripts. Historical `src/mcgc/xai.py` code contains earlier walker-based development machinery and is not the manuscript method.
