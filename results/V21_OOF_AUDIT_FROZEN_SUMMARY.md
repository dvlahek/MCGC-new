# v2.1 strict-OOF label-audit confirmation

Source GitHub Actions run: `39733526319`.

All 12 dataset jobs completed successfully. Each dataset ran 4 corruption conditions across 5 seeds, giving 240 completed experimental units in total. The final Actions aggregate job failed only because the analysis step imported `mcgc` without installing the local package. The numerical dataset artifacts themselves completed successfully.

## Frozen design

- datasets: 12 unseen PMLB datasets
- conditions: symmetric eta=0.10, 0.20, 0.30; pairflip eta=0.20
- seeds: 0,1,2,3,4
- audit folds: 5
- review budgets: 5%, 10%
- methods: `mcgc_oof`, `cl_rf`, `cl_lr`, `knn_oof`, `label_spreading_oof`
- primary inference unit: dataset
- MCGC and Label Spreading are strictly out-of-fold
- no cleaning, relabeling, source attenuation, or harm-priority heuristic is used

## Mean across 12 dataset-level aggregates

| Method | ROC-AUC | PR-AUC | P@5% | P@10% |
|---|---:|---:|---:|---:|
| MCGC OOF | 0.8178 | 0.6010 | 0.6914 | 0.6343 |
| CL-RF | 0.8186 | 0.6146 | 0.7032 | 0.6490 |
| CL-LR | 0.8353 | 0.6224 | 0.7282 | 0.6739 |
| kNN OOF | 0.8042 | 0.5518 | 0.6170 | 0.5907 |
| Label Spreading OOF | 0.8082 | 0.5850 | 0.6631 | 0.6187 |

## Dataset-level paired comparisons: MCGC minus baseline

### ROC-AUC

| Baseline | Mean delta | Median delta | W/T/L | two-sided sign p |
|---|---:|---:|---:|---:|
| CL-RF | -0.0007 | -0.0028 | 6/0/6 | 1.0000 |
| CL-LR | -0.0175 | -0.0060 | 6/0/6 | 1.0000 |
| kNN OOF | +0.0136 | +0.0070 | 10/0/2 | 0.03857 |
| Label Spreading OOF | +0.0097 | +0.0094 | 9/0/3 | 0.145996 |

### PR-AUC

| Baseline | Mean delta | Median delta | W/T/L | two-sided sign p |
|---|---:|---:|---:|---:|
| CL-RF | -0.0135 | -0.0073 | 5/0/7 | 0.774414 |
| CL-LR | -0.0213 | -0.0111 | 5/0/7 | 0.774414 |
| kNN OOF | +0.0492 | +0.0312 | 10/0/2 | 0.03857 |
| Label Spreading OOF | +0.0160 | +0.0259 | 9/0/3 | 0.145996 |

### Precision at 5% review budget

| Baseline | Mean delta | Median delta | W/T/L | two-sided sign p |
|---|---:|---:|---:|---:|
| CL-RF | -0.0118 | -0.0323 | 5/0/7 | 0.774414 |
| CL-LR | -0.0369 | ~0.0000 | 5/2/5 | 1.0000 |
| kNN OOF | +0.0743 | +0.0558 | 9/0/3 | 0.145996 |
| Label Spreading OOF | +0.0282 | +0.0332 | 9/0/3 | 0.145996 |

### Precision at 10% review budget

| Baseline | Mean delta | Median delta | W/T/L | two-sided sign p |
|---|---:|---:|---:|---:|
| CL-RF | -0.0147 | -0.0288 | 5/0/7 | 0.774414 |
| CL-LR | -0.0395 | -0.0320 | 4/0/8 | 0.387695 |
| kNN OOF | +0.0436 | +0.0245 | 7/0/5 | 0.774414 |
| Label Spreading OOF | +0.0157 | +0.0293 | 9/0/3 | 0.145996 |

## Frozen interpretation

The v2.1 result does not support a claim that MCGC is the strongest generic label-error detector. MCGC is statistically indistinguishable from the confident-learning-style RF and LR baselines at the dataset level, and it is slightly lower on their mean PR-AUC and review precision. MCGC is stronger than OOF kNN on ROC-AUC and PR-AUC at the dataset level (10/12 wins, two-sided sign p=0.03857), and is directionally stronger than OOF Label Spreading.

The supported use-case claim is therefore competitive intrinsic label auditing: MCGC provides a useful training-label review ranking from the same local graph-evidence mechanism used for its intrinsic explanations, while its audit accuracy remains competitive with strong confidence-based alternatives.

Provenance hashes of the locally reconstructed combined outputs from the 12 successful Actions artifacts:

- combined JSONL SHA-256: `7dae854eac2e4c8d872ba02b5cadbabd2499a443f50fbe55a28b219e3e6db721`
- combined analysis JSON SHA-256: `fa0825bba1bfb9bd93384a4cd5626d9ac9ab15e1787f84e6c7eaa2a1d1c9323f`
