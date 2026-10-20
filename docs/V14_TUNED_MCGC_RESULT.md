# MCGC v1.4 — fair tuned-vs-tuned confirmation

## Status

Frozen confirmation completed. No post-result grid enlargement or retuning was performed.

Protocol branch: `feature/v1.4-tuned-mcgc`

Frozen checkpoints:

- protocol: `722eb4e9354d0ba76aedc0085d9b6df60078c3d1`
- implementation: `7ee220338b922e2b677acd55cc45eacb7b6c3b98`
- protocol tests: `892066843eecd3ed6f512829f5e51d4f7ba40504`

## Question

Does MCGC's local graph-field margin become more competitive when MCGC itself receives the same nested inner-CV hyperparameter selection used for the comparison models?

## Frozen protocol

Outer split: 70% train / 30% untouched test, with `random_state=5000+seed` for seeds 0–11.

Inner selection: 3-fold stratified CV on outer training only, `random_state=6000+seed`. The selection target is lowest full-OOF E-AURC, with OOF accuracy and deterministic candidate key used only as tie-breaks.

Primary new real-data family:

- `digits`
- `wine`

Previously seen control:

- `breast_cancer`

MCGC grid (18 candidates):

- `n_neighbors ∈ {7,12,17}`
- `diffusion_alpha ∈ {0.75,0.85,0.93}`
- `diffusion_steps ∈ {20,40}`

The previous frozen MCGC setting `(12, 0.85, 40)` is retained as an explicit reference model.

Baselines retain the v1.2 search spaces for tuned kNN, logistic regression, RBF-SVM and Random Forest.

Primary paired advantage is `baseline E-AURC - tuned MCGC E-AURC`; positive values favor tuned MCGC. The tuned-vs-frozen MCGC advantage is `frozen MCGC E-AURC - tuned MCGC E-AURC`; positive values mean tuning helped MCGC.

## Results

### Per-dataset mean E-AURC

| dataset | frozen MCGC | tuned MCGC | tuned kNN | tuned logistic | tuned RBF-SVM | tuned RF |
|---|---:|---:|---:|---:|---:|---:|
| digits | 0.003794 | 0.004224 | 0.001382 | 0.002566 | 0.003888 | 0.000604 |
| wine | 0.002111 | 0.003359 | 0.057190 | 0.000365 | 0.000840 | 0.000716 |
| breast_cancer | 0.005902 | 0.004654 | 0.020937 | 0.002234 | 0.002614 | 0.004775 |

### Tuning effect on MCGC

`digits`:

- mean tuned-vs-frozen E-AURC advantage `-0.000430`
- W/T/L `6/0/6`
- sign-test p `1.0`

`wine`:

- mean advantage `-0.001248`
- W/T/L `2/3/7`
- p `0.179688`

`breast_cancer` control:

- mean advantage `+0.001248`
- W/T/L `11/0/1`
- p `0.006348`

Thus tuning substantially helps the conventional breast-cancer control but does not improve the primary real-data family. On average across `digits` and `wine`, tuned MCGC is slightly worse than the previous frozen setting.

### Primary new real-data pooled result (24 runs)

| contrast | mean E-AURC advantage | median | W/T/L | sign-test p |
|---|---:|---:|---:|---:|
| tuned MCGC vs frozen MCGC | -0.000839 | -0.000384 | 8/3/13 | 0.383310 |
| tuned MCGC vs tuned kNN | +0.025494 | +0.015779 | 13/0/11 | 0.838820 |
| tuned MCGC vs tuned logistic | -0.002326 | -0.001872 | 4/1/19 | 0.002599 |
| tuned MCGC vs tuned RBF-SVM | -0.001428 | -0.001698 | 5/1/18 | 0.010622 |
| tuned MCGC vs tuned RF | -0.003132 | -0.003117 | 1/0/23 | 2.98e-06 |

The positive pooled mean versus kNN is driven by the very large wine advantage and is not a stable paired win: W/T/L is only `13/0/11` and p is `0.8388`.

### Dataset-specific contrasts

`digits`:

- tuned MCGC loses to tuned kNN in `11/12` seeds;
- loses to tuned RF in `12/12` seeds;
- is roughly competitive with tuned SVM in mean E-AURC but loses the paired count `4/0/8`;
- tuning itself does not improve the previous frozen MCGC.

`wine`:

- tuned MCGC strongly beats tuned kNN in `12/12` seeds;
- loses to tuned logistic in `11` non-tied seeds;
- loses to tuned RF in `11/12` seeds;
- loses to tuned SVM in `10` non-tied seeds;
- tuning again does not improve the frozen MCGC.

`breast_cancer` control:

- tuning improves MCGC in `11/12` seeds;
- tuned MCGC still strongly beats tuned kNN (`12/12`);
- it becomes approximately even with tuned RF in mean E-AURC but not in win count (`4/0/8`);
- tuned logistic and tuned SVM remain better.

## Hyperparameter-selection diagnostic

The selected MCGC settings are strongly concentrated on the lower edge of the frozen search grid:

- `diffusion_alpha=0.75` selected in `34/36` dataset×seed runs;
- `diffusion_steps=20` selected in `28/36` runs;
- `n_neighbors=7` selected in `25/36` runs.

Per dataset, `diffusion_alpha=0.75` is selected for all 12 breast-cancer runs, all 12 digits runs and 10/12 wine runs.

This is scientifically informative but does **not** justify post-result expansion of the v1.4 grid. A lower-alpha search could only be a separately preregistered future development study, not a repair of this frozen confirmation.

## Interpretation

v1.4 answers the fairness objection directly: the lack of broad real-data selective-risk superiority in v1.3 is **not explained by leaving MCGC untuned while tuning only the baselines**.

Under the same nested E-AURC selection framework, tuning MCGC does not improve the primary `digits`/`wine` family and does not close the gap to tuned Random Forest, logistic regression or RBF-SVM. It does improve MCGC markedly on the previously seen breast-cancer control, showing that the tuning mechanism is capable of selecting a materially better MCGC configuration when the signal is present.

Therefore the supported performance story remains geometry-dependent rather than universal. The strongest manuscript contribution remains the v0.9 interventionally validated intrinsic path explanation result. Selective-risk performance should be presented as a secondary property with explicit dataset/geometry boundaries.

## Supported reviewer-resistant statement

> When all methods, including MCGC, are tuned by the same nested OOF E-AURC protocol, MCGC does not show general selective-risk superiority on the new `digits` and `wine` real-data family. Tuning materially improves MCGC on the breast-cancer control but not on the primary family. This rules out unequal hyperparameter treatment as the explanation for the negative real-data transfer observed in v1.3.

## Decision

Close the performance-tuning line here. Do not enlarge the v1.4 grid after observing this result. Move next to XAI stability, multiclass explanation validation, path-eligibility analysis, novelty audit and manuscript figures.