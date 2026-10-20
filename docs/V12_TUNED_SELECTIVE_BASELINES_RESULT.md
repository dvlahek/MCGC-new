# MCGC v1.2 — tuned selective-baseline confirmation result

## Status

The frozen v1.2 nested tuning stress test is **mixed but substantively positive**.

The purpose of v1.2 was not to tune MCGC. MCGC remained unchanged from v1.0–v1.1. Instead, kNN, logistic regression, RBF-SVM and Random Forest were allowed to select conventional hyperparameters by 3-fold inner cross-validation on the outer training split, using lowest out-of-fold E-AURC as the primary selection criterion. The outer 30% test split remained untouched until final evaluation.

Primary nonlinear-manifold family:

- `moons_hard`
- `circles_hard`
- `spirals_hard`

Controls:

- `overlap_hd`
- `breast_cancer`

Twelve seeds were used for each dataset.

## Main pooled result

Paired advantage is defined as:

`E-AURC(baseline) - E-AURC(MCGC)`

so positive values favor MCGC.

Across the 36 frozen primary manifold runs:

| baseline | mean E-AURC advantage | median | W/T/L | sign-test p |
|---|---:|---:|---:|---:|
| tuned kNN | `+0.00806614` | `+0.00213883` | `28/0/8` | `0.00119324` |
| tuned logistic | `+0.13530450` | `+0.02771595` | `32/0/4` | `1.94e-06` |
| tuned RBF-SVM | `-0.00210575` | `-0.00016705` | `18/0/18` | `1.0` |
| tuned Random Forest | `+0.00551738` | `+0.00620743` | `26/0/10` | `0.0113310` |

Thus:

1. the v1.0 kNN advantage survives fair inner-CV tuning;
2. the v1.1 Random-Forest advantage also survives tuning;
3. the fixed-SVM aggregate advantage from v1.1 does **not** survive SVM tuning.

This is the main scientific conclusion of v1.2.

## Ordinary AURC

Against tuned Random Forest, ordinary AURC also remains favorable to MCGC:

- mean advantage: `+0.00500993`
- median: `+0.00568940`
- W/T/L: `25/0/11`
- sign-test p: `0.0288167`

Against tuned kNN, mean AURC remains positive (`+0.00656598`) but the sign-test is not significant (`23/0/13`, p `0.13250`).

Against tuned RBF-SVM, ordinary AURC favors SVM in aggregate:

- mean advantage: `-0.00290387`
- W/T/L: `15/0/21`

Therefore the main tuned-baseline claim should remain centered on E-AURC, with tuned RF and tuned kNN as supported positive comparisons.

## Per-dataset E-AURC results

### `moons_hard`

Mean E-AURC:

- MCGC: `0.01880861`
- tuned kNN: `0.02170794`
- tuned RBF-SVM: `0.023661995`
- tuned Random Forest: `0.02439831`

Paired MCGC wins/losses:

- vs tuned kNN: `7/5`
- vs tuned SVM: `10/2`
- vs tuned RF: `9/3`

MCGC remains strongly competitive and beats both tuned nonlinear baselines on moons.

### `circles_hard`

Mean E-AURC:

- MCGC: `0.01725738`
- tuned kNN: `0.02162656`
- tuned RBF-SVM: `0.01498981`
- tuned Random Forest: `0.02886855`

Paired MCGC wins/losses:

- vs tuned kNN: `11/1`
- vs tuned SVM: `3/9`
- vs tuned RF: `11/1`

RBF-SVM remains better on circles. Random Forest and kNN remain worse than MCGC.

### `spirals_hard`

Mean E-AURC:

- MCGC: `0.19206332`
- tuned kNN: `0.20899323`
- tuned RBF-SVM: `0.18316026`
- tuned Random Forest: `0.19141460`

Paired MCGC wins/losses:

- vs tuned kNN: `10/2`
- vs tuned SVM: `5/7`
- vs tuned RF: `6/6`

The tuned SVM is modestly better on average on spirals. Tuned RF is essentially tied with MCGC on this dataset.

## Control datasets

### `overlap_hd`

Mean E-AURC:

- MCGC: `0.07691793`
- tuned kNN: `0.07727432`
- tuned SVM: `0.07459591`
- tuned RF: `0.07044195`

MCGC is approximately tied with tuned kNN and does not beat tuned SVM/RF.

### `breast_cancer`

Mean E-AURC:

- MCGC: `0.00590213`
- tuned kNN: `0.02093747`
- tuned logistic: `0.00223363`
- tuned SVM: `0.00261412`
- tuned RF: `0.00477499`

The conventional tabular controls continue to bound the claim: MCGC is not universally superior.

## Tuning integrity

The v1.2 tuning procedure used only the outer training partition:

- outer split: 70% training / 30% untouched test;
- inner selection: 3-fold stratified CV;
- selection metric: lowest OOF E-AURC;
- tie-break: higher OOF accuracy, then deterministic lexical ordering;
- MCGC tuning: none.

Frozen grids:

- kNN: `n_neighbors` in `{5, 9, 12, 17, 25}`, `weights` in `{uniform, distance}`;
- logistic: `C` in `{0.1, 1, 10}`;
- RBF-SVM: `C` in `{0.1, 1, 10}`, `gamma` in `{scale, 0.1, 1}`;
- Random Forest: `max_depth` in `{None, 6, 12}`, `min_samples_leaf` in `{1, 3, 5}`, `max_features` in `{sqrt, 0.5}`.

No outer-test information was used for baseline selection.

## Interpretation

v1.2 strengthens the selective-risk result, but also narrows it in an important way.

The strongest reviewer-resistant statement is now:

> On the frozen nonlinear-manifold benchmark, the MCGC local graph-field margin retains lower excess selective risk than tuned kNN confidence and tuned Random-Forest confidence under nested inner-CV model selection. Tuned RBF-SVM is competitive overall and removes the fixed-SVM advantage observed in v1.1.

This is stronger than the v1.1 result because the positive kNN and RF comparisons survive baseline-specific hyperparameter selection targeted directly at E-AURC.

At the same time, v1.2 prevents an overclaim: MCGC is not superior to a tuned RBF-SVM on the complete frozen manifold family.

## Claim discipline

Supported:

- MCGC local-margin E-AURC is better than tuned kNN over the frozen manifold family;
- MCGC local-margin E-AURC is better than tuned Random Forest over the frozen manifold family;
- the effect is geometry-dependent rather than universal;
- tuned RBF-SVM is a strong competing method and should be reported prominently.

Not supported:

- universal superiority to tuned nonlinear classifiers;
- superiority to tuned RBF-SVM;
- universal raw-accuracy superiority;
- SOTA selective classification.

## Relation to v0.9

The strongest current MCGC contribution remains two-part:

1. **intrinsic path faithfulness (v0.9)** — representative-path training samples are functionally more decision-relevant than matched and proximity-only controls;
2. **selective confidence (v1.0–v1.2)** — the local graph-field margin provides improved excess selective-risk ranking over tuned kNN and tuned Random Forest on the frozen nonlinear-manifold family.

These two claims are empirically distinct and should remain separated in the paper.