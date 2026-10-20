# MCGC v1.2 — tuned selective-baseline confirmation protocol

## Goal

v1.1 showed that the frozen MCGC local graph-field margin retained a positive selective-risk result on the predefined nonlinear-manifold family against fixed RBF-SVM and Random-Forest baselines. v1.2 asks the stricter reviewer-facing question:

> Does the same frozen MCGC selective-risk signal remain competitive when the comparison baselines are allowed to tune their own hyperparameters using only the outer training data and the same target metric, E-AURC?

MCGC is not tuned in v1.2. No MCGC parameter, confidence definition, dataset definition, seed, split, metric, or primary family may change after this protocol is frozen.

## Outer evaluation protocol

The outer protocol is inherited unchanged from v1.0/v1.1.

Primary nonlinear-manifold family:

- `moons_hard`
- `circles_hard`
- `spirals_hard`

Transfer/control datasets:

- `overlap_hd`
- `breast_cancer`

For every dataset and seed:

- seeds `0..11`
- 70% outer training
- 30% untouched outer test
- stratified split
- outer split seed `5000 + dataset_seed`

The outer test labels are not used for hyperparameter selection.

## Frozen MCGC

Use the same classifier and risk signal as v1.0/v1.1:

- `n_neighbors=12`
- `n_walkers=12`
- `max_steps=10`
- `diffusion_steps=40`
- `random_state=17`
- `adaptive_walker_gate=True`
- risk = `local_margin_risk`

MCGC receives no inner-CV tuning.

## Baseline tuning rule

Each baseline is tuned independently inside the outer training split with three-fold stratified CV:

- `StratifiedKFold(n_splits=3, shuffle=True, random_state=6000 + dataset_seed)`

For every candidate hyperparameter setting, collect out-of-fold predictions and confidence/risk values over the complete outer training split and compute OOF E-AURC.

Selection rule:

1. choose the candidate with lowest OOF E-AURC;
2. if tied to numerical tolerance, choose the candidate with higher OOF accuracy;
3. if still tied, choose the lexicographically first frozen candidate representation.

The selected candidate is then refit on the complete outer training split and evaluated once on the untouched outer test split.

This deliberately gives the baselines a strong advantage: they are tuned for the same selective-risk metric used in the primary comparison, whereas MCGC remains frozen.

## Frozen grids

### Tuned weighted/uniform kNN

- `n_neighbors in {5, 9, 12, 17, 25}`
- `weights in {'uniform', 'distance'}`

Confidence: maximum class probability.

### Tuned RBF SVM

Pipeline includes `StandardScaler` within each inner fold.

- kernel: `rbf`
- `C in {0.1, 1.0, 10.0}`
- `gamma in {'scale', 0.1, 1.0}`

Confidence: absolute decision-function margin for binary classification. Risk is the negative confidence so lower risk means more trusted.

No probability calibration is used.

### Tuned Random Forest

- `n_estimators=300`
- `max_depth in {None, 6, 12}`
- `min_samples_leaf in {1, 3, 5}`
- `max_features in {'sqrt', 0.5}`
- `random_state=97`
- `n_jobs=-1`

Confidence: maximum predicted class probability.

### Tuned logistic reference

This remains a secondary reference rather than a main nonlinear baseline.

Pipeline includes `StandardScaler`.

- `C in {0.1, 1.0, 10.0}`
- `max_iter=3000`
- `random_state=73`

Confidence: maximum predicted class probability.

## Primary metric and contrasts

Primary metric: E-AURC, lower is better.

For baseline `b` define:

`advantage_b = E-AURC(b) - E-AURC(MCGC)`

Positive values favor MCGC.

The main v1.2 contrasts are:

- tuned kNN vs frozen MCGC
- tuned RBF-SVM vs frozen MCGC
- tuned Random Forest vs frozen MCGC

Tuned logistic regression is reported as a secondary contextual reference.

## Secondary metrics

For every fitted outer-test model report:

- raw accuracy
- ordinary AURC
- oracle AURC
- E-AURC
- error-detection ROC-AUC when defined
- selective accuracy at 90%, 80%, 70%, and 50% coverage

Also report the selected hyperparameters and inner-CV OOF E-AURC for every dataset-seed run.

## Statistical reporting

For each dataset and for the pooled 36-run primary manifold family report paired:

- mean advantage
- median advantage
- standard deviation
- min / max
- wins / ties / losses
- two-sided sign-test p-value

Dataset-specific outcomes remain mandatory so that no single benchmark silently drives the pooled conclusion.

## Interpretation rule

A positive result against tuned RBF-SVM and/or tuned Random Forest would substantially strengthen the selective-manifold contribution because those baselines were optimized for the same selective-risk objective using only training data.

A negative result is also scientifically useful and does not erase the frozen v1.0/v1.1 checkpoints. It would delimit the current superiority claim to fixed conventional baselines rather than tuned competitors.

Do not claim universal classifier superiority, universal uncertainty superiority, or SOTA performance from v1.2 alone.
