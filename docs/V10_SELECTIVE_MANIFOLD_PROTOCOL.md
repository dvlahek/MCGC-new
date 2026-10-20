# MCGC v1.0 — frozen selective-manifold benchmark

## Goal

v1.0 tests whether the already-defined MCGC local graph-field margin provides better selective-classification ranking than standard confidence scores on nonlinear manifold problems.

The classifier, graph construction, local-margin definition, benchmark generators, and baseline confidence definitions are frozen before viewing v1.0 results. No dataset-specific retuning is allowed.

## Primary benchmark family

The primary confirmation family contains the three nonlinear synthetic problems already used in prior frozen checkpoints:

- `moons_hard`
- `circles_hard`
- `spirals_hard`

The purpose of this family is explicit: class geometry is nonlinear and local Euclidean proximity can be ambiguous while graph structure remains informative.

Two transfer/control datasets are also reported, but are not part of the primary manifold-family claim:

- `overlap_hd`
- `breast_cancer`

## Frozen models and risk scores

### MCGC

Use the existing MCGC classifier without retuning:

- `n_neighbors=12`
- `n_walkers=12`
- `max_steps=10`
- `diffusion_steps=40`
- `random_state=17`
- `adaptive_walker_gate=True`

Risk score:

`local_margin_risk = -(largest local class evidence - second-largest local class evidence)`

No learned reliability model is fit in v1.0.

### Distance-weighted kNN

- `n_neighbors=12`
- `weights='distance'`

Risk score:

`1 - max(predict_proba)`

### Logistic regression

- `StandardScaler`
- `LogisticRegression(C=1.0, max_iter=3000, random_state=73)`

Risk score:

`1 - max(predict_proba)`

## Splits and seeds

Use the same train/test split for all three models within each run.

- 70% train
- 30% untouched test
- dataset/split seeds: `0..11`
- split seed: `5000 + dataset_seed`
- for fixed real datasets, the underlying data remain fixed while the split changes with seed

No calibration split is needed because all v1.0 metrics depend only on ranking by a frozen risk score.

## Primary metric: E-AURC

For each model, sort test samples from lowest to highest risk. At each prefix, compute selective risk (error rate among retained samples). Empirical AURC is the mean selective risk over all non-empty prefixes.

Because models can have different base error rates, the primary confidence-quality metric is **excess AURC (E-AURC)**:

`E-AURC = observed AURC - oracle AURC`

where oracle AURC is obtained by ranking all correct predictions before all errors for that model's own fixed error count.

Lower E-AURC is better.

### Primary paired contrasts

On the primary nonlinear-manifold family, compute per run:

- `E-AURC(kNN) - E-AURC(MCGC)`
- `E-AURC(logistic) - E-AURC(MCGC)`

Positive values favor MCGC.

The primary v1.0 claim requires positive mean and median paired contrast against both baselines on the pooled manifold-family runs, with direction also reported separately by dataset.

## Mandatory secondary metrics

To prevent a confidence-quality result from hiding weaker classifier performance, also report:

- raw test accuracy;
- ordinary AURC, lower is better;
- selective accuracy at coverages `0.90`, `0.80`, `0.70`, `0.50`;
- error-detection ROC-AUC when both correctness classes are present;
- per-dataset wins/ties/losses for E-AURC contrasts.

A result in which MCGC wins E-AURC but loses ordinary AURC must be described exactly that way: better error ranking, but not better overall selective-classification performance.

## Statistical reporting

The main inferential unit is a dataset-seed run, not an individual test query.

For paired E-AURC differences report:

- n
- mean
- median
- standard deviation
- minimum and maximum
- wins / ties / losses
- two-sided sign-test p-value

No query-level pseudo-replication is used for the primary test.

## Claim discipline

A positive v1.0 result can support:

> On nonlinear manifold benchmarks, MCGC's intrinsic local graph-field margin ranks its prediction errors more effectively than standard distance-weighted kNN and logistic-regression confidence scores.

A stronger statement about practical selective classification is allowed only if ordinary AURC and coverage-specific accuracy also support it.

No claim of general superiority is permitted from this benchmark alone.
