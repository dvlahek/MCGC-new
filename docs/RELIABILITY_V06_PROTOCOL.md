# MCGC v0.6 intrinsic reliability protocol

v0.6 changes the scientific question. v0.4 and v0.5 did not support a claim that trajectory terms improve raw classification accuracy. Instead of retuning the class score, v0.6 asks whether the same trajectory diagnostics identify predictions that are likely to be wrong.

## Principle

The class prediction is left unchanged. Reliability signals are read from quantities already produced during MCGC inference. No secondary risk classifier is fitted and no abstention threshold is selected on the test set.

For each query, v0.6 exposes:

- final winner-versus-runner score margin,
- local contrastive-field margin,
- winner/runner mean trail scores and their difference,
- winner trail-score dispersion,
- winner trail coherence,
- winner and runner lost-trail fractions,
- winner/runner terminal evidence and their difference,
- winner-versus-runner coherence and lost-trail advantages.

Every risk score is oriented so that a larger value means a more suspicious prediction.

## Frozen evaluation

The benchmark uses the existing 12-seed synthetic protocol:

- seeds `0..11`,
- 500 samples per dataset,
- moons, circles, and two-arm spirals,
- test fraction 0.30,
- split seed `1000 + dataset_seed`,
- fixed MCGC hyperparameters and random state,
- adaptive walker gate enabled,
- no per-dataset or per-seed risk tuning.

Two questions are evaluated.

### Error ranking

For each intrinsic risk signal, compute ROC AUC for distinguishing incorrect from correct predictions. Pooled AUC across all 12 test splits is the primary descriptive result; per-seed AUC is reported only for seeds containing both correct and incorrect examples.

### Selective classification

For predeclared coverages `1.0, 0.9, 0.8, 0.7, 0.5`, retain the lowest-risk predictions and report their accuracy. A useful reliability signal should increase retained-set accuracy as coverage decreases.

## Baseline within MCGC

`local_margin_risk` is the primary intrinsic baseline. Walker-derived signals are scientifically interesting only if they provide useful error ranking beyond what is already available from the local contrastive field or final prediction margin.

## Claim discipline

A high AUC for a walker-derived risk is not evidence that walkers improve class accuracy. It would support a narrower claim: trajectory behavior carries information about prediction reliability. Conversely, failure to outperform local-margin risk should be recorded as a negative result rather than repaired by test-set tuning.
