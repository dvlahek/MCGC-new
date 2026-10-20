# MCGC project history and scientific decision log

This file is the canonical running history for the MCGC project. It records code checkpoints, frozen protocols, tests, positive and negative results, and the scientific decisions that followed from them. It should be updated whenever a new frozen version is completed.

The purpose is to prevent claim drift, accidental loss of negative results, and confusion between exploratory single-split observations and confirmed findings.

## Project identity

Repository: `dvlahek/MCGC-new`

Working name: **Multi-Agent Chemotactic Graph Classifier (MCGC)**.

Core idea: labelled samples form a k-nearest-neighbour graph. Each class induces a contrastive evidence field. Class-conditioned walkers follow graph geometry and class evidence, while prediction, reliability and explanation are treated as separate scientific questions.

The project deliberately distinguishes:

1. **classification** — whether walker/path information improves class prediction;
2. **XAI** — whether the actual fitted decision can be traced through exact score components, supporting samples and representative graph paths;
3. **reliability / selective prediction** — whether intrinsic graph quantities rank likely errors and support abstention.

## Current scientific position

As of v1.1 protocol freeze, the strongest supported story is **not** universal raw-accuracy superiority.

The two strongest empirical results are:

1. **v0.9 XAI faithfulness:** for queries with a nontrivial representative path, deleting path-selected training nodes and refitting changes the original class margin more than deleting label-and-distance-matched off-path controls and more than deleting an equal number of nearest off-path examples.
2. **v1.0 selective-manifold confidence:** on a frozen nonlinear-manifold family, the MCGC local graph-field margin achieves better E-AURC than distance-weighted kNN confidence over most paired runs.

Important closed negative results are equally part of the project history:

- the original single-split spiral walker-accuracy gain did not survive multi-seed confirmation;
- uncertainty-gated walker scoring did not repair that failure;
- trail disagreement did not add reliable held-out uncertainty information beyond the local graph-field margin;
- a combined five-feature intrinsic reliability model also failed broader confirmation.

The resulting working positioning is:

> MCGC is an intrinsically traceable graph classifier whose class-conditioned path explanations identify functionally decision-relevant training samples, while its graph-field margin provides useful selective-risk ranking on nonlinear-manifold problems.

This statement must not be expanded to universal accuracy, uncertainty, interpretability or SOTA claims without new evidence.

---

# Version history

## v0.1 — first reproducible classifier prototype

### Goal

Implement the original dog/scent-inspired classifier as a minimal sklearn-compatible research prototype.

### Main code

Core estimator:

- `src/mcgc/classifier.py`
- `MultiAgentChemotacticGraphClassifier`

Other initial files:

- `src/mcgc/__init__.py`
- `tests/test_classifier.py`
- `examples/benchmark_synthetic.py`
- `docs/FORMULATION.md`
- `pyproject.toml`
- `.github/workflows/ci.yml`

Early repository initialization commit:

- `ac8e838501aff1eada17aa853a932d7d83af9e6e` — `Initialize MCGC research prototype`

A complete early v0.1 repository checkpoint was:

- `88588e6203558a3a74ac83ef87c53ff7a338847b`

### Important failure and design correction

The first naive scoring rule rewarded uphill walker movement so strongly that a wrong-class walker could outscore the correct class even when the query was already inside a strong correct-class basin.

The initial smoke behaviour was extremely poor, including approximately zero accuracy on circles and only a few percent on moons.

This led to the key scoring correction that remains conceptually important:

- local contrastive class evidence became the **anchor**;
- walker trajectories became a **refinement**, not the dominant class score.

The resulting v0.1 score used a strong local term plus smaller mean-trail, agreement and lost-trail terms.

### Smoke result after correction

Approximate frozen smoke results:

- circles: kNN `0.9933`, MCGC `0.9933`
- moons: kNN `0.9667`, MCGC `0.9600`

These were explicitly treated as smoke tests, not performance claims.

### Scientific lesson

The biological metaphor is not sufficient justification for a component. Trajectory terms must demonstrate measurable value beyond the underlying graph field.

---

## v0.2 — controlled ablation of the walker stage

### Goal

Separate graph-field performance from trajectory performance and test whether walkers carry predictive information beyond the same fitted field.

### Main code changes

Exposed explicit ablation controls:

- `use_walkers`
- `local_evidence_weight`
- `trail_score_weight`
- `agreement_weight`
- `lost_fraction_weight`

Added:

- `examples/benchmark_ablation_v02.py`
- `docs/ABLATION_V02.md`
- additional regression tests

Important commits:

- `970d75ba989081245ece010f8743557363a6e5fc` — add v0.2 ablation benchmark
- `b48e6d94d1edab8a0e68281c3d09f48be5055c69` — run v0.2 ablation in CI
- `69444556cda0e8c9ad7f09bc0664cb8dbd7e4a7a` — document v0.2 results

### Frozen comparison systems

- uniform kNN
- distance-weighted kNN
- Label Spreading
- `field_only`
- `one_walker`
- `multi_no_gradient`
- `mcgc_no_reliability`
- `mcgc_full`

### Single-split diagnostic result

Accuracy:

| Dataset | field only | one walker | no gradient | full MCGC | distance kNN | Label Spreading |
|---|---:|---:|---:|---:|---:|---:|
| moons | 0.9833 | 0.9889 | 0.9833 | 0.9833 | 0.9778 | 0.9778 |
| circles | 0.9889 | 0.9889 | 0.9889 | 0.9889 | 0.9889 | 0.9889 |
| spirals | 0.7556 | 0.7611 | 0.7667 | 0.7722 | 0.8111 | 0.8111 |

The spiral split suggested a possible walker-specific gain over `field_only`, but MCGC still trailed strong baselines.

### Tests / CI

The v0.2 CI checkpoint passed on Python 3.10, 3.11 and 3.12, with six tests passing at that stage. The smoke benchmark and v0.2 ablation benchmark also completed successfully.

### Decision

Do not claim the spiral gain until it survives multi-seed confirmation.

---

## v0.3 — XAI-first layer without changing prediction

### Goal

Make explanation a first-class output while preserving the v0.2 prediction mechanism.

### Main implementation

Added:

- `src/mcgc/xai.py`
- `tests/test_xai.py`
- `examples/explain_moons.py`
- `docs/XAI_V03.md`

Key implementation commits include:

- `3315a1527edf84a9dc5ea79d0c1580cee8baac22` — faithful XAI layer
- `229cdea59bf8bc45e0d491d375c2a2c7ec9c3801` — deterministic XAI tests
- `55db66d83b15f3e39b37837b0df127ecc47501f6` — v0.3 checkpoint

### Core API

`explain_mcgc(model, X)`

Each explanation separates:

1. **exact score decomposition** used by the stochastic prediction mechanism;
2. **deterministic representative trail**, explicitly presented as a trace through the fitted evidence landscape rather than a fabricated sole causal path.

### Fidelity invariant

For every class, the explanation reconstructs the actual numerical class score from:

- local component
- trail component
- agreement component
- lost-trail component

The explanation raises an error rather than silently returning a decomposition that does not reconstruct the score.

### Tests

Tests check:

- explanation prediction equals ordinary `predict()`;
- exact score reconstruction;
- deterministic representative-path reproducibility;
- support/path structure consistency.

CI passed on Python 3.10, 3.11 and 3.12.

### Decision

Keep prediction and explanation code conceptually separate. Do not describe representative paths as sole causal paths for stochastic multi-agent scoring.

---

## v0.4 — multi-seed confirmation of walker accuracy

### Goal

Test the apparent v0.2 spiral walker gain under a frozen 12-seed protocol.

### Protocol

- 12 seeds: `0..11`
- datasets: circles, moons, two-arm spirals
- fixed hyperparameters
- comparisons: `field_only`, `multi_no_gradient`, `mcgc_full`, distance-weighted kNN

Important commits:

- `d03ad16a5f8de0be7a8a44ba089efab868df3b4f` — benchmark
- `443916f2f5a079e11bf7818206f3a11256877e7f` — CI integration
- `544d15d6c00f9677f8e039174fd139644f347915` — v0.4 checkpoint
- `f611dc65fbcb5da5ffa052fc9b8c31692b4790e6` — result documentation

### Result: walker effect, full minus field-only

| Dataset | mean delta | W/T/L |
|---|---:|---:|
| circles | +0.00056 | 1/11/0 |
| moons | -0.00056 | 0/11/1 |
| spirals | -0.00667 | 3/2/7 |

Gradient effect was also essentially null.

On spirals, full MCGC lost to distance-weighted kNN on all 12 seeds, with mean accuracy delta `-0.0950`.

### Scientific conclusion

**The single-split walker accuracy signal did not replicate.**

This is a closed negative checkpoint. Walkers are not supported as a general raw-accuracy improvement under this formulation.

### Additional XAI progress

Contrastive `why A rather than B` score decomposition was added so component differences reconstruct the requested class-score margin exactly.

---

## v0.5 — ambiguity-gated selective walkers

### Goal

Prevent trajectory corrections from perturbing already-clear local decisions by attenuating walker terms when local class evidence is decisive.

### Gating rule

For local margin `m(q)` and fitted reference scale `m_ref`:

`g(q) = exp(-(m(q) / m_ref) ** p)`

with default `p=1` and no dataset-specific threshold tuning.

Important implementation / test commits:

- `a15177a65f5c8b22dc460b90ba8e14f87d783948` — uncertainty-gated walkers
- `332b990c07382cb3a081d0bd96b41792be213696` — exact XAI under gating
- `d0c4ffec38780b617bdf4f0876a11297efcb7add` — gate behaviour tests
- `0d0e229d89471bd9d2c8bcb33b4b6d909bd545f0` — XAI fidelity tests
- `b102185206fb49202783042d64987e13d87c35f7` — v0.5 confirmation benchmark
- `3cdc3d86411067f4879b60dee77ae7bf1b2fe566` — result documentation

### Result

The gate was numerically active but did not change aggregate predictions relative to unconditional walkers in any of the 36 dataset-seed runs.

Mean gates:

- circles: `0.3836`
- moons: `0.4092`
- spirals: `0.5348`

Selective-minus-unconditional accuracy delta was exactly zero across all runs.

### Tests

The v0.5 checkpoint passed 16 tests and the complete CI suite.

### Scientific conclusion

The gating mechanism is mathematically clean and explainable but does not solve the raw-accuracy failure. Do not tune gate parameters post hoc merely to force a positive result.

Decision: investigate whether intrinsic graph/trail quantities are useful for **reliability** rather than forcing them into the class score.

---

## v0.6 — intrinsic reliability and abstention

### Goal

Test whether quantities already emitted by MCGC identify its own errors without changing class predictions or fitting a secondary risk model.

### Main code

Reliability API introduced in `src/mcgc/reliability.py`.

Important commits:

- `49e8bb2409bea329f031d758a72886f272a2732e` — export reliability API
- `9d9657c2bd789a25cf02fd5b2353221c18f91307` — benchmark
- `5ef70ea8de003ebbcabe782b716930b9ce5288c5` — CI
- `a5d51ab68da77151ecef21990770eb15fcbe08f3` — frozen protocol
- `a6dd12d8fa0b55f1a36aa2f611f4d4671dd35e2f` — result documentation

### Key error-detection results

| Dataset | errors | local-margin AUC | final-margin AUC | trail-disagreement AUC |
|---|---:|---:|---:|---:|
| circles | 6 | 0.9929 | 0.9931 | 0.9936 |
| moons | 42 | **0.9197** | 0.9153 | 0.8847 |
| spirals | 505 | **0.7375** | 0.7257 | 0.6083 |

Circles is underpowered because only six errors occurred.

### Selective classification

On spirals, local-margin filtering increased retained-set accuracy from `0.7194` at 100% coverage to:

- 90%: `0.7475`
- 80%: `0.7722`
- 70%: `0.8040`
- 50%: `0.8711`

### Scientific conclusion

The **local contrastive graph-field margin** is a strong intrinsic reliability signal.

Trail disagreement contains descriptive uncertainty information but is weaker than local margin on the informative datasets. Lost-trail fraction is essentially uninformative in the current implementation.

Next question: does walker disagreement add information **after controlling for local margin**?

---

## v0.7 — calibrated incremental reliability

### Goal

Use a strict train/calibration/test split to test whether walker disagreement contributes held-out reliability information beyond local graph-field ambiguity.

### Frozen split

- 60% classifier training
- 20% reliability calibration
- 20% untouched test

Frozen risk models:

- `local_only`
- `local_plus_final`
- `local_plus_trail_disagreement`
- `combined_intrinsic`

Risk learner: standardized balanced logistic regression with fixed hyperparameters.

Key working checkpoint used for the dedicated confirmation run:

- `a7e7b869b27d27191644bff7b40a5fb5d836c53c`

Dedicated confirmation passed **19/19 tests** before the frozen benchmark.

### Primary result

Primary contrast:

`AUC(local_plus_trail_disagreement) - AUC(local_only)`

On moons:

- local only: `0.9436`
- local + trail disagreement: `0.9223`
- delta: `-0.0213`
- W/T/L: `1/4/5`

On spirals:

- local only: `0.7330`
- local + trail disagreement: `0.7296`
- delta: `-0.00340`
- W/L: `3/9`

### Secondary signal

The preregistered five-feature `combined_intrinsic` model showed a small positive secondary spiral result:

- local: `0.73298`
- combined: `0.74085`
- delta: `+0.007872`
- W/L: `8/4`
- Brier: `0.21245 -> 0.20874`

But the same combined model was negative on moons.

### Scientific conclusion

**Primary v0.7 hypothesis rejected.** Trail disagreement alone does not add reliable held-out information beyond local margin.

The secondary spiral combined signal was treated only as a hypothesis for an independent confirmation, not as a claim.

---

## v0.8 — frozen confirmation of combined intrinsic reliability

### Goal

Carry the exact five-feature `combined_intrinsic` model from v0.7 to a broader confirmation suite with no feature search or retuning.

### Frozen feature set

- `local_margin_risk`
- `final_margin_risk`
- `trail_disagreement_risk`
- `coherence_risk`
- `lost_trail_risk`

Datasets:

- `moons_hard`
- `circles_hard`
- `spirals_hard`
- `overlap_hd`
- `breast_cancer`
- `wine`
- `digits_0_4`

Eight seeds per dataset, with 60/20/20 train/calibration/test splitting.

Important checkpoints:

- `788ef00d2a242c69d569bd2dabd893d83e09148a` — self-contained Windows-compatible benchmark/test fix
- `f7a8be5d3f94b743c7ffe85eb8f4c323992a5812` — frozen negative result documentation

### Tests

After fixing a Windows-only sibling-import issue without changing the science protocol, the suite reached **22 passing tests** before the full confirmation.

### Result

Across 53 evaluable paired runs:

- mean `combined - local` ROC-AUC delta: `-0.0130355`
- median: `-0.00733945`
- W/T/L: `19/2/32`
- sign-test p: `0.0919145`

Every dataset had negative mean paired delta.

### Scientific conclusion

**Combined path-derived reliability did not replicate.**

This is a closed negative checkpoint. No further feature-combination tuning should be used to rescue the same hypothesis.

The main research direction moved decisively to **XAI faithfulness**, while `local_margin_risk` remained the strongest supported reliability signal.

---

## v0.9 — frozen XAI faithfulness confirmation

### Goal

Test whether the representative path is functionally related to the fitted decision, rather than merely being a visually plausible trace.

### Intervention

For an eligible query:

1. fit MCGC;
2. obtain the deterministic representative path for the predicted class;
3. delete those path training nodes;
4. refit the same classifier;
5. measure drop in the original winner-vs-runner margin;
6. compare against equal-size controls.

Primary matched control:

- same class / label structure as appropriate;
- distance-matched;
- off-path.

Secondary baseline:

- equal number of nearest off-path training examples.

Classifier and representative-path construction were frozen before results.

Important checkpoints:

- `9aebb3e51dd40a5b08c9fca1796449befa5b3672` — v0.9 implementation checkpoint
- `133c2810295822e75ee62d2ccae92fc3b71ab76f` — strong frozen result documentation

### Datasets

- `moons_hard`
- `spirals_hard`
- `overlap_hd`
- `breast_cancer`

Five seeds, maximum 12 eligible queries per seed.

### Primary result: path vs matched control

Across 240 eligible queries:

- mean advantage: `+0.0926154467`
- median: `+0.0625828114`
- W/T/L: `202/0/38`
- pooled two-sided sign-test p: `3.377174048e-28`

### Secondary result: path vs proximity-only explanation

- mean advantage: `+0.3593250906`
- median: `+0.3049531662`
- W/T/L: `207/0/33`
- pooled p: `5.430695506e-32`

### Per-dataset primary means

- moons: `+0.1566762324`, `56/4` wins/losses
- spirals: `+0.1088086735`, `49/11`
- overlap_hd: `+0.0430927165`, `47/13`
- breast_cancer: `+0.0618841645`, `50/10`

### Strongest proximity result

On `spirals_hard`:

- path-vs-nearest mean advantage: `+0.5647923100`
- W/L: `59/1`
- path-removal prediction-flip rate: `83.33%`
- nearest-off-path flip rate: `18.33%`

### Exactness

Contrastive margin reconstruction error remained at floating-point scale, approximately `1e-16` to `4e-16`.

### Cluster-aware robustness

Because pooled queries within a seed are dependent, the preferred reviewer-resistant summary aggregates by dataset × seed.

There are 20 such units:

- `20/20` positive mean primary faithfulness advantage;
- `20/20` positive mean path-vs-nearest advantage;
- two-sided sign-test p for 20/20 positives: approximately `1.907348633e-06`.

### Limitation

Eligibility is below 100% because many queries have trivial or very short representative paths. Claims apply to queries admitting a nontrivial representative path.

### Scientific conclusion

**First strong positive MCGC result.**

Supported claim:

> For queries admitting a nontrivial representative path, MCGC identifies training examples whose removal changes the fitted class margin more strongly than matched off-path controls and equal-size nearest-neighbour explanations.

Do not convert this into a universal interpretability or causal-data-generation claim.

---

## v1.0 — frozen selective-manifold benchmark

### Goal

Test whether the already-supported `local_margin_risk` gives better selective-risk ranking than standard confidence signals specifically on nonlinear manifold problems.

### Frozen primary family

- `moons_hard`
- `circles_hard`
- `spirals_hard`

Controls:

- `overlap_hd`
- `breast_cancer`

Twelve seeds, 70/30 train/test split.

Primary metric: **E-AURC**, lower is better.

Baselines:

- distance-weighted 12-NN confidence
- standardized logistic-regression confidence

No confidence calibration, feature search or dataset-specific retuning.

Important code/protocol checkpoint:

- `2517852cafb65a61f69ba211157023dd1b2e48a4`

Frozen result commit:

- `34ceef3e829f819613b9487673a9b2f30aa5a8e1`

### Primary result against weighted kNN

Across 36 frozen manifold runs:

E-AURC advantage `kNN - MCGC`:

- mean: `+0.009157069986134773`
- median: `+0.006353876329219648`
- W/T/L: `29/0/7`
- sign-test p: `0.00031255115754902363`

Ordinary AURC advantage:

- mean: `+0.008125534126382374`
- median: `+0.005130841291406594`
- W/T/L: `25/0/11`
- sign-test p: `0.028816719655878842`

### Per-dataset E-AURC

| Dataset | MCGC | weighted kNN | paired wins/losses |
|---|---:|---:|---:|
| moons_hard | 0.018809 | 0.026451 | 11/1 |
| circles_hard | 0.017257 | 0.023252 | 9/3 |
| spirals_hard | 0.192063 | 0.205898 | 9/3 |

Thus the aggregate result is not driven by one dataset.

### Logistic reference

MCGC also beats linear logistic confidence strongly over the frozen primary family, but this is not the preferred superiority comparison because logistic regression is structurally mismatched to some nonlinear manifold geometries.

### Raw accuracy discipline

This is **not** a universal accuracy win.

Examples:

- moons: MCGC accuracy `0.90278`, kNN `0.89722`
- spirals: MCGC `0.62262`, kNN `0.63056`

Therefore the supported result concerns **risk ranking / selective prediction**, not blanket predictive dominance.

### Controls

- `overlap_hd`: MCGC and weighted kNN are essentially tied in E-AURC on average.
- `breast_cancer`: MCGC beats weighted kNN but loses to logistic confidence.

This supports a geometry-specific interpretation.

### Scientific conclusion

**Second strong positive MCGC result.**

Supported claim:

> On the frozen nonlinear-manifold family, MCGC local graph-field margin provides better selective-risk ranking than distance-weighted kNN confidence.

---

## v1.1 — stronger nonlinear selective baselines

### Status

**Protocol frozen. Results pending.**

Branch:

- `feature/v1.1-stronger-selective-baselines`

Protocol commit:

- `cc2dbe6c3cc4457662e175ad9e7308fc917be074`

Protocol document:

- `docs/V11_STRONGER_SELECTIVE_BASELINES_PROTOCOL.md`

### Goal

Test whether the frozen v1.0 MCGC local-margin confidence remains competitive against stronger nonlinear classifiers, without tuning MCGC after seeing results.

### New baselines

RBF SVM:

- `StandardScaler`
- `SVC(kernel='rbf', C=1.0, gamma='scale', probability=False)`
- confidence from decision-function margin

Random Forest:

- `RandomForestClassifier(n_estimators=300, max_features='sqrt', random_state=97, n_jobs=-1)`
- confidence from maximum predicted class probability

The frozen kNN and logistic references remain for continuity.

### Primary metric

E-AURC, lower is better.

Primary new paired advantages:

- `E-AURC(RBF-SVM) - E-AURC(MCGC)`
- `E-AURC(RandomForest) - E-AURC(MCGC)`

No hyperparameter search is permitted.

### Interpretation rule

A positive result would strengthen the selective-manifold contribution beyond kNN. A mixed or negative result would delimit the current validated advantage to the already-frozen v1.0 kNN comparison and would **not** erase v1.0.

---

# Test and reproducibility policy

The project uses a deliberately failure-aware workflow.

## Rules

1. Freeze protocol and main contrasts before viewing confirmation results.
2. Do not retune thresholds, features or dataset definitions after seeing frozen results.
3. Keep negative confirmations in the repository.
4. Separate exploratory observations from confirmation claims.
5. Keep prediction, XAI and reliability claims separate.
6. Use fixed seeds and explicit train/calibration/test separation where relevant.
7. Run unit/regression tests before confirmation benchmarks.
8. Use GitHub Actions for portable regression testing, but long science benchmarks may be run locally when Actions runtime is inefficient.
9. Store full JSON outputs for science analysis; console logs are summaries.
10. Prefer paired statistics and show per-dataset outcomes so one dataset cannot silently drive an aggregate claim.

## Platform note

Local science runs are currently commonly executed on Windows PowerShell from a clean standalone clone, e.g. `C:\Articles\Clanci\MCGC-new-repo`.

Linux/Bash commands must not be copied verbatim into PowerShell. In particular, PowerShell uses `\.venv\Scripts\Activate.ps1`, backticks for multiline commands when needed, and `Tee-Object` rather than Bash `tee` syntax.

---

# Claim ledger

## Supported now

### Intrinsic XAI / path faithfulness

For eligible queries with nontrivial representative paths, path-selected nodes are more functionally decision-relevant under delete-and-refit intervention than matched off-path controls and equal-size nearest-neighbour explanations.

Evidence: v0.9.

### Intrinsic reliability

Local contrastive graph-field margin is a useful error-ranking / abstention signal.

Evidence: v0.6 and later selective benchmarks.

### Selective risk on nonlinear manifolds

On the frozen moons/circles/spirals family, MCGC local-margin risk achieved lower E-AURC than distance-weighted kNN confidence in 29/36 paired runs.

Evidence: v1.0.

## Not supported / explicitly rejected

- general walker improvement in raw accuracy;
- explicit gradient-term accuracy improvement;
- selective walker gate as an accuracy fix;
- trail disagreement as complementary reliability beyond local margin;
- five-feature combined intrinsic reliability as a generally better uncertainty model;
- universal superiority to kNN, logistic regression or other classifiers;
- universal interpretability or causal explanation of the data-generating process;
- SOTA claims.

## Pending

- selective-risk comparison with RBF SVM and Random Forest: v1.1;
- deeper literature/novelty audit against OPF, path-based classifiers, graph random-walk classifiers and related intrinsic XAI approaches;
- broader real-data validation;
- explanation stability / user-facing usefulness beyond intervention faithfulness;
- counterfactual explanations.

---

# Current development state

Current research branch at the time this history was created:

`feature/v1.1-stronger-selective-baselines`

Frozen v1.1 protocol head before adding this history file:

`cc2dbe6c3cc4457662e175ad9e7308fc917be074`

The next coding task is to implement the v1.1 benchmark while keeping the v1.0 MCGC classifier, local-margin risk, datasets, seeds and metrics unchanged.

When v1.1 completes, update this file with:

- exact implementation/result commit SHAs;
- test count and CI/local-run status;
- full paired E-AURC/AURC result against RBF SVM and Random Forest;
- per-dataset W/T/L;
- the resulting claim boundary and next frozen step.
