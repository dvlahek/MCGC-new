# MCGC project history and scientific decision log

This is the canonical running history for the MCGC project. It records code checkpoints, frozen protocols, tests, positive and negative results, and the scientific decisions that followed from them. Update it whenever a new frozen version is completed.

Repository: `dvlahek/MCGC-new`

Working name: **Multi-Agent Chemotactic Graph Classifier (MCGC)**.

Core idea: labelled samples form a k-nearest-neighbour graph with class-specific contrastive evidence fields. Class-conditioned walkers traverse those fields. Classification, intrinsic XAI and reliability/selective prediction are treated as separate scientific questions.

## Current scientific position

The strongest supported story is not universal raw-accuracy superiority.

The two main positive contributions are now:

1. **v0.9 XAI faithfulness** — for queries with a nontrivial representative path, deleting path-selected training nodes and refitting changes the fitted class margin more than deleting matched off-path controls or an equal number of nearest off-path training examples.
2. **v1.0–v1.1 selective confidence on nonlinear manifolds** — the frozen local graph-field margin gives better excess selective-risk ranking than distance-weighted kNN confidence and, in aggregate over the frozen manifold family, also better ranking than fixed RBF-SVM margin confidence and fixed Random-Forest probability confidence.

Closed negative results remain part of the project story:

- the original single-split spiral walker-accuracy gain did not survive multi-seed confirmation;
- uncertainty-gated walker scoring did not repair that failure;
- trail disagreement did not add held-out reliability information beyond local margin;
- a five-feature combined intrinsic reliability model failed broader confirmation.

Current defensible positioning:

> MCGC is an intrinsically traceable graph classifier whose class-conditioned representative paths identify functionally decision-relevant training samples, while its graph-field margin provides improved selective-risk ranking on the frozen nonlinear-manifold benchmark relative to several conventional confidence baselines.

Do not expand this into universal accuracy, uncertainty, interpretability or SOTA claims without new evidence.

---

# Version history

## v0.1 — first reproducible prototype

### Goal

Implement the original dog/scent-inspired classifier as a minimal sklearn-compatible prototype.

### Main implementation

- `src/mcgc/classifier.py`
- `MultiAgentChemotacticGraphClassifier`
- graph construction, class fields, stochastic walkers and initial scoring
- `examples/benchmark_synthetic.py`
- `tests/test_classifier.py`
- initial CI

Important early commits:

- `ac8e838501aff1eada17aa853a932d7d83af9e6e` — initialize prototype
- `88588e6203558a3a74ac83ef87c53ff7a338847b` — complete early v0.1 repository checkpoint

### Failure that changed the design

The naive score rewarded uphill walker movement so strongly that a wrong-class walker could beat a query already located in a strong correct-class basin.

The fix made **local contrastive evidence the anchor**, with walker terms only refining it.

Post-fix smoke results:

- circles: kNN `0.9933`, MCGC `0.9933`
- moons: kNN `0.9667`, MCGC `0.9600`

These were smoke tests only.

### Decision

Biological plausibility is not enough. Every walker-specific mechanism must demonstrate measurable value beyond the graph field.

---

## v0.2 — walker-stage ablation

### Goal

Separate graph-field prediction from trajectory contribution.

### Added controls

- `use_walkers`
- `local_evidence_weight`
- `trail_score_weight`
- `agreement_weight`
- `lost_fraction_weight`

Compared:

- uniform kNN
- distance-weighted kNN
- Label Spreading
- `field_only`
- `one_walker`
- `multi_no_gradient`
- `mcgc_no_reliability`
- `mcgc_full`

Important commits:

- `970d75ba989081245ece010f8743557363a6e5fc` — ablation benchmark
- `b48e6d94d1edab8a0e68281c3d09f48be5055c69` — run ablation in CI
- `69444556cda0e8c9ad7f09bc0664cb8dbd7e4a7a` — document results

Single-split diagnostic accuracy:

| Dataset | field | one walker | no gradient | full | distance kNN | Label Spreading |
|---|---:|---:|---:|---:|---:|---:|
| moons | 0.9833 | 0.9889 | 0.9833 | 0.9833 | 0.9778 | 0.9778 |
| circles | 0.9889 | 0.9889 | 0.9889 | 0.9889 | 0.9889 | 0.9889 |
| spirals | 0.7556 | 0.7611 | 0.7667 | 0.7722 | 0.8111 | 0.8111 |

CI passed on Python 3.10, 3.11 and 3.12 with six tests at this stage.

### Decision

The apparent spiral walker gain was exploratory only and required independent multi-seed confirmation.

---

## v0.3 — XAI-first layer

### Goal

Expose faithful explanation objects without changing v0.2 prediction.

### Added

- `src/mcgc/xai.py`
- `explain_mcgc(model, X)`
- deterministic representative trails
- supporting training examples
- exact additive class-score decomposition

Important commits:

- `3315a1527edf84a9dc5ea79d0c1580cee8baac22` — faithful XAI layer
- `229cdea59bf8bc45e0d491d375c2a2c7ec9c3801` — deterministic XAI tests
- `55db66d83b15f3e39b37837b0df127ecc47501f6` — v0.3 checkpoint

### Fidelity invariant

Each returned class score must numerically equal the sum of its local, trail, agreement and lost-trail components. The explanation fails rather than silently returning a non-faithful decomposition.

Tests covered prediction identity, exact score reconstruction and deterministic representative paths. CI passed on Python 3.10–3.12.

### Decision

Representative trails are traces through the fitted evidence landscape. They are not described as the sole causal path behind a stochastic multi-agent aggregate.

---

## v0.4 — multi-seed accuracy confirmation

### Goal

Test whether the v0.2 walker signal survives 12 frozen seeds without retuning.

Protocol:

- seeds `0..11`
- circles, moons, two-arm spirals
- fixed hyperparameters
- `field_only`, `multi_no_gradient`, `mcgc_full`, distance-weighted kNN

Important commits:

- `d03ad16a5f8de0be7a8a44ba089efab868df3b4f` — benchmark
- `443916f2f5a079e11bf7818206f3a11256877e7f` — CI integration
- `f611dc65fbcb5da5ffa052fc9b8c31692b4790e6` — result documentation

Full MCGC minus field-only:

| Dataset | mean delta | W/T/L |
|---|---:|---:|
| circles | +0.00056 | 1/11/0 |
| moons | -0.00056 | 0/11/1 |
| spirals | -0.00667 | 3/2/7 |

On spirals, full MCGC lost to distance-weighted kNN on all 12 seeds with mean accuracy delta `-0.0950`.

### Conclusion

**Walker raw-accuracy gain not confirmed.** Explicit gradient gain was also essentially null.

Contrastive `why A rather than B` decomposition was added around this stage and reconstructs the requested class-score difference exactly.

---

## v0.5 — ambiguity-gated walkers

### Goal

Downweight walker corrections when local graph-field evidence is already decisive.

Gate:

`g(q) = exp(-(m(q) / m_ref) ** p)` with `p=1`.

Important commits:

- `a15177a65f5c8b22dc460b90ba8e14f87d783948` — uncertainty-gated walkers
- `332b990c07382cb3a081d0bd96b41792be213696` — XAI under gating
- `d0c4ffec38780b617bdf4f0876a11297efcb7add` — gate tests
- `b102185206fb49202783042d64987e13d87c35f7` — confirmation benchmark

The gate was active but did not change aggregate predictions relative to unconditional walkers in any of the 36 dataset-seed runs.

Mean gates:

- circles `0.3836`
- moons `0.4092`
- spirals `0.5348`

v0.5 passed 16 tests.

### Conclusion

The gate is reproducible and explainable but is not an accuracy fix. Development moved from forcing trajectory terms into prediction toward testing them as reliability information.

---

## v0.6 — intrinsic reliability

### Goal

Test whether internal graph/trail quantities rank MCGC errors without changing prediction or fitting a secondary risk model.

Added `src/mcgc/reliability.py`.

Important commits:

- `49e8bb2409bea329f031d758a72886f272a2732e` — reliability API
- `9d9657c2bd789a25cf02fd5b2353221c18f91307` — benchmark
- `a5d51ab68da77151ecef21990770eb15fcbe08f3` — frozen protocol
- `a6dd12d8fa0b55f1a36aa2f611f4d4671dd35e2f` — results

Error-detection AUC:

| Dataset | errors | local margin | final margin | trail disagreement |
|---|---:|---:|---:|---:|
| circles | 6 | 0.9929 | 0.9931 | 0.9936 |
| moons | 42 | **0.9197** | 0.9153 | 0.8847 |
| spirals | 505 | **0.7375** | 0.7257 | 0.6083 |

On spirals, local-margin selective filtering raised retained-set accuracy from `0.7194` at full coverage to `0.8711` at 50% coverage.

### Conclusion

The **local graph-field margin** is the strongest supported intrinsic reliability signal. Trail disagreement is descriptive but weaker. Lost-trail fraction is essentially uninformative.

---

## v0.7 — calibrated incremental reliability

### Goal

Test whether trail disagreement adds held-out reliability information after controlling for local margin.

Frozen 60/20/20 train/calibration/test split. Frozen risk models:

- `local_only`
- `local_plus_final`
- `local_plus_trail_disagreement`
- `combined_intrinsic`

Working confirmation checkpoint:

- `a7e7b869b27d27191644bff7b40a5fb5d836c53c`

Dedicated confirmation passed 19 tests before the science run.

Primary contrast `local + trail disagreement` minus `local only`:

- moons delta `-0.0213`, W/T/L `1/4/5`
- spirals delta `-0.00340`, W/L `3/9`

Secondary combined-intrinsic spiral signal:

- local AUC `0.73298`
- combined AUC `0.74085`
- delta `+0.007872`
- W/L `8/4`

but the same combined model was negative on moons.

### Conclusion

**Primary v0.7 hypothesis rejected.** The combined spiral signal remained exploratory and was carried forward unchanged to v0.8 for independent confirmation.

---

## v0.8 — frozen combined-intrinsic confirmation

### Goal

Confirm the exact five-feature v0.7 combined model on a broader suite with no feature search or retuning.

Features:

- local margin
- final margin
- trail disagreement
- coherence
- lost-trail risk

Datasets:

- moons_hard
- circles_hard
- spirals_hard
- overlap_hd
- breast_cancer
- wine
- digits_0_4

Important checkpoints:

- `788ef00d2a242c69d569bd2dabd893d83e09148a` — self-contained Windows-compatible benchmark fix
- `f7a8be5d3f94b743c7ffe85eb8f4c323992a5812` — frozen negative result

After the Windows import fix, the suite reached 22 passing tests before the full confirmation.

Across 53 evaluable paired runs:

- mean combined-minus-local ROC-AUC `-0.0130355`
- median `-0.00733945`
- W/T/L `19/2/32`

Every dataset had a negative mean paired delta.

### Conclusion

**Combined path-derived reliability did not replicate.** This is a closed negative checkpoint. Main emphasis moved to XAI faithfulness while local margin remained the reliability signal.

---

## v0.9 — frozen XAI faithfulness

### Goal

Test whether representative-path nodes are functionally linked to the fitted decision.

### Delete-and-refit intervention

For each eligible query:

1. fit MCGC;
2. obtain representative path;
3. delete path nodes;
4. refit the same model;
5. measure original winner-vs-runner margin drop;
6. compare with equal-size controls.

Controls:

- label-and-distance-matched off-path nodes
- equal number of nearest off-path nodes

Important commits:

- `9aebb3e51dd40a5b08c9fca1796449befa5b3672` — implementation checkpoint
- `133c2810295822e75ee62d2ccae92fc3b71ab76f` — result documentation

Datasets: moons_hard, spirals_hard, overlap_hd, breast_cancer. Five seeds, at most 12 eligible queries per seed.

Across 240 eligible queries:

Primary path-vs-matched advantage:

- mean `+0.0926154467`
- median `+0.0625828114`
- W/T/L `202/0/38`

Path-vs-nearest advantage:

- mean `+0.3593250906`
- median `+0.3049531662`
- W/T/L `207/0/33`

At the more reviewer-resistant dataset×seed level, all **20/20 units** had positive mean primary advantage and all 20/20 had positive path-vs-nearest advantage. Two-sided sign-test p is approximately `1.91e-06`.

Contrastive margin reconstruction remained exact to floating-point precision.

### Limitation

Not every query admits a nontrivial representative path. The claim applies only to eligible queries.

### Conclusion

**First strong positive MCGC result.**

Supported wording:

> For queries admitting a nontrivial representative path, MCGC identifies training examples whose removal changes the fitted class margin more strongly than matched off-path controls and equal-size nearest-neighbour explanations.

---

## v1.0 — frozen selective-manifold benchmark

### Goal

Test local graph-field margin against standard confidence baselines specifically on nonlinear manifolds.

Primary family:

- moons_hard
- circles_hard
- spirals_hard

Controls:

- overlap_hd
- breast_cancer

Twelve seeds, 70/30 train/test split. Primary metric: E-AURC.

Baselines:

- distance-weighted 12-NN confidence
- standardized logistic confidence

Important commits:

- `2517852cafb65a61f69ba211157023dd1b2e48a4` — implementation/protocol checkpoint
- `34ceef3e829f819613b9487673a9b2f30aa5a8e1` — positive result documentation

Against weighted kNN over 36 primary runs:

E-AURC advantage `kNN - MCGC`:

- mean `+0.009157069986134773`
- median `+0.006353876329219648`
- W/T/L `29/0/7`
- p `0.00031255115754902363`

Ordinary AURC advantage:

- mean `+0.008125534126382374`
- W/T/L `25/0/11`
- p `0.028816719655878842`

Per-dataset E-AURC MCGC / kNN:

- moons `0.018809 / 0.026451`, W/L `11/1`
- circles `0.017257 / 0.023252`, W/L `9/3`
- spirals `0.192063 / 0.205898`, W/L `9/3`

This is a selective-risk result, not universal raw-accuracy superiority.

### Conclusion

**Second strong positive MCGC result.**

---

## v1.1 — stronger nonlinear selective baselines

### Goal

Stress-test the frozen v1.0 MCGC local-margin confidence against stronger nonlinear baselines without retuning MCGC.

Branch:

- `feature/v1.1-stronger-selective-baselines`

Key commits:

- `cc2dbe6c3cc4457662e175ad9e7308fc917be074` — freeze protocol
- `5981e20267899df029e76276955424e5db59ac4d` — add benchmark implementation
- `2060cce38d7f6e3b102847f445917e36f96d8bec` — add protocol regression tests
- `2e534c6083f7d6ec490c8ae1a3a7be55b566b04b` — document frozen result

New baselines:

RBF SVM:

- StandardScaler
- `SVC(kernel='rbf', C=1.0, gamma='scale', probability=False)`
- confidence from decision-function margin

Random Forest:

- 300 trees
- `max_features='sqrt'`
- confidence from maximum class probability

No hyperparameter search.

### Pooled primary family result

Against RBF-SVM:

E-AURC advantage `SVM - MCGC`:

- mean `+0.014414490689435011`
- median `+0.004798513586381163`
- W/T/L `25/0/11`
- p `0.028816719655878842`

Ordinary AURC:

- mean advantage `+0.016056685079267947`
- W/T/L `22/0/14`
- p `0.2429849540349096`

Against Random Forest:

E-AURC advantage `RF - MCGC`:

- mean `+0.010919444296094165`
- median `+0.009838041947084559`
- W/T/L `31/0/5`
- p `1.2913485988974571e-05`

Ordinary AURC:

- mean advantage `+0.012733213240176582`
- median `+0.010936196421952437`
- W/T/L `29/0/7`
- p `0.00031255115754902363`

### Per-dataset E-AURC

`moons_hard`:

- MCGC `0.018809`
- SVM `0.024504`, MCGC W/L `11/1`
- RF `0.027095`, MCGC W/L `12/0`

`circles_hard`:

- MCGC `0.017257`
- SVM `0.014876`, MCGC W/L `3/9`
- RF `0.032230`, MCGC W/L `11/1`

`spirals_hard`:

- MCGC `0.192063`
- SVM `0.231993`, MCGC W/L `11/1`
- RF `0.201563`, MCGC W/L `8/4`

So MCGC does **not** beat SVM on circles. The pooled SVM claim must remain an aggregate manifold-family claim.

### Controls

`overlap_hd`:

- MCGC `0.076918`
- SVM `0.075446`
- RF `0.072237`

`breast_cancer`:

- MCGC `0.005902`
- SVM `0.002403`
- RF `0.004813`
- logistic `0.002122`

The conventional tabular controls therefore bound the claim clearly: the v1.1 advantage is geometry-specific, not universal.

### Reproducibility note

The local benchmark completed and wrote the full JSON result. The console emitted a scikit-learn deprecation warning involving the `SVC` probability parameter. The frozen code uses `probability=False` and derives SVM risk from the decision-function margin, so probability calibration is not part of the v1.1 comparison.

A dedicated v1.1 test module freezes datasets, seeds, baselines, SVM risk orientation and positive-advantage sign convention. No CI status was available for the final local-run commit at the time of this history update, so this file does not claim a CI pass for v1.1.

### Conclusion

**v1.1 is positive.** The v1.0 selective-manifold result survives two stronger nonlinear baseline challenges in the aggregate, most strongly against Random Forest.

Supported statement:

> On the frozen nonlinear-manifold benchmark, MCGC local graph-field margin provides better excess selective-risk ranking than distance-weighted kNN confidence, fixed RBF-SVM margin confidence and fixed Random-Forest probability confidence in the aggregate, while the effect is not universal across every dataset or conventional tabular control.

---

# Reproducibility policy

1. Freeze protocol and primary contrasts before confirmation results.
2. Do not retune thresholds, features, datasets or metrics after viewing frozen results.
3. Keep negative confirmations in the repository.
4. Separate exploratory observations from confirmation claims.
5. Keep classification, XAI and reliability claims separate.
6. Use fixed seeds and explicit train/calibration/test separation where relevant.
7. Run unit/regression tests before science benchmarks.
8. Use GitHub Actions for portable regression testing when practical. Long science benchmarks may be run locally.
9. Store complete JSON results and use console logs only as summaries.
10. Prefer paired statistics and always report per-dataset outcomes so one benchmark cannot silently drive the aggregate claim.

Local runs are commonly executed in Windows PowerShell from `C:\Articles\Clanci\MCGC-new-repo`.

---

# Claim ledger

## Supported

### Intrinsic path faithfulness

For eligible queries, path-selected training nodes are more functionally decision-relevant under delete-and-refit intervention than matched off-path controls and equal-size nearest-neighbour explanations.

Evidence: v0.9.

### Intrinsic reliability

Local contrastive graph-field margin is a useful error-ranking / abstention signal.

Evidence: v0.6 onward.

### Selective risk on nonlinear manifolds

On the frozen moons/circles/spirals family:

- MCGC beats distance-weighted kNN confidence in E-AURC on `29/36` paired runs;
- MCGC beats fixed RBF-SVM margin confidence in E-AURC on `25/36` paired runs;
- MCGC beats fixed Random-Forest probability confidence in E-AURC on `31/36` paired runs.

Evidence: v1.0 and v1.1.

## Not supported / explicitly rejected

- general walker improvement in raw accuracy;
- explicit gradient-term accuracy improvement;
- selective walker gate as an accuracy fix;
- trail disagreement as complementary reliability beyond local margin;
- five-feature combined intrinsic reliability as a generally better uncertainty model;
- universal selective-risk superiority across arbitrary data;
- universal raw-accuracy superiority;
- universal interpretability;
- SOTA claims.

## Important boundaries

- RBF-SVM beats MCGC on the frozen circles E-AURC comparison.
- RBF-SVM, Random Forest and logistic regression outperform MCGC on the conventional breast-cancer control in E-AURC.
- v0.9 path-faithfulness claims apply only to queries with nontrivial representative paths.

## Next priorities

1. **Tuned-baseline stress test:** keep MCGC frozen, but allow strong baselines to tune their conventional hyperparameters on an inner training/validation split or nested CV. This directly addresses the reviewer objection that fixed default SVM/RF settings may be under-tuned.
2. **Literature/novelty audit:** especially Optimum-Path Forest, path-based classifiers, graph random-walk classifiers, class-conditioned diffusion and intrinsic graph-XAI methods.
3. **Broader real-data validation:** prioritize datasets where nonlinear/manifold structure is plausible rather than claiming universal tabular superiority.
4. **XAI extension:** explanation stability and counterfactual validation, while preserving the strong frozen v0.9 faithfulness result.

---

# Current state

Current branch:

`feature/v1.1-stronger-selective-baselines`

Latest scientific result document:

`docs/V11_STRONGER_SELECTIVE_BASELINES_RESULT.md`

v1.1 result documentation commit:

`2e534c6083f7d6ec490c8ae1a3a7be55b566b04b`

The next recommended frozen checkpoint is a **tuned-baseline stress test** in which MCGC remains unchanged while SVM, RF and kNN receive fair inner-split hyperparameter selection. That is the most direct way to determine whether the v1.1 selective-risk advantage survives a reviewer-grade baseline challenge.
