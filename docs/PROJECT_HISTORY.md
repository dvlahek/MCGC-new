# MCGC project history and scientific decision log

This is the canonical running history for `dvlahek/MCGC-new`. It records frozen code checkpoints, tests, positive and negative results, and the scientific decisions that followed. Update it after every completed frozen version.

Working name: **Multi-Agent Chemotactic Graph Classifier (MCGC)**.

Core separation used throughout the project:

1. **classification** — whether walker/path information improves class prediction;
2. **XAI** — whether the fitted decision can be traced through exact components, supporting samples and graph paths;
3. **reliability/selective prediction** — whether intrinsic graph quantities rank likely errors and support abstention.

## Current scientific position after v1.2

The strongest supported story is not universal raw-accuracy superiority.

Two empirically distinct contributions currently survive frozen confirmation:

1. **Intrinsic path faithfulness (v0.9).** For queries with nontrivial representative paths, deleting path-selected training nodes and refitting changes the fitted class margin more than deleting matched off-path controls or equal-size nearest-neighbour controls.
2. **Selective confidence on nonlinear manifolds (v1.0–v1.2).** The frozen MCGC local graph-field margin provides lower excess selective risk than tuned kNN confidence and tuned Random-Forest confidence on the frozen moons/circles/spirals family. Tuned RBF-SVM is competitive overall and removes the fixed-SVM aggregate advantage seen in v1.1.

Important closed negative results remain part of the contribution boundary:

- walker trajectory terms did not provide a stable raw-accuracy improvement;
- ambiguity gating did not repair that failure;
- trail disagreement did not add held-out reliability information beyond local margin;
- the five-feature combined intrinsic reliability model failed broader confirmation;
- MCGC is not universally better than tuned nonlinear baselines or conventional tabular models.

Current defensible positioning:

> MCGC is an intrinsically traceable graph classifier whose representative paths identify functionally decision-relevant training samples, while its local graph-field margin provides improved selective-risk ranking relative to tuned kNN and tuned Random-Forest confidence on a frozen nonlinear-manifold benchmark.

Do not expand this into universal accuracy, uncertainty, interpretability, causal or SOTA claims without new evidence.

---

# Version history

## v0.1 — first reproducible prototype

Goal: implement the scent/chemotaxis-inspired graph classifier as a sklearn-compatible research prototype.

Core implementation:

- `src/mcgc/classifier.py`
- k-NN graph and locally scaled edge weights
- class-specific evidence fields
- stochastic class-conditioned walkers
- initial score aggregation

Important early commits:

- `ac8e838501aff1eada17aa853a932d7d83af9e6e`
- `88588e6203558a3a74ac83ef87c53ff7a338847b`

Important design correction: naive uphill-walker scoring could let a wrong-class walker beat a query already inside a strong correct-class basin. Local contrastive evidence therefore became the score anchor and walkers became refinements.

Smoke results after the correction:

- circles: kNN `0.9933`, MCGC `0.9933`
- moons: kNN `0.9667`, MCGC `0.9600`

Decision: no performance claim from smoke tests.

---

## v0.2 — controlled walker ablation

Goal: isolate graph-field prediction from trajectory contribution.

Added ablation controls for walker use and score weights. Compared uniform/distance kNN, Label Spreading, field-only, one-walker, no-gradient and full MCGC variants.

Important commits:

- `970d75ba989081245ece010f8743557363a6e5fc`
- `b48e6d94d1edab8a0e68281c3d09f48be5055c69`
- `69444556cda0e8c9ad7f09bc0664cb8dbd7e4a7a`

Single-split spiral diagnostic suggested `field_only 0.7556 -> full 0.7722`, but distance kNN and Label Spreading were both `0.8111`.

Tests/CI: six tests passed at this stage on Python 3.10, 3.11 and 3.12.

Decision: treat the apparent spiral gain as exploratory until multi-seed confirmation.

---

## v0.3 — XAI-first layer

Goal: expose faithful explanation objects without changing prediction.

Added:

- `src/mcgc/xai.py`
- `explain_mcgc()`
- deterministic representative trails
- supporting training examples
- exact additive class-score decomposition

Important commits:

- `3315a1527edf84a9dc5ea79d0c1580cee8baac22`
- `229cdea59bf8bc45e0d491d375c2a2c7ec9c3801`
- `55db66d83b15f3e39b37837b0df127ecc47501f6`

Tests verify prediction identity, exact score reconstruction and deterministic representative-path reproducibility. CI passed on Python 3.10–3.12.

Decision: representative trails are traces through the fitted evidence landscape, not sole causal paths for stochastic scoring.

---

## v0.4 — frozen multi-seed accuracy confirmation

Goal: test the v0.2 walker signal over 12 frozen seeds with no retuning.

Important commits:

- `d03ad16a5f8de0be7a8a44ba089efab868df3b4f`
- `443916f2f5a079e11bf7818206f3a11256877e7f`
- `f611dc65fbcb5da5ffa052fc9b8c31692b4790e6`

Full-minus-field-only accuracy:

| dataset | mean delta | W/T/L |
|---|---:|---:|
| circles | +0.00056 | 1/11/0 |
| moons | -0.00056 | 0/11/1 |
| spirals | -0.00667 | 3/2/7 |

On spirals, full MCGC lost to distance-weighted kNN on all 12 seeds with mean accuracy delta `-0.0950`.

Conclusion: **walker raw-accuracy gain not confirmed**. Explicit gradient contribution was also essentially null.

Around this stage, exact contrastive `why A rather than B` decomposition was added.

---

## v0.5 — ambiguity-gated walkers

Goal: downweight walker corrections when local evidence is already decisive.

Gate: `g(q)=exp(-(m(q)/m_ref)^p)` with frozen `p=1`.

Important commits:

- `a15177a65f5c8b22dc460b90ba8e14f87d783948`
- `332b990c07382cb3a081d0bd96b41792be213696`
- `d0c4ffec38780b617bdf4f0876a11297efcb7add`
- `b102185206fb49202783042d64987e13d87c35f7`

The gate was numerically active but changed no aggregate predictions in the 36 dataset×seed runs.

Tests: 16 passing tests at the frozen checkpoint.

Conclusion: gating is reproducible and explainable but not an accuracy fix.

---

## v0.6 — intrinsic reliability

Goal: test whether internal graph/trail signals identify model errors without fitting a secondary risk model.

Added `src/mcgc/reliability.py`.

Important commits:

- `49e8bb2409bea329f031d758a72886f272a2732e`
- `9d9657c2bd789a25cf02fd5b2353221c18f91307`
- `a5d51ab68da77151ecef21990770eb15fcbe08f3`
- `a6dd12d8fa0b55f1a36aa2f611f4d4671dd35e2f`

Error-detection AUC:

| dataset | local margin | final margin | trail disagreement |
|---|---:|---:|---:|
| circles | 0.9929 | 0.9931 | 0.9936 |
| moons | **0.9197** | 0.9153 | 0.8847 |
| spirals | **0.7375** | 0.7257 | 0.6083 |

On spirals, local-margin filtering raises retained accuracy from `0.7194` at 100% coverage to `0.8711` at 50% coverage.

Conclusion: local graph-field margin is the strongest supported intrinsic reliability signal.

---

## v0.7 — calibrated incremental reliability

Goal: test whether walker disagreement adds reliability information beyond local margin using a strict 60/20/20 train/calibration/test split.

Key checkpoint:

- `a7e7b869b27d27191644bff7b40a5fb5d836c53c`

Tests: 19/19 passed before the dedicated frozen benchmark.

Primary `local + trail disagreement` minus `local only` AUC:

- moons: `-0.0213`, W/T/L `1/4/5`
- spirals: `-0.00340`, W/L `3/9`

Secondary combined-intrinsic spiral signal: `+0.007872`, W/L `8/4`, but negative on moons.

Conclusion: **primary v0.7 hypothesis rejected**. The secondary combined signal was carried forward unchanged to v0.8.

---

## v0.8 — frozen combined-intrinsic confirmation

Goal: confirm the exact five-feature v0.7 combined model on a broader suite without feature search or retuning.

Frozen features: local margin, final margin, trail disagreement, coherence, lost-trail risk.

Datasets: moons_hard, circles_hard, spirals_hard, overlap_hd, breast_cancer, wine, digits_0_4.

Important commits:

- `788ef00d2a242c69d569bd2dabd893d83e09148a` — Windows/self-contained import fix
- `f7a8be5d3f94b743c7ffe85eb8f4c323992a5812` — frozen result

Tests: 22 passed before full confirmation.

Across 53 evaluable paired runs:

- mean combined-minus-local ROC-AUC `-0.0130355`
- median `-0.00733945`
- W/T/L `19/2/32`

Every dataset had a negative mean paired delta.

Conclusion: **combined path-derived reliability did not replicate**. Closed negative checkpoint.

---

## v0.9 — frozen XAI faithfulness

Goal: test whether representative-path nodes are functionally linked to the fitted decision.

Delete-and-refit intervention compares path-node removal against:

- label-and-distance-matched off-path controls
- equal-size nearest off-path controls

Important commits:

- `9aebb3e51dd40a5b08c9fca1796449befa5b3672` — implementation
- `133c2810295822e75ee62d2ccae92fc3b71ab76f` — result documentation

Datasets: moons_hard, spirals_hard, overlap_hd, breast_cancer. Five seeds, at most 12 eligible queries per seed.

Across 240 eligible queries:

- path-vs-matched mean advantage `+0.0926154467`, median `+0.0625828114`, W/T/L `202/0/38`
- path-vs-nearest mean advantage `+0.3593250906`, median `+0.3049531662`, W/T/L `207/0/33`

Reviewer-resistant dataset×seed aggregation:

- `20/20` units positive for matched-control advantage
- `20/20` units positive for nearest-control advantage
- sign-test p approximately `1.91e-06`

Contrastive reconstruction remains exact to floating-point precision.

Limitation: only queries with nontrivial representative paths are eligible.

Conclusion: **first strong positive MCGC result**.

---

## v1.0 — frozen selective-manifold benchmark

Goal: test local-margin risk against standard confidence baselines on a predefined nonlinear-manifold family.

Primary datasets: moons_hard, circles_hard, spirals_hard. Controls: overlap_hd, breast_cancer. Twelve seeds, 70/30 train/test. Primary metric: E-AURC.

Important commits:

- `2517852cafb65a61f69ba211157023dd1b2e48a4`
- `34ceef3e829f819613b9487673a9b2f30aa5a8e1`

Against distance-weighted kNN across 36 primary runs:

- mean E-AURC advantage `+0.00915707`
- median `+0.00635388`
- W/T/L `29/0/7`
- p `0.00031255`

Ordinary AURC: mean advantage `+0.00812553`, W/T/L `25/0/11`, p `0.02882`.

Conclusion: **second strong positive result** — improved selective-risk ranking versus weighted kNN on the frozen manifold family.

---

## v1.1 — stronger fixed nonlinear baselines

Goal: stress-test v1.0 against fixed RBF-SVM and Random Forest without changing MCGC.

Protocol commit:

- `cc2dbe6c3cc4457662e175ad9e7308fc917be074`

Implementation/test head before local science run:

- `2060cce38d7f6e3b102847f445917e36f96d8bec`

Result documentation:

- `2e534c6083f7d6ec490c8ae1a3a7be55b566b04b`

Primary 36-run E-AURC advantages:

- vs weighted kNN: `+0.009157`, W/T/L `29/0/7`
- vs fixed RBF-SVM: `+0.014414`, W/T/L `25/0/11`, p `0.028817`
- vs fixed Random Forest: `+0.010919`, W/T/L `31/0/5`, p `1.29e-05`

Important boundary: fixed RBF-SVM beats MCGC on circles. Conventional control datasets also limit the claim.

Conclusion: v1.1 positive, but reviewer-grade validation requires tuned baselines.

---

## v1.2 — nested tuned-baseline confirmation

### Goal

Keep MCGC completely frozen while giving competing classifiers fair hyperparameter selection targeted directly at selective-risk quality.

Branch:

- `feature/v1.2-tuned-selective-baselines`

Protocol / implementation / test commits:

- `3c1fd98480a4560aac1555c32a3f7b7952fbf144` — protocol freeze
- `ca734a32f26893a012478e5bbcdcee8c5ca4ca1e` — benchmark implementation
- `d4ed7aa0445e3b41aa5e617bc1b1954dc0913de8` — frozen protocol tests

Result document:

- `docs/V12_TUNED_SELECTIVE_BASELINES_RESULT.md`

Result documentation commit:

- `d653e55c5d4eb471557df2995cc0131ff862b15c`

### Tuning protocol

Outer split: 70% train / 30% untouched test.

Baseline selection uses 3-fold stratified inner CV on outer training only. Selection metric is lowest OOF E-AURC, with accuracy and deterministic lexical ordering only as tie-breaks.

MCGC receives **no tuning**.

Frozen grids:

- kNN: `k={5,9,12,17,25}`, uniform/distance weights
- logistic: `C={0.1,1,10}`
- RBF-SVM: `C={0.1,1,10}`, `gamma={scale,0.1,1}`
- Random Forest: depth `{None,6,12}`, leaf `{1,3,5}`, features `{sqrt,0.5}`

### Primary 36-run manifold-family result

Paired E-AURC advantage is baseline minus MCGC, so positive values favor MCGC.

| tuned baseline | mean advantage | median | W/T/L | p |
|---|---:|---:|---:|---:|
| kNN | `+0.00806614` | `+0.00213883` | `28/0/8` | `0.00119324` |
| logistic | `+0.13530450` | `+0.02771595` | `32/0/4` | `1.94e-06` |
| RBF-SVM | `-0.00210575` | `-0.00016705` | `18/0/18` | `1.0` |
| Random Forest | `+0.00551738` | `+0.00620743` | `26/0/10` | `0.0113310` |

Ordinary AURC against tuned RF is also positive:

- mean `+0.00500993`
- median `+0.00568940`
- W/T/L `25/0/11`
- p `0.0288167`

Against tuned kNN, ordinary AURC mean remains positive (`+0.00656598`) but W/T/L `23/0/13` is not sign-test significant (`p=0.13250`).

### Per-dataset E-AURC

`moons_hard`:

- MCGC `0.018809`
- tuned kNN `0.021708`, MCGC W/L `7/5`
- tuned SVM `0.023662`, MCGC W/L `10/2`
- tuned RF `0.024398`, MCGC W/L `9/3`

`circles_hard`:

- MCGC `0.017257`
- tuned kNN `0.021627`, MCGC W/L `11/1`
- tuned SVM `0.014990`, MCGC W/L `3/9`
- tuned RF `0.028869`, MCGC W/L `11/1`

`spirals_hard`:

- MCGC `0.192063`
- tuned kNN `0.208993`, MCGC W/L `10/2`
- tuned SVM `0.183160`, MCGC W/L `5/7`
- tuned RF `0.191415`, MCGC W/L `6/6`

### Controls

`overlap_hd`: MCGC is approximately tied with tuned kNN but worse on average than tuned SVM and RF.

`breast_cancer`: tuned logistic, SVM and RF all beat MCGC in E-AURC, while MCGC strongly beats tuned kNN.

### Conclusion

**v1.2 is a mixed but strengthening result.**

The kNN advantage survives nested tuning. More importantly, the Random-Forest advantage also survives tuning in the pooled manifold family. The fixed-SVM advantage from v1.1 does not survive SVM tuning.

Supported reviewer-resistant statement:

> On the frozen nonlinear-manifold benchmark, MCGC local graph-field margin retains lower excess selective risk than tuned kNN confidence and tuned Random-Forest confidence under nested inner-CV model selection. Tuned RBF-SVM is competitive overall and removes the fixed-SVM advantage observed in v1.1.

This is stronger and more credible than claiming universal nonlinear-baseline superiority.

---

# Reproducibility policy

1. Freeze protocol and primary contrasts before confirmation results.
2. Do not retune MCGC thresholds, features, datasets or metrics after viewing frozen results.
3. Keep negative confirmations in the repository.
4. Separate exploratory observations from confirmation claims.
5. Keep classification, XAI and reliability claims separate.
6. Use fixed seeds and explicit train/calibration/test or nested-selection separation where relevant.
7. Run unit/regression tests before science benchmarks.
8. Use GitHub Actions when practical; long science benchmarks may be run locally.
9. Store complete JSON outputs and use console logs only as summaries.
10. Prefer paired statistics and report per-dataset outcomes so one benchmark cannot silently drive an aggregate claim.

Local science runs are commonly executed in Windows PowerShell from `C:\Articles\Clanci\MCGC-new-repo`.

---

# Claim ledger

## Supported

### Intrinsic XAI / path faithfulness

For eligible queries with nontrivial representative paths, path-selected nodes are more functionally decision-relevant under delete-and-refit intervention than matched off-path controls and equal-size nearest-neighbour explanations.

Evidence: v0.9.

### Intrinsic reliability

Local contrastive graph-field margin is a useful error-ranking / abstention signal.

Evidence: v0.6 onward.

### Selective risk on nonlinear manifolds

On the frozen moons/circles/spirals family:

- MCGC beats **tuned kNN confidence** in E-AURC on `28/36` runs, mean advantage `+0.008066`, p `0.00119`;
- MCGC beats **tuned Random-Forest confidence** in E-AURC on `26/36` runs, mean advantage `+0.005517`, p `0.01133`;
- tuned RBF-SVM is approximately even overall in win count (`18/18`) and slightly better in mean E-AURC.

Evidence: v1.2.

## Not supported / explicitly rejected

- general walker improvement in raw accuracy;
- explicit gradient-term accuracy improvement;
- selective walker gate as an accuracy fix;
- trail disagreement as complementary reliability beyond local margin;
- five-feature combined intrinsic reliability as a generally better uncertainty model;
- superiority to tuned RBF-SVM;
- universal selective-risk superiority;
- universal raw-accuracy superiority;
- universal interpretability or causal explanation;
- SOTA claims.

## Important boundaries

- tuned RBF-SVM beats MCGC on circles and is better on average on spirals;
- tuned SVM/RF/logistic outperform MCGC on the conventional breast-cancer control;
- v0.9 path-faithfulness applies only to queries with nontrivial representative paths;
- the tuned-baseline selective-risk claim is currently tied to the frozen nonlinear-manifold family.

---

# Current development state

Current branch:

`feature/v1.2-tuned-selective-baselines`

Latest result document:

`docs/V12_TUNED_SELECTIVE_BASELINES_RESULT.md`

Latest result documentation commit:

`d653e55c5d4eb471557df2995cc0131ff862b15c`

Recommended next priorities:

1. broaden the tuned-baseline test to independent real datasets with plausible nonlinear/manifold structure, without changing MCGC;
2. perform the deep literature/novelty audit against Optimum-Path Forest, path-based classifiers, random-walk graph classifiers, class-conditioned diffusion and intrinsic graph-XAI;
3. extend XAI validation with stability and counterfactual tests while preserving the frozen v0.9 intervention result;
4. only after these checks decide whether the strongest paper framing should be primarily XAI, selective classification, or a combined intrinsic-XAI + selective-confidence contribution.
