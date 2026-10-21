# MCGC manuscript skeleton v1

## Working positioning

Primary positioning: intrinsic XAI / interpretable-by-design graph classification.

Secondary positioning: geometry-sensitive selective prediction.

The paper should not be framed as a universal accuracy paper. The strongest evidence is that the fitted MCGC decision can be traced through class-conditioned graph evidence, training examples, representative paths, exact winner-runner score decomposition, native feature interventions, interventional delete-and-refit faithfulness, and perturbation stability.

## Working title candidates

1. **Intrinsic Contrastive Explanations in a Class-Conditioned Chemotactic Graph Classifier**
2. **Path-Structured and Feature-Level Explanations in a Class-Conditioned Graph Classifier**
3. **An Intrinsically Explainable Chemotactic Graph Classifier with Faithful Training-Example Paths**
4. **From Local Evidence to Graph Paths: Intrinsic Explanations in Chemotactic Graph Classification**

Preferred current title: **From Local Evidence to Graph Paths: Intrinsic Explanations in Chemotactic Graph Classification**.

## One-sentence paper claim

MCGC builds class-conditioned evidence over a sample graph and exposes the fitted decision through contrastive score components, training examples, representative evidence paths, and native feature interventions, with the resulting explanations validated for functional faithfulness and local stability.

## Claim boundaries

Do not claim:

- first path-based classifier
- first graph classifier with explainability
- universal accuracy superiority
- universal selective-risk superiority
- that the representative path is the single stochastic causal trajectory used by the classifier
- that feature-intervention effects are an additive decomposition of the final score
- biomedical causality from the breast-cancer feature ranking

Safe language:

- intrinsically traceable
- interpretable-by-design
- path-structured explanation
- exact winner-runner score decomposition
- paired feature-intervention effect
- path-geometric feature attribution
- functionally relevant to the fitted decision under delete-and-refit intervention
- query-specific explanation stability

## Abstract skeleton

Graph classifiers can exploit local sample geometry, but their predictions are often difficult to trace back to training examples, paths through the data manifold, and original input features. We introduce MCGC, a class-conditioned multi-agent chemotactic graph classifier that constructs one evidence field per class on a mutualised k-nearest-neighbour graph and combines local contrastive evidence with trajectory-derived terms. The model exposes the same internal quantities used for prediction as an intrinsic explanation: local supporting examples, deterministic representative evidence paths, exact winner-versus-runner score decomposition, and native feature-level interventions. We evaluate explanation faithfulness by deleting path-selected training nodes and refitting the model, and explanation stability by perturbing the same query while comparing against same-class control queries. Across 20 dataset-by-seed units, support explanations remain more similar under within-query perturbation than under same-class controls in all 20 units, while representative-path explanations do so in all 17 evaluable units. Path-selected training examples also induce larger fitted-margin changes under deletion than matched and proximity-only off-path controls. On breast-cancer data, native feature rankings show moderate-to-strong outer-split stability and high robustness to mean-versus-median reference replacement. MCGC additionally provides competitive selective-risk ranking on nonlinear manifold tasks, although this advantage does not transfer universally to real-data benchmarks. These results position MCGC primarily as an intrinsically explainable graph classifier whose decision structure can be inspected at score, path, training-example, and feature levels.

## 1. Introduction

### Problem first

Many classifiers can assign a class label with high predictive accuracy, but the final prediction alone does not reveal how the model moved from local evidence to a class preference. In graph-based classification this problem is particularly important because the decision may depend on both the query neighbourhood and the geometry of paths through the sample graph.

### Limitation

Nearest-neighbour explanations expose proximity but not class-conditioned evidence progression. Post-hoc feature attribution can rank variables but does not necessarily describe the sample-graph structure used by the classifier. Path-based graph methods, in turn, do not automatically provide exact contrastive score decomposition, training-example faithfulness, or original-feature attribution within the same decision mechanism.

### Core intuition

The motivating analogy is chemotaxis: an agent does not only evaluate the signal at its current position, but can follow local increases toward a stronger source. In MCGC, each class induces its own evidence field over the training graph. For a query, the classifier evaluates local class evidence and class-conditioned trajectories through the graph. The same structure naturally exposes a path-based explanation.

### What changed during development

The original hypothesis was that multi-agent trajectories might robustly improve raw predictive accuracy over local graph evidence alone. Frozen ablations did not support that universal claim. This redirected the paper toward the property that remained strong under independent tests: intrinsic explanation structure and its empirical faithfulness.

### Contributions

The final contribution list should be limited to four points:

1. **Class-conditioned graph decision mechanism.** MCGC combines local contrastive evidence with trajectory-derived terms on a sample graph while preserving an explicit winner-versus-runner score.
2. **Multi-level intrinsic explanation.** The fitted model exposes local support examples, representative evidence paths, exact score-component decomposition, and native feature-intervention effects without requiring a separate surrogate explainer.
3. **Explanation validation.** Path-selected training examples are tested by delete-and-refit interventions against matched and nearest off-path controls, and explanation stability is tested against same-class query controls under input perturbation.
4. **Scope and limitations.** Selective-risk improvements are observed on nonlinear manifold benchmarks but do not transfer universally to real-data tasks, defining a clear boundary between the XAI contribution and predictive-performance claims.

## 2. Related work

### 2.1 Graph-based classification and semi-supervised learning

Cover:

- k-NN graph classifiers
- label propagation / harmonic functions / graph diffusion
- random-walk-based classification
- class-conditioned diffusion / personalized propagation where relevant

### 2.2 Path-based classification

Must include a careful comparison with:

- Optimum-Path Forest
- shortest/geodesic path classifiers
- density-ascent / high-density path methods
- graph search over learned potentials

The novelty must not be stated before this audit is complete.

### 2.3 Example-based and graph explanations

Cover:

- prototypes / criticisms / nearest-example explanations
- influence and data attribution
- training-example deletion / retraining tests
- graph path explanations

### 2.4 Feature attribution and intrinsic explainability

Cover:

- permutation / ablation importance
- SHAP / additive post-hoc attribution as contrastive background
- intrinsic interpretable models

Position MCGC feature effects as paired model interventions, not as SHAP-like additive attributions.

## 3. Method

### 3.1 Sample graph

Given training samples \(x_i\in\mathbb{R}^d\), standardize features to \(z_i\) and construct a mutualised k-NN graph.

For connected nodes \(i,j\), define the locally scaled heat-kernel affinity

\[
w_{ij}=\exp\left[-\frac{\lVert z_i-z_j\rVert^2}{\sigma_i\sigma_j}\right].
\]

Let \(W=[w_{ij}]\) and \(P=D^{-1}W\).

### 3.2 Class-conditioned evidence fields

For class \(c\), let \(b^{(c)}\) denote one-hot training evidence. The implemented iterative diffusion is

\[
s_{t+1}^{(c)}=(1-\alpha)b^{(c)}+\alpha P s_t^{(c)}.
\]

After the fixed number of diffusion steps, define the contrastive field

\[
D_i^{(c)}=S_i^{(c)}-\max_{r\neq c}S_i^{(r)}.
\]

This is class evidence relative to the strongest alternative at node \(i\).

### 3.3 Query-local evidence

For a query \(q\), obtain graph-start nodes \(\mathcal N(q)\) and normalized affinity weights \(a_j(q)\). The local evidence for class \(c\) is

\[
L_c(q)=\sum_{j\in\mathcal N(q)}a_j(q)D_j^{(c)}.
\]

### 3.4 Stochastic class-conditioned walkers

A walker at node \(i\) evaluating class \(c\) scores neighbour \(j\) with

\[
H_{ij}^{(c)}=
\beta_1D_j^{(c)}+
\beta_2\big(D_j^{(c)}-D_i^{(c)}\big)+
\beta_3\log(w_{ij}+\varepsilon)-
\beta_4R_{ij},
\]

where \(R_{ij}\) is the revisit term. The next node is sampled by a temperature-controlled softmax over \(H_{ij}^{(c)}\).

Each trail contributes a score based on positive evidence gain, coherence, terminal evidence, graph plausibility, and trail failure.

### 3.5 Adaptive walker gate

The trajectory term is attenuated when the local field is already decisive. Let the local winner-runner margin be \(m(q)\) and let \(m_{\mathrm{ref}}\) be the fitted training-node reference scale. The gate is

\[
g(q)=\exp\left[-\left(\frac{m(q)}{m_{\mathrm{ref}}}\right)^p\right].
\]

### 3.6 Final class score

For class \(c\), the implemented score is

\[
F_c(q)=F_{c,\mathrm{local}}(q)+F_{c,\mathrm{trail}}(q)+F_{c,\mathrm{agreement}}(q)+F_{c,\mathrm{lost}}(q).
\]

For winner \(c^+\) and runner-up \(c^-\), define

\[
M(q)=F_{c^+}(q)-F_{c^-}(q).
\]

Because every class score is stored as the sum of its implemented components,

\[
M(q)=
\Delta F_{\mathrm{local}}+
\Delta F_{\mathrm{trail}}+
\Delta F_{\mathrm{agreement}}+
\Delta F_{\mathrm{lost}},
\]

up to floating-point precision. This is an exact score-level contrastive decomposition.

### 3.7 Representative evidence path

The representative path is a deterministic summary of the fitted evidence landscape, not the unique stochastic trajectory responsible for the prediction.

Starting from the query neighbourhood, choose a class-conditioned initial node from affinity and class evidence. Then greedily select the next node using the same field, gradient, geometry, and revisit ingredients used by the walker desirability until the class basin is reached or no positive evidence ascent remains.

For class \(c\), the representative path is

\[
P_c(q)=(v_0,v_1,\ldots,v_T)
\]

with trace

\[
D_{v_0}^{(c)},D_{v_1}^{(c)},\ldots,D_{v_T}^{(c)}.
\]

### 3.8 Local support examples

Supporting examples are affinity-ranked query neighbours. They must not be described as ranked causal contributors. They provide local training-example context around the query.

### 3.9 Native feature attribution

#### Paired decision-feature intervention

For original winner \(c^+\) and runner-up \(c^-\), keep the class pair fixed and define the original margin

\[
M(q)=F_{c^+}(q)-F_{c^-}(q).
\]

Let \(q^{(-k)}\) replace feature \(k\) by an outer-training reference value. The local feature effect is

\[
I_k(q)=M(q)-M(q^{(-k)}).
\]

Interpretation:

- \(I_k(q)>0\): the observed value of feature \(k\) supports the original winner relative to the runner-up
- \(I_k(q)<0\): the observed value supports the runner-up relative to the winner
- \(|I_k(q)|\): magnitude of the paired intervention effect

Important: \(I_k(q)\) is not an additive decomposition, so generally

\[
M(q)\neq\sum_k I_k(q).
\]

#### Global feature importance

Across test queries,

\[
GI_k=\frac{1}{N}\sum_{q=1}^N |I_k(q)|.
\]

Report split-level variability rather than presenting the exact ordering as invariant.

#### Path-geometric feature attribution

For an edge \((i,j)\) on a representative path,

\[
\lVert z_i-z_j\rVert^2=\sum_{k=1}^d(z_{ik}-z_{jk})^2.
\]

Define the feature share

\[
\rho_{ij,k}=
\frac{(z_{ik}-z_{jk})^2}
{\sum_{\ell=1}^d(z_{i\ell}-z_{j\ell})^2}.
\]

Therefore

\[
\sum_k\rho_{ij,k}=1.
\]

This is an exact decomposition of standardized edge geometry, not an exact decomposition of the prediction score.

## 4. Experimental protocol

### 4.1 Datasets

Separate roles explicitly:

- synthetic nonlinear manifolds: moons, circles, spirals
- controlled overlap dataset
- real-data: breast cancer, wine, digits

### 4.2 Baselines

Prediction/selective-risk baselines:

- distance-weighted kNN
- logistic regression
- RBF SVM
- random forest

Use frozen nested tuning where already established.

### 4.3 Explanation-faithfulness protocol

For eligible queries with nontrivial representative paths:

- delete path-selected training examples
- refit MCGC
- measure change in original winner-runner margin
- matched control: equal-size, label- and distance-matched off-path nodes
- proximity control: equal-size nearest off-path nodes

Primary quantity:

\[
\Delta_{\mathrm{path}}-\Delta_{\mathrm{control}}.
\]

Aggregate at dataset-by-seed level to avoid treating nested query observations as independent experimental units.

### 4.4 Explanation-stability protocol

Perturb each selected query by fixed fractions of training-feature standard deviation. Retain perturbations whose predicted class is unchanged.

Compare:

- same query under perturbation
- different query with the same predicted class

Metrics:

- support-set Jaccard
- representative-path Jaccard
- prediction preservation
- path start / terminal stability where useful

Primary contrast:

\[
J_{\mathrm{within}}-J_{\mathrm{same\ class\ control}}.
\]

### 4.5 Feature-attribution validation

Report:

- split-to-split Spearman rank correlation
- top-5 and top-10 Jaccard overlap
- mean-versus-median reference sensitivity

The feature-ranking claim is robustness of broad ranking structure, not fixed exact rank order.

### 4.6 Selective prediction

Primary metric: E-AURC, lower is better.

Use this section as a secondary analysis. Avoid making the paper depend on universal performance superiority.

## 5. Results

### 5.1 MCGC explanations expose the fitted decision structure

Use Figure 1.

Key result narrative:

- nearest neighbour may belong to runner-up
- majority of top local supports may also favour runner-up
- class-conditioned field and path can still favour the winner
- exact decomposition exposes which mechanism actually carries the score margin

Important wording: the representative path summarizes the fitted evidence landscape. Do not imply that it is the only stochastic path used by the classifier.

### 5.2 Real-data multiclass explanation

Use Figure 2: digits 3 vs 8.

Explain:

- automatically selected representative eligible query
- local training examples include both winner and runner-up evidence
- winner trail ascends toward stronger class-3 evidence
- runner-up path can terminate earlier
- exact score decomposition shows the actual source of the decision

Emphasize that the path need not be the dominant score component.

### 5.3 Native feature-level explanation

Use Figure 3: breast-cancer feature attribution.

Local result:

- signed feature interventions expose which observed values support winner vs runner-up

Path-geometric result:

- original input features can be assigned exact shares of standardized edge geometry

Global result:

- mean absolute margin effect provides a model-native ranking across test queries
- split-to-split rank stability is moderate-to-strong, not perfect
- mean-vs-median reference sensitivity is low

Current validation values:

- pairwise outer-split Spearman mean: 0.7316
- median: 0.7455
- minimum: 0.5288
- top-5 Jaccard mean: 0.6190
- top-10 Jaccard mean: 0.4928
- aggregate mean-vs-median Spearman: 0.9560
- mean-vs-median top-10 Jaccard: 0.8182

### 5.4 Representative-path training examples are functionally relevant

Use Figure 4: interventional faithfulness.

Primary frozen result:

- pooled eligible queries: path-vs-matched mean margin-drop advantage approximately +0.0926
- path-vs-nearest mean approximately +0.3593
- dataset-by-seed units: 20/20 positive for both aggregated contrasts

Interpretation:

Path-selected training examples are functionally relevant to the fitted decision under delete-and-refit intervention.

Do not describe this as causal influence in the data-generating process.

### 5.5 Explanations remain query-specific under local perturbation

Use Figure 5: stability and scope.

Primary frozen result:

- support stability contrast positive in 20/20 dataset-by-seed units
- path stability contrast positive in 17/17 evaluable units
- pooled support advantage approximately +0.868
- pooled path advantage approximately +0.853
- prediction preservation is high across datasets

Eligibility:

- nontrivial representative paths occur for approximately 23-28% of test queries
- eligible queries tend to have lower margins and lower accuracy than trivial-path queries

Interpretation:

Nontrivial paths appear preferentially in harder local decision regions rather than being decorative explanations attached to every prediction.

### 5.6 Selective-risk performance is geometry-dependent

Use Figure 6 only if page budget permits. Otherwise move most of this section to supplement.

Main finding:

- MCGC local graph-field margin outperforms tuned kNN and tuned random forest on frozen nonlinear-manifold E-AURC
- tuned RBF-SVM is competitive and removes a universal nonlinear-baseline superiority claim
- real-data confirmation rejects universal transfer of selective-risk superiority
- tuning MCGC does not overturn this boundary

This negative-transfer result strengthens the XAI-first framing.

## 6. Discussion

### 6.1 What MCGC explains

MCGC exposes four complementary levels:

1. score mechanism
2. training examples
3. graph-path geometry
4. original input features

These are connected but not interchangeable.

### 6.2 Why path explanations matter even when trail score is small

A representative path can remain informative as a summary of fitted class-conditioned geometry even when the trail component contributes little or negatively to the final score. Score importance and structural interpretability are different quantities.

### 6.3 Scope of path eligibility

Nontrivial paths are conditional explanations. For easy queries, the start node may already lie in the class basin. For harder queries, a nontrivial ascent path becomes available and informative.

### 6.4 Feature attribution limitations

Paired feature interventions can alter neighbourhoods, affinities, gates, and paths. Therefore they measure a model-level intervention effect, not a direct additive contribution.

### 6.5 Predictive-performance limitation

MCGC should not be presented as a universal replacement for tuned nonlinear classifiers. Its strongest empirical contribution is explanation structure and validation.

## 7. Conclusion

The conclusion should return to the original problem: a prediction is more useful when its internal support structure can be inspected. MCGC turns class-conditioned graph evidence into a decision that can be traced through local examples, representative paths, exact score components, and feature interventions. The strongest empirical finding is not universal predictive superiority, but that these explanations remain stable under local perturbation and identify training examples whose removal changes the fitted decision more strongly than matched controls.

## Main-paper figure order

### Figure 1 — MCGC explanation anatomy on a nonlinear manifold

Panels:

- class-contrast field
- local graph explanation
- representative evidence ascent
- exact winner-runner decomposition

### Figure 2 — Real-data contrastive explanation on digits

Panels:

- query and decision
- affinity-ranked neighbours
- winner and runner-up representative trails
- evidence trace
- exact score decomposition

### Figure 3 — Native feature attribution

Panels:

- local paired feature effects
- representative-path geometric feature shares
- global decision-feature importance

### Figure 4 — Interventional faithfulness

Panels:

- path vs matched off-path deletion
- path vs nearest off-path deletion

### Figure 5 — Stability and scope

Panels:

- same-query perturbed vs same-class control
- stability vs perturbation magnitude
- eligible vs trivial-path decision margin

### Figure 6 — Selective-risk boundary

Optional main-paper figure. Move to supplement if the target journal or page budget favours a tighter XAI story.

## Supplementary material plan

- full risk-coverage curves
- per-seed selective-risk tables
- v0.4-v0.8 negative reliability/accuracy development results in concise historical table
- extra digits explanation gallery
- one wine multiclass example
- one breast-cancer explanation example
- feature-rank stability matrix
- mean-vs-median reference sensitivity table
- path eligibility distributions
- runtime / complexity analysis

## Immediate writing order

1. Method Section 3.1-3.9
2. Results Sections 5.1-5.5
3. Introduction after Related Work audit
4. Discussion
5. Abstract last

Do not finalize novelty language until the deep literature audit against OPF, graph diffusion, random-walk classifiers, path-based XAI, and training-example attribution is complete.
