# Methods draft v1

## 3. Method

### 3.1 Problem formulation and sample graph

Let \(\mathcal D=\{(x_i,y_i)\}_{i=1}^n\) denote the training set, with \(x_i\in\mathbb R^d\) and class label \(y_i\in\{1,\ldots,C\}\). MCGC represents the training set as a weighted sample graph and evaluates a query through class-conditioned evidence defined on that graph. The objective is not only to assign a class label, but to retain an explicit path from local query evidence to the final class contrast.

Features are standardized using statistics estimated from the training split. A \(k\)-nearest-neighbour graph is then constructed in the standardized space. For connected samples \(i\) and \(j\), the edge affinity is defined by a locally scaled heat kernel

\[
w_{ij}=\exp\left(-\frac{\lVert z_i-z_j\rVert_2^2}{\sigma_i\sigma_j}\right),
\]

where \(z_i\) and \(z_j\) are standardized feature vectors and \(\sigma_i\) and \(\sigma_j\) are local distance scales. The resulting weight matrix is denoted by \(W\). With \(D\) the diagonal degree matrix, the row-normalized propagation matrix is

\[
P=D^{-1}W.
\]

The graph therefore provides two quantities used later by the classifier: local sample geometry through \(w_{ij}\), and a discrete support on which class evidence can be propagated.

### 3.2 Class-conditioned evidence fields

For each class \(c\), MCGC defines an initial label-evidence vector \(b^{(c)}\) over the training nodes. Evidence is propagated by the fixed iterative diffusion

\[
s_{t+1}^{(c)}=(1-\alpha)b^{(c)}+\alpha P s_t^{(c)},
\]

where \(\alpha\in(0,1)\) controls the balance between the observed class labels and graph propagation. After the fixed number of diffusion iterations, the resulting field is denoted by \(S^{(c)}\).

Absolute class evidence is not sufficient for a contrastive decision because several classes can receive large propagated values in the same region. We therefore define the class-conditioned contrastive field

\[
D_i^{(c)}=S_i^{(c)}-\max_{r\neq c}S_i^{(r)}.
\]

A positive value indicates that class \(c\) is locally stronger than its best alternative at node \(i\), while a negative value indicates that another class dominates there. This contrastive field is the central quantity shared by prediction and explanation.

### 3.3 Query-local evidence

For a query \(q\), let \(\mathcal N(q)\) be the selected start nodes in the training graph. Their query affinities are converted to normalized weights \(a_j(q)\), with

\[
\sum_{j\in\mathcal N(q)}a_j(q)=1.
\]

The local class evidence is

\[
L_c(q)=\sum_{j\in\mathcal N(q)}a_j(q)D_j^{(c)}.
\]

This term gives the immediate class preference around the query. It plays a dual role. First, it contributes directly to the class score. Second, it determines how strongly the trajectory mechanism is allowed to modify an already decisive local field.

### 3.4 Class-conditioned chemotactic trajectories

The motivating mechanism is analogous to chemotaxis: an agent does not only evaluate the signal at its current location, but also prefers directions in which the signal increases. In MCGC, each walker is conditioned on a candidate class \(c\). At graph node \(i\), a neighbouring node \(j\) receives desirability

\[
H_{ij}^{(c)}=
\beta_1D_j^{(c)}
+\beta_2\left(D_j^{(c)}-D_i^{(c)}\right)
+\beta_3\log(w_{ij}+\varepsilon)
-\beta_4R_{ij},
\]

where \(R_{ij}\) penalizes revisits. The four terms respectively represent destination evidence, evidence ascent, graph plausibility, and path repetition. A temperature-controlled softmax over \(H_{ij}^{(c)}\) defines the stochastic transition distribution.

Each stochastic trail is summarized by quantities that capture its progress through the evidence field. These include positive evidence gain, coherence of the ascent, terminal evidence, graph plausibility, and trail failure. Multiple walkers are evaluated for every candidate class, which provides a distribution of class-conditioned trail scores rather than a single path realization.

The trajectory mechanism is deliberately secondary to the local field. Frozen ablations showed that walkers do not provide a universal raw-accuracy advantage. They are therefore used as a bounded correction to local evidence, not as an unrestricted replacement for it.

### 3.5 Adaptive walker contribution

Let \(m(q)\) denote the local winner-runner margin and \(m_{\mathrm{ref}}\) a fitted reference scale. The trajectory contribution is attenuated by

\[
g(q)=\exp\left[-\left(\frac{m(q)}{m_{\mathrm{ref}}}\right)^p\right].
\]

When the local field is already decisive, \(g(q)\) is small and the walker terms are suppressed. Ambiguous local regions receive a larger trajectory contribution. This design prevents path evidence from dominating a query for which the graph neighbourhood already gives a clear class preference.

### 3.6 Final class score and exact contrastive decomposition

For class \(c\), the implemented score is written as

\[
F_c(q)=F_{c,\mathrm{local}}(q)
+F_{c,\mathrm{trail}}(q)
+F_{c,\mathrm{agreement}}(q)
+F_{c,\mathrm{lost}}(q).
\]

The local component is a weighted form of \(L_c(q)\). The remaining terms summarize the mean walker trail score, agreement among walkers, and the fraction of failed trails, after application of the adaptive gate. The predicted class is

\[
\hat c(q)=\arg\max_c F_c(q).
\]

Let \(c^+\) be the winner and \(c^-\) the runner-up. Their score contrast is

\[
M(q)=F_{c^+}(q)-F_{c^-}(q).
\]

Because every implemented class score is explicitly stored as the sum of the same four components used for prediction, the contrast can be reconstructed as

\[
M(q)=
\Delta F_{\mathrm{local}}
+\Delta F_{\mathrm{trail}}
+\Delta F_{\mathrm{agreement}}
+\Delta F_{\mathrm{lost}},
\]

where \(\Delta F_x=F_{c^+,x}-F_{c^-,x}\). The reconstruction is exact up to floating-point precision. This property is important for interpretation: the score-level explanation is not obtained from a surrogate model and does not approximate the fitted decision.

### 3.7 Local training-example support

MCGC also exposes the query neighbourhood used to construct local evidence. Supporting examples are ranked by query affinity and reported together with their labels and class-conditioned contrastive evidence. These examples provide direct training-data context around the query.

They are not described as causal contributions ranked by effect size. Affinity answers a different question: which observed training samples are geometrically closest and therefore participate most strongly in the local graph evidence. Functional relevance is tested separately by the intervention experiment described below.

### 3.8 Representative evidence paths

The stochastic walkers used by the classifier are aggregated into trail statistics. For interpretation, MCGC additionally constructs a deterministic representative path for each candidate class. This path summarizes the fitted evidence landscape and is not presented as the unique stochastic trajectory that caused the prediction.

The path starts from the query neighbourhood using both query affinity and class-conditioned evidence. From the current node \(i\), the next node is selected greedily using the same evidence, gradient, graph-affinity, and revisit ingredients that define walker desirability. The procedure stops when the class-specific evidence basin is reached or no admissible positive ascent remains.

For class \(c\), the representative path is

\[
P_c(q)=(v_0,v_1,\ldots,v_T),
\]

with evidence trace

\[
D_{v_0}^{(c)},D_{v_1}^{(c)},\ldots,D_{v_T}^{(c)}.
\]

The resulting explanation therefore exposes both the local training examples and a graph-structured summary of how class evidence changes away from the query neighbourhood.

### 3.9 Native feature-level attribution

The graph and path explanations identify relevant training samples, but they do not by themselves quantify the role of the original input variables. We therefore add a model-native feature intervention that measures the effect of removing one observed feature value from the query while leaving the fitted classifier unchanged.

For the original winner \(c^+\) and runner-up \(c^-\), the baseline contrast is

\[
M(q)=F_{c^+}(q)-F_{c^-}(q).
\]

Let \(q^{(-k)}\) denote the same query after replacing feature \(k\) by a reference value estimated from the outer training split. The class pair is kept fixed to the original winner and runner-up. The paired feature-intervention effect is

\[
I_k(q)=M(q)-M\left(q^{(-k)}\right).
\]

A positive \(I_k(q)\) means that the observed value of feature \(k\) supports the original winner relative to the runner-up. A negative value means that the observed feature value shifts the fitted decision toward the runner-up. The magnitude \(|I_k(q)|\) measures the strength of that paired intervention.

These quantities are not additive feature contributions. In general,

\[
M(q)\neq\sum_{k=1}^{d}I_k(q),
\]

because replacing a feature can change the query neighbourhood, graph affinities, local evidence, walker gate, and trajectory statistics. We therefore use the term \emph{feature-intervention effect} rather than additive attribution.

For a global summary, we report

\[
GI_k=\frac{1}{N}\sum_{q=1}^{N}|I_k(q)|,
\]

computed over the test queries of each frozen outer split. The resulting ranking is interpreted as a model-specific sensitivity ranking and not as a causal ordering of the variables.

### 3.10 Exact feature decomposition of path geometry

The representative path can also be traced back to the original standardized features. For any path edge \((i,j)\),

\[
\lVert z_i-z_j\rVert_2^2
=\sum_{k=1}^{d}(z_{ik}-z_{jk})^2.
\]

We define the share of feature \(k\) in that edge distance as

\[
\rho_{ij,k}=
\frac{(z_{ik}-z_{jk})^2}
{\sum_{\ell=1}^{d}(z_{i\ell}-z_{j\ell})^2}.
\]

By construction,

\[
\sum_{k=1}^{d}\rho_{ij,k}=1.
\]

Unlike the paired decision-feature effect, this is an exact decomposition, but only of the standardized geometric separation represented by that graph edge. It must not be interpreted as an exact decomposition of the prediction score.

## 4. Explanation validation

### 4.1 Delete-and-refit path faithfulness

A graph path can appear plausible without being functionally important to the fitted classifier. We therefore test path explanations by intervention on the training set. For each eligible query with a nontrivial representative path, the training nodes selected by that path are removed and MCGC is refitted. We then measure the change in the original winner-runner margin.

Two equal-size off-path controls are used. The primary control matches path nodes by class label and distance to the query. The secondary control uses the nearest available off-path training samples. If \(\Delta_{\mathrm{path}}\) denotes the margin change after deleting path-selected nodes and \(\Delta_{\mathrm{ctrl}}\) the corresponding control change, the primary faithfulness contrast is

\[
A(q)=\Delta_{\mathrm{path}}(q)-\Delta_{\mathrm{ctrl}}(q).
\]

Positive values indicate that path-selected examples are more influential on the fitted decision margin than the matched off-path alternatives. Because several queries are evaluated within the same fitted split, the primary manuscript summary aggregates results at the dataset-by-seed level rather than treating all query-level observations as independent experimental units.

### 4.2 Perturbation stability and query specificity

A useful intrinsic explanation should respond smoothly to small changes in the same query, but it should not collapse into a class-level template shared by unrelated samples. We therefore perturb each selected query by fixed fractions of the training-feature standard deviation. Only perturbations that preserve the predicted class are retained.

For each retained perturbation, support-set similarity and representative-path similarity are measured by Jaccard overlap with the original explanation. A different query with the same predicted class provides the control. The main contrast is

\[
S(q)=J_{\mathrm{within}}(q)-J_{\mathrm{same\ class}}(q).
\]

A positive value means that the explanation retains more identity under a small perturbation of the same query than it shares with another sample assigned to the same class. Prediction-preservation rate is reported separately so that explanation stability is interpreted only within the region in which the classifier itself remains locally unchanged.

### 4.3 Feature-attribution robustness

The global feature ranking is evaluated across frozen outer splits using pairwise Spearman rank correlation and top-\(k\) Jaccard overlap. We additionally repeat the feature intervention using the outer-training median instead of the mean as the replacement reference. This test evaluates sensitivity to the baseline used for feature removal without changing the classifier, dataset split, or original winner-runner pair.

The resulting claim is intentionally limited. We assess stability of the broad model-specific ranking structure. We do not assume that the exact rank of every feature is invariant across resampling, and we do not interpret the ranking as biomedical causality.

## 5. Predictive evaluation and scope

Prediction experiments use the same frozen train-test protocols established during model development. Raw classification accuracy is reported to characterize predictive behaviour, but it is not the primary claim of the paper. Selective prediction is evaluated using excess area under the risk-coverage curve (E-AURC), where lower values indicate better ranking of reliable and unreliable predictions.

Distance-weighted kNN, logistic regression, RBF SVM, and random forest are included as comparison models. For the tuned comparison, hyperparameters are selected only from the outer training split by nested cross-validation. MCGC results are interpreted by dataset family rather than pooled into a universal superiority statement. The nonlinear-manifold experiments quantify the setting in which the local graph-field margin is useful for selective ranking, while the real-data experiments define the observed transfer boundary.

## Manuscript wording constraints

Use these distinctions consistently in the final paper:

- **exact score decomposition** for the winner-runner split into local, trail, agreement, and lost components
- **representative evidence path** for the deterministic path summary
- **affinity-ranked support examples** for local training neighbours
- **paired feature-intervention effect** for \(I_k(q)\)
- **exact path-geometric feature share** for \(\rho_{ij,k}\)
- **functional faithfulness under delete-and-refit** for the training-example intervention
- **query-specific stability** for perturbation-vs-same-class comparison

Avoid describing the representative path as the unique causal trajectory, the feature effects as additive SHAP-like contributions, or the breast-cancer ranking as causal feature importance.
