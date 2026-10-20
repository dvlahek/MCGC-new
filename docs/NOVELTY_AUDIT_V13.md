# MCGC novelty and prior-art audit — v1.3 checkpoint

## Status

Provisional literature audit performed before manuscript-level novelty claims. This document is deliberately conservative. It identifies the closest known families and records what MCGC must **not** claim as novel.

The audit should be expanded before submission with database-level searches in Scopus/Web of Science/Google Scholar and citation chaining from the closest papers.

## Bottom line

MCGC is **not** novel merely because it uses:

- a k-nearest-neighbour graph;
- graph diffusion / label propagation;
- random walks on a data graph;
- class-specific graph evidence;
- graph paths for classification;
- prototype or example-based explanations;
- deletion/retraining as a faithfulness test.

All of those ingredients have substantial prior art.

The most defensible current novelty candidate is the **specific integration** of:

1. contrastive class-conditioned evidence fields on a sample graph;
2. multi-agent chemotactic trajectories used as intrinsic trace objects rather than merely a propagation solver;
3. an exact additive class-score and class-vs-class decomposition from the same fitted mechanism;
4. representative path / supporting-sample explanations tied directly to the classifier's graph field;
5. explicit delete-and-refit validation showing that path-selected training nodes are more decision-relevant than label-and-distance-matched off-path controls and equal-size nearest-neighbour explanations;
6. a graph-field margin reused as an intrinsic selective-risk signal.

This combination appears distinct from the closest families reviewed below, but **no “first” claim is currently justified**.

---

## 1. Optimum-Path Forest is the strongest conceptual prior

### Core prior art

Papa, Falcão and Suzuki introduced supervised Optimum-Path Forest (OPF), where training samples are graph nodes, class prototypes compete through optimum paths, and test samples receive the class of the tree/path that would conquer them.

Key reference:

- J. P. Papa, A. X. Falcão, C. T. N. Suzuki, “Supervised Pattern Classification Based on Optimum-Path Forest,” *International Journal of Imaging Systems and Technology*, 19(2), 120–131, 2009. DOI: `10.1002/ima.20188`.

Important later references include:

- J. P. Papa et al., “Efficient supervised optimum-path forest classification for large datasets,” *Pattern Recognition*, 45(1), 512–520, 2012. DOI: `10.1016/j.patcog.2011.07.013`.
- OPF variants based on k-neighbourhood / k-connectivity and later surveys in *Optimum-Path Forest: Theory, Algorithms, and Applications*.

### Overlap with MCGC

Substantial overlap:

- graph over training samples;
- non-parametric class geometry;
- class assignment related to graph paths;
- ability to represent non-convex classes;
- prototype/support-sample interpretation.

### Distinction that may matter

OPF is fundamentally an **optimum-path graph partition / prototype competition** classifier. MCGC instead constructs class-conditioned contrastive evidence fields and then uses multiple chemotactic walkers / representative trails over that field. The current strongest MCGC evidence is not that its paths improve raw accuracy, but that its representative trails provide intrinsic, intervention-validated explanations and that its local field margin can rank selective risk.

### Claim restriction

Do **not** claim:

- first graph classifier based on paths;
- first classifier using class-specific routes through training samples;
- first method exploiting arbitrary-shape class connectivity.

Any manuscript must discuss OPF prominently, not bury it among generic graph-learning references.

---

## 2. Graph label propagation / diffusion is established prior art

Two foundational lines are directly relevant:

- X. Zhu, Z. Ghahramani, J. D. Lafferty, “Semi-Supervised Learning Using Gaussian Fields and Harmonic Functions,” ICML 2003.
- D. Zhou, O. Bousquet, T. N. Lal, J. Weston, B. Schölkopf, “Learning with Local and Global Consistency,” NeurIPS 2003/2004.

These methods explicitly use weighted data graphs and smooth/propagate label information over local and global manifold structure. Random-walk interpretations of graph label propagation are also longstanding.

More recent work continues to use random walk with restart for label propagation, including variable-restart formulations.

### Overlap with MCGC

- weighted sample graph;
- diffusion of class evidence;
- manifold smoothness assumption;
- random-walk interpretation.

### Distinction

MCGC's diffusion is an ingredient, not the novelty. The candidate contribution is what is built **on top of** the field: class-contrastive trajectory scoring, intrinsic trails, exact score decomposition, intervention-tested training-node explanations, and selective-risk use of the field margin.

### Claim restriction

Do not describe graph diffusion, label propagation, or random-walk propagation themselves as new.

---

## 3. Random-walk classifiers already exist

Prior work has explicitly proposed graph classification frameworks using random walks and label propagation. A representative example is the 2012 “Random Walk Classifier Framework on Graph,” which derives lazy and single-step random-walk classifiers from graph label propagation.

Therefore, “walker-based classifier” or “random-walk graph classifier” is not sufficient novelty language.

The biological/chemotactic metaphor is useful for formulation and intuition, but it cannot carry the novelty claim by itself.

---

## 4. Training-example explanations and influence are established

Koh and Liang showed how influence functions can trace black-box predictions back to influential training examples:

- P. W. Koh, P. Liang, “Understanding Black-box Predictions via Influence Functions,” ICML 2017, PMLR 70:1885–1894.

Later work such as RelatIF explicitly focuses on selecting training examples that serve as useful explanations:

- E. Barshan, M.-E. Brunet, G. K. Dziugaite, “RelatIF: Identifying Explanatory Training Samples via Relative Influence,” AISTATS 2020, PMLR 108:1899–1909.

Example-based explanations also have a substantial HCI/XAI literature.

### Consequence for MCGC

The fact that MCGC returns supporting training samples is **not** novel by itself.

Potential distinction:

- the support is generated intrinsically by the same class-conditioned graph field / trail mechanism rather than by a post-hoc influence approximation;
- the explanation includes a path structure rather than an unordered set of influential samples;
- v0.9 validates path-selected nodes by delete-and-refit against carefully matched off-path and proximity-only controls.

### Claim restriction

Do not claim “first training-example explanation” or “first influential-sample XAI method.”

---

## 5. Faithfulness-by-intervention is also established

Modern XAI evaluation routinely checks whether masking/removing explanation-selected components changes model predictions. Graph-XAI work likewise evaluates explanation faithfulness by retaining or perturbing selected subgraphs/features.

Deletion diagnostics in classical statistics and influence analysis also use refitting after removal to measure functional importance.

### What is distinctive in v0.9

The strongest MCGC-specific empirical result is therefore not the generic idea of deletion, but the **specific controlled comparison**:

- delete MCGC path-selected training nodes;
- refit the same classifier;
- compare original winner-vs-runner margin drop against equal-size label-and-distance-matched off-path controls;
- separately compare against equal-size nearest-neighbour off-path controls;
- aggregate both per query and at the more conservative dataset×seed level.

The frozen v0.9 result found positive mean effects in all 20 dataset×seed units for both contrasts.

This supports a strong faithfulness statement, but it should not be described as a new general definition of faithfulness.

---

## 6. Graph-XAI literature requires careful terminology

Much graph-XAI literature concerns **classification of graphs or nodes by GNNs**, where an explanation is a subgraph, edge set, feature mask or counterfactual graph. MCGC is different: it classifies vector samples by constructing a graph **over samples**, then explains a query through training-instance paths in that sample graph.

This distinction must be explicit because phrases such as “graph classifier explanation” are otherwise ambiguous.

MCGC should be described as a **sample-graph classifier** or **graph-based classifier over data instances**, not confused with a GNN that classifies graph-structured objects.

---

## 7. Current novelty wording

### Safe working wording

> We introduce a class-conditioned multi-agent graph classifier in which contrastive evidence fields, trajectory traces, exact score decomposition and selective-risk signals arise from the same fitted sample-graph mechanism. Its representative paths provide intrinsic example-based explanations that are validated by controlled delete-and-refit interventions.

### Stronger wording that still needs deeper audit

> The contribution is not graph diffusion or path classification in isolation, but the integration of class-conditioned evidence fields with multi-agent trajectory evidence and intrinsically faithful path explanations in a single sample-graph classifier.

### Avoid for now

- “first scent-based graph classifier”;
- “first path-explainable classifier”;
- “first intrinsically explainable graph classifier”;
- “first faithful training-example explanation”;
- “state of the art”;
- “universally more interpretable”;
- “causal explanation” without qualification.

---

## 8. Manuscript positioning suggested by current evidence

The paper should not be organized as an accuracy paper whose main novelty is a dog/scent metaphor.

A stronger structure is:

1. **Method:** class-conditioned contrastive graph field + multi-agent trail mechanism.
2. **Negative ablation evidence:** walkers do not provide a robust universal raw-accuracy gain.
3. **Intrinsic XAI contribution:** exact score/contrast decomposition plus representative training-instance paths.
4. **Faithfulness result:** v0.9 delete-and-refit matched-control confirmation.
5. **Reliability contribution:** local graph-field margin as intrinsic selective-risk signal.
6. **Performance boundary:** v1.2 beats tuned kNN and tuned RF on the frozen nonlinear-manifold family but is tied/behind tuned RBF-SVM in aggregate.
7. **Real-data transfer:** v1.3, frozen before results.

This framing turns the negative accuracy checkpoints into methodological evidence that the paper separates predictive performance, explanation faithfulness and uncertainty rather than conflating them.

---

## 9. Next literature tasks before submission

Required deeper searches:

- full citation chaining around supervised OPF and OPF-kNN/k-connectivity;
- path-based nearest-neighbour / geodesic classifiers;
- density-ascent / Quick Shift / graph mean-shift classifiers;
- class-conditioned personalized PageRank and random-walk classifiers;
- prototype/path explanation methods for non-neural classifiers;
- deletion-based example faithfulness and representer/influence-point methods;
- intrinsic selective classification / confidence from graph margins;
- chemotaxis/swarm-inspired supervised classifiers.

Until this is completed, novelty claims should remain integration-focused and qualified.
