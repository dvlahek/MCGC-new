# MCGC v0.9 — frozen XAI faithfulness protocol

## Goal

After the negative v0.8 combined-reliability confirmation, v0.9 shifts the primary research question from uncertainty improvement to **intrinsic explanation faithfulness**.

The classifier formulation is not retuned in v0.9. The goal is to test whether the explanations exposed by the existing MCGC XAI layer are faithful, stable, and contrastive in ways that can be measured directly.

## Frozen hypotheses

### H1 — Exact decomposition fidelity

For every explained query and class, the sum of the reported score components must reconstruct the model decision score to numerical tolerance. This is a hard correctness requirement, not a performance metric.

### H2 — Representative-path reproducibility

For a fixed fitted model and query, the deterministic representative path must be identical across repeated explanation calls.

### H3 — Contrastive consistency

For the predicted class `c*` and runner-up class `r`, the reported contrastive component differences must sum exactly to the reported decision margin:

`margin(c*, r) = score(c*) - score(r)`

### H4 — Supporting-sample faithfulness

If the explanation identifies the top supporting training samples for the predicted class, suppressing those supports from the query-local support calculation should reduce the predicted-class margin more than suppressing matched control samples of the same cardinality.

This is the primary empirical v0.9 hypothesis.

### H5 — Small-perturbation stability

For queries whose prediction is unchanged under a small standardized input perturbation, the deterministic explanation should remain substantially similar. Similarity will be reported separately for:

- top-k support overlap;
- representative-path node overlap;
- change in class-score decomposition.

No threshold for a positive claim will be selected after seeing the results; the distributions will be reported directly.

## Primary empirical contrast

The primary paired quantity is:

`drop_in_prediction_margin(top_support_suppression) - drop_in_prediction_margin(matched_control_suppression)`

Positive values support the claim that the samples identified as supports are functionally more important to the explained decision than matched controls.

## Controls

Matched controls must:

- use the same number of training samples as the support set;
- come from the same predicted class where possible;
- be excluded from the reported top-support set;
- be sampled reproducibly from a fixed random seed;
- not trigger classifier retraining.

The intervention operates only on the query-local support contribution used by the already-fitted model/explanation calculation. It must not change graph training, diffusion fields, labels, or fitted parameters.

## Datasets

Use the existing v0.8 confirmation datasets without retuning:

- `moons_hard`
- `circles_hard`
- `spirals_hard`
- `overlap_hd`
- `breast_cancer`
- `wine`
- `digits_0_4`

Use the same 8 dataset seeds where applicable.

## Reporting

Report per dataset and pooled:

- number of evaluable queries;
- mean and median primary paired margin-drop contrast;
- wins / ties / losses;
- bootstrap 95% CI for the mean paired contrast;
- support-overlap stability under small perturbation;
- representative-path overlap stability;
- exact decomposition failure count;
- exact contrastive-margin failure count.

No feature search, no XAI metric selection after test inspection, and no dataset-specific thresholds are allowed.

## Claim discipline

A successful v0.9 result would support a statement such as:

> MCGC exposes intrinsically traceable graph explanations whose reported supporting samples and class-contrastive score decomposition are measurably faithful to the fitted decision mechanism.

It would **not** by itself justify claims that MCGC is fully interpretable, causally explanatory in a general sense, or superior to post-hoc XAI methods.
