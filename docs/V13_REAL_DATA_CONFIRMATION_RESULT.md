# MCGC v1.3 frozen real-data confirmation result

## Status

Completed frozen transfer test on the predeclared new real-data family.

Primary new real datasets:

- `digits` (10 classes)
- `wine` (3 classes)

Previously seen control:

- `breast_cancer`

MCGC remained frozen and untuned. Competing kNN, logistic, RBF-SVM and Random-Forest classifiers used the same nested 3-fold inner-CV E-AURC tuning rule frozen in v1.2. Outer evaluation used the untouched 30% test split over seeds `0..11`.

Paired advantage is defined as baseline E-AURC minus MCGC E-AURC, so positive values favour MCGC.

## Primary pooled new-real-data result

Across the 24 `digits` + `wine` runs:

| tuned baseline | mean E-AURC advantage | median | W/T/L | sign-test p |
|---|---:|---:|---:|---:|
| kNN | `+0.02633347` | `+0.01559138` | `13/0/11` | `0.8388197` |
| logistic | `-0.00148711` | `-0.00189458` | `3/1/20` | `0.0004883` |
| RBF-SVM | `-0.00058878` | `-0.00148237` | `3/1/20` | `0.0004883` |
| Random Forest | `-0.00229231` | `-0.00193582` | `1/0/23` | `2.98e-06` |

The positive pooled kNN mean is not a stable pooled win because the two datasets point in opposite directions and the W/L sign test is `13/11` with `p=0.839`.

The other three tuned baselines are better than MCGC on the pooled new-real-data family, especially Random Forest.

## Dataset-level result

### digits

Mean accuracy / E-AURC:

- MCGC: accuracy `0.95046`, E-AURC `0.003794`
- tuned kNN: `0.98025`, `0.001382`
- tuned logistic: `0.96651`, `0.002566`
- tuned RBF-SVM: `0.94336`, `0.003888`
- tuned Random Forest: `0.97485`, `0.000604`

Paired E-AURC advantage, baseline minus MCGC:

- tuned kNN: `-0.002412`, W/T/L `1/0/11`, p `0.00635`
- tuned logistic: `-0.001228`, W/T/L `3/0/9`
- tuned RBF-SVM: mean `+0.000094`, but W/T/L `2/0/10`; median is negative
- tuned RF: `-0.003190`, W/T/L `0/0/12`, p `0.000488`

Thus `digits` is a clear transfer failure for the broad selective-risk superiority hypothesis. RF is particularly stronger, and tuned kNN also clearly beats MCGC.

### wine

Mean accuracy / E-AURC:

- MCGC: accuracy `0.95370`, E-AURC `0.002111`
- tuned kNN: `0.74383`, `0.057190`
- tuned logistic: `0.98148`, `0.000365`
- tuned RBF-SVM: `0.98457`, `0.000840`
- tuned Random Forest: `0.97994`, `0.000716`

Paired E-AURC advantage:

- tuned kNN: `+0.055079`, W/T/L `12/0/0`, p `0.000488`
- tuned logistic: `-0.001746`, W/T/L `0/1/11`
- tuned RBF-SVM: `-0.001271`, W/T/L `1/1/10`
- tuned RF: `-0.001395`, W/T/L `1/0/11`

The huge kNN advantage for MCGC on wine is real for that baseline but cannot be generalized to the stronger tuned models, all of which outperform MCGC.

### breast-cancer control

The previously seen control reproduces the established boundary:

- MCGC E-AURC `0.005902`
- tuned kNN `0.020937`
- tuned logistic `0.002234`
- tuned RBF-SVM `0.002614`
- tuned RF `0.004775`

MCGC beats tuned kNN strongly but remains worse than tuned logistic, SVM and RF.

## Scientific interpretation

**v1.3 is a negative general-transfer result for selective-risk superiority.**

The v1.2 manifold-specific result remains valid because v1.3 tested a different, predeclared real-data transfer question. However, the selective-risk claim must remain tied to the frozen nonlinear-manifold family. It must not be promoted to broad real-data superiority.

The `digits` result is especially useful in setting the claim boundary: sample-graph local margin does not automatically dominate conventional tuned confidence on high-quality multiclass real data. The `wine` result shows that MCGC can still strongly outperform tuned kNN confidence on some real datasets, but stronger tuned logistic/SVM/RF confidence remains superior there.

Therefore the main paper story should not be built around universal selective-classification performance. The most robust primary contribution remains **intrinsic, interventionally validated path faithfulness**, with selective-risk results presented as a geometry-dependent secondary property.

## Claim boundary after v1.3

Supported:

> MCGC representative-path explanations are interventionally faithful for eligible queries, and its frozen local graph-field margin shows improved selective-risk ranking over tuned kNN and tuned Random-Forest confidence on the predefined nonlinear-manifold benchmark.

Also supported as a boundary:

> That selective-risk advantage does not transfer generally to the predeclared `digits`/`wine` real-data family; tuned logistic, RBF-SVM and Random-Forest confidence are stronger overall on those real-data tasks.

Not supported:

- universal real-data selective-risk superiority
- universal raw-accuracy superiority
- superiority to tuned SVM/RF/logistic across arbitrary datasets
- SOTA claims

## Reproducibility note

The benchmark completed and wrote the full JSON result. The console emitted repeated scikit-learn deprecation warnings mentioning the historical `SVC(probability=...)` parameter interface. The benchmark itself uses decision-function margin confidence for SVM, as frozen in v1.2/v1.3; probability calibration is not the primary SVM confidence definition.
