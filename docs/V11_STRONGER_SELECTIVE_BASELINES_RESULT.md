# MCGC v1.1 — stronger selective-baseline confirmation result

## Status

The frozen v1.1 stronger-baseline confirmation is **positive on the predefined nonlinear-manifold family**.

The MCGC classifier, local-margin risk signal, datasets, seeds, train/test split and metrics were inherited unchanged from v1.0. The only new scientific comparisons were against fixed RBF-SVM and Random-Forest confidence baselines. No MCGC retuning or hyperparameter search was performed after inspecting the result.

Primary family:

- `moons_hard`
- `circles_hard`
- `spirals_hard`

Primary metric: **E-AURC**, lower is better.

Paired advantage is defined as:

`E-AURC(baseline) - E-AURC(MCGC)`

so positive values favour MCGC.

## Primary pooled result: RBF-SVM

Across the 36 frozen manifold runs:

- mean E-AURC advantage: `+0.014414490689435011`
- median: `+0.004798513586381163`
- wins / ties / losses: `25 / 0 / 11`
- two-sided sign-test p: `0.028816719655878842`

Ordinary AURC comparison:

- mean advantage: `+0.016056685079267947`
- median: `+0.004937864065564669`
- wins / ties / losses: `22 / 0 / 14`
- two-sided sign-test p: `0.2429849540349096`

Interpretation: the preregistered primary E-AURC contrast is positive against the fixed RBF-SVM baseline, but ordinary AURC does not show the same level of paired consistency. The claim should therefore remain specifically about **excess selective-risk ranking**, not uniform dominance in every selective metric.

## Primary pooled result: Random Forest

Across the same 36 frozen manifold runs:

- mean E-AURC advantage: `+0.010919444296094165`
- median: `+0.009838041947084559`
- wins / ties / losses: `31 / 0 / 5`
- two-sided sign-test p: `1.2913485988974571e-05`

Ordinary AURC comparison:

- mean advantage: `+0.012733213240176582`
- median: `+0.010936196421952437`
- wins / ties / losses: `29 / 0 / 7`
- two-sided sign-test p: `0.00031255115754902363`

This is the strongest new v1.1 selective-risk result.

## Per-dataset E-AURC results

### `moons_hard`

Mean E-AURC:

- MCGC: `0.01880861195218542`
- RBF-SVM: `0.024504335876055675`
- Random Forest: `0.027094638466386132`

Paired MCGC advantages:

- vs RBF-SVM: mean `+0.005695723923870248`, W/L `11/1`, p `0.00634765625`
- vs Random Forest: mean `+0.008286026514200706`, W/L `12/0`, p `0.00048828125`

Mean raw accuracy:

- MCGC: `0.9027777777777778`
- RBF-SVM: `0.9055555555555556`
- Random Forest: `0.8952380952380953`

Thus the selective-risk result is not reducible to raw-accuracy dominance.

### `circles_hard`

Mean E-AURC:

- MCGC: `0.017257383250261608`
- RBF-SVM: `0.014875667959775187`
- Random Forest: `0.032229529455244106`

Paired MCGC advantages:

- vs RBF-SVM: mean `-0.00238171529048642`, W/L `3/9`, p `0.14599609375`
- vs Random Forest: mean `+0.0149721462049825`, W/L `11/1`, p `0.00634765625`

Mean raw accuracy:

- MCGC: `0.8972222222222223`
- RBF-SVM: `0.9007936507936508`
- Random Forest: `0.8888888888888888`

RBF-SVM is better than MCGC on circles, so the pooled RBF-SVM result must not be described as per-dataset universal superiority.

### `spirals_hard`

Mean E-AURC:

- MCGC: `0.19206331971499538`
- RBF-SVM: `0.23199278314991664`
- Random Forest: `0.20156347988409473`

Paired MCGC advantages:

- vs RBF-SVM: mean `+0.0399294634349212`, W/L `11/1`, p `0.00634765625`
- vs Random Forest: mean `+0.00950016016909929`, W/L `8/4`, p `0.3876953125`

Mean raw accuracy:

- MCGC: `0.6226190476190476`
- RBF-SVM: `0.611904761904762`
- Random Forest: `0.6162698412698412`

The spiral result is particularly relevant to the intended nonlinear-manifold setting because MCGC has both lower E-AURC than both stronger nonlinear baselines and slightly higher mean raw accuracy in this frozen configuration.

## Continuity with v1.0

The v1.0 comparisons reproduce exactly:

### distance-weighted kNN

E-AURC:

- mean advantage: `+0.009157069986134773`
- median: `+0.006353876329219648`
- W/T/L: `29/0/7`
- p: `0.00031255115754902363`

AURC:

- mean advantage: `+0.008125534126382374`
- W/T/L: `25/0/11`

### logistic regression

E-AURC:

- mean advantage: `+0.13541330566873053`
- median: `+0.02758153624699908`
- W/T/L: `32/0/4`
- p: `1.941574737429619e-06`

The logistic contrast remains contextual rather than the preferred superiority claim because a linear classifier is structurally mismatched to parts of the manifold family.

## Transfer / control datasets

The stronger-baseline advantage is **not universal outside the frozen manifold family**.

### `overlap_hd`

Mean E-AURC:

- MCGC: `0.07691793132915233`
- RBF-SVM: `0.07544568196170551`
- Random Forest: `0.07223722388763189`

Paired mean advantages are slightly negative for MCGC:

- vs RBF-SVM: `-0.001472249367446823`
- vs Random Forest: `-0.004680707441520455`

### `breast_cancer`

Mean E-AURC:

- MCGC: `0.00590212975350349`
- RBF-SVM: `0.0024028325420410238`
- Random Forest: `0.004812533748984942`
- logistic regression: `0.0021220332281691704`

MCGC loses to RBF-SVM in all 12 breast-cancer runs and to Random Forest in 10/12 runs. This is an important boundary condition and supports a geometry-specific rather than universal claim.

## Scientific conclusion

v1.1 strengthens the v1.0 selective-manifold result.

The frozen MCGC local graph-field margin is not merely better than proximity-based kNN confidence. Across the predefined nonlinear-manifold family it also achieves lower E-AURC than:

- fixed RBF-SVM margin confidence in `25/36` paired runs;
- fixed Random-Forest probability confidence in `31/36` paired runs.

The strongest supported selective-risk statement is now:

> On the frozen nonlinear-manifold benchmark, MCGC local graph-field margin provides better excess selective-risk ranking than distance-weighted kNN confidence, fixed RBF-SVM margin confidence, and fixed Random-Forest probability confidence in the aggregate, while the effect is not universal across every dataset or conventional tabular control.

Combined with v0.9, the current strongest project story has two distinct contributions:

1. **intrinsic XAI faithfulness:** representative-path nodes are more functionally decision-relevant than matched off-path and proximity-only controls;
2. **geometry-specific selective confidence:** local graph-field margin provides stronger error ranking on the frozen nonlinear-manifold family than several conventional confidence baselines.

## Claim limits

Do not claim:

- universal raw-accuracy superiority;
- universal selective-risk superiority across arbitrary tabular data;
- superiority to all calibrated or modern selective-classification methods;
- SOTA selective classification;
- universal interpretability.

RBF-SVM wins on `circles_hard`, and both stronger nonlinear baselines outperform MCGC on the conventional `breast_cancer` control. These negative sub-results are part of the frozen result and must remain visible.

## Reproducibility note

The full local benchmark completed and wrote `v11_stronger_selective_baselines.json`. The console log also contains a scikit-learn deprecation warning related to the `SVC` `probability` parameter. The frozen benchmark implementation itself uses `probability=False` and derives confidence directly from the decision-function margin, so probability calibration is not part of the v1.1 SVM comparison.
