# MCGC v0.6 intrinsic reliability results

v0.6 tested whether MCGC's existing internal diagnostics can identify likely classification errors without changing the class prediction, fitting a secondary risk model, or tuning an abstention threshold on the test set.

## Main result

The strongest reliability signal is the **local contrastive-field margin**. Walker-derived diagnostics contain some reliability information, especially trail-score disagreement, but they do not outperform the local margin on the datasets with a meaningful number of errors.

| dataset | samples | errors | local-margin AUC | final-margin AUC | trail-disagreement AUC |
|---|---:|---:|---:|---:|---:|
| circles | 1800 | 6 | 0.9929 | 0.9931 | **0.9936** |
| moons | 1800 | 42 | **0.9197** | 0.9153 | 0.8847 |
| spirals | 1800 | 505 | **0.7375** | 0.7257 | 0.6083 |

The circles result is not strong evidence for walker superiority because only six pooled errors occurred. On moons and spirals, where error counts are informative, local margin is clearly stronger than trail disagreement.

## Selective classification

Local-margin risk produces useful risk--coverage behaviour without any fitted risk model.

### Moons

| retained coverage | retained accuracy |
|---:|---:|
| 100% | 0.9767 |
| 90% | 0.9932 |
| 80% | 0.9979 |
| 70% | 0.9976 |
| 50% | 0.9978 |

Trail disagreement is also informative on moons, reaching 0.9926 accuracy at 90% coverage and 0.9978 at 50%, but it remains weaker than local margin in error-ranking AUC.

### Spirals

| retained coverage | local-margin accuracy | trail-disagreement accuracy |
|---:|---:|---:|
| 100% | 0.7194 | 0.7194 |
| 90% | 0.7475 | 0.7142 |
| 80% | 0.7722 | 0.7229 |
| 70% | 0.8040 | 0.7389 |
| 50% | 0.8711 | 0.7856 |

The local graph-field margin therefore provides a practically meaningful intrinsic abstention signal even on the difficult spiral geometry.

## Other walker diagnostics

- `trail_disagreement_risk` is the most promising walker-derived reliability signal: pooled AUC 0.8847 on moons and 0.6083 on spirals.
- trail coherence is only modestly informative.
- lost-trail fraction is essentially uninformative in the current implementation (approximately chance-level AUC).
- the preregistered orientation of `trail_advantage_risk` performs below chance. This is retained as a negative result; its sign will not be flipped post hoc and relabelled as a confirmed finding.

## Scientific conclusion

v0.6 supports a narrower and more defensible statement than the original performance hypothesis:

> MCGC exposes intrinsic graph-based reliability information that can support selective classification, while trajectory disagreement can provide an additional descriptive uncertainty signal.

It does **not** yet support the stronger claim that walker diagnostics improve reliability beyond the local graph-field margin.

## Next confirmation question

The next experiment should test incremental walker information under a clean train/calibration/test separation. A simple calibrated risk model will compare:

1. local margin only,
2. local + final margin,
3. local margin + preregistered walker disagreement,
4. a small preregistered combined intrinsic feature set.

The risk model must be fitted only on a calibration split and evaluated once on an untouched test split. This will directly test whether walker diagnostics contain complementary information after controlling for the strong local-margin baseline.
