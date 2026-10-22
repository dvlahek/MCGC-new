# MCGC v1.11 label-review confirmation

This branch freezes the label-review prioritization line developed after the v1.8/v1.9 label-noise diagnostics.

## Question

Can the intrinsic MCGC graph-evidence inconsistency score prioritize corrupted training labels for human review under a limited inspection budget?

The audit score is computed out of fold:

`A_i = best rival local evidence - observed-label local evidence`

Larger values are more suspicious. Clean labels are used only to synthesize benchmark corruption and evaluate ranking quality. They are never used to rank samples or tune the audit score.

## Frozen confirmation protocol

- Datasets: `digits`, `breast_cancer`, `wine`
- Noise conditions: boundary-targeted 5%, boundary-targeted 10%, symmetric 30%
- Seeds: 0, 1, 2, 3, 4
- Review budgets: 1%, 2%, 5%, 10%
- Outer split: 70% train / 30% untouched clean test per seed
- Primary outcomes: precision at review budget, corrupted-label recall, precision enrichment over random review, review-AUC precision
- Random reference: analytical expected precision equal to the empirical corruption fraction
- No simulated repair is used in the primary confirmation

## Frozen global result

Across the 9 dataset/noise conditions and 5 seeds:

| Review budget | Mean precision | Mean recall | Mean enrichment vs random | Conditions above random in all 5 seeds |
|---:|---:|---:|---:|---:|
| 1% | 0.853 | 0.082 | 8.43x | 7/9 |
| 2% | 0.879 | 0.170 | 8.97x | 8/9 |
| 5% | 0.765 | 0.349 | 7.02x | 9/9 |
| 10% | 0.692 | 0.595 | 5.99x | 9/9 |

The 5% and 10% review budgets are the cleanest confirmatory operating points because every dataset/noise condition exceeded random precision in every seed.

## Reproduction

Run:

```powershell
python -u .\examples\confirm_label_review_multiseed_v111.py `
    --output .\v111_label_review_multiseed_confirmation.json
```

The optional `--with-repair` mode is secondary and substantially slower.

## Frozen artifacts

- `examples/confirm_label_review_multiseed_v111.py`
- `results/v111_label_review_multiseed_confirmation_summary.json`
- `results/v111_label_review_multiseed_confirmation.log`

The compact summary records SHA-256 hashes for the full local JSON and log used to produce the frozen aggregates.

## Scope

This is a data-auditing / human-review prioritization result. It does not claim that every identified corruption is harmful to classifier performance, nor that automatic label correction is uniformly beneficial.
