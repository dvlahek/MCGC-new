# CGEC v3.3 matched-tuning audit protocol

Status: frozen before inspecting any tuned-CGEC result.

Purpose: secondary matched-tuning analysis on the pre-specified eight-dataset real-data evaluation panel. The primary manuscript result remains the fixed CGEC configuration (K=12, alpha=0.85, n_D=40). This audit asks how CGEC performs when given the same inner-CV model-selection opportunity as the tuned predictive baselines.

## Dataset panel

The exact v3.2c panel is reused without modification:

- credit_approval_australia
- credit_approval_germany
- dermatology
- haberman
- heart_disease_cleveland
- ionosphere
- penguins
- segmentation

Data loading, complete-case handling, pinned PMLB commit, and dataset fingerprints are inherited from `benchmark_real_data_confirmation_v32c.py`.

## Outer evaluation

- 12 seeds: 0..11
- stratified 70/30 outer train-test split
- `random_state = 5000 + seed`
- outer test data are never used for hyperparameter selection

## Inner tuning

Tuned CGEC uses stratified 3-fold CV on the outer-training partition only, with `random_state = 6000 + seed`.

Frozen grid:

- K in {8, 12, 17, 25}
- alpha in {0.50, 0.75, 0.85}
- n_D in {20, 40, 80}

Total: 36 candidate configurations per outer dataset-seed unit.

Selection rule, matching the predictive baseline protocol:

1. lowest inner out-of-fold E-AURC;
2. higher inner out-of-fold accuracy as tie-breaker;
3. deterministic lexical candidate key as final tie-breaker.

After selection, tuned CGEC is refitted on the complete outer-training partition and evaluated once on the untouched outer-test partition.

## Reporting

The fixed CGEC result remains primary and pre-specified. Tuned CGEC is reported only as a secondary matched-tuning analysis. Dataset means across the 12 seeds are the primary real-data comparison units. The script also reports parameter-selection frequencies and tuned-vs-fixed changes.

No grid, dataset, seed, split, selection metric, or tie-break rule is to be changed after tuned results are inspected.
