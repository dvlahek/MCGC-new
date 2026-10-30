# v3.4 Runtime + matched-tuning audit protocol

Status: secondary reviewer-facing audit, frozen before v3.4 timing results are inspected.

This audit preserves the primary pre-specified fixed-CGEC real-data evaluation. It adds a secondary matched-tuning analysis and runtime measurements on exactly the same eight PMLB datasets, 12 seeds, and 70/30 outer splits used by v3.2c.

## CGEC matched tuning

- K in {8, 12, 17, 25}
- alpha in {0.50, 0.75, 0.85}
- n_D in {20, 40, 80}
- 36 candidate configurations
- inner CV: 3-fold stratified on outer-training data only, shuffle=True, random_state=6000+seed
- selection: lowest inner OOF E-AURC; higher inner OOF accuracy as the first tie-break; deterministic lexical candidate key as the final tie-break
- selected configuration is refit on the full outer-training split and evaluated once on the untouched outer test split

The fixed CGEC result K=12, alpha=0.85, n_D=40 remains the primary pre-specified manuscript result. Tuned CGEC is a secondary matched-tuning analysis.

## Runtime environment

The timing script forces these thread counts before NumPy/scikit-learn are imported:

- OMP_NUM_THREADS=1
- MKL_NUM_THREADS=1
- OPENBLAS_NUM_THREADS=1
- NUMEXPR_NUM_THREADS=1

For runtime only, the random-forest implementation is also forced from its benchmark default `n_jobs=-1` to `n_jobs=1`. This changes only parallel execution, not the estimator, grid, random state, selected hyperparameters, predictions, or metrics. The correction was made before timing results were inspected so that all methods are measured under a single-thread execution policy.

Wall-clock time is measured with time.perf_counter(). Hardware/software metadata and thread settings are stored in the output.

## Runtime quantities

For each dataset-seed outer split:

1. Fixed CGEC final-model cost
   - fit time
   - test prediction/evidence time
   - fit+prediction time
   - three independent timing repetitions; report median, with all repetitions retained

2. Tuned CGEC end-to-end cost
   - sum of the 36 inner-CV candidate-evaluation wall times
   - selected-model final refit time
   - outer-test prediction/evidence time
   - end-to-end = tuning + one final refit + one test prediction
   - candidate-level times are checkpointed so interrupted runs can resume without losing completed candidate measurements
   - selected final-model fit and prediction are additionally timed over three repetitions; report medians separately from end-to-end tuning cost

3. Tuned baseline end-to-end cost
   - rerun the same v2.6 inner 3-fold tuning routine for scaled kNN, RBF-SVM, random forest, and scaled Label Spreading on the same outer-training split
   - time the complete tuning routine, including its final refit
   - time outer-test prediction separately
   - end-to-end = tuning routine + test prediction
   - selected hyperparameters must match the deterministic selection stored in the v3.2c checkpoint; mismatches are flagged

4. Baseline selected final-model cost
   - refit the selected baseline configuration from scratch three times
   - time fit and outer-test prediction separately
   - report medians and retain all repetitions

## Interpretation

Runtime is descriptive for the tested implementation and hardware. It is not a scaling-complexity experiment. The dense CGEC implementation still has O(n^2) memory requirements as stated in the manuscript. Fixed-CGEC timing and matched-tuned timing answer different questions and must not be conflated.
