# v3.6 Medium runtime audit protocol

Status: secondary descriptive runtime audit, frozen before v3.6 timing results are inspected.

This audit preserves the primary predictive evidence and the full v3.3 matched-tuning analysis. It evaluates computational cost on the same eight PMLB real-data datasets, using five fixed outer-split seeds per dataset: 0, 2, 5, 8, and 11. This gives 40 dataset-seed runtime units.

## Dataset and split scope

- all eight v3.2c real-data datasets
- runtime seeds: {0, 2, 5, 8, 11}
- 70/30 stratified outer split, random_state=5000+seed
- all methods use exactly the same outer split within each dataset-seed unit

## CGEC matched tuning

- K in {8, 12, 17, 25}
- alpha in {0.50, 0.75, 0.85}
- n_D in {20, 40, 80}
- 36 candidate configurations
- inner CV: 3-fold stratified on outer-training data only, shuffle=True, random_state=6000+seed
- selection: lowest inner OOF E-AURC; higher inner OOF accuracy as first tie-break; deterministic lexical candidate key as final tie-break
- selected configuration is refit on the full outer-training split and evaluated once on the untouched outer test split

The fixed CGEC configuration K=12, alpha=0.85, n_D=40 remains the primary pre-specified predictive result. Tuned CGEC remains a secondary matched-tuning analysis.

## Runtime environment

The timing implementation inherits the v3.4b single-thread policy before numerical libraries are imported:

- OMP_NUM_THREADS=1
- MKL_NUM_THREADS=1
- OPENBLAS_NUM_THREADS=1
- NUMEXPR_NUM_THREADS=1
- random forest is forced to n_jobs=1 for runtime measurements

Wall-clock time is measured with time.perf_counter(). Hardware/software metadata and thread settings are retained in the output.

## Runtime quantities

For each of the 40 dataset-seed units, the audit measures:

1. fixed CGEC final-model fit and outer-test prediction/evidence time;
2. tuned CGEC end-to-end cost, including 36 inner-CV candidates, final refit, and outer-test prediction;
3. tuned baseline end-to-end cost for scaled kNN, RBF-SVM, random forest, and scaled Label Spreading under their existing inner-CV protocol;
4. selected final-model fit and prediction time for CGEC and baselines, using three repetitions and the median.

Candidate-level and component-level checkpoints are retained so interrupted runs can resume.

## Interpretation

Runtime results are descriptive for the tested implementation and hardware. They are not a scaling-complexity experiment and are not pooled with timing measurements from another computer. All methods reported in the runtime comparison must be measured on the same dedicated workstation under the same single-thread policy.
