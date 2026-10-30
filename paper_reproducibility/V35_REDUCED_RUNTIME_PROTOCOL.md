# v3.5 Reduced runtime audit protocol

Status: secondary descriptive runtime audit, fixed before v3.5 runtime results are inspected.

The predictive matched-tuning analysis remains v3.3 and uses all eight PMLB real-data evaluation datasets and all 12 outer seeds. v3.5 is only a reduced-cost runtime audit and must not replace the 12-seed predictive results.

## Scope

Runtime is measured on all eight datasets from the v3.2c/v3.3 real-data panel using three fixed outer seeds:

- 0
- 5
- 11

This gives 24 dataset-seed timing units. The dataset panel, outer 70/30 stratified split definition, CGEC tuning grid, inner three-fold CV, selection metric, and baseline grids are unchanged from the matched-tuning audit.

## CGEC matched tuning

- K in {8, 12, 17, 25}
- alpha in {0.50, 0.75, 0.85}
- n_D in {20, 40, 80}
- 36 candidate configurations
- inner CV: 3-fold stratified on outer-training data only, shuffle=True, random_state=6000+seed
- selection: lowest inner OOF E-AURC; higher inner OOF accuracy as first tie-break; deterministic lexical candidate key as final tie-break

## Timing policy

The runtime implementation is inherited from v3.4b:

- wall-clock time measured with time.perf_counter()
- OMP_NUM_THREADS=1
- MKL_NUM_THREADS=1
- OPENBLAS_NUM_THREADS=1
- NUMEXPR_NUM_THREADS=1
- random forest timing forced to n_jobs=1
- fixed CGEC final fit/predict measured over three repetitions and summarized by the median
- tuned CGEC complete inner-CV search measured once per dataset-seed unit, followed by final refit and outer-test prediction
- selected tuned CGEC final fit/predict additionally measured over three repetitions and summarized by the median
- tuned baseline end-to-end cost measured using the same inner-CV tuning protocol, with selected final-model fit/predict also measured over three repetitions

## Interpretation

v3.5 is descriptive implementation-level timing evidence, not a scaling-complexity experiment and not an inferential runtime superiority test. All eight datasets are retained to preserve coverage of the real-data evaluation panel; three fixed seeds are used to reduce computational cost while avoiding a single-split timing result. Runtime measurements should be obtained on an otherwise idle machine. Running another CPU- or memory-intensive benchmark concurrently invalidates the timing comparison because of resource contention.
