# CGEC / MCGC research repository

## Paper reproducibility branch

This branch is the frozen reproducibility branch for the manuscript **“Chemotactic Graph Evidence for Prediction Explanation and Training-Label Review.”**

The manuscript method is **field-only CGEC**:

- prediction is obtained only from affinity-weighted local class evidence on the fitted graph-evidence field;
- stochastic walkers are **not** part of the manuscript classifier, confidence score, representative path, feature intervention, or label-review procedure;
- the representative chemotactic path is a deterministic post-prediction inspection of the fitted graph-evidence representation;
- training-label review is evaluated strictly out of fold.

The canonical paper artifacts are in [`paper_reproducibility/`](paper_reproducibility/). Start with:

- [`paper_reproducibility/README.md`](paper_reproducibility/README.md) — reproduction instructions and experiment map;
- [`paper_reproducibility/REPRODUCIBILITY_LOCK.md`](paper_reproducibility/REPRODUCIBILITY_LOCK.md) — manuscript-to-code/result mapping;
- [`paper_reproducibility/results/final_manuscript_results.json`](paper_reproducibility/results/final_manuscript_results.json) — compact numerical lock for the reported manuscript results.

The paper-specific implementation is isolated in `paper_reproducibility/xai/` so that historical development code cannot silently change the manuscript definition.

## Historical development code

The repository also retains the earlier MCGC development line, including stochastic-walker experiments and negative confirmation results. That code is preserved for provenance and research history, but it is **not** the method evaluated in the final manuscript.

Legacy documentation remains under `docs/` and legacy package code remains under `src/mcgc/`.

## Minimal paper setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -U pip
pip install -e ".[dev]"
pip install -r paper_reproducibility/requirements.txt
```

Then follow [`paper_reproducibility/README.md`](paper_reproducibility/README.md).
