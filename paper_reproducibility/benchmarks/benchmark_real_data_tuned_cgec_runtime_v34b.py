from __future__ import annotations

# Keep the runtime audit single-threaded before importing numerical libraries.
import os
import sys
from pathlib import Path

for _name in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_name] = "1"

import importlib.util


HERE = Path(__file__).resolve().parent
V34_SCRIPT = HERE / "benchmark_real_data_tuned_cgec_runtime_v34.py"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


v34 = _load_module(V34_SCRIPT, "benchmark_real_data_tuned_cgec_runtime_v34b_base")

# The historical RF benchmark uses n_jobs=-1. For the runtime audit only,
# force n_jobs=1 so all methods follow the same single-thread timing policy.
# This does not alter RF hyperparameters, random_state, predictions, or metrics.
_original_classifier_for = v34.v26.classifier_for


def _single_thread_classifier_for(kind: str, params: dict):
    model = _original_classifier_for(kind, params)
    if kind == "rf":
        model.set_params(n_jobs=1)
    return model


v34.v26.classifier_for = _single_thread_classifier_for


def _inject_versioned_defaults():
    args = sys.argv[1:]
    if "--candidate-checkpoint" not in args:
        sys.argv.extend([
            "--candidate-checkpoint",
            str(v34.v32.PAPER_ROOT / "results" / "v34b_timed_cgec_candidates.jsonl"),
        ])
    if "--component-checkpoint" not in args:
        sys.argv.extend([
            "--component-checkpoint",
            str(v34.v32.PAPER_ROOT / "results" / "v34b_runtime_components.jsonl"),
        ])
    if "--output" not in args:
        sys.argv.extend([
            "--output",
            str(v34.v32.PAPER_ROOT / "results" / "v34b_runtime_matched_tuning_summary.json"),
        ])


if __name__ == "__main__":
    _inject_versioned_defaults()
    v34.main()
