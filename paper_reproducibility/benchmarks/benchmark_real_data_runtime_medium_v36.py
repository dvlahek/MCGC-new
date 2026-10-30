from __future__ import annotations

# v3.6: medium descriptive runtime audit.
# Predictive matched-tuning evidence remains v3.3 (all 12 seeds).
# Runtime uses all eight datasets and fixed seeds 0, 2, 5, 8, and 11.

import importlib.util
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
V34B_SCRIPT = HERE / "benchmark_real_data_tuned_cgec_runtime_v34b.py"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


v34b = _load_module(V34B_SCRIPT, "benchmark_real_data_tuned_cgec_runtime_v34b_v36")
v34 = v34b.v34

RUNTIME_SEEDS = "0,2,5,8,11"


def _inject_v36_defaults():
    args = sys.argv[1:]

    if "--seeds" not in args:
        sys.argv.extend(["--seeds", RUNTIME_SEEDS])

    if "--candidate-checkpoint" not in args:
        sys.argv.extend([
            "--candidate-checkpoint",
            str(v34.v32.PAPER_ROOT / "results" / "v36_runtime_candidates.jsonl"),
        ])

    if "--component-checkpoint" not in args:
        sys.argv.extend([
            "--component-checkpoint",
            str(v34.v32.PAPER_ROOT / "results" / "v36_runtime_components.jsonl"),
        ])

    if "--output" not in args:
        sys.argv.extend([
            "--output",
            str(v34.v32.PAPER_ROOT / "results" / "v36_runtime_summary.json"),
        ])


if __name__ == "__main__":
    _inject_v36_defaults()
    v34b._inject_versioned_defaults()
    v34.main()
