from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
V32B_SCRIPT = HERE / "benchmark_real_data_confirmation_v32b.py"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


v32b = _load_module(V32B_SCRIPT, "benchmark_real_data_confirmation_v32b")
v32 = v32b.v32

# Second pre-result eligibility correction. horse_colic_outcome has extensive
# missingness; under the predeclared complete-case/no-missing protocol only
# seven rows remain, violating n >= 250. The loader failed before the benchmark
# entered the dataset/seed evaluation loop, so no CGEC or baseline result had
# been produced when this replacement was made. credit_approval_germany
# satisfies the frozen size/feature/class criteria and is disjoint from the
# 12 label-review datasets and the original digits/wine predictive pair.
v32.DATASETS = (
    "credit_approval_australia",
    "credit_approval_germany",
    "dermatology",
    "haberman",
    "heart_disease_cleveland",
    "ionosphere",
    "penguins",
    "segmentation",
)


def _inject_versioned_defaults():
    """Keep v32c artifacts separate unless the caller supplied explicit paths."""
    args = sys.argv[1:]
    if "--cache-dir" not in args:
        sys.argv.extend([
            "--cache-dir",
            str(v32.PAPER_ROOT / "source_data" / "pmlb_real_confirmation_v32c"),
        ])
    if "--checkpoint" not in args:
        sys.argv.extend([
            "--checkpoint",
            str(v32.PAPER_ROOT / "results" / "v32c_real_data_confirmation_runs.jsonl"),
        ])
    if "--output" not in args:
        sys.argv.extend([
            "--output",
            str(v32.PAPER_ROOT / "results" / "v32c_real_data_confirmation.json"),
        ])


if __name__ == "__main__":
    _inject_versioned_defaults()
    v32.main()
