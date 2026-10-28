from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

from mcgc import MultiAgentChemotacticGraphClassifier

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / "examples" / "benchmark_xai_stability_v15ter_tail.py"

def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module

v15t = load_module(PARENT, "benchmark_xai_stability_v15ter_tail")

def field_classifier():
    return MultiAgentChemotacticGraphClassifier(
        n_neighbors=12,
        diffusion_alpha=0.85,
        diffusion_steps=40,
        random_state=17,
        standardize=True,
        use_walkers=False,
    )

# The frozen v15-ter protocol obtains predictions/explanations through this
# factory. Replacing only the model factory preserves datasets, splits,
# perturbations, query matching, path construction, and inference units.
v15t.v15.mcgc_classifier = field_classifier

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "v30_field_only_path_stability.json",
    )
    args = ap.parse_args()

    datasets = {}
    for name in v15t.v15b.DATASETS:
        print(f"DATASET {name}", flush=True)
        runs = []
        for seed in v15t.v15b.SEEDS:
            print(f"  seed={seed}", flush=True)
            run = v15t.evaluate_seed(name, seed)
            runs.append(run)
            s = run["summary"]
            print(
                "    "
                f"full={s['full_path_advantage']['mean']} "
                f"tail={s['tail_path_advantage']['mean']} "
                f"same_start={s['same_start_tail_advantage']['mean']}",
                flush=True,
            )
        datasets[name] = {"runs": runs}

    result = {
        "protocol": {
            "version": "v3.0-path",
            "status": "final field-only path-stability confirmation",
            "parent":
                "benchmark_xai_stability_v15ter_tail.py",
            "only_change":
                "use_walkers=False; graph/path protocol unchanged",
            "datasets": list(v15t.v15b.DATASETS),
            "seeds": list(v15t.v15b.SEEDS),
            "cross_query_control":
                "nearest same-predicted-class test query in standardized feature space",
            "tail_definition":
                "representative path after removing first node",
            "primary_inference_unit": "dataset x seed",
        },
        "datasets": datasets,
        "dataset_seed_inference": v15t.aggregate(datasets),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print("\n=== FINAL FIELD-ONLY PATH STABILITY ===")
    for key, block in result["dataset_seed_inference"].items():
        s = block["summary"]
        print(
            f"{key:28s} n={s['n']} mean={s['mean']} "
            f"W/T/L={s['wins']}/{s['ties']}/{s['losses']} "
            f"p={s['sign_test_p_two_sided']}"
        )
    print(f"FULL_JSON={args.output.resolve()}")

if __name__ == "__main__":
    main()
