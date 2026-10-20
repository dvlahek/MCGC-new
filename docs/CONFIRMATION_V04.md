# MCGC v0.4 multi-seed confirmation

This checkpoint was designed to test whether the apparent walker benefit seen in the original single-split spiral experiment survives independent dataset/split seeds without per-seed retuning.

## Frozen protocol

- 12 seeds: `0..11`
- 500 samples per synthetic dataset
- datasets: moons, circles, two-arm spirals
- test fraction: 0.30
- split seed: `1000 + dataset_seed`
- MCGC random state: 17
- fixed hyperparameters across all runs
- comparisons: `field_only`, `multi_no_gradient`, `mcgc_full`, and distance-weighted k-NN

The benchmark is implemented in `examples/benchmark_multiseed_v04.py` and runs in GitHub Actions on Python 3.11.

## Paired results

### Walker effect: full MCGC minus field-only

| dataset | mean delta | wins | ties | losses |
|---|---:|---:|---:|---:|
| circles | +0.00056 | 1 | 11 | 0 |
| moons | -0.00056 | 0 | 11 | 1 |
| spirals | -0.00667 | 3 | 2 | 7 |

The original positive single-split spiral effect does **not** survive this confirmation protocol. The current walker correction is therefore not supported as a general accuracy improvement over the same fitted contrastive field.

### Gradient effect: full MCGC minus multi-walker/no-gradient

| dataset | mean delta | wins | ties | losses |
|---|---:|---:|---:|---:|
| circles | 0.00000 | 0 | 12 | 0 |
| moons | -0.00056 | 0 | 11 | 1 |
| spirals | ~0.00000 | 4 | 3 | 5 |

There is no stable evidence yet that the explicit gradient term improves aggregate accuracy.

### Full MCGC minus distance-weighted k-NN

On spirals, full MCGC lost on all 12 seeds with mean accuracy delta `-0.0950`. This confirms that the current prototype is not competitive with this baseline on the frozen spiral protocol.

## Interpretation

This is a useful failure-aware checkpoint rather than a reason to discard the project.

The fitted contrastive graph field remains strong on the easy datasets, while unconditional walker corrections can perturb already-good local decisions. A more defensible next hypothesis is therefore **selective trajectory use**: trajectory evidence should have most influence when the local graph field is ambiguous, and little or no influence when local evidence is already decisive.

The XAI contribution is logically separate from the accuracy result. v0.4 also adds exact contrastive explanations (`why A rather than B`) whose component deltas reconstruct the requested class-score margin exactly.

No performance or novelty claim should be made from v0.4 beyond these measured observations.
