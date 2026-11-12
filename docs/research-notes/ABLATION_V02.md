# MCGC v0.2 ablation protocol

The purpose of v0.2 is not to tune MCGC to beat a baseline. It is to isolate which parts of the prototype contribute predictive information.

## Frozen synthetic protocol

Datasets:

- noisy two moons,
- noisy concentric circles,
- noisy two-arm spirals.

Each dataset uses 600 samples, a stratified 70/30 train/test split, split seed 17, and fixed dataset seeds. MCGC inference uses random seed 17.

This is a diagnostic benchmark, not a generalization claim.

## Compared systems

1. `knn_uniform`: ordinary 9-neighbour k-NN.
2. `knn_distance`: distance-weighted 9-neighbour k-NN.
3. `label_spreading`: k-NN graph Label Spreading baseline.
4. `field_only`: MCGC graph, diffusion, contrastive field and local query evidence, with walkers disabled.
5. `one_walker`: full scoring structure with one class-conditioned walker per class.
6. `multi_no_gradient`: multiple walkers, but the explicit gradient term is removed from transition desirability.
7. `mcgc_no_reliability`: full multiple-walker model with agreement and lost-trail contributions removed from the final class score.
8. `mcgc_full`: current complete v0.2 prototype.

## Primary scientific comparisons

The most important comparison is:

`mcgc_full` versus `field_only`.

If there is no reproducible advantage over `field_only`, then the class-conditioned trajectories have not yet demonstrated useful predictive information beyond the graph field itself.

Secondary comparisons test more specific mechanisms:

- `one_walker` versus `mcgc_full`: benefit of multiple walkers,
- `multi_no_gradient` versus `mcgc_full`: benefit of explicit gradient following,
- `mcgc_no_reliability` versus `mcgc_full`: benefit of inter-walker agreement and lost-trail evidence.

## Diagnostic trail outputs

For the winning class, the benchmark records:

- mean lost-trail fraction,
- mean trail coherence,
- mean trail length.

These diagnostics are descriptive. They must not be interpreted as evidence of better classification unless linked to out-of-sample predictive performance.

## Interpretation rule

No component is retained as scientifically useful merely because it matches the biological metaphor. A walker-specific component should eventually be justified by a reproducible ablation effect, improved selective classification/rejection behaviour, or another independently measured benefit.
