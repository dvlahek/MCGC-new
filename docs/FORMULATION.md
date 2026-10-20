# MCGC v0.1 formulation

MCGC is currently an experimental classifier built from four deliberately separated components.

## 1. Graph

Given standardized training samples, build a mutualised k-nearest-neighbour graph with locally scaled heat-kernel weights.

## 2. Class evidence fields

For each class, diffuse one-hot class evidence over the graph. The v0.1 implementation then forms a contrastive field for class `c` as its diffused evidence minus the strongest rival class evidence at the same node.

This diffusion step is not claimed as novel.

## 3. Class-conditioned chemotactic walkers

For a query, choose nearby training nodes as possible starting nodes. For every candidate class, launch multiple stochastic walkers. At a node `i`, a walker evaluates neighbour `j` using:

- destination contrastive evidence,
- positive change in that evidence,
- graph-edge plausibility,
- a revisit penalty.

The next node is sampled with a temperature-controlled softmax. A walker can terminate because it reaches a representative class basin, reaches a local positive basin, or loses the trail.

## 4. Query score

The local contrastive field around the query is the anchor signal. Walker trajectories refine that signal using mean trail score, inter-walker agreement, and the fraction of lost walkers. This design is intentional: an earlier naive prototype over-rewarded large uphill trajectories for the wrong class and could invert otherwise easy classifications.

## Research questions

The current implementation is only a falsifiable v0.1 prototype. The next experiments should test whether trajectory-derived information adds measurable value beyond:

1. k-NN,
2. weighted k-NN,
3. label propagation / label spreading,
4. random-walk graph classifiers,
5. the same MCGC local field with walker terms ablated.

Useful future ablations include the contrastive field, gradient term, number of walkers, trail-loss term, agreement term, and temperature.
