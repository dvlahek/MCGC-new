from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class CGECConfig:
    n_neighbors: int = 12
    diffusion_alpha: float = 0.85
    diffusion_steps: int = 40
    field_weight: float = 1.0
    gradient_weight: float = 1.5
    geometry_weight: float = 0.35
    revisit_penalty: float = 0.75
    max_steps: int = 10
    trail_floor: float = 1e-8
    eps: float = 1e-12


class FieldOnlyCGEC:
    """Paper-locked field-only CGEC used for reproducibility and XAI.

    Prediction is based only on affinity-weighted local class evidence L_c(q).
    Support examples, exact neighbour margin contributions, paired feature
    interventions, and deterministic representative paths are exposed from the
    same fitted graph-evidence representation after prediction.
    """

    def __init__(self, config: CGECConfig | None = None):
        self.config = config or CGECConfig()

    def fit(self, X: np.ndarray, y: np.ndarray) -> "FieldOnlyCGEC":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if X.ndim != 2 or len(X) != len(y):
            raise ValueError("X must be 2-D and X/y lengths must match")

        cfg = self.config
        self.scaler_ = StandardScaler().fit(X)
        self.X_train_ = X.copy()
        self.Z_ = self.scaler_.transform(X)
        self.y_ = y.copy()
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        self.class_to_index_ = {c: i for i, c in enumerate(self.classes_)}
        self.raw_feature_mean_ = X.mean(axis=0)

        n = len(X)
        dist = cdist(self.Z_, self.Z_)
        np.fill_diagonal(dist, np.inf)
        directed_neighbors = np.argsort(dist, axis=1)[:, : cfg.n_neighbors]
        neighbor_distances = np.take_along_axis(dist, directed_neighbors, axis=1)
        sigma = neighbor_distances[:, -1]

        w_directed = np.zeros((n, n), dtype=float)
        for i in range(n):
            js = directed_neighbors[i]
            denom = np.maximum(sigma[i] * sigma[js], cfg.eps)
            w_directed[i, js] = np.exp(-(dist[i, js] ** 2) / denom)

        self.W_ = np.maximum(w_directed, w_directed.T)
        row_sum = self.W_.sum(axis=1, keepdims=True)
        self.P_ = np.divide(
            self.W_, row_sum, out=np.zeros_like(self.W_), where=row_sum > 0
        )
        self.graph_neighbors_ = [np.flatnonzero(self.W_[i] > 0) for i in range(n)]

        B = np.column_stack([(y == c).astype(float) for c in self.classes_])
        S = B.copy()
        for _ in range(cfg.diffusion_steps):
            S = (1.0 - cfg.diffusion_alpha) * B + cfg.diffusion_alpha * (self.P_ @ S)
        self.class_field_ = S

        D = np.empty_like(S)
        for c in range(self.n_classes_):
            rivals = np.delete(S, c, axis=1)
            D[:, c] = S[:, c] - np.max(rivals, axis=1)
        self.contrast_field_ = D

        self.target_threshold_ = np.array(
            [np.median(D[y == cls, c]) for c, cls in enumerate(self.classes_)],
            dtype=float,
        )
        return self

    @staticmethod
    def _py(value):
        return value.item() if hasattr(value, "item") else value

    def _query_neighbourhood(self, x: np.ndarray):
        cfg = self.config
        x = np.asarray(x, dtype=float).reshape(1, -1)
        zq = self.scaler_.transform(x)[0]
        distances = np.linalg.norm(self.Z_ - zq, axis=1)
        starts = np.argsort(distances)[: cfg.n_neighbors]
        sigma_q = max(float(np.sort(distances)[cfg.n_neighbors - 1]), cfg.eps)
        raw = np.exp(-(distances[starts] ** 2) / (sigma_q**2))
        affinity = raw / raw.sum()
        return zq, distances[starts], starts, affinity

    def local_evidence(self, x: np.ndarray):
        _, distances, starts, affinity = self._query_neighbourhood(x)
        L = np.array(
            [
                affinity @ self.contrast_field_[starts, c]
                for c in range(self.n_classes_)
            ],
            dtype=float,
        )
        return distances, starts, affinity, L

    def margin_contributions(self, x: np.ndarray) -> Dict[str, object]:
        """Exact neighbour decomposition of the original winner-runner margin.

        C_j(q) = a_j(q) [D_j^(c+) - D_j^(c-)], and sum_j C_j(q) = M(q).
        """
        distances, starts, affinity, L = self.local_evidence(x)
        order = np.argsort(L)[::-1]
        winner = int(order[0])
        runner = int(order[1]) if len(order) > 1 else winner
        contributions = affinity * (
            self.contrast_field_[starts, winner] - self.contrast_field_[starts, runner]
        )
        margin = float(L[winner] - L[runner])
        if not np.isclose(float(np.sum(contributions)), margin, rtol=1e-11, atol=1e-12):
            raise RuntimeError("Neighbour contributions do not reconstruct the margin")
        return {
            "winner_index": winner,
            "runner_index": runner,
            "winner": self._py(self.classes_[winner]),
            "runner_up": self._py(self.classes_[runner]),
            "margin": margin,
            "nodes": [int(v) for v in starts],
            "distances": [float(v) for v in distances],
            "affinities": [float(v) for v in affinity],
            "contributions": [float(v) for v in contributions],
            "local_evidence_vector": [float(v) for v in L],
        }

    def feature_intervention_effects(
        self,
        x: np.ndarray,
        reference: np.ndarray | None = None,
    ) -> Dict[str, object]:
        """Paired feature interventions on the fixed original winner-runner margin."""
        x = np.asarray(x, dtype=float).copy()
        record = self.margin_contributions(x)
        winner = int(record["winner_index"])
        runner = int(record["runner_index"])
        base_margin = float(record["margin"])
        ref = self.raw_feature_mean_ if reference is None else np.asarray(reference, dtype=float)
        if ref.shape != x.shape:
            raise ValueError("reference must have the same shape as x")

        effects = np.empty(x.shape[0], dtype=float)
        intervened_margins = np.empty_like(effects)
        for k in range(x.shape[0]):
            xk = x.copy()
            xk[k] = ref[k]
            _, _, _, Lk = self.local_evidence(xk)
            mk = float(Lk[winner] - Lk[runner])
            intervened_margins[k] = mk
            effects[k] = base_margin - mk

        return {
            "winner_index": winner,
            "runner_index": runner,
            "winner": record["winner"],
            "runner_up": record["runner_up"],
            "base_margin": base_margin,
            "reference": [float(v) for v in ref],
            "effects": [float(v) for v in effects],
            "intervened_margins": [float(v) for v in intervened_margins],
        }

    def representative_path(
        self,
        starts: np.ndarray,
        affinity: np.ndarray,
        class_index: int,
    ) -> Dict[str, object]:
        cfg = self.config
        D = self.contrast_field_
        local = D[starts, class_index]
        shifted = local - np.min(local) + cfg.eps
        current = int(starts[np.argmax(affinity * shifted)])

        path = [current]
        evidence_trace = [float(D[current, class_index])]
        visited = {current}
        reason = (
            "threshold_reached"
            if evidence_trace[-1] >= self.target_threshold_[class_index]
            else None
        )

        if reason is None:
            for _ in range(cfg.max_steps):
                neigh = self.graph_neighbors_[current]
                if len(neigh) == 0:
                    reason = "no_graph_neighbour"
                    break

                current_e = D[current, class_index]
                next_e = D[neigh, class_index]
                grad = next_e - current_e
                if float(np.max(grad)) <= cfg.trail_floor:
                    reason = "no_evidence_increase_above_floor"
                    break

                revisit = np.array(
                    [1.0 if int(node) in visited else 0.0 for node in neigh]
                )
                desirability = (
                    cfg.field_weight * next_e
                    + cfg.gradient_weight * grad
                    + cfg.geometry_weight * np.log(self.W_[current, neigh] + cfg.eps)
                    - cfg.revisit_penalty * revisit
                )
                current = int(neigh[np.argmax(desirability)])
                visited.add(current)
                path.append(current)
                evidence_trace.append(float(D[current, class_index]))

                if evidence_trace[-1] >= self.target_threshold_[class_index]:
                    reason = "threshold_reached"
                    break
            else:
                reason = "max_steps"

        return {
            "class_index": int(class_index),
            "class_label": self._py(self.classes_[class_index]),
            "entry_node": int(path[0]),
            "path": [int(v) for v in path],
            "evidence_trace": [float(v) for v in evidence_trace],
            "threshold": float(self.target_threshold_[class_index]),
            "termination_reason": reason,
        }

    def explain_query(self, x: np.ndarray, top_support: int = 5) -> Dict[str, object]:
        distances, starts, affinity, L = self.local_evidence(x)
        order = np.argsort(L)[::-1]
        winner = int(order[0])
        runner = int(order[1]) if len(order) > 1 else winner

        support_order = np.argsort(-affinity)[: min(top_support, len(starts))]
        support = []
        for pos in support_order:
            node = int(starts[pos])
            label = self._py(self.y_[node])
            support.append(
                {
                    "node_index": node,
                    "distance": float(distances[pos]),
                    "affinity": float(affinity[pos]),
                    "training_label": label,
                    "contrastive_evidence": {
                        str(self.classes_[c]): float(self.contrast_field_[node, c])
                        for c in range(self.n_classes_)
                    },
                }
            )

        paths = {
            str(self.classes_[c]): self.representative_path(starts, affinity, c)
            for c in range(self.n_classes_)
        }
        contribution_record = self.margin_contributions(x)
        intervention_record = self.feature_intervention_effects(x)

        return {
            "prediction": self._py(self.classes_[winner]),
            "runner_up": self._py(self.classes_[runner]),
            "local_evidence": {
                str(self.classes_[c]): float(L[c]) for c in range(self.n_classes_)
            },
            "margin": float(L[winner] - L[runner]),
            "query_neighbour_nodes": [int(v) for v in starts],
            "query_affinities": [float(v) for v in affinity],
            "supporting_neighbors": support,
            "neighbour_margin_contributions": contribution_record,
            "feature_interventions": intervention_record,
            "paths": paths,
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        preds = []
        for x in X:
            _, _, _, L = self.local_evidence(x)
            preds.append(self.classes_[int(np.argmax(L))])
        return np.asarray(preds)
