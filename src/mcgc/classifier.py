from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


@dataclass(frozen=True)
class TrailSummary:
    score: float
    coherence: float
    terminal_evidence: float
    lost: bool
    length: int


class MultiAgentChemotacticGraphClassifier(BaseEstimator, ClassifierMixin):
    """Experimental multi-agent chemotactic graph classifier.

    During ``fit`` the estimator builds a mutualised k-NN graph over the
    training set and diffuses one evidence field per class. During prediction,
    several stochastic class-conditioned walkers are launched from each
    query's local neighbourhood. A class is scored from local contrastive
    evidence together with path coherence, terminal evidence, geometry, walker
    agreement, and the fraction of walkers that lose the trail.

    The top-level score weights and ``use_walkers`` are deliberately exposed so
    the walker contribution can be ablated without changing the fitted graph or
    class-field construction.

    This is a compact research prototype, not a performance-optimised model.
    """

    def __init__(
        self,
        n_neighbors: int = 12,
        n_walkers: int = 12,
        max_steps: int = 10,
        diffusion_alpha: float = 0.85,
        diffusion_steps: int = 40,
        temperature: float = 0.35,
        random_state: int | None = 0,
        standardize: bool = True,
        field_weight: float = 1.0,
        gradient_weight: float = 1.5,
        geometry_weight: float = 0.35,
        revisit_penalty: float = 0.75,
        coherence_weight: float = 0.75,
        terminal_weight: float = 0.5,
        lost_penalty: float = 0.75,
        trail_floor: float = 1e-8,
        use_walkers: bool = True,
        local_evidence_weight: float = 2.0,
        trail_score_weight: float = 0.25,
        agreement_weight: float = 0.10,
        lost_fraction_weight: float = 0.20,
    ) -> None:
        self.n_neighbors = n_neighbors
        self.n_walkers = n_walkers
        self.max_steps = max_steps
        self.diffusion_alpha = diffusion_alpha
        self.diffusion_steps = diffusion_steps
        self.temperature = temperature
        self.random_state = random_state
        self.standardize = standardize
        self.field_weight = field_weight
        self.gradient_weight = gradient_weight
        self.geometry_weight = geometry_weight
        self.revisit_penalty = revisit_penalty
        self.coherence_weight = coherence_weight
        self.terminal_weight = terminal_weight
        self.lost_penalty = lost_penalty
        self.trail_floor = trail_floor
        self.use_walkers = use_walkers
        self.local_evidence_weight = local_evidence_weight
        self.trail_score_weight = trail_score_weight
        self.agreement_weight = agreement_weight
        self.lost_fraction_weight = lost_fraction_weight

    def fit(self, X: np.ndarray, y: np.ndarray):
        X, y = check_X_y(X, y)
        if len(X) < 3:
            raise ValueError("MCGC requires at least three training samples.")
        if self.n_neighbors < 1:
            raise ValueError("n_neighbors must be >= 1.")
        if self.n_walkers < 1 or self.max_steps < 1:
            raise ValueError("n_walkers and max_steps must be >= 1.")
        if not (0.0 <= self.diffusion_alpha < 1.0):
            raise ValueError("diffusion_alpha must be in [0, 1).")
        if self.temperature <= 0:
            raise ValueError("temperature must be > 0.")

        self.classes_, y_index = np.unique(y, return_inverse=True)
        self.n_classes_ = len(self.classes_)
        if self.n_classes_ < 2:
            raise ValueError("MCGC requires at least two classes.")

        self.scaler_ = StandardScaler() if self.standardize else None
        Xs = self.scaler_.fit_transform(X) if self.scaler_ is not None else np.asarray(X, dtype=float)
        self.X_ = Xs
        self.y_index_ = y_index
        self.n_features_in_ = X.shape[1]

        k = min(self.n_neighbors, len(X) - 1)
        self.k_ = k
        self.nn_ = NearestNeighbors(n_neighbors=k + 1)
        self.nn_.fit(Xs)
        distances, indices = self.nn_.kneighbors(Xs)

        distances = distances[:, 1:]
        indices = indices[:, 1:]
        local_sigma = np.maximum(distances[:, -1], 1e-12)
        W = np.zeros((len(X), len(X)), dtype=float)
        for i in range(len(X)):
            for d, j in zip(distances[i], indices[i]):
                denom = max(local_sigma[i] * local_sigma[j], 1e-12)
                W[i, j] = np.exp(-(d * d) / denom)
        W = np.maximum(W, W.T)
        row_sum = W.sum(axis=1, keepdims=True)
        self.P_ = np.divide(W, row_sum, out=np.zeros_like(W), where=row_sum > 0)
        self.W_ = W
        self.graph_neighbors_ = [np.flatnonzero(W[i] > 0) for i in range(len(X))]

        base = np.zeros((len(X), self.n_classes_), dtype=float)
        base[np.arange(len(X)), y_index] = 1.0
        field = base.copy()
        a = self.diffusion_alpha
        for _ in range(self.diffusion_steps):
            field = (1.0 - a) * base + a * (self.P_ @ field)
        self.class_field_ = field

        contrast = np.empty_like(field)
        for c in range(self.n_classes_):
            rivals = np.delete(field, c, axis=1)
            contrast[:, c] = field[:, c] - np.max(rivals, axis=1)
        self.contrast_field_ = contrast
        self.target_threshold_ = np.empty(self.n_classes_, dtype=float)
        for c in range(self.n_classes_):
            own = contrast[y_index == c, c]
            self.target_threshold_[c] = float(np.quantile(own, 0.50))
        return self

    def _scaled(self, X: np.ndarray) -> np.ndarray:
        X = check_array(X)
        return self.scaler_.transform(X) if self.scaler_ is not None else np.asarray(X, dtype=float)

    def _softmax_sample(self, values: np.ndarray, rng: np.random.Generator) -> int:
        z = (values - np.max(values)) / self.temperature
        p = np.exp(z)
        p /= p.sum()
        return int(rng.choice(len(values), p=p))

    def _walk(self, start: int, class_index: int, rng: np.random.Generator) -> TrailSummary:
        current = int(start)
        visited = {current}
        gains: List[float] = []
        geom: List[float] = []
        evidence_trace = [float(self.contrast_field_[current, class_index])]
        lost = False

        if evidence_trace[-1] >= self.target_threshold_[class_index]:
            terminal = evidence_trace[-1]
            return TrailSummary(
                score=0.5 + self.terminal_weight * terminal,
                coherence=1.0,
                terminal_evidence=terminal,
                lost=False,
                length=0,
            )

        for _ in range(self.max_steps):
            neigh = self.graph_neighbors_[current]
            if len(neigh) == 0:
                lost = True
                break

            current_e = self.contrast_field_[current, class_index]
            next_e = self.contrast_field_[neigh, class_index]
            grad = next_e - current_e
            weights = self.W_[current, neigh]
            revisit = np.array([1.0 if int(j) in visited else 0.0 for j in neigh])
            desirability = (
                self.field_weight * next_e
                + self.gradient_weight * grad
                + self.geometry_weight * np.log(weights + 1e-12)
                - self.revisit_penalty * revisit
            )

            if float(np.max(grad)) <= self.trail_floor:
                lost = bool(current_e <= 0.0)
                break

            pick = self._softmax_sample(desirability, rng)
            nxt = int(neigh[pick])
            gains.append(float(grad[pick]))
            geom.append(float(np.log(weights[pick] + 1e-12)))
            current = nxt
            visited.add(current)
            evidence_trace.append(float(self.contrast_field_[current, class_index]))
            if evidence_trace[-1] >= self.target_threshold_[class_index]:
                break

        if not gains:
            terminal = evidence_trace[-1]
            return TrailSummary(
                score=self.terminal_weight * terminal - (self.lost_penalty if lost else 0.0),
                coherence=1.0 if terminal > 0 else 0.0,
                terminal_evidence=terminal,
                lost=lost,
                length=0,
            )

        gains_arr = np.asarray(gains)
        coherence = float(np.mean(gains_arr > 0.0))
        positive_gain = float(np.sum(np.maximum(gains_arr, 0.0)))
        terminal = evidence_trace[-1]
        geometry = float(np.mean(geom))
        score = (
            positive_gain
            + self.coherence_weight * coherence
            + self.terminal_weight * terminal
            + self.geometry_weight * geometry
            - (self.lost_penalty if lost else 0.0)
        )
        return TrailSummary(score, coherence, terminal, lost, len(gains))

    def _query_start_nodes(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        distances, indices = self.nn_.kneighbors(
            x.reshape(1, -1), n_neighbors=min(self.k_, len(self.X_))
        )
        return distances[0], indices[0]

    def _score_query(
        self, x: np.ndarray, rng: np.random.Generator
    ) -> Tuple[np.ndarray, Dict[int, List[TrailSummary]]]:
        distances, starts = self._query_start_nodes(x)
        sigma = max(float(distances[-1]) if len(distances) else 1.0, 1e-12)
        start_affinity = np.exp(-(distances**2) / (sigma**2))
        start_affinity /= start_affinity.sum()

        scores = np.empty(self.n_classes_, dtype=float)
        detail: Dict[int, List[TrailSummary]] = {}
        for c in range(self.n_classes_):
            local_evidence = float(np.sum(start_affinity * self.contrast_field_[starts, c]))
            if not self.use_walkers:
                detail[c] = []
                scores[c] = float(self.local_evidence_weight * local_evidence)
                continue

            trails: List[TrailSummary] = []
            for _ in range(self.n_walkers):
                s = int(rng.choice(starts, p=start_affinity))
                trails.append(self._walk(s, c, rng))
            detail[c] = trails
            trail_scores = np.asarray([t.score for t in trails])
            lost_fraction = float(np.mean([t.lost for t in trails]))
            agreement = 1.0 / (1.0 + float(np.std(trail_scores)))
            scores[c] = float(
                self.local_evidence_weight * local_evidence
                + self.trail_score_weight * np.mean(trail_scores)
                + self.agreement_weight * agreement
                - self.lost_fraction_weight * lost_fraction
            )
        return scores, detail

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, ["class_field_", "contrast_field_", "nn_"])
        Xs = self._scaled(X)
        rng = np.random.default_rng(self.random_state)
        out = np.empty((len(Xs), self.n_classes_), dtype=float)
        for i, x in enumerate(Xs):
            out[i], _ = self._score_query(x, rng)
        return out

    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = self.decision_function(X)
        return self.classes_[np.argmax(scores, axis=1)]

    def predict_with_diagnostics(self, X: np.ndarray):
        """Return predictions plus per-query class scores and trail diagnostics."""
        check_is_fitted(self, ["class_field_", "contrast_field_", "nn_"])
        Xs = self._scaled(X)
        rng = np.random.default_rng(self.random_state)
        predictions = []
        diagnostics = []
        for x in Xs:
            scores, detail = self._score_query(x, rng)
            winner = int(np.argmax(scores))
            predictions.append(self.classes_[winner])
            diagnostics.append(
                {
                    "class_scores": {
                        self.classes_[c].item() if hasattr(self.classes_[c], "item") else self.classes_[c]: float(scores[c])
                        for c in range(self.n_classes_)
                    },
                    "winner": self.classes_[winner].item() if hasattr(self.classes_[winner], "item") else self.classes_[winner],
                    "trails": {
                        self.classes_[c].item() if hasattr(self.classes_[c], "item") else self.classes_[c]: [t.__dict__ for t in detail[c]]
                        for c in range(self.n_classes_)
                    },
                }
            )
        return np.asarray(predictions), diagnostics
