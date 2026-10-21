"""Cold/warm rank-fusion hybrid."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..data.features import Catalog, UserHistory
from .base import Recommender
from .collaborative import CollaborativeRecommender
from .content_based import ContentBasedRecommender
from .popularity import TimeDecayedPopularityRecommender
from .recency import RecencyRecommender


def _rank_percentile(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype="float64")
    n = scores.size
    if n <= 1:
        return np.zeros(n, dtype="float64")
    if np.allclose(scores, scores[0]):
        return np.zeros(n, dtype="float64")
    order = np.argsort(scores, kind="stable")
    ranks = np.empty(n, dtype="float64")
    ranks[order] = np.arange(n, dtype="float64")
    return ranks / float(n - 1)


class HybridRecommender(Recommender):
    name = "hybrid"

    def __init__(
        self,
        components: dict[str, Recommender] | None = None,
        cold_weights: dict[str, float] | None = None,
        warm_weights: dict[str, float] | None = None,
        normalization: str = "rank",
        affinity_half_life_days: float = 30.0,
        use_metadata: bool = True,
        cold_bonus: float = 0.0,
    ) -> None:
        self.components = components or {
            "recency": RecencyRecommender(),
            "popularity": TimeDecayedPopularityRecommender(half_life_days=7.0),
            "content": ContentBasedRecommender(),
            "collaborative": CollaborativeRecommender(),
        }
        self.cold_weights = cold_weights or {
            "recency": 1.0,
            "popularity": 0.5,
            "content": 1.0,
            "language": 0.2,
        }
        self.warm_weights = warm_weights or {
            "recency": 0.5,
            "popularity": 0.5,
            "content": 1.0,
            "collaborative": 1.0,
            "language": 0.2,
            "author": 0.1,
            "source": 0.1,
            "region": 0.0,
        }
        self.normalization = normalization
        self.affinity_half_life_days = float(affinity_half_life_days)
        self.use_metadata = bool(use_metadata)
        self.cold_bonus = float(cold_bonus)
        self.catalog: Catalog | None = None
        self.history: UserHistory | None = None
        self.warm_items: set[int] = set()
        self.signals = sorted(set(self.cold_weights) | set(self.warm_weights))

    def fit(self, events: pd.DataFrame, catalog: Catalog) -> "HybridRecommender":
        self.catalog = catalog
        self.history = UserHistory(events, catalog)
        self.warm_items = set(int(item) for item in events["contentId"].unique())
        for component in self.components.values():
            component.fit(events, catalog)
        return self

    def _normalize(self, scores: np.ndarray) -> np.ndarray:
        if self.normalization == "zscore":
            scores = np.asarray(scores, dtype="float64")
            std = scores.std()
            if std <= 1e-12:
                return np.zeros_like(scores)
            return (scores - scores.mean()) / std
        return _rank_percentile(scores)

    def _metadata_signal(
        self,
        user_id: int,
        candidate_indices: np.ndarray,
        timestamp: pd.Timestamp | int | float,
        field: str,
    ) -> np.ndarray:
        if self.catalog is None or self.history is None:
            return np.zeros(len(candidate_indices), dtype="float64")
        events = self.history.user_events(user_id)
        if events is None or events.empty:
            return np.zeros(len(candidate_indices), dtype="float64")
        history_indices = self.catalog.indices(events["contentId"].to_numpy())
        if field == "author":
            history_values = self.catalog.author[history_indices]
            candidate_values = self.catalog.author[candidate_indices]
        elif field == "source":
            history_values = self.catalog.source[history_indices]
            candidate_values = self.catalog.source[candidate_indices]
        else:
            raise ValueError(f"Unknown affinity field: {field}")
        totals = self.history.affinity(
            user_id,
            history_values,
            timestamp,
            self.affinity_half_life_days,
        )
        return np.asarray([totals.get(str(value), 0.0) for value in candidate_values], dtype="float64")

    def score(self, user_id: int, candidate_ids: np.ndarray, timestamp: pd.Timestamp | int | float) -> np.ndarray:
        if self.catalog is None or self.history is None:
            raise RuntimeError("Model is not fitted")
        candidate_indices = self.catalog.indices(candidate_ids)
        signal_values: dict[str, np.ndarray] = {}
        for name, component in self.components.items():
            signal_values[name] = self._normalize(component.score(user_id, candidate_ids, timestamp))

        if self.use_metadata:
            language = self.history.preferred_language(user_id)
            region = self.history.preferred_region(user_id)
            signal_values["language"] = self.catalog.language_match(candidate_indices, language)
            signal_values["region"] = self.catalog.region_match(candidate_indices, region)
            signal_values["author"] = self._normalize(
                self._metadata_signal(user_id, candidate_indices, timestamp, "author")
            )
            signal_values["source"] = self._normalize(
                self._metadata_signal(user_id, candidate_indices, timestamp, "source")
            )

        n_candidates = len(candidate_ids)
        cold_mask = np.asarray([int(item) not in self.warm_items for item in candidate_ids])
        output = np.zeros(n_candidates, dtype="float64")
        for signal, values in signal_values.items():
            cold_weight = float(self.cold_weights.get(signal, 0.0))
            warm_weight = float(self.warm_weights.get(signal, 0.0))
            weights = np.where(cold_mask, cold_weight, warm_weight)
            output += weights * np.asarray(values, dtype="float64")
        if self.cold_bonus:
            output += np.where(cold_mask, self.cold_bonus, 0.0)
        return output

    def get_params(self) -> dict[str, Any]:
        return {
            "cold_weights": self.cold_weights,
            "warm_weights": self.warm_weights,
            "normalization": self.normalization,
            "use_metadata": self.use_metadata,
            "cold_bonus": self.cold_bonus,
        }
