"""Global and time-decayed popularity baselines."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.features import Catalog, to_epoch_seconds
from .base import Recommender


class GlobalPopularityRecommender(Recommender):
    name = "popularity"

    def __init__(self) -> None:
        self.item_scores = np.zeros(0, dtype="float64")
        self.catalog: Catalog | None = None

    def fit(self, events: pd.DataFrame, catalog: Catalog) -> "GlobalPopularityRecommender":
        self.catalog = catalog
        scores = np.zeros(catalog.n_items, dtype="float64")
        counts = events.groupby("contentId")["weight"].sum()
        for item_id, value in counts.items():
            index = catalog.id_to_index.get(int(item_id))
            if index is not None:
                scores[index] = float(value)
        self.item_scores = scores
        return self

    def score(self, user_id: int, candidate_ids: np.ndarray, timestamp: pd.Timestamp | int | float) -> np.ndarray:
        if self.catalog is None:
            raise RuntimeError("Model is not fitted")
        return self.item_scores[self.catalog.indices(candidate_ids)]


class TimeDecayedPopularityRecommender(Recommender):
    name = "time_decayed_popularity"

    def __init__(self, half_life_days: float = 7.0) -> None:
        self.half_life_days = float(half_life_days)
        self.item_scores = np.zeros(0, dtype="float64")
        self.catalog: Catalog | None = None
        self.reference_time: int | None = None

    def fit(self, events: pd.DataFrame, catalog: Catalog) -> "TimeDecayedPopularityRecommender":
        self.catalog = catalog
        scores = np.zeros(catalog.n_items, dtype="float64")
        if events.empty:
            self.item_scores = scores
            return self
        reference = events["timestamp"].max()
        self.reference_time = int(reference)
        half_life_seconds = max(self.half_life_days * 86400.0, 1.0)
        ages = np.maximum(reference - events["timestamp"].to_numpy(dtype="int64"), 0) / half_life_seconds
        decayed = events["weight"].to_numpy(dtype="float64") * np.exp2(-ages)
        counts = pd.Series(decayed).groupby(events["contentId"].to_numpy()).sum()
        for item_id, value in counts.items():
            index = catalog.id_to_index.get(int(item_id))
            if index is not None:
                scores[index] = float(value)
        self.item_scores = scores
        return self

    def score(self, user_id: int, candidate_ids: np.ndarray, timestamp: pd.Timestamp | int | float) -> np.ndarray:
        if self.catalog is None:
            raise RuntimeError("Model is not fitted")
        return self.item_scores[self.catalog.indices(candidate_ids)]
