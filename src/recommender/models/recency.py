"""Recency baselines, including a language-match variant."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.features import Catalog, UserHistory, to_epoch_seconds
from .base import Recommender


class RecencyRecommender(Recommender):
    name = "recency"

    def __init__(self, half_life_days: float | None = None) -> None:
        self.half_life_days = half_life_days
        self.catalog: Catalog | None = None

    def fit(self, events: pd.DataFrame, catalog: Catalog) -> "RecencyRecommender":
        self.catalog = catalog
        return self

    def score(self, user_id: int, candidate_ids: np.ndarray, timestamp: pd.Timestamp | int | float) -> np.ndarray:
        if self.catalog is None:
            raise RuntimeError("Model is not fitted")
        indices = self.catalog.indices(candidate_ids)
        if self.half_life_days is None:
            return self.catalog.share_ts[indices].astype("float64")
        epoch = to_epoch_seconds(timestamp)
        age_days = np.maximum(epoch - self.catalog.share_ts[indices], 0) / 86400.0
        return np.exp2(-age_days / float(self.half_life_days))


class RecencyLanguageRecommender(Recommender):
    name = "recency_language"

    def __init__(self, half_life_days: float = 30.0, language_weight: float = 1.0) -> None:
        self.half_life_days = float(half_life_days)
        self.language_weight = float(language_weight)
        self.catalog: Catalog | None = None
        self.history: UserHistory | None = None

    def fit(self, events: pd.DataFrame, catalog: Catalog) -> "RecencyLanguageRecommender":
        self.catalog = catalog
        self.history = UserHistory(events, catalog)
        return self

    def score(self, user_id: int, candidate_ids: np.ndarray, timestamp: pd.Timestamp | int | float) -> np.ndarray:
        if self.catalog is None or self.history is None:
            raise RuntimeError("Model is not fitted")
        indices = self.catalog.indices(candidate_ids)
        epoch = to_epoch_seconds(timestamp)
        age_days = np.maximum(epoch - self.catalog.share_ts[indices], 0) / 86400.0
        recency = np.exp2(-age_days / self.half_life_days)
        language = self.history.preferred_language(user_id)
        match = self.catalog.language_match(indices, language)
        return recency * (1.0 + self.language_weight * match)
