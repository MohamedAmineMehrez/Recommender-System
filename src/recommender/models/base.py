"""Common recommender interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterable

import numpy as np
import pandas as pd

from ..data.features import Catalog


class Recommender(ABC):
    name: str = "recommender"

    @abstractmethod
    def fit(self, events: pd.DataFrame, catalog: Catalog) -> "Recommender":
        raise NotImplementedError

    @abstractmethod
    def score(
        self,
        user_id: int,
        candidate_ids: np.ndarray,
        timestamp: pd.Timestamp | int | float,
    ) -> np.ndarray:
        raise NotImplementedError

    def recommend(
        self,
        user_id: int,
        k: int,
        timestamp: pd.Timestamp | int | float,
        candidate_ids: Iterable[int] | None = None,
        exclude: Iterable[int] | None = None,
        catalog: Catalog | None = None,
    ) -> np.ndarray:
        if candidate_ids is None:
            if catalog is None:
                raise ValueError("candidate_ids or catalog is required")
            candidate_ids = catalog.available_ids(timestamp)
        candidates = np.asarray(list(candidate_ids), dtype="int64")
        if exclude is not None:
            excluded = set(int(item) for item in exclude)
            candidates = np.asarray([item for item in candidates if int(item) not in excluded], dtype="int64")
        if candidates.size == 0:
            return candidates
        scores = np.asarray(self.score(user_id, candidates, timestamp), dtype="float64")
        scores = np.nan_to_num(scores, nan=-np.inf, posinf=np.inf, neginf=-np.inf)
        order = np.lexsort((candidates, -scores))
        return candidates[order[:k]]

    def get_params(self) -> dict[str, Any]:
        return {}
