"""Random recommendation baseline."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.features import Catalog, to_epoch_seconds
from .base import Recommender


class RandomRecommender(Recommender):
    name = "random"

    def __init__(self, seed: int = 42) -> None:
        self.seed = int(seed)
        self.catalog: Catalog | None = None

    def fit(self, events: pd.DataFrame, catalog: Catalog) -> "RandomRecommender":
        self.catalog = catalog
        return self

    def score(self, user_id: int, candidate_ids: np.ndarray, timestamp: pd.Timestamp | int | float) -> np.ndarray:
        epoch = to_epoch_seconds(timestamp)
        seed = (self.seed + int(user_id) * 1_000_003 + epoch) % (2**32)
        rng = np.random.default_rng(seed)
        return rng.random(len(candidate_ids))
