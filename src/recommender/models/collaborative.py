"""Implicit-feedback collaborative filtering models."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import normalize

from ..data.features import Catalog, UserHistory
from .base import Recommender

try:
    from implicit.als import AlternatingLeastSquares
    from implicit.bpr import BayesianPersonalizedRanking

    # threadpoolctl can fail to introspect some Windows BLAS installs. The
    # check only prints a warning, so bypass it to keep ALS usable here.
    try:
        import implicit.cpu.als as _implicit_cpu_als

        _implicit_cpu_als.check_blas_config = lambda: None
    except Exception:  # pragma: no cover
        pass
except Exception:  # pragma: no cover
    AlternatingLeastSquares = None
    BayesianPersonalizedRanking = None


def _build_user_item_matrix(
    events: pd.DataFrame,
    user_index: dict[int, int],
    item_index: dict[int, int],
    alpha: float,
) -> sparse.csr_matrix:
    aggregated = events.groupby(["personId", "contentId"])["weight"].sum().reset_index()
    rows = aggregated["personId"].map(user_index).to_numpy(dtype="int64")
    cols = aggregated["contentId"].map(item_index).to_numpy(dtype="int64")
    weights = aggregated["weight"].to_numpy(dtype="float64")
    confidence = 1.0 + alpha * np.log1p(weights)
    return sparse.csr_matrix(
        (confidence.astype(np.float32), (rows, cols)),
        shape=(len(user_index), len(item_index)),
    )


class CollaborativeRecommender(Recommender):
    name = "collaborative"

    def __init__(
        self,
        factors: int = 64,
        regularization: float = 0.1,
        alpha: float = 40.0,
        iterations: int = 20,
        model_type: str = "als",
        seed: int = 42,
    ) -> None:
        self.factors = int(factors)
        self.regularization = float(regularization)
        self.alpha = float(alpha)
        self.iterations = int(iterations)
        self.model_type = model_type
        self.seed = int(seed)
        self.catalog: Catalog | None = None
        self.history: UserHistory | None = None
        self.user_ids = np.empty(0, dtype="int64")
        self.item_ids = np.empty(0, dtype="int64")
        self.user_index: dict[int, int] = {}
        self.item_index: dict[int, int] = {}
        self.model: Any = None
        self.user_factors: np.ndarray | None = None
        self.item_factors: np.ndarray | None = None

    def fit(self, events: pd.DataFrame, catalog: Catalog) -> "CollaborativeRecommender":
        if AlternatingLeastSquares is None:
            raise ImportError("implicit is required for collaborative filtering")
        self.catalog = catalog
        self.history = UserHistory(events, catalog)
        self.user_ids = np.sort(events["personId"].unique().astype("int64"))
        self.item_ids = np.sort(events["contentId"].unique().astype("int64"))
        self.user_index = {int(user): index for index, user in enumerate(self.user_ids)}
        self.item_index = {int(item): index for index, item in enumerate(self.item_ids)}
        matrix = _build_user_item_matrix(events, self.user_index, self.item_index, self.alpha)

        if self.model_type == "bpr":
            self.model = BayesianPersonalizedRanking(
                factors=self.factors,
                learning_rate=0.05,
                regularization=self.regularization,
                iterations=self.iterations,
                random_state=self.seed,
            )
            binary = matrix.copy()
            binary.data = np.ones_like(binary.data, dtype=np.float32)
            self.model.fit(binary, show_progress=False)
        else:
            self.model = AlternatingLeastSquares(
                factors=self.factors,
                regularization=self.regularization,
                iterations=self.iterations,
                random_state=self.seed,
                use_gpu=False,
            )
            self.model.fit(matrix, show_progress=False)
        self.user_factors = np.asarray(self.model.user_factors, dtype="float64")
        self.item_factors = np.asarray(self.model.item_factors, dtype="float64")
        return self

    def score(self, user_id: int, candidate_ids: np.ndarray, timestamp: pd.Timestamp | int | float) -> np.ndarray:
        if self.catalog is None or self.user_factors is None or self.item_factors is None:
            raise RuntimeError("Model is not fitted")
        scores = np.zeros(len(candidate_ids), dtype="float64")
        user_position = self.user_index.get(int(user_id))
        if user_position is None:
            return scores
        item_positions = np.fromiter(
            (self.item_index.get(int(item), -1) for item in candidate_ids),
            dtype="int64",
            count=len(candidate_ids),
        )
        valid = item_positions >= 0
        if valid.any():
            scores[valid] = self.item_factors[item_positions[valid]].dot(self.user_factors[user_position])
        return scores

    def get_params(self) -> dict[str, Any]:
        return {
            "factors": self.factors,
            "regularization": self.regularization,
            "alpha": self.alpha,
            "iterations": self.iterations,
            "model_type": self.model_type,
        }


class ItemItemCFRecommender(Recommender):
    name = "item_item_cf"

    def __init__(self, top_neighbors: int | None = None) -> None:
        self.top_neighbors = top_neighbors
        self.catalog: Catalog | None = None
        self.history: UserHistory | None = None
        self.item_similarity: sparse.csr_matrix | None = None

    def fit(self, events: pd.DataFrame, catalog: Catalog) -> "ItemItemCFRecommender":
        self.catalog = catalog
        self.history = UserHistory(events, catalog)
        aggregated = events.groupby(["contentId", "personId"])["weight"].sum().reset_index()
        item_ids = np.sort(aggregated["contentId"].unique().astype("int64"))
        user_ids = np.sort(aggregated["personId"].unique().astype("int64"))
        item_index = {int(item): index for index, item in enumerate(item_ids)}
        user_index = {int(user): index for index, user in enumerate(user_ids)}
        rows = aggregated["contentId"].map(item_index).to_numpy(dtype="int64")
        cols = aggregated["personId"].map(user_index).to_numpy(dtype="int64")
        values = np.log1p(aggregated["weight"].to_numpy(dtype="float64")).astype(np.float32)
        item_user = sparse.csr_matrix(
            (values, (rows, cols)),
            shape=(len(item_ids), len(user_ids)),
        )
        item_user = normalize(item_user, norm="l2", axis=1)
        similarity = (item_user @ item_user.T).tocsr()
        similarity.setdiag(0.0)
        similarity.eliminate_zeros()

        catalog_positions = np.asarray([catalog.id_to_index[int(item)] for item in item_ids], dtype="int64")
        sub = similarity.tocoo()
        catalog_similarity = sparse.csr_matrix(
            (
                sub.data.astype(np.float32),
                (catalog_positions[sub.row], catalog_positions[sub.col]),
            ),
            shape=(catalog.n_items, catalog.n_items),
        )
        self.item_similarity = catalog_similarity
        return self

    def score(self, user_id: int, candidate_ids: np.ndarray, timestamp: pd.Timestamp | int | float) -> np.ndarray:
        if self.catalog is None or self.history is None or self.item_similarity is None:
            raise RuntimeError("Model is not fitted")
        history_items = self.history.seen_before(user_id, timestamp)
        if history_items.size == 0:
            return np.zeros(len(candidate_ids), dtype="float64")
        candidate_indices = self.catalog.indices(candidate_ids)
        history_indices = np.unique(self.catalog.indices(history_items))
        scores = self.item_similarity[candidate_indices][:, history_indices].sum(axis=1)
        return np.asarray(scores).ravel().astype("float64")

    def get_params(self) -> dict[str, Any]:
        return {"top_neighbors": self.top_neighbors}
