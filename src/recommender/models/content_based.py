"""TF-IDF content-based recommender with recency-weighted user profiles."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.preprocessing import normalize

from ..data.features import Catalog, UserHistory
from .base import Recommender

try:  # pragma: no cover - depends on local NLTK data
    from nltk.corpus import stopwords as _nltk_stopwords

    _PORTUGUESE_STOPWORDS = set(_nltk_stopwords.words("portuguese"))
except Exception:  # pragma: no cover
    _PORTUGUESE_STOPWORDS = {
        "a", "à", "ao", "aos", "as", "às", "um", "uma", "uns", "umas", "de", "do", "da",
        "dos", "das", "em", "no", "na", "nos", "nas", "por", "para", "com", "sem", "que",
        "e", "ou", "é", "são", "foi", "ser", "se", "não", "mais", "como", "mas", "seu",
        "sua", "seus", "suas", "este", "esta", "isso", "isto", "ele", "ela", "eles", "elas",
    }

STOPWORDS = sorted(set(ENGLISH_STOP_WORDS) | _PORTUGUESE_STOPWORDS)


class ContentBasedRecommender(Recommender):
    name = "content"

    def __init__(
        self,
        half_life_days: float = 30.0,
        word_max_features: int = 20000,
        char_max_features: int = 10000,
        min_df: int = 3,
        max_df: float = 0.5,
        use_char: bool = True,
        use_word: bool = True,
        profile_aggregation: str = "decayed",
    ) -> None:
        self.half_life_days = float(half_life_days)
        self.word_max_features = int(word_max_features)
        self.char_max_features = int(char_max_features)
        self.min_df = int(min_df)
        self.max_df = float(max_df)
        self.use_char = bool(use_char)
        self.use_word = bool(use_word)
        self.profile_aggregation = profile_aggregation
        self.catalog: Catalog | None = None
        self.history: UserHistory | None = None
        self.item_vectors: sparse.csr_matrix | None = None
        self.word_vectorizer: TfidfVectorizer | None = None
        self.char_vectorizer: TfidfVectorizer | None = None
        self.reference_time: int | None = None
        self._profiles: dict[tuple[int, int], sparse.csr_matrix] = {}

    def fit(self, events: pd.DataFrame, catalog: Catalog) -> "ContentBasedRecommender":
        self.catalog = catalog
        self.history = UserHistory(events, catalog)
        self.reference_time = int(events["timestamp"].max()) if len(events) else 0
        train_mask = catalog.share_ts <= self.reference_time
        train_text = catalog.text_all[train_mask]
        all_text = catalog.text_all

        matrices: list[sparse.csr_matrix] = []
        if self.use_word:
            self.word_vectorizer = TfidfVectorizer(
                analyzer="word",
                ngram_range=(1, 2),
                min_df=self.min_df,
                max_df=self.max_df,
                max_features=self.word_max_features,
                sublinear_tf=True,
                stop_words=STOPWORDS,
                strip_accents="unicode",
                dtype=np.float32,
            )
            self.word_vectorizer.fit(train_text)
            matrices.append(self.word_vectorizer.transform(all_text).tocsr())
        if self.use_char:
            self.char_vectorizer = TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 5),
                min_df=self.min_df,
                max_df=self.max_df,
                max_features=self.char_max_features,
                sublinear_tf=True,
                strip_accents="unicode",
                dtype=np.float32,
            )
            self.char_vectorizer.fit(train_text)
            matrices.append(self.char_vectorizer.transform(all_text).tocsr())

        if not matrices:
            raise ValueError("At least one of use_word or use_char must be enabled")
        item_vectors = sparse.hstack(matrices, format="csr") if len(matrices) > 1 else matrices[0]
        self.item_vectors = normalize(item_vectors, norm="l2", axis=1, copy=False)
        self._profiles.clear()
        return self

    def _user_profile(self, user_id: int, timestamp: pd.Timestamp | int | float) -> sparse.csr_matrix:
        if self.catalog is None or self.history is None or self.item_vectors is None:
            raise RuntimeError("Model is not fitted")
        cache_key = (int(user_id), int(pd.Timestamp(timestamp).timestamp()) // 86400)
        if cache_key in self._profiles:
            return self._profiles[cache_key]
        indices, weights = self.history.weighted_item_weights(
            user_id,
            timestamp,
            self.half_life_days if self.profile_aggregation == "decayed" else 0.0,
        )
        if indices.size == 0:
            profile = sparse.csr_matrix((1, self.item_vectors.shape[1]), dtype=np.float32)
        else:
            aggregated: dict[int, float] = {}
            for index, weight in zip(indices.tolist(), weights.tolist()):
                aggregated[index] = aggregated.get(index, 0.0) + float(weight)
            ordered = sorted(aggregated.items())
            row_indices = np.asarray([item for item, _ in ordered], dtype="int64")
            row_weights = np.asarray([weight for _, weight in ordered], dtype="float32")
            sub = self.item_vectors[row_indices]
            weighted = sub.multiply(row_weights[:, None])
            profile = sparse.csr_matrix(np.asarray(weighted.sum(axis=0)))
            profile = normalize(profile, norm="l2", axis=1, copy=False)
        self._profiles[cache_key] = profile
        return profile

    def score(self, user_id: int, candidate_ids: np.ndarray, timestamp: pd.Timestamp | int | float) -> np.ndarray:
        if self.catalog is None or self.item_vectors is None:
            raise RuntimeError("Model is not fitted")
        profile = self._user_profile(user_id, timestamp)
        if profile.nnz == 0:
            return np.zeros(len(candidate_ids), dtype="float64")
        indices = self.catalog.indices(candidate_ids)
        scores = profile.dot(self.item_vectors[indices].T)
        return np.asarray(scores.toarray()).ravel().astype("float64")

    def get_params(self) -> dict[str, Any]:
        return {
            "half_life_days": self.half_life_days,
            "word_max_features": self.word_max_features,
            "char_max_features": self.char_max_features,
            "min_df": self.min_df,
            "max_df": self.max_df,
            "use_char": self.use_char,
            "use_word": self.use_word,
        }
