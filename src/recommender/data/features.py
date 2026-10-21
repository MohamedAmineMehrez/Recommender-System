"""Fast catalog and user-history accessors used by models and the evaluator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


def to_epoch_seconds(timestamp: pd.Timestamp | int | float) -> int:
    if isinstance(timestamp, (int, np.integer)):
        return int(timestamp)
    if isinstance(timestamp, (float, np.floating)):
        return int(timestamp)
    ts = pd.Timestamp(timestamp)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return int(ts.timestamp())


class Catalog:
    """Immutable item metadata with fast time-aware availability lookups."""

    def __init__(self, articles: pd.DataFrame):
        self.articles = articles.reset_index(drop=True).copy()
        self.item_ids = self.articles["contentId"].to_numpy(dtype="int64")
        self.n_items = len(self.item_ids)
        self.id_to_index = {int(item): index for index, item in enumerate(self.item_ids)}
        self.index_to_id = self.item_ids.copy()
        self.share_ts = self.articles["share_ts"].to_numpy(dtype="int64")
        self.share_timestamp = self.articles["share_timestamp"].to_numpy()
        self.lang = self.articles["lang"].astype(str).to_numpy()
        self.author = self.articles["author_id"].astype(str).to_numpy()
        self.source = self.articles["source_domain"].astype(str).to_numpy()
        self.region = self.articles["authorRegion"].astype(str).to_numpy()
        self.country = self.articles["authorCountry"].astype(str).to_numpy()
        self.content_type = self.articles["contentType"].astype(str).to_numpy()
        self.title = self.articles["title"].astype(str).to_numpy()
        self.text_all = self.articles["text_all"].astype(str).to_numpy()

        order = np.argsort(self.share_ts, kind="stable")
        self._share_order = order
        self._sorted_share_ts = self.share_ts[order]
        self._languages = sorted({str(value) for value in self.lang})

    def indices(self, item_ids: Iterable[int] | np.ndarray) -> np.ndarray:
        if isinstance(item_ids, np.ndarray) and item_ids.dtype.kind in "iu":
            values = item_ids
        else:
            values = np.asarray(list(item_ids), dtype="int64")
        return np.fromiter(
            (self.id_to_index[int(item)] for item in values),
            dtype="int64",
            count=len(values),
        )

    def available_ids(self, timestamp: pd.Timestamp | int | float) -> np.ndarray:
        epoch = to_epoch_seconds(timestamp)
        count = int(np.searchsorted(self._sorted_share_ts, epoch, side="right"))
        return self.item_ids[self._share_order[:count]]

    def available_mask(self, timestamp: pd.Timestamp | int | float) -> np.ndarray:
        epoch = to_epoch_seconds(timestamp)
        return self.share_ts <= epoch

    def age_days(self, indices: np.ndarray, timestamp: pd.Timestamp | int | float) -> np.ndarray:
        epoch = to_epoch_seconds(timestamp)
        return (epoch - self.share_ts[indices]) / 86400.0

    def language_match(self, indices: np.ndarray, language: str) -> np.ndarray:
        return (self.lang[indices] == language).astype("float64")

    def region_match(self, indices: np.ndarray, region: str) -> np.ndarray:
        return (self.region[indices] == region).astype("float64")

    def country_match(self, indices: np.ndarray, country: str) -> np.ndarray:
        return (self.country[indices] == country).astype("float64")


class UserHistory:
    """Per-user event history, restricted to the events supplied at fit time."""

    def __init__(self, interactions: pd.DataFrame, catalog: Catalog):
        self.catalog = catalog
        events = interactions.sort_values(["personId", "timestamp"], kind="stable")
        self.events = events
        self._by_user: dict[int, pd.DataFrame] = {
            int(user_id): group for user_id, group in events.groupby("personId", sort=False)
        }

    def users(self) -> np.ndarray:
        return np.asarray(list(self._by_user.keys()), dtype="int64")

    def user_events(self, user_id: int) -> pd.DataFrame | None:
        return self._by_user.get(int(user_id))

    def seen_before(self, user_id: int, timestamp: pd.Timestamp | int | float) -> np.ndarray:
        events = self._by_user.get(int(user_id))
        if events is None or events.empty:
            return np.empty(0, dtype="int64")
        epoch = to_epoch_seconds(timestamp)
        times = events["timestamp"].to_numpy(dtype="int64")
        count = int(np.searchsorted(times, epoch, side="left"))
        return events["contentId"].to_numpy(dtype="int64")[:count]

    def last_interaction(self, user_id: int) -> int | None:
        events = self._by_user.get(int(user_id))
        if events is None or events.empty:
            return None
        return int(events["timestamp"].max())

    def preferred_language(self, user_id: int) -> str:
        events = self._by_user.get(int(user_id))
        if events is None or events.empty:
            return "unknown"
        item_indices = self.catalog.indices(events["contentId"].to_numpy())
        languages = self.catalog.lang[item_indices]
        weights = events["weight"].to_numpy(dtype="float64")
        totals: dict[str, float] = {}
        for language, weight in zip(languages, weights):
            totals[str(language)] = totals.get(str(language), 0.0) + float(weight)
        return max(totals.items(), key=lambda pair: pair[1])[0] if totals else "unknown"

    def preferred_region(self, user_id: int) -> str:
        events = self._by_user.get(int(user_id))
        if events is None or events.empty:
            return "unknown"
        values = events["userRegion"].fillna("unknown").astype(str).to_numpy()
        weights = events["weight"].to_numpy(dtype="float64")
        totals: dict[str, float] = {}
        for value, weight in zip(values, weights):
            totals[value] = totals.get(value, 0.0) + float(weight)
        return max(totals.items(), key=lambda pair: pair[1])[0] if totals else "unknown"

    def preferred_country(self, user_id: int) -> str:
        events = self._by_user.get(int(user_id))
        if events is None or events.empty:
            return "unknown"
        values = events["userCountry"].fillna("unknown").astype(str).to_numpy()
        weights = events["weight"].to_numpy(dtype="float64")
        totals: dict[str, float] = {}
        for value, weight in zip(values, weights):
            totals[value] = totals.get(value, 0.0) + float(weight)
        return max(totals.items(), key=lambda pair: pair[1])[0] if totals else "unknown"

    def interaction_stats(self, user_id: int) -> dict[str, float]:
        events = self._by_user.get(int(user_id))
        if events is None or events.empty:
            return {"count": 0.0, "distinct_items": 0.0, "mean_weight": 0.0}
        return {
            "count": float(len(events)),
            "distinct_items": float(events["contentId"].nunique()),
            "mean_weight": float(events["weight"].mean()),
        }

    def weighted_item_weights(
        self,
        user_id: int,
        reference_time: pd.Timestamp | int | float,
        half_life_days: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return catalog indices and decayed weights for a user profile."""
        events = self._by_user.get(int(user_id))
        if events is None or events.empty:
            return np.empty(0, dtype="int64"), np.empty(0, dtype="float64")
        reference_epoch = to_epoch_seconds(reference_time)
        ages_days = np.maximum(reference_epoch - events["timestamp"].to_numpy(dtype="int64"), 0) / 86400.0
        if half_life_days and half_life_days > 0:
            decay = np.exp2(-ages_days / float(half_life_days))
        else:
            decay = np.ones_like(ages_days)
        weights = events["weight"].to_numpy(dtype="float64") * decay
        indices = self.catalog.indices(events["contentId"].to_numpy())
        return indices, weights

    def affinity(
        self,
        user_id: int,
        values: np.ndarray,
        reference_time: pd.Timestamp | int | float,
        half_life_days: float,
    ) -> dict[str, float]:
        events = self._by_user.get(int(user_id))
        if events is None or events.empty:
            return {}
        reference_epoch = to_epoch_seconds(reference_time)
        ages_days = np.maximum(reference_epoch - events["timestamp"].to_numpy(dtype="int64"), 0) / 86400.0
        decay = np.exp2(-ages_days / float(half_life_days)) if half_life_days > 0 else np.ones_like(ages_days)
        weights = events["weight"].to_numpy(dtype="float64") * decay
        totals: dict[str, float] = {}
        for value, weight in zip(values, weights):
            key = str(value)
            totals[key] = totals.get(key, 0.0) + float(weight)
        return totals
