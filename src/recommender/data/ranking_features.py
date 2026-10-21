"""Feature construction for the learning-to-rank stage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..models.hybrid import _rank_percentile
from .features import Catalog, UserHistory, to_epoch_seconds

FEATURE_NAMES: list[str] = [
    "user_count",
    "user_distinct",
    "user_mean_weight",
    "user_days_since_last",
    "user_span_days",
    "user_lang_code",
    "user_region_code",
    "user_country_code",
    "item_age_days",
    "item_share_hour",
    "item_share_weekday",
    "item_title_len",
    "item_text_len",
    "item_lang_code",
    "item_content_type_code",
    "item_is_cold",
    "item_count",
    "item_users",
    "item_weight",
    "lang_match",
    "region_match",
    "country_match",
    "recency_pct",
    "recency_age_days",
    "pop_0.25d_pct",
    "pop_0.5d_pct",
    "pop_1d_pct",
    "pop_3d_pct",
    "pop_7d_pct",
    "pop_all_pct",
    "pop_lang_7d_pct",
    "pop_source_7d_pct",
    "pop_region_7d_pct",
    "content_pct",
    "cf_pct",
    "itemitem_pct",
    "author_affinity",
    "author_affinity_pct",
    "author_seen",
    "author_days_since",
    "source_affinity",
    "source_affinity_pct",
    "source_seen",
    "source_days_since",
]

CATEGORICAL_FEATURES = [
    "user_lang_code",
    "user_region_code",
    "user_country_code",
    "item_lang_code",
    "item_content_type_code",
]

_SIGNAL_TO_FEATURE = {
    "recency": "recency_pct",
    "pop_0.25d": "pop_0.25d_pct",
    "pop_0.5d": "pop_0.5d_pct",
    "pop_1d": "pop_1d_pct",
    "pop_3d": "pop_3d_pct",
    "pop_7d": "pop_7d_pct",
    "pop_all": "pop_all_pct",
    "content": "content_pct",
    "cf": "cf_pct",
    "itemitem": "itemitem_pct",
}


@dataclass
class UserStats:
    count: int
    distinct: int
    mean_weight: float
    first_ts: int
    last_ts: int
    lang_code: int
    region_code: int
    country_code: int


class RankingFeatureBuilder:
    def __init__(
        self,
        catalog: Catalog,
        history_events: pd.DataFrame,
        models: dict[str, Any],
        affinity_half_life_days: float = 30.0,
    ) -> None:
        self.catalog = catalog
        self.events = history_events.sort_values("timestamp", kind="stable")
        self.models = models
        self.history = UserHistory(self.events, catalog)
        self.reference_time = int(self.events["timestamp"].max()) if len(self.events) else 0
        self.affinity_half_life_days = float(affinity_half_life_days)
        self._build_item_features()
        self._build_user_features()
        self._build_group_popularity()

    def _build_item_features(self) -> None:
        n = self.catalog.n_items
        self.item_counts = np.zeros(n, dtype="float64")
        self.item_users = np.zeros(n, dtype="float64")
        self.item_weight = np.zeros(n, dtype="float64")
        if len(self.events):
            grouped = self.events.groupby("contentId").agg(
                count=("weight", "size"),
                users=("personId", "nunique"),
                weight=("weight", "sum"),
            )
            for item_id, row in grouped.iterrows():
                index = self.catalog.id_to_index.get(int(item_id))
                if index is not None:
                    self.item_counts[index] = float(row["count"])
                    self.item_users[index] = float(row["users"])
                    self.item_weight[index] = float(row["weight"])
        self.item_is_cold = self.item_counts == 0

        share = pd.to_datetime(self.catalog.share_ts, unit="s", utc=True)
        self.item_share_hour = share.hour.to_numpy(dtype="float32")
        self.item_share_weekday = share.weekday.to_numpy(dtype="float32")
        self.item_title_len = np.log1p(
            np.asarray([len(value) for value in self.catalog.title], dtype="float32")
        )
        self.item_text_len = np.log1p(
            np.asarray([len(value) for value in self.catalog.text_all], dtype="float32")
        )
        languages = sorted(set(self.catalog.lang.tolist()))
        content_types = sorted(set(self.catalog.content_type.tolist()))
        self.lang_codes = {value: index for index, value in enumerate(languages)}
        self.content_type_codes = {value: index for index, value in enumerate(content_types)}
        self.item_lang_code = np.asarray(
            [self.lang_codes.get(value, 0) for value in self.catalog.lang], dtype="float32"
        )
        self.item_content_type_code = np.asarray(
            [self.content_type_codes.get(value, 0) for value in self.catalog.content_type],
            dtype="float32",
        )

    def _build_user_features(self) -> None:
        self.user_stats: dict[int, UserStats] = {}
        self.user_author_totals: dict[int, dict[str, float]] = {}
        self.user_source_totals: dict[int, dict[str, float]] = {}
        self.user_last_author: dict[int, dict[str, int]] = {}
        self.user_last_source: dict[int, dict[str, int]] = {}
        if self.events.empty:
            self.user_lang_codes: dict[str, int] = {}
            self.user_region_codes: dict[str, int] = {}
            self.user_country_codes: dict[str, int] = {}
            return

        grouped = self.events.groupby("personId").agg(
            count=("weight", "size"),
            distinct=("contentId", "nunique"),
            mean_weight=("weight", "mean"),
            first_ts=("timestamp", "min"),
            last_ts=("timestamp", "max"),
        )
        languages = sorted(set(self.catalog.lang.tolist()))
        self.user_lang_codes = {value: index for index, value in enumerate(languages)}
        regions = {str(value) for value in self.events["userRegion"].fillna("unknown").unique()}
        countries = {str(value) for value in self.events["userCountry"].fillna("unknown").unique()}
        self.user_region_codes = {value: index for index, value in enumerate(sorted(regions))}
        self.user_country_codes = {value: index for index, value in enumerate(sorted(countries))}

        for user_id, row in grouped.iterrows():
            user_id = int(user_id)
            events = self.events.loc[self.events["personId"] == user_id]
            indices = self.catalog.indices(events["contentId"].to_numpy())
            language = self.history.preferred_language(user_id)
            region = self.history.preferred_region(user_id)
            country = self.history.preferred_country(user_id)
            self.user_stats[user_id] = UserStats(
                count=int(row["count"]),
                distinct=int(row["distinct"]),
                mean_weight=float(row["mean_weight"]),
                first_ts=int(row["first_ts"]),
                last_ts=int(row["last_ts"]),
                lang_code=self.user_lang_codes.get(language, 0),
                region_code=self.user_region_codes.get(region, 0),
                country_code=self.user_country_codes.get(country, 0),
            )
            ages = np.maximum(self.reference_time - events["timestamp"].to_numpy(dtype="int64"), 0) / 86400.0
            weights = events["weight"].to_numpy(dtype="float64") * np.exp2(-ages / self.affinity_half_life_days)
            author_totals: dict[str, float] = {}
            source_totals: dict[str, float] = {}
            last_author: dict[str, int] = {}
            last_source: dict[str, int] = {}
            for author, source, weight, timestamp in zip(
                self.catalog.author[indices],
                self.catalog.source[indices],
                weights,
                events["timestamp"].to_numpy(dtype="int64"),
            ):
                author_totals[str(author)] = author_totals.get(str(author), 0.0) + float(weight)
                source_totals[str(source)] = source_totals.get(str(source), 0.0) + float(weight)
                last_author[str(author)] = max(last_author.get(str(author), 0), int(timestamp))
                last_source[str(source)] = max(last_source.get(str(source), 0), int(timestamp))
            self.user_author_totals[user_id] = author_totals
            self.user_source_totals[user_id] = source_totals
            self.user_last_author[user_id] = last_author
            self.user_last_source[user_id] = last_source

    def _build_group_popularity(self) -> None:
        n = self.catalog.n_items
        self.item_lang_pop = np.zeros(n, dtype="float64")
        self.item_source_pop = np.zeros(n, dtype="float64")
        self.item_region_pop = np.zeros(n, dtype="float64")
        if self.events.empty:
            return
        ages = np.maximum(self.reference_time - self.events["timestamp"].to_numpy(dtype="int64"), 0) / 86400.0
        decayed = self.events["weight"].to_numpy(dtype="float64") * np.exp2(-ages / 7.0)
        indices = self.catalog.indices(self.events["contentId"].to_numpy())
        languages = self.catalog.lang[indices]
        sources = self.catalog.source[indices]
        regions = self.catalog.region[indices]
        lang_totals: dict[str, float] = {}
        source_totals: dict[str, float] = {}
        region_totals: dict[str, float] = {}
        for language, source, region, value in zip(languages, sources, regions, decayed):
            lang_totals[str(language)] = lang_totals.get(str(language), 0.0) + float(value)
            source_totals[str(source)] = source_totals.get(str(source), 0.0) + float(value)
            region_totals[str(region)] = region_totals.get(str(region), 0.0) + float(value)
        for index in range(n):
            self.item_lang_pop[index] = lang_totals.get(str(self.catalog.lang[index]), 0.0)
            self.item_source_pop[index] = source_totals.get(str(self.catalog.source[index]), 0.0)
            self.item_region_pop[index] = region_totals.get(str(self.catalog.region[index]), 0.0)

    def transform(
        self,
        user_id: int,
        candidate_ids: np.ndarray,
        timestamp: pd.Timestamp | int | float,
    ) -> np.ndarray:
        if self.catalog is None:
            raise RuntimeError("Feature builder is not initialized")
        candidate_ids = np.asarray(candidate_ids, dtype="int64")
        n = len(candidate_ids)
        indices = self.catalog.indices(candidate_ids)
        epoch = to_epoch_seconds(timestamp)
        features = np.zeros((n, len(FEATURE_NAMES)), dtype="float32")
        column = {name: index for index, name in enumerate(FEATURE_NAMES)}

        stats = self.user_stats.get(int(user_id))
        if stats is not None:
            features[:, column["user_count"]] = np.log1p(stats.count)
            features[:, column["user_distinct"]] = np.log1p(stats.distinct)
            features[:, column["user_mean_weight"]] = stats.mean_weight
            features[:, column["user_days_since_last"]] = min(max(epoch - stats.last_ts, 0) / 86400.0, 3650.0)
            features[:, column["user_span_days"]] = min(max(stats.last_ts - stats.first_ts, 0) / 86400.0, 3650.0)
            features[:, column["user_lang_code"]] = stats.lang_code
            features[:, column["user_region_code"]] = stats.region_code
            features[:, column["user_country_code"]] = stats.country_code

        age_days = np.maximum(epoch - self.catalog.share_ts[indices], 0) / 86400.0
        features[:, column["item_age_days"]] = np.minimum(age_days, 3650.0)
        features[:, column["item_share_hour"]] = self.item_share_hour[indices]
        features[:, column["item_share_weekday"]] = self.item_share_weekday[indices]
        features[:, column["item_title_len"]] = self.item_title_len[indices]
        features[:, column["item_text_len"]] = self.item_text_len[indices]
        features[:, column["item_lang_code"]] = self.item_lang_code[indices]
        features[:, column["item_content_type_code"]] = self.item_content_type_code[indices]
        features[:, column["item_is_cold"]] = self.item_is_cold[indices].astype("float32")
        features[:, column["item_count"]] = np.log1p(self.item_counts[indices])
        features[:, column["item_users"]] = np.log1p(self.item_users[indices])
        features[:, column["item_weight"]] = np.log1p(self.item_weight[indices])

        language = self.history.preferred_language(int(user_id))
        region = self.history.preferred_region(int(user_id))
        country = self.history.preferred_country(int(user_id))
        features[:, column["lang_match"]] = self.catalog.language_match(indices, language)
        features[:, column["region_match"]] = self.catalog.region_match(indices, region)
        features[:, column["country_match"]] = self.catalog.country_match(indices, country)

        for name, model in self.models.items():
            target = _SIGNAL_TO_FEATURE.get(name)
            if target is None:
                continue
            raw = np.asarray(model.score(int(user_id), candidate_ids, timestamp), dtype="float64")
            features[:, column[target]] = _rank_percentile(raw).astype("float32")
        features[:, column["recency_age_days"]] = np.minimum(age_days, 3650.0)

        features[:, column["pop_lang_7d_pct"]] = _rank_percentile(self.item_lang_pop[indices]).astype("float32")
        features[:, column["pop_source_7d_pct"]] = _rank_percentile(self.item_source_pop[indices]).astype("float32")
        features[:, column["pop_region_7d_pct"]] = _rank_percentile(self.item_region_pop[indices]).astype("float32")

        author_totals = self.user_author_totals.get(int(user_id), {})
        source_totals = self.user_source_totals.get(int(user_id), {})
        last_author = self.user_last_author.get(int(user_id), {})
        last_source = self.user_last_source.get(int(user_id), {})
        candidate_authors = self.catalog.author[indices]
        candidate_sources = self.catalog.source[indices]
        author_raw = np.asarray([author_totals.get(str(value), 0.0) for value in candidate_authors], dtype="float64")
        source_raw = np.asarray([source_totals.get(str(value), 0.0) for value in candidate_sources], dtype="float64")
        features[:, column["author_affinity"]] = np.log1p(author_raw).astype("float32")
        features[:, column["source_affinity"]] = np.log1p(source_raw).astype("float32")
        features[:, column["author_affinity_pct"]] = _rank_percentile(author_raw).astype("float32")
        features[:, column["source_affinity_pct"]] = _rank_percentile(source_raw).astype("float32")
        features[:, column["author_seen"]] = (author_raw > 0).astype("float32")
        features[:, column["source_seen"]] = (source_raw > 0).astype("float32")
        author_days = np.asarray(
            [
                min(max(epoch - last_author.get(str(value), epoch), 0) / 86400.0, 3650.0)
                if str(value) in last_author
                else 3650.0
                for value in candidate_authors
            ],
            dtype="float32",
        )
        source_days = np.asarray(
            [
                min(max(epoch - last_source.get(str(value), epoch), 0) / 86400.0, 3650.0)
                if str(value) in last_source
                else 3650.0
                for value in candidate_sources
            ],
            dtype="float32",
        )
        features[:, column["author_days_since"]] = author_days
        features[:, column["source_days_since"]] = source_days
        return features

    def categorical_indices(self) -> list[int]:
        return [FEATURE_NAMES.index(name) for name in CATEGORICAL_FEATURES]


def sample_training_queries(
    builder: RankingFeatureBuilder,
    positives: pd.DataFrame,
    all_interactions: pd.DataFrame,
    negatives_hard: int = 50,
    negatives_random: int = 50,
    negatives_uniform: int = 50,
    seed: int = 42,
    max_queries: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build (features, labels, groups) for LambdaRank training."""
    catalog = builder.catalog
    all_history = UserHistory(all_interactions, catalog)
    rng = np.random.default_rng(seed)
    rows: list[np.ndarray] = []
    labels: list[float] = []
    groups: list[int] = []
    positives = positives.sort_values("first_timestamp")
    if max_queries is not None and len(positives) > max_queries:
        positives = (
            positives.sample(n=max_queries, random_state=seed)
            .sort_values("first_timestamp")
            .reset_index(drop=True)
        )
    for row in positives.itertuples(index=False):
        user_id = int(row.personId)
        positive = int(row.contentId)
        timestamp = row.first_timestamp
        available = catalog.available_ids(timestamp)
        seen = all_history.seen_before(user_id, timestamp)
        if seen.size:
            available = available[~np.isin(available, seen)]
        if positive not in available:
            continue
        candidates = available
        # Hard negatives: top items from each retrieval signal.
        hard: list[int] = []
        for name in ("recency", "pop_1d", "content", "cf"):
            model = builder.models.get(name)
            if model is None:
                continue
            raw = np.asarray(model.score(user_id, candidates, timestamp), dtype="float64")
            order = np.argsort(-raw, kind="stable")[: max(negatives_hard * 4, 100)]
            hard.extend(int(item) for item in candidates[order] if int(item) != positive)
        hard = list(dict.fromkeys(hard))
        hard_negative = (
            rng.choice(hard, size=min(negatives_hard, len(hard)), replace=False).tolist()
            if hard
            else []
        )

        # Time-matched random negatives, stratified across cold and warm items.
        positive_index = catalog.id_to_index[positive]
        positive_share = int(catalog.share_ts[positive_index])
        candidate_indices = catalog.indices(candidates)
        candidate_share = catalog.share_ts[candidate_indices]
        candidate_lang = catalog.lang[candidate_indices]
        positive_lang = catalog.lang[positive_index]
        candidate_counts = builder.item_counts[candidate_indices]

        time_match = np.abs(candidate_share - positive_share) <= 3 * 86400
        lang_match = candidate_lang == positive_lang
        pool = candidates[time_match & lang_match & (candidates != positive)]
        if len(pool) < negatives_random:
            pool = candidates[time_match & (candidates != positive)]
        if len(pool) < negatives_random:
            pool = candidates[candidates != positive]
        pool_indices = catalog.indices(pool)
        pool_counts = builder.item_counts[pool_indices]
        cold_pool = pool[pool_counts == 0]
        warm_pool = pool[pool_counts > 0]

        def pick(pool_values: np.ndarray, size: int) -> list[int]:
            if size <= 0 or len(pool_values) == 0:
                return []
            return rng.choice(
                pool_values,
                size=min(size, len(pool_values)),
                replace=False,
            ).tolist()

        cold_random = pick(cold_pool, negatives_random // 2)
        warm_random = pick(warm_pool, negatives_random - len(cold_random))
        random_negative = list(dict.fromkeys(cold_random + warm_random))
        if len(random_negative) < negatives_random:
            already_selected = set(random_negative)
            remaining_pool = np.asarray(
                [item for item in pool if int(item) not in already_selected],
                dtype="int64",
            )
            random_negative.extend(pick(remaining_pool, negatives_random - len(random_negative)))

        # Uniform full-catalog negatives teach the model that warm items are
        # not automatically better than cold items outside the sampled set.
        uniform_pool = candidates[candidates != positive]
        uniform_negative = pick(uniform_pool, negatives_uniform)

        negatives = list(dict.fromkeys(hard_negative + random_negative + uniform_negative))
        negatives = [item for item in negatives if item != positive]
        target_negatives = negatives_hard + negatives_random + negatives_uniform
        if len(negatives) < target_negatives:
            remaining = np.asarray(
                [
                    item
                    for item in candidates
                    if item != positive and item not in set(negatives)
                ],
                dtype="int64",
            )
            if len(remaining):
                extra = min(target_negatives - len(negatives), len(remaining))
                negatives.extend(rng.choice(remaining, size=extra, replace=False).tolist())
        if not negatives:
            continue
        candidate_set = np.asarray([positive] + negatives, dtype="int64")
        features = builder.transform(user_id, candidate_set, timestamp)
        rows.append(features)
        labels.extend([1.0] + [0.0] * len(negatives))
        groups.append(1 + len(negatives))
    if not rows:
        raise ValueError("No training queries could be built")
    return np.vstack(rows), np.asarray(labels, dtype="float32"), np.asarray(groups, dtype="int64")
