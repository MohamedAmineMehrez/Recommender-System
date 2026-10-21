"""LightGBM LambdaRank model over cold/warm retrieval signals."""

from __future__ import annotations

from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from ..data.features import Catalog
from ..data.ranking_features import (
    CATEGORICAL_FEATURES,
    FEATURE_NAMES,
    RankingFeatureBuilder,
    sample_training_queries,
)
from .base import Recommender
from .collaborative import CollaborativeRecommender, ItemItemCFRecommender
from .content_based import ContentBasedRecommender
from .popularity import GlobalPopularityRecommender, TimeDecayedPopularityRecommender
from .recency import RecencyRecommender


def build_signal_models(config: dict[str, Any], history: pd.DataFrame, catalog: Catalog) -> dict[str, Recommender]:
    models: dict[str, Recommender] = {
        "recency": RecencyRecommender(),
        "pop_0.25d": TimeDecayedPopularityRecommender(0.25),
        "pop_0.5d": TimeDecayedPopularityRecommender(0.5),
        "pop_1d": TimeDecayedPopularityRecommender(1.0),
        "pop_3d": TimeDecayedPopularityRecommender(3.0),
        "pop_7d": TimeDecayedPopularityRecommender(7.0),
        "pop_all": GlobalPopularityRecommender(),
        "content": ContentBasedRecommender(**config["content"]),
        "cf": CollaborativeRecommender(**config["collaborative"]),
        "itemitem": ItemItemCFRecommender(),
    }
    for model in models.values():
        model.fit(history, catalog)
    return models


def _first_events(events: pd.DataFrame) -> pd.DataFrame:
    first = (
        events.sort_values(["timestamp", "personId", "contentId"], kind="stable")
        .drop_duplicates(["personId", "contentId"], keep="first")
        .copy()
        .reset_index(drop=True)
    )
    if "datetime" in first.columns:
        first["first_timestamp"] = first["datetime"]
    else:
        first["first_timestamp"] = pd.to_datetime(first["timestamp"], unit="s", utc=True)
    first["first_ts"] = first["timestamp"].astype("int64")
    return first


class RankerRecommender(Recommender):
    name = "ranker"

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        folds: list[tuple[float, float]] | None = None,
        negatives_hard: int = 50,
        negatives_random: int = 50,
        negatives_uniform: int = 50,
        seed: int = 42,
        max_queries_per_fold: int | None = None,
        max_validation_queries: int | None = None,
    ) -> None:
        self.config = config or {}
        self.folds = folds or [(0.6, 0.7), (0.7, 0.8), (0.8, 0.9)]
        self.negatives_hard = int(negatives_hard)
        self.negatives_random = int(negatives_random)
        self.negatives_uniform = int(negatives_uniform)
        self.seed = int(seed)
        self.max_queries_per_fold = max_queries_per_fold
        self.max_validation_queries = max_validation_queries
        self.catalog: Catalog | None = None
        self.model: lgb.LGBMRanker | None = None
        self.feature_builder: RankingFeatureBuilder | None = None
        self.best_iteration_: int | None = None
        self.training_stats: dict[str, Any] = {}

    def _build_queries(
        self,
        history: pd.DataFrame,
        positives: pd.DataFrame,
        all_interactions: pd.DataFrame,
        catalog: Catalog,
        seed_offset: int,
        max_queries: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        models = build_signal_models(self.config, history, catalog)
        builder = RankingFeatureBuilder(
            catalog,
            history,
            models,
            affinity_half_life_days=float(self.config.get("affinity_half_life_days", 30.0)),
        )
        return sample_training_queries(
            builder,
            positives,
            all_interactions,
            negatives_hard=self.negatives_hard,
            negatives_random=self.negatives_random,
            negatives_uniform=self.negatives_uniform,
            seed=self.seed + seed_offset,
            max_queries=max_queries,
        )

    def fit(
        self,
        events: pd.DataFrame,
        catalog: Catalog,
        all_interactions: pd.DataFrame | None = None,
        validation_events: pd.DataFrame | None = None,
        num_boost_round: int | None = None,
    ) -> "RankerRecommender":
        self.catalog = catalog
        all_interactions = all_interactions if all_interactions is not None else events
        first = _first_events(events)
        if first.empty:
            raise ValueError("Cannot fit ranker without first-time events")

        feature_blocks: list[np.ndarray] = []
        label_blocks: list[np.ndarray] = []
        group_blocks: list[np.ndarray] = []
        for fold_index, (start_quantile, end_quantile) in enumerate(self.folds):
            start = first["first_timestamp"].quantile(start_quantile)
            end = first["first_timestamp"].quantile(end_quantile)
            history = events.loc[events["timestamp"] < int(start.timestamp())].copy()
            positives = first.loc[
                (first["first_timestamp"] >= start) & (first["first_timestamp"] < end)
            ].copy()
            if len(positives) < 20 or len(history) < 100:
                continue
            print(
                f"Ranker fold {fold_index}: history={len(history)}, positives={len(positives)}",
                flush=True,
            )
            features, labels, groups = self._build_queries(
                history,
                positives,
                all_interactions,
                catalog,
                seed_offset=fold_index * 1000,
                max_queries=self.max_queries_per_fold,
            )
            feature_blocks.append(features)
            label_blocks.append(labels)
            group_blocks.append(groups)
            print(f"  built {len(groups)} groups / {len(labels)} rows", flush=True)
        if not feature_blocks:
            raise ValueError("No ranker folds produced training data")

        train_features = np.vstack(feature_blocks)
        train_labels = np.concatenate(label_blocks)
        train_groups = np.concatenate(group_blocks)

        eval_set: list[tuple[np.ndarray, np.ndarray]] = []
        eval_groups: list[np.ndarray] = []
        validation_builder: RankingFeatureBuilder | None = None
        if validation_events is not None and len(validation_events):
            print("Building ranker validation queries...", flush=True)
            validation_models = build_signal_models(self.config, events, catalog)
            validation_builder = RankingFeatureBuilder(
                catalog,
                events,
                validation_models,
                affinity_half_life_days=float(self.config.get("affinity_half_life_days", 30.0)),
            )
            validation_features, validation_labels, validation_groups = sample_training_queries(
                validation_builder,
                validation_events,
                all_interactions,
                negatives_hard=self.negatives_hard,
                negatives_random=self.negatives_random,
                negatives_uniform=self.negatives_uniform,
                seed=self.seed + 9000,
                max_queries=self.max_validation_queries,
            )
            eval_set.append((validation_features, validation_labels))
            eval_groups.append(validation_groups)

        ranker_config = self.config.get("ranker", self.config)
        categorical = [FEATURE_NAMES.index(name) for name in CATEGORICAL_FEATURES]
        n_estimators = int(num_boost_round or ranker_config.get("num_boost_round", 500))
        self.model = lgb.LGBMRanker(
            objective="lambdarank",
            n_estimators=n_estimators,
            metric="ndcg",
            ndcg_eval_at=[5, 10, 20],
            learning_rate=float(ranker_config.get("learning_rate", 0.03)),
            num_leaves=int(ranker_config.get("num_leaves", 31)),
            min_data_in_leaf=int(ranker_config.get("min_data_in_leaf", 50)),
            feature_fraction=float(ranker_config.get("feature_fraction", 0.8)),
            bagging_fraction=float(ranker_config.get("bagging_fraction", 0.8)),
            bagging_freq=int(ranker_config.get("bagging_freq", 1)),
            lambda_l2=float(ranker_config.get("lambda_l2", 1.0)),
            random_state=int(ranker_config.get("seed", self.seed)),
            n_jobs=int(ranker_config.get("n_jobs", -1)),
            verbose=-1,
        )
        callbacks = []
        rounds = n_estimators
        if eval_set:
            callbacks.append(
                lgb.early_stopping(
                    int(ranker_config.get("early_stopping_rounds", 100)),
                    verbose=False,
                )
            )
        self.model.fit(
            train_features,
            train_labels,
            group=train_groups,
            eval_set=eval_set if eval_set else None,
            eval_group=eval_groups if eval_groups else None,
            eval_at=[5, 10, 20],
            categorical_feature=categorical,
            callbacks=callbacks,
        )
        self.best_iteration_ = getattr(self.model, "best_iteration_", None) or rounds
        self.training_stats = {
            "train_rows": int(len(train_labels)),
            "train_groups": int(len(train_groups)),
            "best_iteration": int(self.best_iteration_),
        }
        if validation_builder is not None:
            self.feature_builder = validation_builder
        else:
            scoring_models = build_signal_models(self.config, events, catalog)
            self.feature_builder = RankingFeatureBuilder(
                catalog,
                events,
                scoring_models,
                affinity_half_life_days=float(self.config.get("affinity_half_life_days", 30.0)),
            )
        return self

    def score(self, user_id: int, candidate_ids: np.ndarray, timestamp: pd.Timestamp | int | float) -> np.ndarray:
        if self.model is None or self.feature_builder is None:
            raise RuntimeError("Model is not fitted")
        features = self.feature_builder.transform(int(user_id), candidate_ids, timestamp)
        return np.asarray(self.model.predict(features), dtype="float64")

    def get_params(self) -> dict[str, Any]:
        return {
            "folds": self.folds,
            "negatives_hard": self.negatives_hard,
            "negatives_random": self.negatives_random,
            "negatives_uniform": self.negatives_uniform,
            "best_iteration": self.best_iteration_,
            "max_queries_per_fold": self.max_queries_per_fold,
            "max_validation_queries": self.max_validation_queries,
        }
