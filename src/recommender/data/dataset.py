"""Dataset container and chronological split logic."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .clean import build_first_events, clean_articles, clean_interactions
from .load import load_raw_articles, load_raw_interactions


@dataclass
class Dataset:
    articles: pd.DataFrame
    interactions: pd.DataFrame
    first_events: pd.DataFrame
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    cold_user_events: pd.DataFrame
    cutoffs: dict[str, pd.Timestamp]
    eligible_users: np.ndarray
    stats: dict[str, Any] = field(default_factory=dict)

    def split_events(self, split: str) -> pd.DataFrame:
        if split == "train":
            return self.train
        if split in {"validation", "val"}:
            return self.validation
        if split == "test":
            return self.test
        if split in {"cold_users", "cold_user"}:
            return self.cold_user_events
        raise ValueError(f"Unknown split: {split}")

    def training_interactions(self, split: str) -> pd.DataFrame:
        """Interactions available to a model before the evaluation period.

        Validation models see only data before the train cutoff. Test models
        are refit on train + validation, i.e. data before the validation cutoff.
        """
        if split in {"train", "validation", "val"}:
            cutoff = self.cutoffs["train"]
        elif split == "test":
            cutoff = self.cutoffs["validation"]
        else:
            raise ValueError(f"Unknown split: {split}")
        return self.interactions.loc[self.interactions["timestamp"] < int(cutoff.timestamp())].copy()

    def reference_time(self, split: str) -> pd.Timestamp:
        if split in {"train", "validation", "val"}:
            return self.cutoffs["train"]
        if split == "test":
            return self.cutoffs["validation"]
        raise ValueError(f"Unknown split: {split}")

    def item_interaction_counts(self, split: str) -> pd.Series:
        events = self.training_interactions(split)
        return events.groupby("contentId")["weight"].sum()

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.articles.to_parquet(directory / "articles.parquet", index=False)
        self.interactions.to_parquet(directory / "interactions.parquet", index=False)
        self.first_events.to_parquet(directory / "first_events.parquet", index=False)
        self.train.to_parquet(directory / "train.parquet", index=False)
        self.validation.to_parquet(directory / "validation.parquet", index=False)
        self.test.to_parquet(directory / "test.parquet", index=False)
        self.cold_user_events.to_parquet(directory / "cold_user_events.parquet", index=False)
        metadata = {
            "cutoffs": {key: value.isoformat() for key, value in self.cutoffs.items()},
            "eligible_users": [int(user) for user in self.eligible_users.tolist()],
            "stats": self.stats,
        }
        with (directory / "metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, default=str)

    @classmethod
    def load(cls, directory: str | Path) -> "Dataset":
        directory = Path(directory)
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        cutoffs = {
            key: pd.Timestamp(value).tz_convert("UTC") if pd.Timestamp(value).tzinfo else pd.Timestamp(value, tz="UTC")
            for key, value in metadata["cutoffs"].items()
        }
        return cls(
            articles=pd.read_parquet(directory / "articles.parquet"),
            interactions=pd.read_parquet(directory / "interactions.parquet"),
            first_events=pd.read_parquet(directory / "first_events.parquet"),
            train=pd.read_parquet(directory / "train.parquet"),
            validation=pd.read_parquet(directory / "validation.parquet"),
            test=pd.read_parquet(directory / "test.parquet"),
            cold_user_events=pd.read_parquet(directory / "cold_user_events.parquet"),
            cutoffs=cutoffs,
            eligible_users=np.asarray(metadata["eligible_users"], dtype="int64"),
            stats=metadata["stats"],
        )


def chronological_split(
    first_events: pd.DataFrame,
    train_quantile: float = 0.7,
    val_quantile: float = 0.8,
    min_user_interactions: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, pd.Timestamp], np.ndarray]:
    timestamps = first_events["first_timestamp"]
    train_cutoff = timestamps.quantile(train_quantile)
    val_cutoff = timestamps.quantile(val_quantile)
    if train_cutoff >= val_cutoff:
        raise ValueError("train cutoff must be before validation cutoff")

    train = first_events.loc[timestamps < train_cutoff].copy()
    validation = first_events.loc[(timestamps >= train_cutoff) & (timestamps < val_cutoff)].copy()
    test = first_events.loc[timestamps >= val_cutoff].copy()

    user_item_counts = train.groupby("personId")["contentId"].nunique()
    eligible_users = np.sort(
        user_item_counts.loc[user_item_counts >= min_user_interactions].index.to_numpy(dtype="int64")
    )

    validation = validation.loc[validation["personId"].isin(eligible_users)].copy()
    test = test.loc[test["personId"].isin(eligible_users)].copy()

    first_user_time = first_events.groupby("personId")["first_timestamp"].min()
    cold_user_ids = first_user_time.loc[first_user_time >= train_cutoff].index
    cold_user_events = first_events.loc[
        first_events["personId"].isin(cold_user_ids)
        & (first_events["first_timestamp"] >= train_cutoff)
    ].copy()

    cutoffs = {"train": train_cutoff, "validation": val_cutoff}
    return train, validation, test, cold_user_events, cutoffs, eligible_users


def build_dataset(config: dict[str, Any]) -> Dataset:
    raw_articles = load_raw_articles(config)
    raw_interactions = load_raw_interactions(config)
    articles, removed_ids, article_stats = clean_articles(raw_articles)
    interactions, interaction_stats = clean_interactions(raw_interactions, articles, removed_ids)
    first_events = build_first_events(interactions)

    split_config = config.get("split", {})
    train, validation, test, cold_user_events, cutoffs, eligible_users = chronological_split(
        first_events,
        train_quantile=float(split_config.get("train_quantile", 0.7)),
        val_quantile=float(split_config.get("val_quantile", 0.8)),
        min_user_interactions=int(split_config.get("min_user_interactions", 5)),
    )

    train_cutoff_ts = int(cutoffs["train"].timestamp())
    fit_items = interactions.loc[interactions["timestamp"] < train_cutoff_ts, "contentId"]
    all_test_items = test["contentId"]
    cold_rate = float((~all_test_items.isin(set(fit_items.tolist()))).mean()) if len(test) else 0.0

    stats: dict[str, Any] = {
        "articles": article_stats,
        "interactions": interaction_stats,
        "first_events": int(len(first_events)),
        "train_events": int(len(train)),
        "validation_events": int(len(validation)),
        "test_events": int(len(test)),
        "cold_user_events": int(len(cold_user_events)),
        "eligible_users": int(len(eligible_users)),
        "test_cold_item_rate": cold_rate,
    }
    return Dataset(
        articles=articles,
        interactions=interactions,
        first_events=first_events,
        train=train,
        validation=validation,
        test=test,
        cold_user_events=cold_user_events,
        cutoffs=cutoffs,
        eligible_users=eligible_users,
        stats=stats,
    )
