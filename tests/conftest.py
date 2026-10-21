from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.recommender.data.dataset import Dataset
from src.recommender.data.features import Catalog


def make_articles(rows: list[dict]) -> pd.DataFrame:
    articles = pd.DataFrame(rows)
    articles["share_ts"] = articles["timestamp"].astype("int64")
    articles["share_timestamp"] = pd.to_datetime(articles["share_ts"], unit="s", utc=True)
    def _column(name: str, default: str) -> pd.Series:
        if name in articles.columns:
            return articles[name].fillna(default).astype(str)
        return pd.Series([default] * len(articles), index=articles.index)

    articles["author_id"] = _column("authorPersonId", "author")
    articles["authorRegion"] = _column("authorRegion", "unknown")
    articles["authorCountry"] = _column("authorCountry", "unknown")
    articles["source_domain"] = _column("source_domain", "example.com")
    articles["lang"] = _column("lang", "en")
    articles["contentType"] = _column("contentType", "HTML")
    articles["title"] = _column("title", "")
    articles["text"] = _column("text", "")
    articles["text_all"] = (articles["title"] + " " + articles["text"]).str.strip()
    return articles


def make_events(rows: list[dict]) -> pd.DataFrame:
    events = pd.DataFrame(rows)
    events["weight"] = events["weight"].astype(float) if "weight" in events.columns else 1.0
    events["timestamp"] = events["timestamp"].astype("int64")
    events["datetime"] = pd.to_datetime(events["timestamp"], unit="s", utc=True)
    events["userRegion"] = (
        events["userRegion"].fillna("unknown").astype(str)
        if "userRegion" in events.columns
        else "unknown"
    )
    events["userCountry"] = (
        events["userCountry"].fillna("unknown").astype(str)
        if "userCountry" in events.columns
        else "unknown"
    )
    return events


@pytest.fixture
def small_catalog() -> Catalog:
    articles = make_articles(
        [
            {"contentId": 1, "timestamp": 1000, "title": "alpha beta", "text": "alpha beta gamma", "lang": "en"},
            {"contentId": 2, "timestamp": 1100, "title": "beta gamma", "text": "beta gamma delta", "lang": "en"},
            {"contentId": 3, "timestamp": 1200, "title": "future token", "text": "futuretoken uniqueterm", "lang": "en"},
        ]
    )
    return Catalog(articles)


@pytest.fixture
def small_dataset(small_catalog: Catalog) -> Dataset:
    articles = small_catalog.articles
    train = make_events(
        [
            {"personId": 10, "contentId": 1, "timestamp": 1300, "weight": 1.0},
            {"personId": 10, "contentId": 2, "timestamp": 1400, "weight": 2.0},
            {"personId": 11, "contentId": 1, "timestamp": 1500, "weight": 1.0},
            {"personId": 11, "contentId": 2, "timestamp": 1600, "weight": 1.0},
        ]
    )
    validation = make_events(
        [
            {"personId": 10, "contentId": 3, "timestamp": 2000, "weight": 1.0},
        ]
    )
    test = make_events(
        [
            {"personId": 11, "contentId": 3, "timestamp": 2100, "weight": 1.0},
        ]
    )
    interactions = pd.concat([train, validation, test], ignore_index=True).sort_values("timestamp")
    first = interactions.rename(columns={"timestamp": "first_ts", "datetime": "first_timestamp"}).copy()
    first["eventStrength"] = first["weight"]
    cutoffs = {
        "train": pd.Timestamp(1700, unit="s", tz="UTC"),
        "validation": pd.Timestamp(2050, unit="s", tz="UTC"),
    }
    empty = first.iloc[0:0].copy()
    return Dataset(
        articles=articles,
        interactions=interactions,
        first_events=first,
        train=first,
        validation=empty,
        test=first,
        cold_user_events=empty,
        cutoffs=cutoffs,
        eligible_users=np.asarray([10, 11], dtype="int64"),
        stats={},
    )
