from __future__ import annotations

import numpy as np
import pandas as pd

from src.recommender.data.features import Catalog
from src.recommender.models.content_based import ContentBasedRecommender
from src.recommender.models.popularity import (
    GlobalPopularityRecommender,
    TimeDecayedPopularityRecommender,
)

from .conftest import make_articles, make_events


def _catalog_and_events(future_share_ts: int = 1000):
    articles = make_articles(
        [
            {"contentId": 1, "timestamp": 100, "title": "alpha", "text": "alpha article", "lang": "en"},
            {"contentId": 2, "timestamp": 200, "title": "beta", "text": "beta article", "lang": "en"},
            {
                "contentId": 3,
                "timestamp": future_share_ts,
                "title": "zzzz",
                "text": "zzzz uniquetesttoken",
                "lang": "en",
            },
        ]
    )
    catalog = Catalog(articles)
    train_events = make_events(
        [
            {"personId": 10, "contentId": 1, "timestamp": 1200, "weight": 2.0},
            {"personId": 10, "contentId": 2, "timestamp": 1300, "weight": 1.0},
            {"personId": 11, "contentId": 2, "timestamp": 1400, "weight": 1.0},
        ]
    )
    test_events = make_events(
        [
            {"personId": 10, "contentId": 3, "timestamp": 1500, "weight": 1.0},
        ]
    )
    return catalog, train_events, test_events


def test_user_profile_excludes_test_items() -> None:
    catalog, train_events, test_events = _catalog_and_events(future_share_ts=900)
    model = ContentBasedRecommender(min_df=1, max_df=1.0, use_char=False, half_life_days=30)
    model.fit(train_events, catalog)
    history_items = set(model.history.user_events(10)["contentId"].tolist())
    assert 3 not in history_items
    assert history_items == {1, 2}

    profile = model._user_profile(10, pd.Timestamp(1500, unit="s", tz="UTC"))
    item_index = catalog.id_to_index[3]
    similarity = profile.dot(model.item_vectors[item_index].T).toarray().ravel()[0]
    assert abs(float(similarity)) < 1e-9


def test_popularity_uses_train_only() -> None:
    catalog, train_events, _ = _catalog_and_events(future_share_ts=900)
    model = GlobalPopularityRecommender().fit(train_events, catalog)
    scores = model.score(10, np.asarray([1, 2, 3]), pd.Timestamp(1500, unit="s", tz="UTC"))
    assert scores[2] == 0.0
    assert scores[0] > 0.0
    assert scores[1] > 0.0

    decayed = TimeDecayedPopularityRecommender(half_life_days=7).fit(train_events, catalog)
    decayed_scores = decayed.score(10, np.asarray([1, 2, 3]), pd.Timestamp(1500, unit="s", tz="UTC"))
    assert decayed_scores[2] == 0.0


def test_candidates_only_include_available_items() -> None:
    catalog, _, _ = _catalog_and_events(future_share_ts=1200)
    available = catalog.available_ids(pd.Timestamp(1150, unit="s", tz="UTC"))
    assert set(available.tolist()) == {1, 2}
    available_later = catalog.available_ids(pd.Timestamp(1250, unit="s", tz="UTC"))
    assert set(available_later.tolist()) == {1, 2, 3}
    assert catalog.available_ids(pd.Timestamp(50, unit="s", tz="UTC")).size == 0


def test_vectorizer_fit_on_train_catalog() -> None:
    catalog, train_events, _ = _catalog_and_events(future_share_ts=3000)
    model = ContentBasedRecommender(min_df=1, max_df=1.0, use_char=False)
    model.fit(train_events, catalog)
    assert "uniquetesttoken" not in model.word_vectorizer.vocabulary_
    assert "futuretoken" not in model.word_vectorizer.vocabulary_


def test_no_test_events_in_features() -> None:
    catalog, train_events, _ = _catalog_and_events(future_share_ts=900)
    model = ContentBasedRecommender(min_df=1, max_df=1.0, use_char=False)
    model.fit(train_events, catalog)
    for user_id in (10, 11):
        events = model.history.user_events(user_id)
        assert events is not None
        assert 3 not in set(events["contentId"].tolist())
    assert model.history.last_interaction(10) is not None
    assert model.history.last_interaction(10) <= 1300
