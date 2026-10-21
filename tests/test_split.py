from __future__ import annotations

import numpy as np
import pandas as pd

from src.recommender.data.dataset import chronological_split


def test_chronological_split_has_no_overlap() -> None:
    timestamps = np.arange(1, 101, dtype="int64")
    first = pd.DataFrame(
        {
            "personId": np.tile(np.arange(1, 11), 10),
            "contentId": np.repeat(np.arange(1, 11), 10),
            "first_ts": timestamps,
            "first_timestamp": pd.to_datetime(timestamps, unit="s", utc=True),
        }
    )
    train, validation, test, cold, cutoffs, eligible = chronological_split(first)
    assert train["first_ts"].max() < cutoffs["train"].timestamp() + 1
    assert validation["first_ts"].min() >= int(cutoffs["train"].timestamp())
    assert validation["first_ts"].max() < cutoffs["validation"].timestamp() + 1
    assert test["first_ts"].min() >= int(cutoffs["validation"].timestamp())
    assert set(train.index).isdisjoint(validation.index)
    assert set(train.index).isdisjoint(test.index)
    assert set(validation.index).isdisjoint(test.index)
    assert len(eligible) > 0


def test_eligible_users_require_minimum_interactions() -> None:
    first = pd.DataFrame(
        {
            "personId": [1, 1, 1, 1, 1, 2, 2],
            "contentId": [10, 11, 12, 13, 14, 20, 21],
            "first_ts": [1, 2, 3, 4, 5, 6, 7],
            "first_timestamp": pd.to_datetime([1, 2, 3, 4, 5, 6, 7], unit="s", utc=True),
        }
    )
    _, _, _, _, _, eligible = chronological_split(first, train_quantile=0.7, val_quantile=0.8)
    assert 2 not in eligible.tolist()
