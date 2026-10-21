from __future__ import annotations

import numpy as np
import pandas as pd

from src.recommender.evaluation.evaluator import Evaluator

from .conftest import make_events


class ItemIdModel:
    name = "item_id"

    def score(self, user_id: int, candidate_ids: np.ndarray, timestamp) -> np.ndarray:
        return candidate_ids.astype("float64")


def test_evaluator_excludes_seen_and_unavailable_items(small_dataset) -> None:
    events = make_events(
        [
            {"personId": 10, "contentId": 3, "timestamp": 2000, "weight": 1.0},
        ]
    )
    small_dataset.test = events
    result = Evaluator(small_dataset, bootstrap_samples=10, seed=1).evaluate(
        ItemIdModel(), "test", bootstrap=False
    )
    assert len(result.events) == 1
    assert result.events["n_candidates"].iloc[0] == 1
    assert result.events["rank"].iloc[0] == 1
    assert result.summary["micro"]["recall@10"] == 1.0
    assert result.summary["skipped_events"] == 0


def test_evaluator_skips_unavailable_positive(small_dataset) -> None:
    events = make_events(
        [
            {"personId": 10, "contentId": 3, "timestamp": 1150, "weight": 1.0},
        ]
    )
    small_dataset.test = events
    evaluator = Evaluator(small_dataset, bootstrap_samples=10, seed=1)
    try:
        evaluator.evaluate(ItemIdModel(), "test", bootstrap=False)
    except ValueError as error:
        assert "No evaluable events" in str(error)
    else:  # pragma: no cover
        raise AssertionError("Expected ValueError for unavailable positive")
