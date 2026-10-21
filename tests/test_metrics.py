from __future__ import annotations

import math

import numpy as np
import pytest

from src.recommender.evaluation.metrics import (
    intra_list_diversity,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
)


def test_metrics_known_values() -> None:
    assert recall_at_k(1, 5) == 1.0
    assert recall_at_k(5, 5) == 1.0
    assert recall_at_k(6, 5) == 0.0
    assert recall_at_k(0, 5) == 0.0

    assert ndcg_at_k(1, 10) == pytest.approx(1.0)
    assert ndcg_at_k(2, 10) == pytest.approx(1.0 / math.log2(3))
    assert ndcg_at_k(6, 5) == 0.0
    assert ndcg_at_k(6, 10) == pytest.approx(1.0 / math.log2(7))
    assert ndcg_at_k(11, 10) == 0.0

    assert mrr_at_k(1, 10) == 1.0
    assert mrr_at_k(4, 10) == 0.25
    assert mrr_at_k(11, 10) == 0.0


def test_intra_list_diversity_known_values() -> None:
    similarity = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert intra_list_diversity(similarity) == pytest.approx(1.0)
    similarity = np.ones((2, 2))
    assert intra_list_diversity(similarity) == pytest.approx(0.0)
    assert intra_list_diversity(np.ones((1, 1))) == 0.0
