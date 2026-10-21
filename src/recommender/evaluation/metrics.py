"""Ranking metrics for one-positive-per-query evaluation."""

from __future__ import annotations

import math

import numpy as np


def recall_at_k(rank: int, k: int) -> float:
    return 1.0 if 1 <= rank <= k else 0.0


def hit_rate_at_k(rank: int, k: int) -> float:
    return recall_at_k(rank, k)


def ndcg_at_k(rank: int, k: int) -> float:
    if not 1 <= rank <= k:
        return 0.0
    return 1.0 / math.log2(rank + 1)


def mrr_at_k(rank: int, k: int) -> float:
    if not 1 <= rank <= k:
        return 0.0
    return 1.0 / rank


def average_precision_at_k(rank: int, k: int) -> float:
    # With a single relevant item, AP@k equals reciprocal rank when hit.
    return mrr_at_k(rank, k)


def precision_at_k(rank: int, k: int) -> float:
    return recall_at_k(rank, k) / float(k)


def self_information(probability: float, eps: float = 1e-12) -> float:
    return -math.log2(max(probability, eps))


def intra_list_diversity(similarity: np.ndarray) -> float:
    n = similarity.shape[0]
    if n < 2:
        return 0.0
    upper = np.triu_indices(n, k=1)
    distances = 1.0 - similarity[upper]
    return float(np.clip(distances, 0.0, 2.0).mean())
