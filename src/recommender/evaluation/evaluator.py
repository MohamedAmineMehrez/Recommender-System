"""Full-catalog, time-aware evaluator."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..data.dataset import Dataset
from ..data.features import Catalog, UserHistory, to_epoch_seconds
from .bootstrap import bootstrap_mean_ci, bootstrap_ratio_ci, paired_bootstrap_difference
from .metrics import (
    average_precision_at_k,
    intra_list_diversity,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
    self_information,
)


@dataclass
class EvaluationResult:
    name: str
    split: str
    events: pd.DataFrame
    summary: dict[str, Any]
    per_user: pd.DataFrame
    recommended_by_user: dict[int, set[int]] = field(default_factory=dict)

    def to_json(self, path: str) -> None:
        import json

        with open(path, "w", encoding="utf-8") as handle:
            json.dump({"name": self.name, "split": self.split, "summary": self.summary}, handle, indent=2, default=str)


class Evaluator:
    def __init__(
        self,
        dataset: Dataset,
        ks: tuple[int, ...] = (5, 10, 20),
        bootstrap_samples: int = 1000,
        seed: int = 42,
        catalog: Catalog | None = None,
    ):
        self.dataset = dataset
        self.catalog = catalog or Catalog(dataset.articles)
        self.history = UserHistory(dataset.interactions, self.catalog)
        self.ks = tuple(sorted(set(ks)))
        self.max_k = max(max(self.ks), 10)
        self.bootstrap_samples = bootstrap_samples
        self.seed = seed
        self._item_index = self.catalog.id_to_index

    def evaluate(
        self,
        model: Any,
        split: str,
        events: pd.DataFrame | None = None,
        item_vectors: Any | None = None,
        bootstrap: bool = True,
        max_events: int | None = None,
        name: str | None = None,
    ) -> EvaluationResult:
        started = time.perf_counter()
        if events is None:
            events = self.dataset.split_events(split)
        if max_events is not None:
            events = events.head(max_events)
        if events.empty:
            raise ValueError("No events to evaluate")

        item_counts = self.dataset.item_interaction_counts(split)
        warm_items = set(int(item) for item in item_counts.index.tolist())
        cold_items = set(int(item) for item in self.catalog.item_ids.tolist()) - warm_items

        fit_events = self.dataset.training_interactions(split)
        popularity = fit_events.groupby("contentId")["weight"].sum()
        total_popularity = float(popularity.sum())
        n_items = self.catalog.n_items
        popularity_array = np.zeros(n_items, dtype="float64")
        for item_id, value in popularity.items():
            index = self._item_index.get(int(item_id))
            if index is not None:
                popularity_array[index] = float(value)
        probabilities = (popularity_array + 1.0) / (total_popularity + n_items)

        records: list[dict[str, Any]] = []
        recommended_by_user: dict[int, set[int]] = {}
        recommended_by_user_segment: dict[tuple[int, str], set[int]] = {}
        skipped = 0
        candidate_counts: list[int] = []

        for row in events.itertuples(index=False):
            user_id = int(row.personId)
            positive = int(row.contentId)
            timestamp = row.first_timestamp if hasattr(row, "first_timestamp") else row.timestamp
            available = self.catalog.available_ids(timestamp)
            seen = self.history.seen_before(user_id, timestamp)
            if seen.size:
                available = available[~np.isin(available, seen)]
            if positive not in available:
                skipped += 1
                continue
            scores = np.asarray(model.score(user_id, available, timestamp), dtype="float64")
            if scores.shape != available.shape:
                raise ValueError(
                    f"Model {getattr(model, 'name', type(model).__name__)} returned scores with shape "
                    f"{scores.shape}, expected {available.shape}"
                )
            scores = np.nan_to_num(scores, nan=-np.inf, posinf=np.inf, neginf=-np.inf)
            order = np.lexsort((available, -scores))
            positive_positions = np.flatnonzero(available == positive)
            if positive_positions.size == 0:
                skipped += 1
                continue
            positive_position = int(positive_positions[0])
            rank = int(np.flatnonzero(order == positive_position)[0]) + 1
            top = available[order[: self.max_k]]
            top10 = top[:10]
            top10_indices = self.catalog.indices(top10)

            record: dict[str, Any] = {
                "user_id": user_id,
                "item_id": positive,
                "timestamp": timestamp,
                "segment": "warm" if positive in warm_items else "cold",
                "rank": rank,
                "n_candidates": int(len(available)),
            }
            for k in self.ks:
                record[f"recall@{k}"] = recall_at_k(rank, k)
                record[f"ndcg@{k}"] = ndcg_at_k(rank, k)
                record[f"mrr@{k}"] = mrr_at_k(rank, k)
                record[f"map@{k}"] = average_precision_at_k(rank, k)
            record["ndcg@10"] = ndcg_at_k(rank, 10)
            record["mrr@10"] = mrr_at_k(rank, 10)
            record["map@10"] = average_precision_at_k(rank, 10)
            record["novelty@10"] = float(
                np.mean([self_information(float(probabilities[index])) for index in top10_indices])
            )
            if item_vectors is not None and len(top10_indices) > 1:
                sub = item_vectors[top10_indices]
                similarity = sub.dot(sub.T)
                if hasattr(similarity, "toarray"):
                    similarity = similarity.toarray()
                record["diversity@10"] = intra_list_diversity(np.asarray(similarity))
            else:
                record["diversity@10"] = 0.0
            records.append(record)
            candidate_counts.append(len(available))
            top10_ids = {int(item) for item in top10}
            recommended_by_user.setdefault(user_id, set()).update(top10_ids)
            recommended_by_user_segment.setdefault((user_id, record["segment"]), set()).update(top10_ids)

        if not records:
            raise ValueError("No evaluable events; candidate sets never contained the positive item")

        event_frame = pd.DataFrame(records)
        per_user = (
            event_frame.groupby("user_id")
            .agg(
                n_events=("item_id", "size"),
                mean_rank=("rank", "mean"),
                **{f"mean_{metric}": (metric, "mean") for metric in self._metric_columns() if metric in event_frame},
            )
            .reset_index()
        )

        summary: dict[str, Any] = {
            "split": split,
            "n_events": int(len(event_frame)),
            "n_users": int(event_frame["user_id"].nunique()),
            "n_catalog": int(self.catalog.n_items),
            "skipped_events": int(skipped),
            "mean_candidates": float(np.mean(candidate_counts)) if candidate_counts else 0.0,
            "runtime_seconds": float(time.perf_counter() - started),
        }
        summary["micro"] = self._aggregate_micro(event_frame)
        summary["macro"] = self._aggregate_macro(event_frame)
        if bootstrap:
            summary["ci"] = {
                "micro": self._bootstrap_micro_ci(event_frame),
                "macro": self._bootstrap_macro_ci(event_frame),
            }
        summary["coverage@10"] = self._coverage(recommended_by_user)
        if bootstrap:
            summary["coverage@10_ci"] = self._bootstrap_coverage_ci(
                event_frame, recommended_by_user
            )

        segments: dict[str, Any] = {}
        for segment in ("cold", "warm"):
            segment_events = event_frame.loc[event_frame["segment"] == segment]
            if segment_events.empty:
                continue
            segment_ranges: dict[int, set[int]] = {}
            for user_id, items in recommended_by_user_segment.items():
                if user_id[1] == segment:
                    segment_ranges.setdefault(user_id[0], set()).update(items)
            segment_summary: dict[str, Any] = {
                "n_events": int(len(segment_events)),
                "n_users": int(segment_events["user_id"].nunique()),
                "micro": self._aggregate_micro(segment_events),
                "macro": self._aggregate_macro(segment_events),
                "coverage@10": self._coverage(segment_ranges),
            }
            if bootstrap:
                segment_summary["ci"] = {
                    "micro": self._bootstrap_micro_ci(segment_events),
                    "macro": self._bootstrap_macro_ci(segment_events),
                }
            segments[segment] = segment_summary
        summary["segments"] = segments

        return EvaluationResult(
            name=name or getattr(model, "name", type(model).__name__),
            split=split,
            events=event_frame,
            summary=summary,
            per_user=per_user,
            recommended_by_user=recommended_by_user,
        )

    def _metric_columns(self) -> list[str]:
        columns = []
        for k in self.ks:
            columns.extend([f"recall@{k}", f"ndcg@{k}", f"mrr@{k}", f"map@{k}"])
        columns.extend(["ndcg@10", "mrr@10", "map@10", "novelty@10", "diversity@10"])
        return sorted(set(columns))

    def _aggregate_micro(self, events: pd.DataFrame) -> dict[str, float]:
        return {
            metric: float(events[metric].mean())
            for metric in self._metric_columns()
            if metric in events.columns
        }

    def _aggregate_macro(self, events: pd.DataFrame) -> dict[str, float]:
        user_means = events.groupby("user_id")[self._metric_columns()].mean()
        return {metric: float(user_means[metric].mean()) for metric in user_means.columns}

    def _bootstrap_micro_ci(self, events: pd.DataFrame) -> dict[str, list[float]]:
        user_stats = events.groupby("user_id")[self._metric_columns()].agg(["sum", "count"])
        result: dict[str, list[float]] = {}
        for metric in self._metric_columns():
            if metric not in user_stats.columns.get_level_values(0):
                continue
            numerators = user_stats[(metric, "sum")].to_numpy(dtype="float64")
            denominators = user_stats[(metric, "count")].to_numpy(dtype="float64")
            low, high = bootstrap_ratio_ci(
                numerators,
                denominators,
                n_samples=self.bootstrap_samples,
                seed=self.seed,
            )
            result[metric] = [low, high]
        return result

    def _bootstrap_macro_ci(self, events: pd.DataFrame) -> dict[str, list[float]]:
        user_means = events.groupby("user_id")[self._metric_columns()].mean()
        result: dict[str, list[float]] = {}
        for metric in user_means.columns:
            low, high = bootstrap_mean_ci(
                user_means[metric].to_numpy(dtype="float64"),
                n_samples=self.bootstrap_samples,
                seed=self.seed,
            )
            result[metric] = [low, high]
        return result

    def _coverage(self, recommended_by_user: dict[int, set[int]]) -> float:
        if not recommended_by_user:
            return 0.0
        union: set[int] = set()
        for items in recommended_by_user.values():
            union.update(items)
        return float(len(union) / self.catalog.n_items)

    def _bootstrap_coverage_ci(
        self,
        events: pd.DataFrame,
        recommended_by_user: dict[int, set[int]],
    ) -> list[float]:
        users = events["user_id"].drop_duplicates().to_numpy(dtype="int64")
        if users.size == 0:
            return [float("nan"), float("nan")]
        rng = np.random.default_rng(self.seed)
        estimates = np.empty(self.bootstrap_samples, dtype="float64")
        for index in range(self.bootstrap_samples):
            sample = users[rng.integers(0, users.size, size=users.size)]
            union: set[int] = set()
            for user_id in sample:
                union.update(recommended_by_user.get(int(user_id), set()))
            estimates[index] = len(union) / self.catalog.n_items
        low, high = np.quantile(estimates, [0.025, 0.975])
        return [float(low), float(high)]


def compare_results(
    result_a: EvaluationResult,
    result_b: EvaluationResult,
    metrics: tuple[str, ...] = ("recall@5", "recall@10", "recall@20", "ndcg@5", "ndcg@10", "ndcg@20", "mrr@10"),
    bootstrap_samples: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """Paired bootstrap comparison on aligned events."""
    keys = ["user_id", "item_id", "timestamp"]
    left = result_a.events.set_index(keys)
    right = result_b.events.set_index(keys)
    common = left.index.intersection(right.index)
    if len(common) == 0:
        raise ValueError("No common events to compare")
    left = left.loc[common]
    right = right.loc[common]
    rows = []
    for metric in metrics:
        if metric not in left.columns or metric not in right.columns:
            continue
        left_user = left.groupby(level="user_id")[metric].agg(["sum", "count"])
        right_user = right.groupby(level="user_id")[metric].agg(["sum", "count"])
        users = left_user.index.union(right_user.index)
        left_user = left_user.reindex(users, fill_value=0.0)
        right_user = right_user.reindex(users, fill_value=0.0)
        comparison = paired_bootstrap_difference(
            left_user["sum"].to_numpy(dtype="float64"),
            left_user["count"].to_numpy(dtype="float64"),
            right_user["sum"].to_numpy(dtype="float64"),
            right_user["count"].to_numpy(dtype="float64"),
            n_samples=bootstrap_samples,
            seed=seed,
        )
        baseline = right_user["sum"].sum() / right_user["count"].sum() if right_user["count"].sum() else 0.0
        comparison["metric"] = metric
        comparison["model"] = result_a.name
        comparison["baseline"] = result_b.name
        comparison["relative_improvement"] = (
            comparison["difference"] / baseline if baseline else float("nan")
        )
        rows.append(comparison)
    return pd.DataFrame(rows)
