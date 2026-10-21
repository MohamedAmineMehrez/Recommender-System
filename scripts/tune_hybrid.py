"""Tune the cold/warm hybrid on the validation split.

The script caches per-event candidate signals once, then searches non-negative
fusion weights and a cold-item prior. Test data is never touched.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.recommender.config import load_config  # noqa: E402
from src.recommender.data.dataset import Dataset  # noqa: E402
from src.recommender.data.features import Catalog, UserHistory  # noqa: E402
from src.recommender.evaluation.metrics import ndcg_at_k, recall_at_k  # noqa: E402
from src.recommender.models.collaborative import (  # noqa: E402
    CollaborativeRecommender,
    ItemItemCFRecommender,
)
from src.recommender.models.content_based import ContentBasedRecommender  # noqa: E402
from src.recommender.models.hybrid import _rank_percentile  # noqa: E402
from src.recommender.models.popularity import (  # noqa: E402
    GlobalPopularityRecommender,
    TimeDecayedPopularityRecommender,
)
from src.recommender.models.recency import RecencyRecommender  # noqa: E402


def build_models(config: dict) -> dict:
    return {
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


def build_cache(dataset: Dataset, catalog: Catalog, events: pd.DataFrame, models: dict) -> list[dict]:
    history = UserHistory(dataset.interactions, catalog)
    fit_events = dataset.training_interactions("validation")
    warm_items = set(int(item) for item in fit_events["contentId"].unique())
    cache: list[dict] = []
    for row in events.itertuples(index=False):
        user_id = int(row.personId)
        positive = int(row.contentId)
        timestamp = row.first_timestamp
        candidates = catalog.available_ids(timestamp)
        seen = history.seen_before(user_id, timestamp)
        if seen.size:
            candidates = candidates[~np.isin(candidates, seen)]
        if positive not in candidates:
            continue
        signals: dict[str, np.ndarray] = {}
        for name, model in models.items():
            signals[name] = _rank_percentile(model.score(user_id, candidates, timestamp)).astype("float32")
        candidate_indices = catalog.indices(candidates)
        language = history.preferred_language(user_id)
        region = history.preferred_region(user_id)
        signals["language"] = catalog.language_match(candidate_indices, language).astype("float32")
        signals["region"] = catalog.region_match(candidate_indices, region).astype("float32")
        user_events = history.user_events(user_id)
        history_indices = catalog.indices(user_events["contentId"].to_numpy())
        for field in ("author", "source"):
            history_values = catalog.author[history_indices] if field == "author" else catalog.source[history_indices]
            totals = history.affinity(user_id, history_values, timestamp, 30.0)
            candidate_values = catalog.author[candidate_indices] if field == "author" else catalog.source[candidate_indices]
            raw = np.asarray([totals.get(str(value), 0.0) for value in candidate_values], dtype="float64")
            signals[field] = _rank_percentile(raw).astype("float32")
        names = sorted(signals)
        matrix = np.vstack([signals[name] for name in names])
        cache.append(
            {
                "user_id": user_id,
                "timestamp": timestamp,
                "candidates": candidates,
                "positive_index": int(np.where(candidates == positive)[0][0]),
                "cold_mask": np.asarray([int(item) not in warm_items for item in candidates], dtype=bool),
                "signal_names": names,
                "signals": matrix,
                "segment": "warm" if positive in warm_items else "cold",
            }
        )
    return cache


def evaluate_cache(
    cache: list[dict],
    cold_weights: dict[str, float],
    warm_weights: dict[str, float],
    cold_bonus: float = 0.0,
    indices: np.ndarray | None = None,
) -> dict[str, dict[str, float]]:
    if indices is None:
        indices = np.arange(len(cache))
    records = []
    for index in indices:
        event = cache[index]
        names = event["signal_names"]
        cold = np.zeros(event["signals"].shape[1], dtype="float64")
        warm = np.zeros(event["signals"].shape[1], dtype="float64")
        cold_total = sum(cold_weights.get(name, 0.0) for name in names)
        warm_total = sum(warm_weights.get(name, 0.0) for name in names)
        for row, name in enumerate(names):
            cold += (cold_weights.get(name, 0.0) / cold_total if cold_total else 0.0) * event["signals"][row]
            warm += (warm_weights.get(name, 0.0) / warm_total if warm_total else 0.0) * event["signals"][row]
        scores = np.where(event["cold_mask"], cold + cold_bonus, warm)
        order = np.lexsort((event["candidates"], -scores))
        rank = int(np.flatnonzero(order == event["positive_index"])[0]) + 1
        records.append(
            {
                "segment": event["segment"],
                "rank": rank,
                "recall@5": recall_at_k(rank, 5),
                "recall@10": recall_at_k(rank, 10),
                "recall@20": recall_at_k(rank, 20),
                "ndcg@5": ndcg_at_k(rank, 5),
                "ndcg@10": ndcg_at_k(rank, 10),
                "ndcg@20": ndcg_at_k(rank, 20),
                "mrr@10": 1.0 / rank if rank <= 10 else 0.0,
            }
        )
    frame = pd.DataFrame(records)
    output: dict[str, dict[str, float]] = {}
    for segment in ("overall", "cold", "warm"):
        subset = frame if segment == "overall" else frame.loc[frame["segment"] == segment]
        if subset.empty:
            continue
        output[segment] = {column: float(subset[column].mean()) for column in subset.columns if column != "segment"}
        output[segment]["n_events"] = int(len(subset))
    return output


def random_search(
    cache: list[dict],
    warm_indices: np.ndarray,
    trials: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    signal_names = [
        "recency",
        "pop_0.25d",
        "pop_0.5d",
        "pop_1d",
        "pop_3d",
        "pop_7d",
        "pop_all",
        "content",
        "cf",
        "itemitem",
        "language",
        "region",
        "author",
        "source",
    ]
    seeds = [
        {"recency": 1.0},
        {"pop_0.25d": 1.0},
        {"pop_0.5d": 1.0},
        {"pop_1d": 1.0},
        {"pop_3d": 1.0},
        {"author": 1.0},
        {"source": 1.0},
        {"pop_0.25d": 1.0, "author": 0.25, "source": 0.25},
        {"recency": 0.5, "pop_0.25d": 0.5, "author": 0.2, "source": 0.2},
    ]
    rows = []
    for weights in seeds:
        metrics = evaluate_cache(cache, {"recency": 1.0}, weights, 0.0, warm_indices)["warm"]
        rows.append({"weights": weights, **metrics})
    for _ in range(trials):
        concentration = rng.choice([0.3, 0.5, 1.0])
        values = rng.dirichlet(np.full(len(signal_names), concentration))
        values[values < 0.02] = 0.0
        if values.sum() == 0:
            continue
        weights = {name: float(value) for name, value in zip(signal_names, values) if value > 0}
        metrics = evaluate_cache(cache, {"recency": 1.0}, weights, 0.0, warm_indices)["warm"]
        rows.append({"weights": weights, **metrics})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--trials", type=int, default=1500)
    args = parser.parse_args()

    config = load_config(args.config)
    dataset = Dataset.load(config["data"]["processed_dir"])
    catalog = Catalog(dataset.articles)
    fit_events = dataset.training_interactions("validation")
    models = build_models(config)
    for name, model in models.items():
        print(f"Fitting {name}...", flush=True)
        model.fit(fit_events, catalog)

    events = dataset.validation
    print("Building validation signal cache...", flush=True)
    started = time.perf_counter()
    cache = build_cache(dataset, catalog, events, models)
    print(f"Cached {len(cache)} events in {time.perf_counter() - started:.1f}s", flush=True)

    warm_indices = np.asarray([index for index, event in enumerate(cache) if event["segment"] == "warm"])
    search = random_search(cache, warm_indices, args.trials, int(config["seed"]))
    search = search.sort_values("ndcg@10", ascending=False).reset_index(drop=True)
    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    search.head(200).to_csv(results_dir / "validation_hybrid_warm_search.csv", index=False)

    best = search.iloc[0]
    warm_weights = best["weights"]
    print("\nBest warm weights:", json.dumps(warm_weights, indent=2), flush=True)
    print("Warm metrics:", best[["ndcg@10", "recall@10", "mrr@10"]].to_dict(), flush=True)

    cold_weight_variants = {
        "recency": {"recency": 1.0},
        "recency_language": {"recency": 1.0, "language": 0.05},
        "recency_region": {"recency": 1.0, "region": 0.05},
        "recency_pop": {"recency": 1.0, "pop_0.25d": 0.05},
    }
    rows = []
    for cold_name, cold_weights in cold_weight_variants.items():
        for bonus in (0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0):
            metrics = evaluate_cache(cache, cold_weights, warm_weights, bonus)
            rows.append(
                {
                    "cold_variant": cold_name,
                    "cold_bonus": bonus,
                    **{f"overall_{key}": value for key, value in metrics["overall"].items()},
                    **{f"cold_{key}": value for key, value in metrics["cold"].items()},
                    **{f"warm_{key}": value for key, value in metrics["warm"].items()},
                }
            )
    sweep = pd.DataFrame(rows)
    sweep.to_csv(results_dir / "validation_hybrid_cold_bonus.csv", index=False)
    print("\nCold bonus sweep:")
    print(
        sweep.sort_values("overall_ndcg@10", ascending=False)
        .head(15)[
            [
                "cold_variant",
                "cold_bonus",
                "overall_ndcg@10",
                "overall_recall@10",
                "cold_ndcg@10",
                "cold_recall@10",
                "warm_ndcg@10",
                "warm_recall@10",
            ]
        ]
        .to_string(index=False)
    )

    best_row = sweep.sort_values("overall_ndcg@10", ascending=False).iloc[0]
    best_config = {
        "cold_variant": best_row["cold_variant"],
        "cold_bonus": float(best_row["cold_bonus"]),
        "cold_weights": cold_weight_variants[str(best_row["cold_variant"])],
        "warm_weights": warm_weights,
        "validation_metrics": {
            "overall_ndcg@10": float(best_row["overall_ndcg@10"]),
            "overall_recall@10": float(best_row["overall_recall@10"]),
            "cold_ndcg@10": float(best_row["cold_ndcg@10"]),
            "cold_recall@10": float(best_row["cold_recall@10"]),
            "warm_ndcg@10": float(best_row["warm_ndcg@10"]),
            "warm_recall@10": float(best_row["warm_recall@10"]),
        },
    }
    with (results_dir / "validation_hybrid_config.json").open("w", encoding="utf-8") as handle:
        json.dump(best_config, handle, indent=2)
    print("\nSaved", results_dir / "validation_hybrid_config.json", flush=True)


if __name__ == "__main__":
    main()
