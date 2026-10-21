"""Freeze the selected validation model and run the one-time test evaluation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.recommender.config import load_config  # noqa: E402
from src.recommender.data.dataset import Dataset  # noqa: E402
from src.recommender.data.features import Catalog  # noqa: E402
from src.recommender.evaluation.evaluator import Evaluator, compare_results  # noqa: E402
from src.recommender.models.hybrid import HybridRecommender  # noqa: E402
from src.recommender.models.popularity import TimeDecayedPopularityRecommender  # noqa: E402
from src.recommender.models.recency import RecencyRecommender  # noqa: E402


def metrics_frame(result) -> dict:
    summary = result.summary
    return {
        "n_events": summary["n_events"],
        "n_users": summary["n_users"],
        "mean_candidates": summary["mean_candidates"],
        "coverage@10": summary["coverage@10"],
        "overall": summary["micro"],
        "cold": summary["segments"]["cold"]["micro"],
        "warm": summary["segments"]["warm"]["micro"],
    }


def main() -> None:
    config = load_config("configs/base.yaml")
    dataset = Dataset.load(config["data"]["processed_dir"])
    catalog = Catalog(dataset.articles)
    evaluator = Evaluator(
        dataset,
        ks=tuple(config["evaluation"]["ks"]),
        bootstrap_samples=int(config["evaluation"]["bootstrap_samples"]),
        seed=int(config["evaluation"]["seed"]),
        catalog=catalog,
    )

    metrics_dir = ROOT / "artifacts" / "metrics"
    results_dir = ROOT / "results"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    print("Evaluating frozen recency on validation...", flush=True)
    recency = RecencyRecommender().fit(dataset.training_interactions("validation"), catalog)
    recency_validation = evaluator.evaluate(recency, "validation", bootstrap=True, name="recency")
    recency_validation.to_json(metrics_dir / "validation_recency_final.json")
    recency_validation.events.to_parquet(
        metrics_dir / "validation_recency_final_events.parquet", index=False
    )

    cold_total = 1.0 + 0.05
    fusion = HybridRecommender(
        components={
            "recency": RecencyRecommender(),
            "pop_0.25d": TimeDecayedPopularityRecommender(half_life_days=0.25),
        },
        cold_weights={"recency": 1.0 / cold_total, "pop_0.25d": 0.05 / cold_total},
        warm_weights={"pop_0.25d": 1.0},
        normalization="rank",
        use_metadata=False,
        cold_bonus=0.05,
    )
    fusion.name = "fusion"
    fusion.fit(dataset.training_interactions("validation"), catalog)

    print("Evaluating frozen fusion on validation...", flush=True)
    fusion_validation = evaluator.evaluate(fusion, "validation", bootstrap=True, name="fusion")
    fusion_validation.to_json(metrics_dir / "validation_fusion.json")
    fusion_validation.events.to_parquet(metrics_dir / "validation_fusion_events.parquet", index=False)

    comparison = compare_results(
        recency_validation,
        fusion_validation,
        bootstrap_samples=int(config["evaluation"]["bootstrap_samples"]),
        seed=int(config["evaluation"]["seed"]),
    )
    comparison.to_csv(results_dir / "validation_recency_vs_fusion_bootstrap.csv", index=False)
    print(comparison.to_string(index=False), flush=True)

    print("Evaluating frozen recency on test (one-time)...", flush=True)
    recency_test = RecencyRecommender().fit(dataset.training_interactions("test"), catalog)
    test_result = evaluator.evaluate(recency_test, "test", bootstrap=True, name="recency")
    test_result.to_json(metrics_dir / "test_recency.json")
    test_result.events.to_parquet(metrics_dir / "test_recency_events.parquet", index=False)

    final = {
        "selected_model": "recency",
        "selection_split": "validation",
        "test_evaluated": True,
        "validation": metrics_frame(recency_validation),
        "validation_fusion": metrics_frame(fusion_validation),
        "test": metrics_frame(test_result),
    }
    with (results_dir / "final_model.json").open("w", encoding="utf-8") as handle:
        json.dump(final, handle, indent=2)

    print(
        json.dumps(
            {
                "validation_ndcg@10": final["validation"]["overall"]["ndcg@10"],
                "validation_recall@10": final["validation"]["overall"]["recall@10"],
                "test_ndcg@10": final["test"]["overall"]["ndcg@10"],
                "test_recall@10": final["test"]["overall"]["recall@10"],
                "test_cold_ndcg@10": final["test"]["cold"]["ndcg@10"],
                "test_warm_ndcg@10": final["test"]["warm"]["ndcg@10"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
