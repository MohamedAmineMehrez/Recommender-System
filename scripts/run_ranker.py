"""Train and evaluate the Phase 2 ranker on validation (test stays frozen)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.recommender.config import load_config  # noqa: E402
from src.recommender.data.dataset import Dataset  # noqa: E402
from src.recommender.data.features import Catalog  # noqa: E402
from src.recommender.evaluation.evaluator import Evaluator  # noqa: E402
from src.recommender.models.ranker import RankerRecommender  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--split", default="validation", choices=["validation", "test"])
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--max-validation-queries", type=int, default=None)
    parser.add_argument("--max-eval-events", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    dataset = Dataset.load(config["data"]["processed_dir"])
    catalog = Catalog(dataset.articles)
    split = args.split
    fit_events = dataset.training_interactions(split)
    print(f"Fitting ranker for split={split} on {len(fit_events)} interactions...", flush=True)
    ranker_config = config.get("ranker", {})
    model = RankerRecommender(
        config=config,
        negatives_hard=int(ranker_config.get("negatives_hard", 50)),
        negatives_random=int(ranker_config.get("negatives_random", 50)),
        negatives_uniform=int(ranker_config.get("negatives_uniform", 50)),
        seed=int(config["seed"]),
        max_queries_per_fold=args.max_queries,
        max_validation_queries=args.max_validation_queries,
    )
    validation_events = dataset.validation if split == "validation" else None
    model.fit(
        fit_events,
        catalog,
        all_interactions=dataset.interactions,
        validation_events=validation_events,
        num_boost_round=args.rounds,
    )
    print("Training stats:", json.dumps(model.training_stats, indent=2), flush=True)

    artifacts = ROOT / "artifacts" / "models"
    artifacts.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model.model,
            "feature_builder": model.feature_builder,
            "training_stats": model.training_stats,
            "config": model.get_params(),
        },
        artifacts / f"ranker_{split}.joblib",
    )

    evaluator = Evaluator(
        dataset,
        ks=tuple(config["evaluation"]["ks"]),
        bootstrap_samples=args.bootstrap,
        seed=int(config["evaluation"]["seed"]),
        catalog=catalog,
    )
    item_vectors = None
    if model.feature_builder is not None and "content" in model.feature_builder.models:
        item_vectors = model.feature_builder.models["content"].item_vectors
    print("Evaluating ranker...", flush=True)
    result = evaluator.evaluate(
        model,
        split,
        item_vectors=item_vectors,
        bootstrap=args.bootstrap > 0,
        max_events=args.max_eval_events,
        name="ranker",
    )
    metrics_dir = ROOT / "artifacts" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    result.to_json(metrics_dir / f"{split}_ranker.json")
    result.events.to_parquet(metrics_dir / f"{split}_ranker_events.parquet", index=False)
    summary = result.summary
    print(json.dumps({k: v for k, v in summary.items() if k not in {"segments", "ci", "macro"}}, indent=2), flush=True)
    print("Segments:", flush=True)
    for segment, values in summary.get("segments", {}).items():
        print(
            segment,
            {
                key: round(values["micro"][key], 4)
                for key in ("recall@5", "recall@10", "recall@20", "ndcg@10", "mrr@10")
            },
            flush=True,
        )


if __name__ == "__main__":
    main()
