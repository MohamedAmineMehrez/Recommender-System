"""Run Phase 0 baselines on the validation split."""

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
from src.recommender.evaluation.evaluator import Evaluator  # noqa: E402
from src.recommender.models.collaborative import (  # noqa: E402
    CollaborativeRecommender,
    ItemItemCFRecommender,
)
from src.recommender.models.content_based import ContentBasedRecommender  # noqa: E402
from src.recommender.models.popularity import (  # noqa: E402
    GlobalPopularityRecommender,
    TimeDecayedPopularityRecommender,
)
from src.recommender.models.random_baseline import RandomRecommender  # noqa: E402
from src.recommender.models.recency import (  # noqa: E402
    RecencyLanguageRecommender,
    RecencyRecommender,
)


def main() -> None:
    config = load_config(sys.argv[1] if len(sys.argv) > 1 else "configs/base.yaml")
    dataset = Dataset.load(config["data"]["processed_dir"])
    catalog = Catalog(dataset.articles)
    split = "validation"
    train_events = dataset.training_interactions(split)
    eval_config = config["evaluation"]
    evaluator = Evaluator(
        dataset,
        ks=tuple(eval_config["ks"]),
        bootstrap_samples=int(eval_config["bootstrap_samples"]),
        seed=int(eval_config["seed"]),
        catalog=catalog,
    )

    content_model = ContentBasedRecommender(**config["content"]).fit(train_events, catalog)
    item_vectors = content_model.item_vectors

    models = {
        "random": RandomRecommender(seed=int(config["seed"])),
        "popularity": GlobalPopularityRecommender(),
        "recency": RecencyRecommender(),
        "recency_language": RecencyLanguageRecommender(
            half_life_days=float(config["baselines"]["recency_language"]["half_life_days"]),
            language_weight=float(config["baselines"]["recency_language"]["language_weight"]),
        ),
        "content": content_model,
        "collaborative_als": CollaborativeRecommender(**config["collaborative"]),
        "item_item_cf": ItemItemCFRecommender(),
    }
    for half_life in config["baselines"]["popularity_half_lives_days"]:
        models[f"popularity_decay_{half_life}d"] = TimeDecayedPopularityRecommender(
            half_life_days=float(half_life)
        )
    for half_life in config["baselines"]["recency_half_lives_days"]:
        models[f"recency_decay_{half_life}d"] = RecencyRecommender(half_life_days=float(half_life))

    metrics_dir = ROOT / "artifacts" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, model in models.items():
        model.name = name
        print(f"Fitting {name}...", flush=True)
        model.fit(train_events, catalog)
        print(f"Evaluating {name}...", flush=True)
        result = evaluator.evaluate(
            model,
            split,
            item_vectors=item_vectors,
            bootstrap=True,
            name=name,
        )
        result.to_json(metrics_dir / f"validation_{name}.json")
        result.events.to_parquet(metrics_dir / f"validation_{name}_events.parquet", index=False)
        summary = result.summary
        row = {
            "model": name,
            "split": split,
            "n_events": summary["n_events"],
            "n_users": summary["n_users"],
            "mean_candidates": summary["mean_candidates"],
            "coverage@10": summary["coverage@10"],
        }
        for metric, value in summary["micro"].items():
            row[metric] = value
            ci = summary.get("ci", {}).get("micro", {}).get(metric)
            if ci:
                row[f"{metric}_ci_low"] = ci[0]
                row[f"{metric}_ci_high"] = ci[1]
        rows.append(row)
        print(json.dumps({key: row[key] for key in ("model", "n_events", "recall@10", "ndcg@10", "mrr@10")}, indent=2), flush=True)

    results = pd.DataFrame(rows)
    results.to_csv(ROOT / "results" / "validation_baselines.csv", index=False)
    print(results[["model", "n_events", "recall@5", "recall@10", "recall@20", "ndcg@10", "mrr@10"]].to_string(index=False))


if __name__ == "__main__":
    main()
