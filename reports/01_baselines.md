# Clean Baselines

All numbers are full-catalog, time-aware **validation** metrics. The test split is untouched.

## Overall results

| Model | Recall@5 | Recall@10 | Recall@20 | NDCG@10 | MRR@10 | Coverage@10 |
|---|---:|---:|---:|---:|---:|---:|
| **Recency** | **0.5152** | **0.7017** | **0.8056** | **0.4376** | **0.3566** | 0.0873 |
| Recency + language | 0.4357 | 0.5163 | 0.5559 | 0.3504 | 0.2983 | 0.0916 |
| Popularity, 1-day decay | 0.0371 | 0.0413 | 0.0486 | 0.0304 | 0.0269 | 0.0069 |
| Popularity, 7-day decay | 0.0183 | 0.0329 | 0.0472 | 0.0147 | 0.0093 | 0.0075 |
| Content-based | 0.0104 | 0.0166 | 0.0317 | 0.0076 | 0.0048 | 0.1152 |
| Popularity, 30-day decay | 0.0048 | 0.0101 | 0.0284 | 0.0049 | 0.0034 | 0.0089 |
| ALS collaborative | 0.0048 | 0.0101 | 0.0140 | 0.0045 | 0.0028 | 0.2888 |
| Item-item CF | 0.0039 | 0.0067 | 0.0115 | 0.0032 | 0.0022 | 0.2816 |
| Global popularity | 0.0017 | 0.0056 | 0.0081 | 0.0023 | 0.0013 | 0.0085 |
| Random | 0.0022 | 0.0037 | 0.0079 | 0.0014 | 0.0007 | 0.8293 |

## Cold-item results

Cold items have no training interactions and account for 3,024 of 3,560 validation events (84.94%).

| Model | Recall@10 | NDCG@10 | MRR@10 |
|---|---:|---:|---:|
| **Recency** | **0.7874** | **0.5002** | **0.4120** |
| Recency + language | 0.5926 | 0.4052 | 0.3463 |
| Content-based | 0.0179 | 0.0081 | 0.0051 |
| Random | 0.0040 | 0.0015 | 0.0008 |
| Popularity (all variants) | 0.0000 | 0.0000 | 0.0000 |
| ALS / item-item CF | 0.0000 | 0.0000 | 0.0000 |

## Warm-item results

Warm items have at least one training interaction and account for 536 of 3,560 validation events (15.06%).

| Model | Recall@10 | NDCG@10 | MRR@10 |
|---|---:|---:|---:|
| **Popularity, 1-day decay** | **0.2743** | **0.2016** | **0.1788** |
| Popularity, 7-day decay | 0.2183 | 0.0979 | 0.0616 |
| Recency | 0.2183 | 0.0845 | 0.0443 |
| Popularity, 30-day decay | 0.0672 | 0.0325 | 0.0225 |
| ALS collaborative | 0.0672 | 0.0298 | 0.0187 |
| Item-item CF | 0.0448 | 0.0214 | 0.0145 |
| Global popularity | 0.0373 | 0.0150 | 0.0087 |
| Content-based | 0.0093 | 0.0047 | 0.0033 |
| Recency + language | 0.0858 | 0.0414 | 0.0274 |

## Observations

- **Recency is the strongest overall model by a wide margin.** It wins because cold items dominate the evaluation and the newest available item is frequently the held-out positive.
- **Recency is weak on warm items** (warm NDCG@10 `0.0845`), while 1-day popularity is much stronger there (`0.2016`). This is the expected cold/warm split.
- **Content and CF cannot carry the full catalog.** Content has a non-zero cold signal but very low absolute ranking quality; ALS and item-item CF are exactly zero on cold items because they have no factors or neighbors for them.
- **Recency + language hurts.** The language weight suppresses newer items and reduces cold NDCG@10 from `0.5002` to `0.4052`.
- **Coverage is inversely related to concentration.** Random covers 82.9% of the catalog at 10, while recency covers 8.7%. Coverage should not be optimized at the cost of the primary ranking metric.
- The global popularity and 30-day decay models are nearly indistinguishable because old interactions dominate both.

## Frozen test result

Recency was selected on validation, refit on train + validation, and evaluated once on test:

| Segment | Events | Recall@5 | Recall@10 | Recall@20 | NDCG@10 | MRR@10 |
|---|---:|---:|---:|---:|---:|---:|
| Overall | 6,667 | 0.6082 | 0.7702 | 0.8443 | 0.4937 | 0.4078 |
| Cold | 6,053 | 0.6679 | 0.8429 | 0.9209 | 0.5414 | 0.4478 |
| Warm | 614 | 0.0195 | 0.0537 | 0.0896 | 0.0229 | 0.0138 |

Overall NDCG@10 95% CI: `[0.4665, 0.5261]`. Coverage@10 is `0.1687`.

The test set has 6,053 cold and 614 warm events against the final train + validation fitting window. Relative to the train cutoff alone, the cold rate is 94.02%; after the required refit on train + validation, it is 90.80%.

## Phase 0 verdict

Phase 0 passes. The clean baseline is reproducible and the old leaked/sampled figures are not used. The target for Phase 1 was explicit: improve warm-item quality without losing cold-item recency performance. No tested model did so.
