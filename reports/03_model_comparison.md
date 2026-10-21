# Ranker Comparison and Phase 2 Decision

## Candidate

The Phase 2 candidate is a LightGBM `LGBMRanker` over 44 content, CF, popularity, recency, metadata, affinity, and cold/warm features.

Training design:

- Nested temporal folds: `(0.6, 0.7)`, `(0.7, 0.8)`, `(0.8, 0.9)`.
- Labels are first-time user-item events in each outer fold.
- Per query: 1 positive, 50 hard negatives, 50 time-matched random negatives, 50 or 300 uniform negatives.
- `objective="lambdarank"`, `metric="ndcg"`, early stopping on validation.
- Full-catalog scoring at evaluation time.

The training code is in `src/recommender/models/ranker.py`; the runner is `scripts/run_ranker.py`.

## Training-data bug found and fixed

The first implementation sampled only the first `max_queries` positives with `head()`. Because validation events are ordered by time, the first 100 rows of the final fold were only about 2% cold, while the full fold was 84.9% cold. The smoke model therefore looked artificially strong on warm queries and collapsed on cold ones.

The fix:

- `max_queries` now uses a seeded random sample.
- Random negatives are stratified half cold and half warm.
- Uniform negatives are added from the full candidate catalog.
- The default reproducible setting is now `negatives_uniform: 50`.

This fixed the sampling bias but not the underlying ranking failure.

## Validation results

Latest run: 150 rounds, 1,000 queries per fold, 500 validation queries, 500 evaluation events, `negatives_uniform=300`.

Training: 1,203,000 rows, 3,000 groups, best iteration 129.

| Model | Overall NDCG@10 | Overall Recall@10 | Cold NDCG@10 | Cold Recall@10 | Warm NDCG@10 | Warm Recall@10 |
|---|---:|---:|---:|---:|---:|---:|
| Recency (same 500 events) | **0.4728** | **0.7900** | **0.6479** | **0.9424** | 0.2208 | 0.5707 |
| Popularity 1-day (same 500) | 0.1929 | 0.2520 | 0.0000 | 0.0000 | **0.4706** | **0.6146** |
| Ranker | 0.0573 | 0.1380 | 0.0000 | 0.0000 | 0.1398 | 0.3366 |

The ranker's evaluation covered 295 cold and 205 warm events. It lost to recency on both segments overall and lost decisively on cold items. It also lost to 1-day popularity on warm items.

Earlier full runs were no better: overall NDCG@10 was `0.0094` before the sampling fix and `0.0323` with 50 uniform negatives. The 300-negative run improved warm NDCG@10 to `0.1398` but collapsed cold to `0.0`.

## Train/serve mismatch

On the same 100 validation events:

- Sampled validation NDCG@10: `0.9839`
- Full-catalog NDCG@10: `0.0395`

The ranker learned the sampled 100-negative distribution rather than the ~2,200-candidate catalog. Early stopping on sampled queries is misleading; only full-catalog validation is meaningful.

## Feature importance

Latest model, top features by split gain:

| Feature | Gain |
|---|---:|
| `pop_lang_7d_pct` | 552 |
| `source_affinity_pct` | 251 |
| `item_age_days` | 248 |
| `author_affinity_pct` | 225 |
| `item_text_len` | 160 |
| `recency_pct` | 150 |
| `pop_0.25d_pct` | 142 |
| `source_affinity` | 133 |
| `user_distinct` | 129 |
| `content_pct` | 119 |

`item_is_cold` had importance 9 and ranked 35th of 44. The model did not learn an effective cold-item routing signal; it relied on popularity and metadata affinities that cold items cannot match.

## Phase 2 exit gate

Required: test NDCG@10 at least 10% above the best clean baseline, paired-bootstrap CI above zero, cold performance no regression, warm performance improvement, coverage@10 at least 0.20.

| Requirement | Result | Status |
|---|---|---|
| Beat best clean baseline | `0.0573` vs `0.4376` overall | Fail |
| Cold-item performance no regression | `0.0000` vs recency `0.6479` on the subset | Fail |
| Warm-item performance improves | `0.1398` vs recency `0.2208`; below 1-day popularity `0.4706` | Fail |
| Coverage@10 >= 0.20 | `0.0656` | Fail |
| Test evaluation | Not run for the ranker | Correctly excluded |

Phase 2 **does not pass**. The ranker is not selected and was not evaluated on the test set.

## Decision

Ship the validated recency baseline, not the ranker. The ranker artifact remains under `artifacts/models/ranker_validation.joblib` as a documented negative result, not as a production model.

The one-time test evaluation was reserved for the frozen recency model: overall NDCG@10 `0.4937`, Recall@10 `0.7702`, cold NDCG@10 `0.5414`, warm NDCG@10 `0.0229`.

## Why the ranker failed

1. **Cold items dominate** and have no CF or historical item popularity. The strongest cold signal is recency, which the ranker already has but does not beat.
2. **The global model optimizes a mixed cold/warm objective.** With 85% cold positives and many warm uniform negatives, it learns popularity/metadata shortcuts that do not generalize to cold candidates.
3. **Sampled negatives are not representative** of the full catalog. Fixing the cold/warm mix improves calibration but not full-catalog personalization.
4. **Warm items are only 15% of validation events.** A single global ranker cannot specialize on them without an explicit router or separate warm model.
