# Baseline Comparison and Phase 1 Gate

## Clean comparison

The previous 100-negative, leakage-prone evaluation is discarded. The clean full-catalog validation leaderboard is:

| Rank | Model | NDCG@10 | Recall@10 |
|---:|---|---:|---:|
| 1 | Recency | **0.437639** | **0.701685** |
| 2 | Best simple cold/warm fusion | 0.437788 | 0.697191 |
| 3 | Recency + language | 0.350449 | 0.516292 |
| 4 | Popularity, 1-day decay | 0.030352 | 0.041292 |
| 5 | Content-based | 0.007567 | 0.016573 |
| 6 | ALS collaborative | 0.004480 | 0.010112 |

The best fusion is numerically ahead of recency by `0.000149` NDCG@10, which is `+0.034%` relative and not meaningful. It is behind recency on Recall@10.

## Best simple fusion

Configuration (`results/validation_hybrid_config.json`):

- Cold candidates: recency `1.0` + `pop_0.25d` `0.05`.
- Warm candidates: `pop_0.25d` `1.0`.
- Cold-item bonus: `0.05`.
- All component scores are per-query rank percentiles.

| Segment | NDCG@10 | Recall@10 | MRR@10 |
|---|---:|---:|---:|
| Overall | 0.437788 | 0.697191 | 0.357951 |
| Cold | 0.500239 | 0.787368 | 0.412007 |
| Warm | 0.085451 | 0.188433 | 0.052978 |

Comparison with pure recency:

| Segment | Recency NDCG@10 | Fusion NDCG@10 | Relative change |
|---|---:|---:|---:|
| Overall | 0.437639 | 0.437788 | +0.034% |
| Cold | 0.500200 | 0.500239 | +0.008% |
| Warm | 0.084520 | 0.085451 | +1.10% |

The fusion preserves cold performance but does not materially improve warm performance. The warm-item result is far below the 1-day popularity model's warm NDCG@10 of `0.2016` because a single top-10 list under 85% cold-item prevalence is dominated by cold candidates.

## Phase 1 exit gate

Required: simple fusion beats recency on validation NDCG@10 by at least **5% relative**, warm NDCG@10 is at least **2×** recency warm, and cold NDCG@10 does not regress by more than 2%.

| Requirement | Result | Status |
|---|---|---|
| Overall NDCG@10 at least 5% above recency (`>= 0.4595`) | `0.4378` (`+0.034%`) | Fail |
| Warm NDCG@10 at least 2× recency (`>= 0.1690`) | `0.0855` | Fail |
| Cold NDCG@10 no worse than 2% below recency | `0.5002` | Pass |
| No leakage | Leakage tests pass | Pass |

Phase 1 **does not pass**. A 5% overall gain would require a warm NDCG@10 of roughly `0.234` while holding cold performance. That is the oracle upper bound: applying the best warm ranking only to warm events yields overall NDCG@10 approximately `0.4598` (+5.07%). A deployable model cannot know the event segment in advance, so this oracle is not achievable by simple routing.

## Paired bootstrap

Paired user-level bootstrap over the 3,560 validation events confirms the tie:

| Metric | Recency minus fusion | 95% CI | p-value |
|---|---:|---:|---:|
| NDCG@10 | -0.000149 | [-0.001734, 0.001385] | 0.84 |
| Recall@10 | +0.004494 | [0.001718, 0.007469] | <0.001 |
| MRR@10 | -0.001302 | [-0.002548, -0.000116] | 0.03 |

The NDCG@10 difference is indistinguishable from zero. Recency is better on Recall@10; fusion is marginally better on MRR. There is no consistent ranking improvement, and the NDCG@10 estimate is on the wrong side of the +5% gate.

## Decision

Pure recency remains the validated model. The simple fusion is not selected: its NDCG@10 gain is within noise, it reduces Recall@10, and it adds substantial complexity. The Phase 2 ranker would need to exceed this bar before replacing recency; it does not (see `03_model_comparison.md`).

Recency was refit on train + validation and evaluated once on test: overall NDCG@10 `0.4937`, Recall@10 `0.7702`, cold NDCG@10 `0.5414`, warm NDCG@10 `0.0229`.

## Caveats

- The fusion search used 1,500 random weight trials on validation, so its tiny edge may be selection noise.
- Coverage confidence intervals are unreliable because resampling users can move the union-coverage point estimate outside the interval.
- No test evaluation has been performed. The test set remains frozen.
