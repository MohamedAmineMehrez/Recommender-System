# Error Analysis

All detailed slices are validation only. The test split was evaluated once with the frozen recency model; its summary is included at the end.

## 1. The evaluation is cold-item and fresh-item dominated

- Cold items: 3,024 / 3,560 events (84.94%).
- Warm items: 536 / 3,560 events (15.06%).
- Cold-user events: 0. Every evaluated user has training history.

Item age at prediction time explains most of the result:

| Segment | Item age | Events | Recall@10 | NDCG@10 | Median rank |
|---|---|---:|---:|---:|---:|
| Cold | 1-7 days | 2,922 | 0.8149 | 0.5177 | 4 |
| Cold | 7-30 days | 101 | 0.0000 | 0.0000 | 73 |
| Warm | 1-7 days | 164 | 0.7134 | 0.2760 | 6 |
| Warm | 7-30 days | 88 | 0.0000 | 0.0000 | 126 |
| Warm | 30-90 days | 148 | 0.0000 | 0.0000 | 476 |
| Warm | 90+ days | 136 | 0.0000 | 0.0000 | 1,244 |

**96.6% of cold events are items shared within the previous seven days.** Recency is not a shortcut inherited from a bad split; it matches the actual exposure distribution. Any model that down-weights freshness loses most of the evaluation.

This also explains why warm items are hard: many warm positives are old items that receive their first user interaction long after publication. Recency cannot rank them, but they are a small minority of events.

## 2. Recency failure is concentrated in old warm items

Recency achieves warm NDCG@10 `0.0845` overall but `0.2760` on warm items aged 1-7 days. On warm items older than seven days it is effectively zero. The 1-day popularity model is better on warm items (`0.2016`) because it captures recent engagement rather than publication freshness, but it is zero on cold items.

There is no single global signal that serves both:

| Signal | Cold NDCG@10 | Warm NDCG@10 | Overall NDCG@10 |
|---|---:|---:|---:|
| Recency | **0.5002** | 0.0845 | **0.4376** |
| Popularity 1-day | 0.0000 | **0.2016** | 0.0304 |
| ALS | 0.0000 | 0.0298 | 0.0045 |
| Content | 0.0081 | 0.0047 | 0.0076 |

## 3. Language and activity slices

Language of the positive item:

| Language | Events | Recall@10 | NDCG@10 | MRR@10 |
|---|---:|---:|---:|---:|
| English | 2,259 | 0.7118 | 0.4519 | 0.3718 |
| Portuguese | 1,301 | 0.6841 | 0.4129 | 0.3304 |

Recency is weaker on Portuguese content. The language bonus in `recency_language` over-corrects and reduces cold NDCG@10 from `0.5002` to `0.4052`, so a simple additive language match is not the right fix. A language-specific or calibrated freshness model would be needed.

User activity (training interactions):

| Training interactions | Events | Recall@10 | NDCG@10 | MRR@10 |
|---|---:|---:|---:|---:|
| 5-9 | 198 | 0.6364 | 0.3464 | 0.2585 |
| 10-19 | 343 | 0.5714 | 0.3426 | 0.2731 |
| 20-49 | 708 | 0.6610 | 0.3613 | 0.2701 |
| 50+ | 2,311 | 0.7391 | 0.4829 | 0.4040 |

Heavy users are easier: more history means more candidate exclusions and more stable preferences. The 10-19 bucket is the weakest. A personalized model should be evaluated with this slice held out rather than assuming uniform gains.

Source-domain slices vary substantially:

| Source | Events | NDCG@10 |
|---|---:|---:|
| `googlediscovery.com` | 115 | 0.5344 |
| `realm.io` | 113 | 0.5393 |
| `cloudplatform.googleblog.com` | 75 | 0.4785 |
| `oreilly.com` | 70 | 0.4721 |
| `blog.caelum.com.br` | 79 | 0.4726 |
| `hackernoon.com` | 127 | 0.4477 |
| `techcrunch.com` | 127 | 0.4203 |
| `medium.com` | 120 | 0.4211 |
| `br.blog.trello.com` | 72 | 0.3319 |

Source-domain affinity is one of the ranker's most important features (`source_affinity_pct`, gain 251), but it is learned mostly from warm items and does not transfer to cold candidates.

## 4. Ranker-specific errors

The ranker's latest full-catalog validation result:

- Cold NDCG@10: `0.0000` on 295 cold events.
- Warm NDCG@10: `0.1398` on 205 warm events.
- Overall NDCG@10: `0.0573` on 500 events.
- Cold median rank: 724 (minimum 33); cold Recall@10 is exactly zero.
- `item_is_cold` importance: 9, rank 35 of 44 features.

The model did not learn an effective cold routing rule. Its top features are language popularity, source affinity, item age, author affinity, and text length. These favor warm items with history and metadata, while cold items have no historical popularity or affinity.

The sampled-vs-full diagnostic shows the mechanism:

| Validation view | NDCG@10 |
|---|---:|
| Sampled 100-negative queries | 0.9839 |
| Full catalog, same events | 0.0395 |

The model is calibrated to the sampled group size and negative mix, not to the full catalog.

## 5. Coverage and diversity

| Model | Coverage@10 |
|---|---:|
| Random | 0.8293 |
| ALS | 0.2888 |
| Item-item CF | 0.2816 |
| Content | 0.1152 |
| Recency | 0.0873 |
| Ranker | 0.0656 |

Recency and the ranker concentrate recommendations on a small set of fresh items. That is consistent with the exposure distribution but bad for catalog coverage. The coverage bootstrap interval is not reliable because user resampling changes the recommended-item union; do not use it as a confirmatory interval.

## 6. Frozen test result

Recency was refit on train + validation and evaluated once on test:

| Segment | Events | NDCG@10 | Recall@10 | MRR@10 |
|---|---:|---:|---:|---:|
| Overall | 6,667 | 0.4937 | 0.7702 | 0.4078 |
| Cold | 6,053 | 0.5414 | 0.8429 | 0.4478 |
| Warm | 614 | 0.0229 | 0.0537 | 0.0138 |

The test set is more cold-dominated than validation after the train + validation refit (90.80% cold), and warm performance is correspondingly lower. This confirms the central finding: the task is mostly freshness ranking for items with no interaction history.

## 7. Implications

1. **Do not replace recency with a global ranker.** It loses on both cold and warm slices.
2. **Do not add more negative sampling variants as the primary fix.** Uniform 50 produced overall NDCG@10 `0.0323`; uniform 300 produced `0.0573` and cold `0.0000`. The failure is structural, not just a negative-count issue.
3. **If personalization is pursued, route explicitly.** A cold model must remain recency-dominated; a warm model could use 1-day popularity or a warm-only ranker. The two score scales must be calibrated before merging into one list.
4. **Evaluate freshness-aware baselines first.** Any proposed model must beat recency on warm items without losing the 1-7 day cold items. Report the age and language slices above alongside overall NDCG.
5. **Do not rerun test after seeing these numbers.** The frozen recency test result is the final reported result; further changes require a new protocol and a new holdout.
