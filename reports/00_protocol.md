# Evaluation Protocol

Status: final model frozen. The test split was evaluated **once** with recency after model selection.

## Scope

This report defines the leakage-free, time-aware protocol used in this repository. All model selection and hyperparameter tuning use the validation split only. The test split is used once, after the model is frozen.

## Data sources and cleaning

Only the raw files are used:

- `data/shared_articles.csv`
- `data/users_interactions.csv`

Cleaning steps:

1. Drop `CONTENT REMOVED` articles.
2. Deduplicate articles by `contentId`, keeping one canonical row.
3. Strip `U+FFFD` replacement characters from text; 232 cells were affected.
4. Parse timestamps as UTC.
5. Drop invalid or missing user, item, and timestamp rows.
6. Drop interactions on removed articles (74 deleted IDs; 408 interactions).
7. Drop the 95 interactions dated before the article share timestamp.
8. Keep event weights: VIEW 1.0, LIKE 2.0, BOOKMARK 2.5, FOLLOW 3.0, COMMENT CREATED 4.0.
9. Build first-time user-item events for splitting and test positives.
10. Training may aggregate all events before the cutoff; evaluation only uses first-time events.

Resulting data:

| Quantity | Value |
|---|---:|
| Clean articles | 3,047 |
| Clean interactions | 71,809 |
| First-time user-item events | 40,420 |
| Events in train/validation/test after eligibility filtering | 38,521 |
| Eligible users | 915 |

The 40,420 first-time events include 885 events from cold users and events from users with fewer than five training interactions. The eligibility filter removes those users, leaving 38,521 evaluable events and 915 eligible users.

## Chronological split

Cutoffs are the 70th and 80th percentiles of first-time user-item events.

| Split | Period | Events |
|---|---|---:|
| Train | 2016-03-14 to `2016-09-15 22:12:39.8Z` | 28,294 |
| Validation | `2016-09-15 22:12:39.8Z` to `2016-10-20 17:06:33.6Z` | 3,560 |
| Test | `2016-10-20 17:06:33.6Z` to 2017-02-28 | 6,667 |

The main evaluation population is users with at least five interactions before the train cutoff. The validation and test evaluation sets contain no cold users: every evaluated user has training history.

Item segments are defined from the model's fitting window:

- Validation model: interactions before the train cutoff.
- Test model: interactions before the validation cutoff (train + validation), as required by the final refit.

| Segment | Validation | Test |
|---|---:|---:|
| Cold item (no interactions in fitting window) | 3,024 events (84.94%) | 6,053 events (90.80%) |
| Warm item (at least one interaction in fitting window) | 536 events (15.06%) | 614 events (9.20%) |

Relative to the **train cutoff only**, 94.02% of test items are cold. The final test model is refit on train + validation, so the evaluated test cold rate is 90.80%.

Cold users are users with no interactions before the split cutoff. They are not part of the current evaluated sets, so the cold-item results are **warm-user x cold-item** results.

## Ranking protocol

For each validation event `(user, item, t)`:

1. Candidate set = all articles with `share_timestamp <= t`.
2. Remove items the user interacted with before `t`, including events outside the training window.
3. Score every candidate with the model.
4. Sort descending by score; break ties by ascending `contentId` using `np.lexsort((candidates, -scores))`.
5. Compute metrics from the rank of the single held-out positive.

The average candidate set has 2,310.69 items on validation. Full-catalog scoring is feasible at this scale and avoids sampled-negative artifacts.

## Metrics

Primary:

- Recall@5/10/20 (with one positive, this equals HitRate@k)
- NDCG@5/10/20 (primary ranking metric: NDCG@10)
- MRR@10
- Coverage@10
- Novelty@10 and intra-list diversity@10 are reported as secondary diagnostics.

Aggregation:

- Micro-average over events is the headline number.
- Macro-average over users is retained in artifacts.
- Bootstrap 95% confidence intervals use 1,000 user-level resamples and seed 42.

Known limitation: the coverage interval resamples users and recomputes the union of recommended items, so the point estimate can fall outside the interval. Coverage is reported without treating that interval as confirmatory.

## Leakage controls

Passing tests in `tests/test_no_leakage.py`:

- `test_user_profile_excludes_test_items`
- `test_popularity_uses_train_only`
- `test_candidates_only_include_available_items`
- `test_vectorizer_fit_on_train_catalog`
- `test_no_test_events_in_features`

Additional guarantees:

- Every model is fit on `dataset.training_interactions(split)`.
- Candidate availability is checked at the event timestamp, not at the split cutoff.
- The content vectorizer is fit only on the training catalog.
- CF and popularity are trained only on the fitting window.
- The test split is not read for model selection.

## Reproducible commands

```powershell
& "C:\Users\Mega Pc\anaconda3\python.exe" -m src.recommender.pipelines.make_dataset --config configs/base.yaml
& "C:\Users\Mega Pc\anaconda3\python.exe" scripts\run_baselines.py configs/base.yaml
& "C:\Users\Mega Pc\anaconda3\python.exe" scripts\tune_hybrid.py --config configs/base.yaml --trials 1500
& "C:\Users\Mega Pc\anaconda3\python.exe" scripts\run_ranker.py --split validation --rounds 200 --max-validation-queries 1000 --max-eval-events 500 --bootstrap 0
& "C:\Users\Mega Pc\anaconda3\python.exe" scripts\finalize_model.py
& "C:\Users\Mega Pc\anaconda3\python.exe" -m pytest -q
```

`scripts/finalize_model.py` is the single test-entry point. It refits recency on train + validation, evaluates test once, and saves the frozen result. Do not rerun it after inspecting test metrics.

## Frozen test result

The selected model is recency, refit on train + validation. The test set was evaluated once:

| Segment | Events | NDCG@10 | Recall@10 | MRR@10 |
|---|---:|---:|---:|---:|
| Overall | 6,667 | 0.4937 | 0.7702 | 0.4078 |
| Cold | 6,053 | 0.5414 | 0.8429 | 0.4478 |
| Warm | 614 | 0.0229 | 0.0537 | 0.0138 |

Overall NDCG@10 bootstrap 95% CI: `[0.4665, 0.5261]`. Coverage@10 is `0.1687`; average candidates per event is `2,680.59`. The ranker was not evaluated on test.

Artifacts: `artifacts/metrics/test_recency.json`, `artifacts/metrics/test_recency_events.parquet`, `results/final_model.json`.

## Phase 0 exit gate

| Requirement | Status |
|---|---|
| Leakage tests pass | Pass |
| Recency baseline reproduced | Pass |
| Popularity baselines reproduced | Pass |
| Cold/warm counts documented | Pass |
| Full-catalog, time-aware evaluation | Pass |
| Bootstrap intervals | Pass, with coverage caveat |

Phase 0 is complete. The clean recency validation result is NDCG@10 `0.437639`, far below the previous leaked 100-negative numbers, as expected. The frozen test result is NDCG@10 `0.493690`.
