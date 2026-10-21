# Recommender System — Improvement Plan

Status: Phase 0 complete. Phase 1 and Phase 2 were evaluated on validation; the selected model is the recency baseline because the simple fusion and LightGBM ranker did not pass their gates. Recency was refit on train + validation and evaluated once on test (NDCG@10 `0.4937`, Recall@10 `0.7702`). See `reports/`.
Scope: data pipeline, evaluation protocol, modeling architecture, engineering/reproducibility.

## Guiding principle

Do **not** chase the current 0.633/0.753 Recall@10 numbers. Those results are inflated by test leakage and by a lenient 100-negative evaluation protocol. The first clean run will produce lower scores, and that is expected.

The target is to beat a **leakage-free, time-aware, full-catalog baseline** — especially a recency baseline — with a model that handles the approximately 90% cold-item rate in the test set.

Priority order:

1. Fix evaluation and data leakage.
2. Establish strong clean baselines, especially recency and time-decayed popularity.
3. Build a cold/warm hybrid: recency/content for cold items, CF/content for warm items.
4. Add a learning-to-rank layer over all signals.
5. Package, test, document, and freeze the test set.

---

## Phase 0 — Evaluation and data foundation (1 week, mandatory)

This phase determines whether any later result is believable.

### 0.1 Canonical data pipeline

- Use **raw** `users_interactions.csv` and `shared_articles.csv` as the only sources.
- Clean articles:
  - Drop `CONTENT REMOVED` rows.
  - Deduplicate by `contentId`; keep one canonical article row.
  - Fix encoding: the raw CSV contains `U+FFFD` replacement characters (`Psic��loga`, `Rob��`). Detect and repair, or re-download the source. At minimum, strip `\ufffd` and log the count.
  - Parse timestamps as UTC.
- Clean interactions:
  - Drop interactions with missing or invalid `personId`, `contentId`, or timestamps.
  - Drop interactions on removed articles.
  - Keep event weights: VIEW 1.0, LIKE 2.0, BOOKMARK 2.5, FOLLOW 3.0, COMMENT CREATED 4.0.
- Build **first-time user-item events**: the first interaction timestamp for each `(personId, contentId)`. Use these for splitting and test positives. Training can still aggregate all events before the cutoff.
- Drop the 4 events where the article share timestamp is after the interaction timestamp, or fix those timestamps.

### 0.2 Chronological split

Use a 70/10/20 split by first-time user-item events:

| Split | Period | Purpose |
|---|---|---|
| Train | 2016-03-14 → 2016-09-15 | Fit features and models |
| Validation | 2016-09-15 → 2016-10-20 | Tune hyperparameters and weights |
| Test | 2016-10-20 → 2017-02-28 | Final report only |

These cutoffs are the 70th and 80th percentiles of first-time user-item pairs. They give roughly 28k train, 4k validation, and 8k test events. The cold-item rate is approximately 90% in all periods, so this is an intrinsic property of the data, not a split artifact.

Rules:

- Main evaluation: users with at least 5 interactions before the train cutoff.
- Cold-user segment: users whose first interaction happens after the cutoff; report separately.
- Test positives are first-time user-item interactions in the test period. Do not reuse pairs.
- Never use test labels, test interactions, or future articles in any feature.
- After tuning, retrain on train + validation and evaluate on the test set **once**.

### 0.3 Evaluation protocol

Primary evaluation: **full-catalog, time-aware ranking**.

For each test event `(user, item, t)`:

- Candidate set = all articles with `share_timestamp <= t`, excluding items the user interacted with before `t`.
- Score every candidate, rank, and compute metrics.
- There are approximately 2,800 candidates on average, so full-catalog scoring is feasible at this scale.
- Report cold items (0 train interactions) and warm items (at least 1 train interaction) separately.

Metrics:

| Metric | Notes |
|---|---|
| Recall@5/10/20 | With one positive per event, this equals HitRate@k |
| NDCG@5/10/20 | Primary ranking metric |
| MRR@10 | Rank quality |
| MAP@10 | Optional |
| Coverage@10 | Fraction of the catalog recommended |
| Novelty | Mean self-information of recommended items |
| Diversity | Intra-list cosine distance |

Reporting:

- Micro-average over events and macro-average over users.
- Bootstrap 95% confidence intervals over users (1,000 resamples).
- Paired bootstrap or Wilcoxon signed-rank test for model comparisons.
- Keep the existing 100-negative sampled metric only as a secondary diagnostic for comparability. Do not use it as the primary metric.

Baselines to implement in this phase:

- Random.
- Train-only global popularity.
- Time-decayed popularity (half-life 1, 7, 30 days).
- Recency: newest available items.
- Recency plus language match.
- Content-only.
- CF-only.

Quick full-catalog diagnostics from the current split provide a sanity check:

| Baseline | Recall@5 | Recall@10 | NDCG@10 | MRR@10 |
|---|---:|---:|---:|---:|
| Recency, all test | 0.523 | 0.689 | 0.420 | 0.337 |
| Recency, cold items (n=6,634) | 0.593 | 0.778 | 0.475 | 0.381 |
| Recency, warm items (n=914) | 0.019 | 0.043 | 0.019 | 0.012 |

The central modeling fact is that **recency is strong on cold items and useless on warm items**. Any useful model must beat recency on warm items without losing cold-item performance.

### 0.4 Leakage tests (must pass in CI)

Add explicit tests:

- `test_user_profile_excludes_test_items`
- `test_popularity_uses_train_only`
- `test_candidates_only_include_available_items`
- `test_vectorizer_fit_on_train_catalog`
- `test_no_test_events_in_features`
- `test_metrics_known_values`

### Phase 0 deliverables

- `reports/00_protocol.md`
- `reports/01_baselines.md`
- `src/recommender/evaluation/`
- `tests/test_no_leakage.py`
- Corrected baseline leaderboard with confidence intervals.

**Exit gate:** leakage tests pass; recency and popularity baselines reproduced; cold/warm counts documented.

---

## Phase 1 — Strong baselines and simple cold/warm hybrid (1 week)

### 1.1 Content-based model

- Fit TF-IDF only on the train catalog.
- Use word n-grams (1–2) plus char n-grams (3–5), `sublinear_tf=True`, `min_df=3`, `max_df=0.5`, maximum 20k word features and 10k char features.
- Use language-specific stopwords. Prefer char n-grams for robustness to the corrupted Portuguese text.
- L2-normalize item vectors.
- User profile = weighted average of item vectors using:
  - event strength;
  - exponential recency decay (half-life 7, 30, or 90 days, tuned on validation).
- Score with cosine similarity.
- Add metadata signals: article age at prediction time, language match, author affinity, source-domain affinity, and region match.
- Optional: multilingual sentence embeddings such as `paraphrase-multilingual-MiniLM-L12-v2` as a second content signal. Compare with TF-IDF; do not replace it initially.

### 1.2 Collaborative filtering

Replace SVD-on-zeros with an implicit-feedback model:

- Use `implicit.als.AlternatingLeastSquares` or `implicit.bpr.BayesianPersonalizedRanking`.
- Tune factors {32, 64, 128}, regularization {0.01, 0.1, 1.0}, alpha {1, 10, 40}, and 15–30 iterations.
- Use confidence = `1 + alpha * log1p(weight)`.
- Train on train interactions only.
- Add item-item cosine as a simple warm-item baseline.
- CF score is 0 for cold items. Include an `is_cold` flag.

### 1.3 Popularity and recency

- Time-decayed popularity: `sum(weight * exp(-(t - event_time) / tau))`, with tau in {1, 7, 30} days.
- Compute popularity per language and per region.
- Recency score: `exp(-(t - share_time) / tau)`, with tau tuned.
- Add a "trending last 24h/7d" feature.

### 1.4 Simple hybrid

Use cold/warm routing with rank fusion:

- Cold item (0 train interactions): combine content similarity, recency, time-decayed popularity, and language match.
- Warm item (at least 1 train interaction): combine content, CF/ALS, time-decayed popularity, recency, and author/source affinity.
- Normalize each signal by per-query rank percentile or z-score, not raw score.
- Tune non-negative weights on validation with Optuna or grid search.
- This is the minimum viable replacement for the current `HybridRecommender` with its inverted 1:100 weights.

### Phase 1 deliverables

- `src/recommender/models/{popularity,content_based,collaborative,hybrid}.py`
- `reports/02_baseline_comparison.md`
- Best simple model beating recency on validation NDCG@10 by at least 5% relative.

**Exit gate:** best clean baseline established; cold/warm segment metrics reported; no leakage.

---

## Phase 2 — Recommended architecture: retrieval plus LTR ranking (1–2 weeks)

### 2.1 Architecture

```
Raw events/articles
      ↓
clean / dedup / encoding / time-aware split
      ↓
┌─────────────────────────────────────────────┐
│ signal layer                                │
│  • item text + metadata + age               │
│  • recency-weighted user profiles           │
│  • time-decayed popularity (global/lang/reg)│
│  • ALS/BPR user-item factors                │
│  • session co-occurrence                    │
└─────────────────────────────────────────────┘
      ↓
┌─────────────────────────────────────────────┐
│ candidate generation                        │
│  recency top-300                            │
│  popularity top-300                         │
│  content top-300                            │
│  ALS top-300                                │
│  session top-100                            │
│  → union, dedup, cap ~1,000                 │
└─────────────────────────────────────────────┘
      ↓
┌─────────────────────────────────────────────┐
│ feature builder (user, item, timestamp)     │
│  content, CF, recency, popularity, language │
│  author, source, session, cold/warm flags   │
└─────────────────────────────────────────────┘
      ↓
┌─────────────────────────────────────────────┐
│ LightGBM LambdaRank                         │
│  objective=lambdarank, metric=ndcg@10       │
│  early stopping on validation               │
└─────────────────────────────────────────────┘
      ↓
┌─────────────────────────────────────────────┐
│ re-ranking                                  │
│  MMR diversity, freshness guardrail,        │
│  language/region filters, dedup, top-K      │
└─────────────────────────────────────────────┘
      ↓
┌─────────────────────────────────────────────┐
│ full-catalog evaluator                      │
│  cold/warm segments, bootstrap CIs          │
└─────────────────────────────────────────────┘
```

At the current scale of approximately 3k items, you can skip candidate generation and score all available items in the ranker. Use the retrieval layer for production and to measure retrieval recall.

### 2.2 Feature set (approximately 30–40 features)

User features:

- Interaction count, distinct items, average strength.
- Days since last interaction.
- Preferred language and region/country.
- Author affinity and source-domain affinity.
- Session length and last-N item vectors.
- Activity bucket.

Item features:

- Age at prediction time in hours; share hour and weekday.
- Language, content type, source domain, author.
- Title and text length.
- Interaction count before `t`; unique users before `t`.
- Time-decayed popularity (global, language, region, source).
- Trending score over 24 hours and 7 days.
- Cold flag.

Interaction features:

- Content cosine (TF-IDF, char n-grams, optional embeddings).
- ALS/BPR score.
- Item-item CF score.
- Author and source affinity.
- Language match and region match.
- Session similarity.
- Recency rank, popularity rank, and content rank.
- Time since the user's last interaction with the same author, source, or topic.

### 2.3 Training data and negative sampling

For each training positive `(user, item, t)`:

- 50 hard negatives from the top of recency, popularity, content, and CF lists, excluding items the user already interacted with.
- 50 time-matched random negatives: articles shared within ±3 days, same language, and similar popularity decile.
- Exclude items the user interacted with before `t`.

This is critical: random negatives from the full catalog make recency look artificially strong. Time-matched negatives force the ranker to learn personalization.

To train on cold items, use a nested temporal split inside train: features at `t0`, labels from `(t0, t1]`. This simulates the cold-item distribution at test time. Oversample cold positives if needed.

### 2.4 Ranker

- LightGBM `LGBMRanker`:
  - `objective='lambdarank'`
  - `metric='ndcg'`, `ndcg_eval_at=[5,10,20]`
  - `learning_rate=0.03`, `num_leaves=31–63`, `min_data_in_leaf=50–100`
  - `feature_fraction=0.8`, `bagging_fraction=0.8`, `lambda_l2=1–10`
  - `num_boost_round=1000`, early stopping at 100 rounds.
- Groups = one query per positive (1 positive + 100 negatives).
- Tune with Optuna on validation NDCG@10.
- Alternative: logistic regression ranker for interpretability.
- Include `is_cold` and `is_warm` interaction features. The ranker learns the routing implicitly. Keep an explicit cold fallback if cold-item performance lags.

### 2.5 Ablations

Run on validation only:

- Remove recency features.
- Remove content features.
- Remove CF features.
- Remove popularity features.
- Remove session features.
- Compare cold/warm routing with a single global model.
- Compare TF-IDF with sentence embeddings.
- Compare ALS, BPR, and item-item CF.

### Phase 2 deliverables

- `src/recommender/models/ranker.py`
- `src/recommender/pipelines/train_ranker.py`
- `reports/03_model_comparison.md`
- Feature-importance and ablation tables.

**Exit gate:** test NDCG@10 improves by at least 10% relative over the best clean baseline with a paired-bootstrap 95% CI excluding zero; cold-item performance does not regress; warm-item performance improves.

---

## Phase 3 — Error analysis, packaging, and reproducibility (1 week)

### 3.1 Error analysis

Slice results by:

- Cold versus warm items.
- Cold versus warm users.
- Language (English/Portuguese).
- User activity bucket.
- Item age bucket.
- Source domain.
- Author popularity.

Look for:

- Cold-item failures where recency wins but content should help.
- Warm-item failures where CF is noisy.
- Language or region mismatch.
- Popularity bias and low coverage.
- Cases where the ranker over-prioritizes freshness.

Optional improvements if justified by error analysis:

- Multilingual sentence embeddings.
- Session-based recommender (item co-occurrence or SASRec).
- Cross-lingual user profiles.
- Diversity re-ranking.

### 3.2 Repository architecture

```
recommender-system/
├── README.md
├── requirements.txt
├── pyproject.toml
├── .gitignore
├── configs/
│   ├── base.yaml
│   └── experiments/
│       ├── content.yaml
│       ├── cf.yaml
│       ├── fusion.yaml
│       └── ranker.yaml
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── src/recommender/
│   ├── __init__.py
│   ├── config.py
│   ├── data/
│   │   ├── load.py
│   │   ├── clean.py
│   │   ├── split.py
│   │   └── features.py
│   ├── models/
│   │   ├── base.py
│   │   ├── popularity.py
│   │   ├── recency.py
│   │   ├── content_based.py
│   │   ├── collaborative.py
│   │   ├── hybrid.py
│   │   └── ranker.py
│   ├── evaluation/
│   │   ├── metrics.py
│   │   ├── evaluator.py
│   │   ├── segments.py
│   │   ├── bootstrap.py
│   │   └── report.py
│   └── pipelines/
│       ├── make_dataset.py
│       ├── train.py
│       ├── evaluate.py
│       └── recommend.py
├── scripts/
│   ├── run_all.py
│   └── run_experiments.py
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_error_analysis.ipynb
│   └── 03_results.ipynb
├── tests/
│   ├── test_metrics.py
│   ├── test_split.py
│   ├── test_no_leakage.py
│   └── test_models.py
├── artifacts/
│   ├── models/
│   ├── metrics/
│   └── figures/
└── reports/
    ├── 00_protocol.md
    ├── 01_baselines.md
    ├── 02_baseline_comparison.md
    ├── 03_model_comparison.md
    └── 04_error_analysis.md
```

Key interfaces:

- `Recommender.fit(train_data, catalog)`
- `Recommender.score(user_id, candidate_ids, timestamp)`
- `Recommender.recommend(user_id, k, timestamp)`
- `Evaluator.evaluate(model, events, catalog, k, segments)`
- `Evaluator.compare(models)` with paired bootstrap.

### 3.3 Experiment tracking and CI

- YAML config per experiment: seed, cutoffs, model parameters, feature flags.
- `results/experiments.csv` columns: run_id, git_sha, config_hash, model, split, segment, Recall@5/10/20, NDCG@5/10/20, MRR, coverage, novelty, CI low/high.
- Save model artifacts with joblib or the LightGBM native format.
- GitHub Actions:
  - install dependencies;
  - `ruff check .`;
  - `mypy src`;
  - `pytest -q`;
  - run a smoke experiment on a small sample.
- One-command reproduction:
  - `python -m src.recommender.pipelines.make_dataset --config configs/base.yaml`
  - `python -m src.recommender.pipelines.train --config configs/experiments/ranker.yaml`
  - `python -m src.recommender.pipelines.evaluate --config configs/experiments/ranker.yaml --split test`
  - `pytest -q`

### 3.4 Cleanup

- Remove `.history` from git.
- Stop committing raw data and notebook outputs; use `.gitignore` or Git LFS/DVC.
- Keep only a small sample dataset in the repository, or document the download.
- Pin dependencies.
- Rewrite the README with the problem statement, data, protocol, results, and how to run.
- Delete the empty cells at the end of the preparation notebook; move the final comparison into a report.

---

## Exact acceptance gates

| Phase | Gate |
|---|---|
| 0 | Leakage tests pass; recency/popularity baselines reproduced; cold/warm counts and confidence intervals reported. |
| 1 | Simple fusion beats recency on validation NDCG@10 by at least 5% relative; warm-item NDCG@10 at least 2× recency warm; cold-item NDCG@10 no worse than recency by more than 2%. |
| 2 | Test NDCG@10 improves by at least 10% relative over the best clean baseline, with paired-bootstrap 95% CI greater than 0; overall Recall@10 at least 0.72 (indicative); cold Recall@10 at least 0.65 (indicative); warm NDCG@10 at least 2× baseline; coverage@10 at least 0.20; latency below 100 ms per user in batch. |
| 3 | Clean checkout reproduces results with one command; tests and CI pass; final report complete with ablations and error analysis. |

The 0.72 and 0.65 numbers are indicative. Finalize absolute targets after Phase 0 baselines, but keep the relative gates fixed.

---

## Timeline

| Week | Focus | Output |
|---|---|---|
| 1 | Phase 0: data, evaluation, leakage tests | Corrected protocol and baselines |
| 2 | Phase 1: recency/popularity/content/ALS + cold/warm fusion | Best simple model and baseline report |
| 3 | Phase 2: features + LightGBM ranker + ablations | Final model candidate |
| 4 | Phase 3: error analysis, packaging, CI, report | Reproducible final deliverable |

Part-time estimate: 6–8 weeks.

---

## Minimum viable path

If time is limited, do only this:

1. Phase 0: fix leakage and switch to full-catalog, time-aware evaluation.
2. Phase 1: build **recency for cold items plus content/popularity for warm items**, with validation-tuned rank fusion.
3. Report cold/warm metrics with bootstrap confidence intervals.

This alone should produce a credible improvement over the current CF-dominant hybrid. Only then add ALS and the LightGBM ranker.

---

## Main risks

- **Cold-item dominance:** approximately 90% of test interactions are cold items. CF alone will never work; content and recency must carry cold items.
- **Recency shortcut:** random negatives make recency look artificially strong. Use time-matched negatives and full-catalog evaluation.
- **Corrected numbers drop:** the old 0.633/0.753 is not a valid target. Expect lower clean scores.
- **Text corruption:** Portuguese text contains replacement characters; fix before tuning TF-IDF.
- **Small dataset:** approximately 3k items and 1k users. Prefer simple models, regularization, and confidence intervals over deep learning.
- **Test reuse:** freeze the test set; tune only on validation.
- **Reproducibility:** pin dependencies, seed everything, and save configs and artifacts.
