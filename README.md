# Recommender System

[![CI](https://github.com/MohamedAmineMehrez/Recommender-System/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/MohamedAmineMehrez/Recommender-System/actions/workflows/ci.yml)

Leakage-free, time-aware recommender experiments on the shared-articles interaction dataset.

**Selected model:** recency. **Frozen test:** NDCG@10 `0.4937`, Recall@10 `0.7702`.

## Problem

Given a user and a prediction time, rank the articles that are available at that time and that the user has not seen before. The dataset is strongly cold-item dominated: about 85% of validation positives and 91% of test positives have no interactions in the final fitting window.

The original project reported high Recall@10 using a 100-negative sample that included future information. This repository replaces that protocol with full-catalog, time-aware evaluation and uses the test set once for the frozen model.

## Protocol

- Clean articles/interactions, remove `CONTENT REMOVED`, strip U+FFFD, parse UTC.
- Split first-time user-item events chronologically at the 70th and 80th percentiles.
- Fit models on the training window only.
- Candidate set = articles shared at or before the event time, minus items the user saw before that time.
- Rank all candidates with deterministic item-ID tie-breaking.
- Tune on validation only; do not touch test until the model is frozen.
- Report cold-item and warm-item metrics separately.

Details: `reports/00_protocol.md`.

## Results

Validation, full-catalog, micro-averaged:

| Model | NDCG@10 | Recall@10 | Warm NDCG@10 | Cold NDCG@10 |
|---|---:|---:|---:|---:|
| **Recency (selected)** | **0.4376** | **0.7017** | 0.0845 | **0.5002** |
| Popularity, 1-day decay | 0.0304 | 0.0413 | **0.2016** | 0.0000 |
| Content-based | 0.0076 | 0.0166 | 0.0047 | 0.0081 |
| ALS collaborative | 0.0045 | 0.0101 | 0.0298 | 0.0000 |
| Ranker (failed candidate) | 0.0573* | 0.1380* | 0.1398* | 0.0000* |

\*Ranker numbers are on a 500-event validation subset; it lost to recency on cold and warm segments.

The simple cold/warm fusion reaches NDCG@10 `0.4378`, a `+0.034%` change versus recency. Paired bootstrap gives an NDCG@10 difference of `-0.000149` with 95% CI `[-0.001734, 0.001385]`; this is a tie, not the required +5%. Fusion is not selected.

The LightGBM ranker failed full-catalog validation. Sampled validation NDCG@10 was `0.9839` but full-catalog NDCG@10 was `0.0395`; with larger uniform negatives it still scored `0.0573` overall and `0.0000` on cold items. The ranker was not evaluated on test.

Frozen recency, one-time test evaluation:

| Segment | Events | NDCG@10 | Recall@10 | MRR@10 |
|---|---:|---:|---:|---:|
| Overall | 6,667 | 0.4937 | 0.7702 | 0.4078 |
| Cold | 6,053 | 0.5414 | 0.8429 | 0.4478 |
| Warm | 614 | 0.0229 | 0.0537 | 0.0138 |

Overall NDCG@10 95% CI: `[0.4665, 0.5261]`. Coverage@10: `0.1687`.

Full analysis:

- `reports/01_baselines.md`
- `reports/02_baseline_comparison.md`
- `reports/03_model_comparison.md`
- `reports/04_error_analysis.md`

## Setup

Python 3.11 or 3.12 is supported. The validation results were produced with 3.12; CI uses 3.11. Run all commands from the repository root.

```bash
python -m pip install -r requirements.txt
```

On the original Windows machine, use `& "C:\Users\Mega Pc\anaconda3\python.exe"` if `python` does not resolve to Anaconda.

## Reproduce

```bash
python -m src.recommender.pipelines.make_dataset --config configs/base.yaml
python scripts/run_baselines.py configs/base.yaml
python scripts/tune_hybrid.py --config configs/base.yaml --trials 1500
python -m pytest -q
```

Ranker validation only:

```bash
python scripts/run_ranker.py --split validation --rounds 200 --max-validation-queries 1000 --max-eval-events 500 --bootstrap 0
```

Freeze recency and run the one-time test evaluation:

```bash
python scripts/finalize_model.py
```

## Repository layout

```text
configs/                 YAML configs
data/                    raw inputs; processed/ is generated and ignored
src/recommender/
  data/                  loading, cleaning, splitting, features
  evaluation/            metrics, bootstrap, full-catalog evaluator
  models/                baselines, hybrid, LightGBM ranker
  pipelines/             dataset construction
scripts/                 baseline, fusion, ranker, and final-evaluation entry points
tests/                   leakage, metric, split, and evaluator tests
results/                 validation leaderboard and fusion outputs
artifacts/               saved metrics/events; large model binaries are ignored
reports/                 protocol and result write-ups
```

`data/processed/` and `artifacts/models/*.joblib` are intentionally not committed. Rebuild the processed dataset with `make_dataset`; the ranker artifact is retained locally only as a negative-result diagnostic.

## Known limitations

- The test set was evaluated once with frozen recency; it must not be reused for further tuning.
- The ranker is a documented negative result, not a production model.
- Coverage bootstrap intervals are unstable because resampling users changes the recommended-item union.
- The dataset is small and cold-item dominated; deep models are not justified.
- The current best validated model is pure recency. Improving warm-item performance without losing cold items remains open.
