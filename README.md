# News Source Classification (CIS 4190/5190 Final Project, Track B)

Binary text classification: given a news headline, predict Fox News or NBC News.

## Setup

```bash
uv sync
uv run python -m pytest tests/
```

## Pipeline

```bash
# 1. scrape (drop the course URL CSV at data/starter/starter_urls.csv first)
uv run python -m src.scrape.starter_urls --use-wayback

# 2. clean + split
uv run python -m src.data.clean
uv run python -m src.data.splits

# 3. classical baselines
uv run python -m src.models.classical --variant v0 --split random
uv run python -m src.models.classical --variant v2 --split random

# 4. transformer fine-tuning (Colab GPU)
#    see notebooks/04_distilbert_finetune.ipynb

# 5. temporal robustness analysis
uv run python -m src.eval.temporal --do-classical

# 6. report figures
uv run python -m src.report.make_figures
```

## Results

| Model | Random val | Random test | Hidden val |
|---|---:|---:|---:|
| Course baseline (TF-IDF top-100 + LR) | - | 0.6649 | - |
| V0 (replica) | 0.6979 | 0.7112 | - |
| V1 (word 1-2gram + balanced LR) | 0.7888 | 0.8529 | - |
| V2 (word + char n-grams) | 0.8128 | 0.8583 | - |
| V3 (200k features, C=2) | 0.8235 | 0.8663 | 0.7308 (initial) |
| DistilBERT-base (3 epochs) | 0.8209 | 0.8396 | 0.4817 |
| **V2+V3 cat-aware ensemble (final)** | - | - | **0.93** |

V3 trained on full data ≤2022 → temporal test (≥2024): **0.6733** (19pp drop, the headline finding of the temporal exploratory section).

The category-aware ensemble (final submission) prepends URL path-category tokens (`news/world`, `politics`, `select/shopping`, ...) to both training (cleaned headline) and inference (URL slug) text. Fox single-level paths (`politics`, `media`, `lifestyle`) and NBC two-level paths (`news/world`, `select/shopping`) carry source-specific vocabulary that headlines can't. See `reports/final_report.tex` §6 for the full diagnostic.

## Submission

`submission/model.py` (V2+V3 averaged predict_proba, pickled+gzipped+base64) and `submission/preprocess.py` (extracts category prefix + slug) go on the leaderboard. `eval_local.py` mirrors the backend evaluator.

```bash
# Build the final submission from saved joblibs
uv run python tools/train_category_enriched.py
uv run python tools/build_ensemble_submission.py \
  --joblibs models/classical_v2_cat_all.joblib models/classical_v3_cat_all.joblib

# Sanity check
uv run python eval_local.py --csv data/processed/splits_random/test.csv
uv run pytest tests/
```

## Repository layout

```
data/processed/headlines.csv      cleaned dataset (3,734 rows)
data/processed/splits_*/          stratified random + temporal splits
src/scrape/                       scraper + parsers
src/data/                         cleaning + splits
src/models/                       classical (v0-v3) + transformer
src/eval/                         metrics + temporal analysis
src/report/                       figure generation
notebooks/                        Colab orchestration
submission/                       leaderboard upload (model.py, preprocess.py)
reports/                          final report + figures
tests/                            pytest smoke tests
```
