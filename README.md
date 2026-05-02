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

| Model | Random val | Random test |
|---|---:|---:|
| Course baseline (TF-IDF top-100 + LR) | - | 0.6649 |
| V0 (replica) | 0.6979 | 0.7112 |
| V1 (word 1-2gram + balanced LR) | 0.7888 | 0.8529 |
| V2 (word + char n-grams) | 0.8128 | 0.8583 |
| V3 (200k features, C=2) | 0.8235 | 0.8663 |
| DistilBERT-base (3 epochs) | 0.8209 | 0.8396 |

V2 trained on full data ≤2022 → temporal test (≥2024): **0.6733** (19pp drop, the headline finding of the exploratory section).

## Submission

`submission/model.py` and `submission/preprocess.py` go on the leaderboard. `eval_local.py` mirrors the backend evaluator.

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
