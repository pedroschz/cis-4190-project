# Quickstart

Walks the team from a fresh clone to a leaderboard submission.

## 0. Prerequisites

- Python 3.11+
- `uv` (or `pip + venv`) for env management
- A Colab Pro/Pro+ account for transformer fine-tuning
- The course-provided **starter URL CSV** (3,815 rows). Drop it at `data/starter/starter_urls.csv`.

## 1. Local environment

```bash
uv sync                      # installs deps from pyproject.toml
uv run python -m pytest      # smoke tests should all pass
```

If you don't have `uv`:

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e .
pytest
```

## 2. Scrape (Day 1)

The full scrape is ~2-3 hours wall time across all three sources. Resumable — re-running picks up where it left off.

```bash
# Provided URLs
uv run python -m src.scrape.starter_urls --use-wayback

# Historical expansion (run in parallel across team members on different machines)
uv run python -m src.scrape.sitemap_expand --source FoxNews --years 2019-2024 --per-year 800 --use-wayback
uv run python -m src.scrape.sitemap_expand --source NBC --years 2019-2024 --per-year 800 --use-wayback
```

After scraping, `data/interim/` should contain `starter_scraped.csv`, `historical_FoxNews.csv`, `historical_NBC.csv`.

## 3. Clean + split (Day 2 morning)

```bash
uv run python -m src.data.clean
uv run python -m src.data.splits
```

This produces `data/processed/headlines.csv` and the two split directories.

**Sanity checks before moving on:**

```bash
# Class balance ~ 50/50?
uv run python -c "import pandas as pd; df=pd.read_csv('data/processed/headlines.csv'); print(df['source'].value_counts(normalize=True))"

# No "Fox News" / "NBC News" leftover in headlines?
grep -i "fox news\|nbc news" data/processed/headlines.csv | head
```

## 4. Classical baselines (Day 2 afternoon)

```bash
uv run python -m src.models.classical --variant v0 --split random   # should hit ~66.5%
uv run python -m src.models.classical --variant v1 --split random
uv run python -m src.models.classical --variant v2 --split random
uv run python -m src.models.classical --variant v2 --split temporal
```

Results are appended to `reports/figures/results_table.csv`.

## 5. Transformer fine-tuning (Day 3, Colab)

Open `notebooks/04_distilbert_finetune.ipynb` in Colab. Steps:

1. Mount Drive.
2. Clone the repo.
3. Install transformers stack.
4. Smoke test on 200 rows.
5. Full run with default hyperparams (3 epochs, lr 2e-5, batch 32, max_len 128).
6. (Optional) hyperparameter sweep.
7. (Optional) RoBERTa via `notebooks/05_roberta_finetune.ipynb`.
8. Copy best model dir into `models/distilbert_final/` (or rename to `models/roberta_final/`).
9. Train the same model on the temporal split — needed for the analysis.

## 6. Temporal robustness analysis (Day 4)

```bash
# Classical decay curve (CPU, fast)
uv run python -m src.eval.temporal --do-classical
```

Then in `notebooks/06_temporal_decay.ipynb`, run the per-year DistilBERT decay block on Colab.

## 7. Error analysis + figures

```bash
uv run python -m src.report.make_figures
```

Open `notebooks/07_error_analysis.ipynb` for the per-token / per-year breakdown.

## 8. Predict on a new headline

```bash
echo "Trump rallies in Iowa ahead of primaries" | \
    uv run python predict.py --model-path models/classical_v2_random.joblib

# Or via the transformer
uv run python predict.py --model-path models/distilbert_final --input some_headlines.csv --output preds.csv
```

## 9. Leaderboard submission

Format TBD per the course PDF. Once announced (check Ed/Gradescope on Day 3), generate predictions on the provided test CSV with `predict.py` and submit.

## 10. Final deliverables (Day 5)

- [ ] `data/processed/headlines.csv` — committed in repo, also zip for submission.
- [ ] Best model — `models/<best>/` zipped (or HF Hub link).
- [ ] `reports/final_report.pdf` — compile from `final_report.tex`.
- [ ] Leaderboard entry confirmed.
