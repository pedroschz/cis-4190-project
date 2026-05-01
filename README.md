# CIS 4190/5190 Final Project — Track B: News Source Classification

Binary text classification: given a news headline, predict whether it came from **Fox News** or **NBC News**.

Course-provided baseline (TF-IDF + Logistic Regression): **66.49%**.
Our target: **80–83%** with a fine-tuned transformer.
Exploratory component: **temporal robustness** — how does source-classification accuracy decay as the gap between training and test data grows?

## Repository layout

```
cis-4190-project/
├── data/
│   ├── starter/          # provided 3,815-URL CSV (drop here)
│   ├── raw/              # scraped HTML (gitignored)
│   ├── interim/          # parsed but unclean CSVs (gitignored)
│   └── processed/        # final clean CSVs (committed)
├── src/
│   ├── scrape/           # URL + sitemap scrapers
│   ├── data/             # cleaning + splits
│   ├── models/           # classical + transformer
│   ├── eval/             # metrics + temporal analysis
│   └── report/           # figure generation
├── notebooks/            # Colab-friendly orchestration
├── reports/              # final paper + figures
└── tests/                # pipeline smoke tests
```

## Setup

```bash
# Local (CPU work — scraping, classical baseline, EDA, figures):
uv sync
uv run python -c "import sklearn; print(sklearn.__version__)"

# Colab (transformer fine-tuning):
!pip install -q transformers datasets accelerate evaluate scikit-learn pandas matplotlib seaborn
```

## Pipeline

The pipeline is six stages, all runnable as Python modules under `src/`:

1. **Scrape** — `python -m src.scrape.starter_urls` (provided URLs) and `python -m src.scrape.sitemap_expand` (historical expansion).
2. **Clean** — `python -m src.data.clean` → `data/processed/headlines.csv`.
3. **Split** — `python -m src.data.splits` → `data/processed/splits_random/` and `splits_temporal/`.
4. **Classical** — `python -m src.models.classical --variant {v0,v1,v2}`.
5. **Transformer** — `python -m src.models.transformer --model distilbert-base-uncased` (Colab GPU recommended).
6. **Temporal analysis** — `python -m src.eval.temporal` → decay-curve figures.

See [`docs/QUICKSTART.md`](docs/QUICKSTART.md) for a step-by-step walkthrough.

## Reproducibility

- Seed 42 is set throughout (`random`, `numpy`, `torch`, `transformers.set_seed`).
- Pinned versions in `pyproject.toml`.
- Final dataset (`data/processed/headlines.csv`) is committed to the repo so model training is byte-reproducible.

## Results

Filled in after experiments — see [`reports/figures/results_table.csv`](reports/figures/results_table.csv).

| Model | Random-split val acc | Temporal-split test acc (2024) |
|---|---|---|
| TF-IDF + LR (course baseline replica) | TBD | TBD |
| TF-IDF + LR (word + char n-grams) | TBD | TBD |
| DistilBERT-base-uncased (fine-tuned) | TBD | TBD |
| RoBERTa-base (fine-tuned) | TBD | TBD |

## Team

3-person team. Roles described in [the implementation plan](../../.claude/plans/users-pedro-downloads-cis-5190-final-pr-curious-squirrel.md).

## License

Course project — not for redistribution outside CIS 4190/5190.
