# `data/starter/`

Drop the course-provided **3,815-URL CSV** here as `starter_urls.csv`.

The CSV must have at minimum a `url` column. If it also has a `source` column it'll be used; otherwise the source is inferred from the URL host (foxnews.com → `FoxNews`, nbcnews.com → `NBC`).

Once the file is in place, run:

```bash
uv run python -m src.scrape.starter_urls --use-wayback
```
