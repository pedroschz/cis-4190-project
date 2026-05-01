"""Scrape the course-provided starter list of URLs into a per-row CSV.

Input:  data/starter/starter_urls.csv  (columns: url, source — source is FoxNews/NBC)
Output: data/interim/starter_scraped.csv (columns: url, source, headline, publish_date, fetch_status)

Resumable: rows already in the output CSV are skipped on re-run.

Usage:
    python -m src.scrape.starter_urls --input data/starter/starter_urls.csv \
                                       --output data/interim/starter_scraped.csv
    python -m src.scrape.starter_urls --use-wayback   # fall back to archive.org for failures
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd
from tqdm import tqdm

from src.scrape._http import get
from src.scrape.archive_fallback import fetch_via_wayback
from src.scrape.parsers import detect_source, parse_article

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("scrape.starter_urls")

OUT_FIELDS = ["url", "source", "headline", "publish_date", "fetch_status"]


def _load_seen(out_path: Path) -> set[str]:
    if not out_path.exists():
        return set()
    df = pd.read_csv(out_path)
    return set(df["url"].astype(str).tolist())


def _open_out(out_path: Path) -> tuple[csv.DictWriter, "object"]:
    new = not out_path.exists()
    f = out_path.open("a", newline="", encoding="utf-8")
    writer = csv.DictWriter(f, fieldnames=OUT_FIELDS)
    if new:
        writer.writeheader()
    return writer, f


def _normalize_input(input_path: Path) -> pd.DataFrame:
    """Accept either {url} or {url, source}; infer source from URL if missing."""
    df = pd.read_csv(input_path)
    cols = {c.lower(): c for c in df.columns}
    url_col = cols.get("url") or cols.get("link")
    if not url_col:
        raise SystemExit(f"Input CSV must have a 'url' or 'link' column, got: {df.columns.tolist()}")
    df = df.rename(columns={url_col: "url"})
    if "source" not in cols:
        df["source"] = df["url"].map(lambda u: detect_source(str(u)))
    else:
        df = df.rename(columns={cols["source"]: "source"})
    return df[["url", "source"]].dropna(subset=["url"])


def scrape_urls(
    urls: Iterable[str], use_wayback: bool, writer: csv.DictWriter, out_fh
) -> None:
    for url in tqdm(list(urls), desc="scraping"):
        url = str(url).strip()
        if not url:
            continue
        try:
            source = detect_source(url)
        except ValueError:
            logger.warning("skipping unknown-source url: %s", url)
            continue

        html = get(url)
        status = "ok"
        if html is None and use_wayback:
            html = fetch_via_wayback(url)
            status = "wayback" if html else "wayback_fail"
        elif html is None:
            status = "fetch_fail"

        if html is None:
            writer.writerow(
                dict(url=url, source=source, headline=None, publish_date=None, fetch_status=status)
            )
            out_fh.flush()
            continue

        parsed = parse_article(html, url)
        writer.writerow(
            dict(
                url=url,
                source=source,
                headline=parsed.headline,
                publish_date=parsed.publish_date,
                fetch_status=status,
            )
        )
        out_fh.flush()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--input", type=Path, default=Path("data/starter/starter_urls.csv"))
    p.add_argument("--output", type=Path, default=Path("data/interim/starter_scraped.csv"))
    p.add_argument(
        "--use-wayback", action="store_true", help="fall back to archive.org on fetch failure"
    )
    p.add_argument("--limit", type=int, default=None, help="only scrape the first N URLs")
    args = p.parse_args(argv)

    if not args.input.exists():
        logger.error(
            "Starter URL CSV not found at %s. Drop the course-provided file there first.",
            args.input,
        )
        return 2

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df = _normalize_input(args.input)
    seen = _load_seen(args.output)
    todo = df[~df["url"].astype(str).isin(seen)]
    if args.limit:
        todo = todo.head(args.limit)
    logger.info(
        "Total URLs: %d | already scraped: %d | scraping now: %d", len(df), len(seen), len(todo)
    )

    writer, fh = _open_out(args.output)
    try:
        scrape_urls(todo["url"].tolist(), args.use_wayback, writer, fh)
    finally:
        fh.close()

    logger.info("Done. Output: %s", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
