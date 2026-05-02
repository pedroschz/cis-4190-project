"""Expand the dataset by walking each outlet's sitemap index.

Both Fox News and NBC News expose XML sitemap indexes that link to per-month or
per-year sitemaps of all article URLs they've published. We:
  1. Fetch each outlet's sitemap index.
  2. Discover sub-sitemaps and filter by year.
  3. Sample N URLs per (source, year) bucket.
  4. Scrape each sampled URL using the same parser pipeline as starter_urls.

Output: data/interim/historical_<source>.csv with the same schema as
        starter_scraped.csv, plus a `bucket_year` column we filtered against.

Usage:
    python -m src.scrape.sitemap_expand --source FoxNews --years 2019-2024 --per-year 1000
    python -m src.scrape.sitemap_expand --source NBC --years 2019-2024 --per-year 1000

Notes on resilience:
  - Sitemap structure changes; selectors are intentionally permissive.
  - We rely on the URL itself to assign a bucket year (most outlets put YYYY/MM in path).
    If we can't infer a year from the URL, we drop the URL from sampling.
"""

from __future__ import annotations

import argparse
import csv
import logging
import random
import re
import sys
from pathlib import Path
from typing import Iterable, Optional
from xml.etree import ElementTree as ET

from tqdm import tqdm

from src.scrape._http import get
from src.scrape.archive_fallback import fetch_via_wayback
from src.scrape.parsers import detect_source, parse_article

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("scrape.sitemap_expand")

NS = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}

# Per-outlet sitemap entrypoints. Validated 2026-05-01.
#
# Fox's sitemap is "fast-changing" — only recent articles. For historical
# (2019-2024) data, prefer src.scrape.wayback_expand which uses Wayback CDX.
#
# NBC's sitemap-index lists per-month sub-sitemaps named
# `sitemap-YYYY-MM-article.xml`, ideal for year-stratified sampling.
SITEMAP_INDEX = {
    "FoxNews": "https://www.foxnews.com/sitemap.xml?type=articles",
    "NBC": "https://www.nbcnews.com/sitemap/nbcnews/sitemap-index",
}

OUT_FIELDS = ["url", "source", "headline", "publish_date", "fetch_status", "bucket_year"]

# Year embedded somewhere in a sub-sitemap URL, e.g. `sitemap-2024-06-article.xml` or `/2020/12/`.
SITEMAP_YEAR_RE = re.compile(r"(?:[/-])(20\d{2})(?:[/-]|$)")
# Year inside an article URL — Fox uses `/2020/12/...`, NBC does NOT (rcna IDs have no year).
ARTICLE_URL_YEAR_RE = re.compile(r"/(20\d{2})/")


def _parse_year_range(s: str) -> list[int]:
    if "-" in s:
        a, b = s.split("-", 1)
        return list(range(int(a), int(b) + 1))
    return [int(s)]


def _xml_locs(xml_text: str) -> list[str]:
    """Return all <loc> values in a sitemap or sitemapindex."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        logger.warning("xml parse error: %s", e)
        return []
    return [el.text.strip() for el in root.findall(".//sm:loc", NS) if el.text]


def discover_article_urls(source: str, years: list[int]) -> dict[int, list[str]]:
    """Walk the sitemap index, return {year: [article URLs]} filtered to `years`."""
    index_url = SITEMAP_INDEX[source]
    logger.info("Fetching sitemap index: %s", index_url)
    idx_xml = get(index_url)
    if not idx_xml:
        logger.error("Could not fetch sitemap index for %s. Trying Wayback.", source)
        idx_xml = fetch_via_wayback(index_url)
    if not idx_xml:
        raise SystemExit(f"Sitemap index unreachable for {source}; try a different sitemap URL.")

    sub_locs = _xml_locs(idx_xml)
    if not sub_locs:
        # Index XML may itself list article URLs directly (not nested)
        sub_locs = [index_url]

    by_year: dict[int, list[str]] = {y: [] for y in years}

    # Skip non-article sub-sitemaps. NBC has sitemap-{curations,select,video,slideshow,...}
    # which we don't want; only year-stamped 'article' sub-sitemaps are useful.
    def _is_article_sitemap(url: str) -> bool:
        if any(skip in url for skip in ("curation", "select", "video", "slideshow", "image")):
            return False
        return SITEMAP_YEAR_RE.search(url) is not None

    candidate_subs = [s for s in sub_locs if _is_article_sitemap(s)]
    if not candidate_subs:
        # Fallback: maybe the sitemap URL itself is the urlset (Fox's flat sitemap)
        candidate_subs = sub_locs

    for sub in tqdm(candidate_subs, desc=f"sitemaps[{source}]"):
        # Determine the year from the sub-sitemap URL itself.
        sub_year_match = SITEMAP_YEAR_RE.search(sub)
        sub_year = int(sub_year_match.group(1)) if sub_year_match else None
        if sub_year is not None and sub_year not in years:
            continue

        sub_xml = get(sub)
        if not sub_xml:
            continue

        for art_url in _xml_locs(sub_xml):
            # First try to read year from the article URL (Fox-style /YYYY/MM/),
            # otherwise fall back to the sub-sitemap's year (NBC-style rcna IDs).
            art_match = ARTICLE_URL_YEAR_RE.search(art_url)
            y = int(art_match.group(1)) if art_match else sub_year
            if y is None:
                continue
            if y in by_year:
                by_year[y].append(art_url)

    for y, urls in by_year.items():
        logger.info("source=%s year=%d candidate_urls=%d", source, y, len(urls))
    return by_year


def sample_urls(by_year: dict[int, list[str]], per_year: int, seed: int = 42) -> list[tuple[int, str]]:
    rng = random.Random(seed)
    out: list[tuple[int, str]] = []
    for y, urls in by_year.items():
        if not urls:
            continue
        rng.shuffle(urls)
        out.extend((y, u) for u in urls[:per_year])
    return out


def _open_out(out_path: Path) -> tuple[csv.DictWriter, "object", set[str]]:
    seen: set[str] = set()
    new = not out_path.exists()
    if not new:
        try:
            with out_path.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                seen = {row["url"] for row in reader if row.get("url")}
        except Exception:
            seen = set()
    f = out_path.open("a", newline="", encoding="utf-8")
    writer = csv.DictWriter(f, fieldnames=OUT_FIELDS)
    if new:
        writer.writeheader()
    return writer, f, seen


def scrape_sample(
    sample: Iterable[tuple[int, str]],
    use_wayback: bool,
    writer: csv.DictWriter,
    out_fh,
    seen: set[str],
) -> None:
    for year, url in tqdm(list(sample), desc="scrape"):
        if url in seen:
            continue
        try:
            source = detect_source(url)
        except ValueError:
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
                dict(
                    url=url,
                    source=source,
                    headline=None,
                    publish_date=None,
                    fetch_status=status,
                    bucket_year=year,
                )
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
                bucket_year=year,
            )
        )
        out_fh.flush()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--source", choices=["FoxNews", "NBC"], required=True)
    p.add_argument("--years", default="2019-2024", help="e.g. 2019-2024 or 2022")
    p.add_argument("--per-year", type=int, default=1000)
    p.add_argument("--use-wayback", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args(argv)

    out_path: Path = args.output or Path(f"data/interim/historical_{args.source}.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    years = _parse_year_range(args.years)
    by_year = discover_article_urls(args.source, years)
    sampled = sample_urls(by_year, args.per_year, seed=args.seed)
    logger.info("Sampled %d URLs total across %d years", len(sampled), len(years))

    writer, fh, seen = _open_out(out_path)
    try:
        scrape_sample(sampled, args.use_wayback, writer, fh, seen)
    finally:
        fh.close()
    logger.info("Done. Output: %s", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
