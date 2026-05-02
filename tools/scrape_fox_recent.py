"""Scrape recent Fox News articles by reading the flat urlset sitemap.

Fox's sitemap (https://www.foxnews.com/sitemap.xml?type=articles) is a single
flat urlset of recent article URLs with <lastmod> dates, NOT an index of
year-stamped sub-sitemaps. src.scrape.sitemap_expand expects the latter, so we
parse the urlset directly here, filter by lastmod year, sample, and scrape.
"""

from __future__ import annotations

import argparse
import csv
import logging
import random
import re
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

from tqdm import tqdm

from src.scrape._http import get
from src.scrape.archive_fallback import fetch_via_wayback
from src.scrape.parsers import detect_source, parse_article

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("scrape_fox_recent")

NS = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
LASTMOD_YEAR_RE = re.compile(r"^(20\d{2})-")


def fetch_url_dates(sitemap_url: str) -> list[tuple[str, int]]:
    """Return [(article_url, year), ...] from a flat urlset sitemap."""
    xml = get(sitemap_url, timeout=30.0)
    if not xml:
        raise SystemExit(f"Could not fetch {sitemap_url}")
    root = ET.fromstring(xml)
    out = []
    for url_el in root.findall(".//sm:url", NS):
        loc = url_el.find("sm:loc", NS)
        lastmod = url_el.find("sm:lastmod", NS)
        if loc is None or loc.text is None:
            continue
        year = None
        if lastmod is not None and lastmod.text:
            m = LASTMOD_YEAR_RE.match(lastmod.text.strip())
            if m:
                year = int(m.group(1))
        if year is None:
            continue
        out.append((loc.text.strip(), year))
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument(
        "--sitemap",
        default="https://www.foxnews.com/sitemap.xml?type=articles",
    )
    p.add_argument("--years", default="2025-2026", help="e.g. 2025-2026 or 2026")
    p.add_argument("--per-year", type=int, default=500)
    p.add_argument("--use-wayback", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=Path, default=Path("data/interim/recent_FoxNews.csv"))
    args = p.parse_args()

    years = (
        list(range(int(args.years.split("-")[0]), int(args.years.split("-")[1]) + 1))
        if "-" in args.years
        else [int(args.years)]
    )

    logger.info("Fetching Fox sitemap (flat urlset): %s", args.sitemap)
    pairs = fetch_url_dates(args.sitemap)
    logger.info("Total URLs in sitemap: %d", len(pairs))

    by_year: dict[int, list[str]] = {y: [] for y in years}
    for url, y in pairs:
        if y in by_year:
            by_year[y].append(url)
    for y, urls in by_year.items():
        logger.info("  year=%d candidate_urls=%d", y, len(urls))

    rng = random.Random(args.seed)
    sampled: list[tuple[int, str]] = []
    for y, urls in by_year.items():
        rng.shuffle(urls)
        sampled.extend((y, u) for u in urls[: args.per_year])
    logger.info("Sampled %d URLs total", len(sampled))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    new = not args.output.exists()
    seen: set[str] = set()
    if not new:
        with args.output.open("r", encoding="utf-8") as f:
            seen = {row["url"] for row in csv.DictReader(f) if row.get("url")}

    with args.output.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["url", "source", "headline", "publish_date", "fetch_status", "bucket_year"],
        )
        if new:
            writer.writeheader()

        for year, url in tqdm(sampled, desc="scrape[Fox]"):
            if url in seen:
                continue
            try:
                source = detect_source(url)
            except ValueError:
                continue
            html = get(url)
            status = "ok"
            if html is None and args.use_wayback:
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
                fh.flush()
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
            fh.flush()

    logger.info("Done. Output: %s", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
