"""Discover historical article URLs via the Wayback CDX API."""

from __future__ import annotations

import argparse
import csv
import logging
import random
import sys
from pathlib import Path
from typing import Iterable, Optional

import requests
from tqdm import tqdm

from src.scrape._http import USER_AGENT, get
from src.scrape.archive_fallback import fetch_via_wayback
from src.scrape.parsers import detect_source, parse_article

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("scrape.wayback_expand")

CDX_ENDPOINT = "https://web.archive.org/cdx/search/cdx"

# Hostname per source. We use matchType=domain to capture all subdomains and paths.
HOSTNAMES = {
    "FoxNews": "www.foxnews.com",
    "NBC": "www.nbcnews.com",
}

OUT_FIELDS = ["url", "source", "headline", "publish_date", "fetch_status", "bucket_year"]


def discover_via_cdx(source: str, year: int, limit: int = 1000, seed: int = 42) -> list[str]:
    """Query Wayback CDX for snapshots in [year, year+1), return deduped original URLs.

    Wayback CDX can be slow or 504 under load. We retry a few times before giving up.
    """
    host = HOSTNAMES[source]
    params = {
        "url": host,
        "matchType": "domain",
        "from": f"{year}0101",
        "to": f"{year}1231",
        "output": "json",
        "filter": "statuscode:200",
        "fl": "original",
        "collapse": "urlkey",
        "limit": str(limit * 4),  # over-sample for path filtering + dedup
    }
    rows = None
    for attempt in range(3):
        try:
            resp = requests.get(
                CDX_ENDPOINT,
                params=params,
                headers={"User-Agent": USER_AGENT},
                timeout=180,
            )
        except requests.RequestException as e:
            logger.warning("CDX network error (try %d/3): %s", attempt + 1, e)
            continue
        if resp.status_code == 200:
            try:
                rows = resp.json()
                break
            except ValueError:
                logger.warning("CDX returned non-JSON (try %d/3)", attempt + 1)
                continue
        if resp.status_code in (429, 502, 503, 504):
            logger.warning(
                "CDX returned %d (try %d/3); backing off", resp.status_code, attempt + 1
            )
            import time

            time.sleep(15 * (attempt + 1))
            continue
        logger.error("CDX returned unexpected %d", resp.status_code)
        return []

    if not rows or len(rows) < 2:
        logger.error("CDX returned no results for %s %d", source, year)
        return []
    urls = [r[0] for r in rows[1:]]
    # Filter to plausible article URLs: hyphenated slug as the last path segment,
    # not a section/feed/sitemap page.
    def _looks_like_article(u: str) -> bool:
        if any(seg in u for seg in ("/sitemap", "/feed", "/rss", "/category/", "/tag/", "/section/")):
            return False
        last = u.rstrip("/").rsplit("/", 1)[-1]
        # Article slugs are hyphenated and >=20 chars; section pages are single words.
        return "-" in last and len(last) >= 20

    urls = [u for u in urls if _looks_like_article(u)]
    seen: set[str] = set()
    deduped = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            deduped.append(u)
    rng = random.Random(seed)
    rng.shuffle(deduped)
    return deduped[:limit]


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


def scrape_with_wayback_fallback(
    urls: Iterable[str], year: int, writer: csv.DictWriter, out_fh, seen: set[str]
) -> None:
    for url in tqdm(list(urls), desc=f"scrape[{year}]"):
        if url in seen:
            continue
        try:
            source = detect_source(url)
        except ValueError:
            continue

        html = get(url)
        status = "ok"
        if html is None:
            html = fetch_via_wayback(url)
            status = "wayback" if html else "wayback_fail"

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
    p.add_argument("--year", type=int, required=True)
    p.add_argument("--limit", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args(argv)

    out_path: Path = args.output or Path(f"data/interim/historical_{args.source}.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Querying Wayback CDX for %s articles in %d (limit=%d)", args.source, args.year, args.limit)
    urls = discover_via_cdx(args.source, args.year, args.limit, args.seed)
    logger.info("Discovered %d article URLs", len(urls))
    if not urls:
        return 1

    writer, fh, seen = _open_out(out_path)
    try:
        scrape_with_wayback_fallback(urls, args.year, writer, fh, seen)
    finally:
        fh.close()

    logger.info("Done. Output: %s", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
