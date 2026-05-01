"""Per-site headline + publish-date extractors.

The two outlets present headlines and dates differently. We try several extraction
paths in order of robustness, falling back gracefully. The single most important
thing we DO NOT keep is the site suffix (e.g. " | Fox News") — that's stripped
later in src.data.clean, but we already strip the obvious ones here.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Optional

from bs4 import BeautifulSoup

Source = Literal["FoxNews", "NBC"]

SITE_SUFFIX_PATTERNS = [
    r"\s*[\|\-—–]\s*Fox\s*News.*$",
    r"\s*[\|\-—–]\s*NBC\s*News.*$",
    r"\s*[\|\-—–]\s*NBCNews\.com.*$",
    r"\s*[\|\-—–]\s*foxnews\.com.*$",
]


@dataclass
class ParsedArticle:
    headline: Optional[str]
    publish_date: Optional[str]  # ISO 8601 string or None
    source: Source
    url: str


def detect_source(url: str) -> Source:
    if "foxnews.com" in url.lower():
        return "FoxNews"
    if "nbcnews.com" in url.lower():
        return "NBC"
    raise ValueError(f"Unknown source for url: {url}")


def _strip_suffix(text: str) -> str:
    out = text
    for pat in SITE_SUFFIX_PATTERNS:
        out = re.sub(pat, "", out, flags=re.IGNORECASE)
    return out.strip()


def _parse_date(raw: str) -> Optional[str]:
    """Best-effort ISO-8601 normalization. Returns YYYY-MM-DD or None."""
    if not raw:
        return None
    raw = raw.strip()
    # Common formats from JSON-LD / meta tags
    for fmt in (
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d",
    ):
        try:
            return datetime.strptime(raw[: len(fmt) + 5], fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    # Fallback: regex YYYY-MM-DD anywhere
    m = re.search(r"(\d{4}-\d{2}-\d{2})", raw)
    return m.group(1) if m else None


def _extract_jsonld(soup: BeautifulSoup) -> tuple[Optional[str], Optional[str]]:
    """Try every <script type=application/ld+json> looking for headline + date."""
    for tag in soup.find_all("script", attrs={"type": "application/ld+json"}):
        try:
            payload = json.loads(tag.string or "{}")
        except (json.JSONDecodeError, TypeError):
            continue
        # Some sites wrap a list of objects
        candidates = payload if isinstance(payload, list) else [payload]
        for obj in candidates:
            if not isinstance(obj, dict):
                continue
            # Sometimes nested under @graph
            inner = obj.get("@graph", [obj])
            if isinstance(inner, dict):
                inner = [inner]
            for item in inner:
                if not isinstance(item, dict):
                    continue
                t = item.get("@type", "")
                types = t if isinstance(t, list) else [t]
                if any(kind in types for kind in ("NewsArticle", "Article", "WebPage")):
                    headline = item.get("headline") or item.get("name")
                    date = item.get("datePublished") or item.get("dateCreated")
                    if headline:
                        return str(headline), _parse_date(str(date or ""))
    return None, None


def _extract_meta(soup: BeautifulSoup) -> tuple[Optional[str], Optional[str]]:
    headline = None
    date = None
    og_title = soup.find("meta", property="og:title")
    if og_title and og_title.get("content"):
        headline = og_title["content"]
    art_pub = soup.find("meta", property="article:published_time")
    if art_pub and art_pub.get("content"):
        date = _parse_date(art_pub["content"])
    if not date:
        time_tag = soup.find("time", attrs={"datetime": True})
        if time_tag:
            date = _parse_date(time_tag["datetime"])
    return headline, date


def _extract_h1(soup: BeautifulSoup) -> Optional[str]:
    for h1 in soup.find_all("h1"):
        text = h1.get_text(strip=True)
        if text and len(text) > 5:
            return text
    return None


def _extract_title(soup: BeautifulSoup) -> Optional[str]:
    if soup.title and soup.title.string:
        return soup.title.string.strip()
    return None


def parse_article(html: str, url: str) -> ParsedArticle:
    """Parse an article HTML page into a ParsedArticle.

    Order of preference for the headline:
      1. JSON-LD NewsArticle.headline
      2. <meta property="og:title">
      3. <h1>
      4. <title>

    For the date:
      1. JSON-LD datePublished
      2. <meta property="article:published_time">
      3. <time datetime="...">
    """
    soup = BeautifulSoup(html, "lxml")
    source = detect_source(url)

    h_jsonld, d_jsonld = _extract_jsonld(soup)
    h_meta, d_meta = _extract_meta(soup)

    headline = h_jsonld or h_meta or _extract_h1(soup) or _extract_title(soup)
    date = d_jsonld or d_meta

    if headline:
        headline = _strip_suffix(headline)

    return ParsedArticle(
        headline=headline or None,
        publish_date=date,
        source=source,
        url=url,
    )
