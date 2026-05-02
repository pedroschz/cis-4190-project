"""Combine scraped CSVs into a clean canonical dataset."""

from __future__ import annotations

import argparse
import logging
import re
import sys
import unicodedata
from pathlib import Path
from typing import Optional

import pandas as pd

from src.scrape.parsers import SITE_SUFFIX_PATTERNS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("data.clean")

DEFAULT_INPUTS = [
    ("starter", Path("data/interim/starter_scraped.csv")),
    ("historical_FoxNews", Path("data/interim/historical_FoxNews.csv")),
    ("historical_NBC", Path("data/interim/historical_NBC.csv")),
]
DEFAULT_OUTPUT = Path("data/processed/headlines.csv")

LEAKAGE_TERMS = ["fox news", "nbc news", "foxnews.com", "nbcnews.com"]
WHITESPACE_RE = re.compile(r"\s+")
PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)


def _strip_suffixes(text: str) -> str:
    out = text
    for pat in SITE_SUFFIX_PATTERNS:
        out = re.sub(pat, "", out, flags=re.IGNORECASE)
    return out


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace(" ", " ").replace("​", "")
    text = WHITESPACE_RE.sub(" ", text).strip()
    return text


def _dedup_key(text: str) -> str:
    return PUNCT_RE.sub("", text.lower()).strip()


def _load_inputs(input_paths: list[tuple[str, Path]]) -> pd.DataFrame:
    frames = []
    for tag, p in input_paths:
        if not p.exists():
            logger.warning("Input not found, skipping: %s", p)
            continue
        df = pd.read_csv(p)
        df["source_split"] = tag
        frames.append(df)
    if not frames:
        raise SystemExit("No input CSVs found in data/interim/. Run scrape steps first.")
    return pd.concat(frames, ignore_index=True)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    n0 = len(df)
    logger.info("Loaded %d raw rows", n0)

    df = df.dropna(subset=["headline", "publish_date", "source"]).copy()
    logger.info("After drop missing headline/date/source: %d (%+d)", len(df), len(df) - n0)

    df["headline"] = df["headline"].astype(str).map(_strip_suffixes).map(_normalize)
    df = df[df["headline"].str.len().between(10, 300)]
    logger.info("After length filter [10, 300]: %d", len(df))

    # Leakage check
    lower = df["headline"].str.lower()
    leakage_mask = pd.Series(False, index=df.index)
    for term in LEAKAGE_TERMS:
        leakage_mask = leakage_mask | lower.str.contains(re.escape(term), na=False)
    if leakage_mask.any():
        n_leak = int(leakage_mask.sum())
        logger.warning("Dropping %d rows that still contain source-name leakage", n_leak)
        df = df[~leakage_mask]

    # Dedup
    df["_dedup_key"] = df["headline"].map(_dedup_key)
    df = df.drop_duplicates(subset=["_dedup_key"]).drop(columns=["_dedup_key"])
    logger.info("After dedup: %d", len(df))

    # Year
    df["publish_date"] = pd.to_datetime(df["publish_date"], errors="coerce")
    df = df.dropna(subset=["publish_date"])
    df["year"] = df["publish_date"].dt.year
    df["publish_date"] = df["publish_date"].dt.strftime("%Y-%m-%d")
    logger.info("After date parse: %d", len(df))

    # Normalize source label spelling
    df["source"] = df["source"].astype(str).replace({"foxnews": "FoxNews", "Fox News": "FoxNews", "nbc": "NBC", "NBC News": "NBC"})
    df = df[df["source"].isin(["FoxNews", "NBC"])]

    # Class balance log
    counts = df["source"].value_counts().to_dict()
    by_year = df.groupby(["year", "source"]).size().unstack(fill_value=0)
    logger.info("Final class balance: %s", counts)
    logger.info("Class balance by year:\n%s", by_year)

    cols = ["headline", "source", "publish_date", "year", "url", "source_split"]
    return df[cols].reset_index(drop=True)


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument(
        "--input",
        type=Path,
        action="append",
        help="add an extra input CSV (can be passed multiple times)",
    )
    args = p.parse_args(argv)

    inputs = list(DEFAULT_INPUTS)
    if args.input:
        for i, p in enumerate(args.input):
            inputs.append((f"extra_{i}", p))

    df = _load_inputs(inputs)
    cleaned = clean(df)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(args.output, index=False)
    logger.info("Wrote %d rows to %s", len(cleaned), args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
