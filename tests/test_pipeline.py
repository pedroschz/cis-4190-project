"""Smoke tests for the data + classical pipeline. Run via:

    uv run python -m pytest tests/

These tests don't hit the network. They use a small synthetic dataset to verify
that cleaning, splitting, and classical training all run end-to-end.
"""

from __future__ import annotations

import sys
from pathlib import Path
from datetime import date

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.clean import clean
from src.data.splits import random_split, temporal_split
from src.eval.metrics import evaluate_predictions, labels_to_ints
from src.models.classical import build_pipeline
from src.scrape.parsers import parse_article, _strip_suffix


def _synthetic_df(n: int = 200) -> pd.DataFrame:
    """Build a small balanced fake dataset for smoke testing."""
    rows = []
    for i in range(n):
        is_fox = i % 2 == 0
        year = 2019 + (i % 6)
        # Token salt makes the classical model actually learnable.
        salt = "trump rally" if is_fox else "biden senate"
        rows.append(
            dict(
                headline=f"{salt} headline number {i}",
                source="FoxNews" if is_fox else "NBC",
                publish_date=f"{year}-06-15",
                url=f"https://www.{'foxnews' if is_fox else 'nbcnews'}.com/x/{i}",
                source_split="synthetic",
            )
        )
    return pd.DataFrame(rows)


def test_strip_suffix_handles_common_patterns():
    assert _strip_suffix("Big news today | Fox News") == "Big news today"
    assert _strip_suffix("Big news today - NBC News") == "Big news today"
    assert _strip_suffix("Plain title with no suffix") == "Plain title with no suffix"


def test_parse_article_extracts_jsonld_headline():
    html = """
    <html><head>
      <script type="application/ld+json">
      {"@type": "NewsArticle", "headline": "Some headline | Fox News",
       "datePublished": "2024-03-15T10:00:00Z"}
      </script>
    </head><body><h1>fallback</h1></body></html>
    """
    parsed = parse_article(html, "https://www.foxnews.com/x/y")
    assert parsed.headline == "Some headline"
    assert parsed.publish_date == "2024-03-15"
    assert parsed.source == "FoxNews"


def test_clean_drops_leakage_and_dedups():
    raw = pd.DataFrame(
        [
            dict(headline="A leaky one Fox News", source="FoxNews", publish_date="2023-01-01", url="x", source_split="t"),
            dict(headline="Trump speaks to crowd", source="FoxNews", publish_date="2022-05-01", url="x", source_split="t"),
            dict(headline="Trump speaks to crowd", source="FoxNews", publish_date="2022-05-01", url="y", source_split="t"),
            dict(headline="Biden meets allies", source="NBC", publish_date="2024-02-02", url="z", source_split="t"),
            dict(headline=None, source="NBC", publish_date="2024-02-02", url="w", source_split="t"),
        ]
    )
    cleaned = clean(raw)
    # Leakage dropped, dup dropped, missing-headline dropped
    assert len(cleaned) == 2
    assert "year" in cleaned.columns
    assert set(cleaned["source"]) <= {"FoxNews", "NBC"}


def test_random_split_is_stratified():
    df = _synthetic_df(500)
    s = random_split(df)
    for name, sub in s.items():
        ratio = (sub["source"] == "FoxNews").mean()
        # 50/50 source balance preserved within ±5pp.
        assert 0.45 <= ratio <= 0.55, f"{name} source balance off: {ratio}"


def test_temporal_split_separates_years():
    df = _synthetic_df(600)
    s = temporal_split(df)
    if len(s["train"]):
        assert pd.to_datetime(s["train"]["publish_date"]).dt.year.max() <= 2022
    if len(s["val"]):
        assert (pd.to_datetime(s["val"]["publish_date"]).dt.year == 2023).all()
    if len(s["test"]):
        assert pd.to_datetime(s["test"]["publish_date"]).dt.year.min() >= 2024


def test_classical_v0_trains_and_beats_chance():
    df = _synthetic_df(400)
    s = random_split(df)
    pipe = build_pipeline("v0")
    pipe.fit(s["train"]["headline"].tolist(), labels_to_ints(s["train"]["source"].tolist()))
    preds = pipe.predict(s["val"]["headline"].tolist())
    y_true = labels_to_ints(s["val"]["source"].tolist())
    bundle = evaluate_predictions("smoke", y_true, preds)
    assert bundle.accuracy > 0.6, f"Smoke test classical accuracy too low: {bundle.accuracy}"


def test_classical_v2_handles_realistic_inputs():
    df = _synthetic_df(400)
    s = random_split(df)
    pipe = build_pipeline("v2")
    pipe.fit(s["train"]["headline"].tolist(), labels_to_ints(s["train"]["source"].tolist()))
    preds = pipe.predict(s["val"]["headline"].tolist())
    assert len(preds) == len(s["val"])
