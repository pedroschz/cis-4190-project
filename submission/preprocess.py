"""Submission preprocess.py — Read the test CSV, return (X, y).

Per the Final Project Submission Guideline §3.2:
  def prepare_data(csv_path: str) -> (X, y)
    X: sequence/array/tensor of inputs suitable for the model.
    y: sequence of labels (strings or integer class ids).

We:
  1. Pick a headline column from a generous list of candidates.
  2. Pick a label column from a similar list (or return None y if absent).
  3. Apply the same lightweight cleaning we used at training time
     (strip site suffixes, normalise whitespace) so distribution-shift
     between scrape contexts doesn't bite us.

We DO NOT scrape URLs — the backend has no network and the leaderboard
example UIs imply scraped text is provided. If only URLs are given, we
just return them as the input string; the model will still produce a
prediction, just a less reliable one.
"""

from __future__ import annotations

import re
import unicodedata
from typing import List, Optional, Tuple

import pandas as pd


HEADLINE_CANDIDATES = (
    "headline",
    "scraped_headline",
    "alternative_headline",
    "title",
    "text",
    "h1",
)

LABEL_CANDIDATES = (
    "source",
    "label",
    "class",
    "target",
    "y",
)

URL_CANDIDATES = (
    "url",
    "link",
    "article_url",
)

# Same suffix patterns as src/scrape/parsers.py — defensive in case
# the test set leaks them.
SITE_SUFFIX_PATTERNS = [
    r"\s*[\|\-—–]\s*Fox\s*News.*$",
    r"\s*[\|\-—–]\s*NBC\s*News.*$",
    r"\s*[\|\-—–]\s*NBCNews\.com.*$",
    r"\s*[\|\-—–]\s*foxnews\.com.*$",
]
WHITESPACE_RE = re.compile(r"\s+")


def _find_col(df: pd.DataFrame, candidates) -> Optional[str]:
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None


def _strip_suffixes(text: str) -> str:
    out = text
    for pat in SITE_SUFFIX_PATTERNS:
        out = re.sub(pat, "", out, flags=re.IGNORECASE)
    return out


def _normalise(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text))
    text = WHITESPACE_RE.sub(" ", text).strip()
    return _strip_suffixes(text)


def _slug_to_text(url: str) -> str:
    """Last-resort fallback: derive pseudo-text from a URL slug."""
    if not url:
        return ""
    slug = url.rstrip("/").rsplit("/", 1)[-1]
    return slug.replace("-", " ").replace("_", " ")


def prepare_data(csv_path: str) -> Tuple[List[str], List[str]]:
    df = pd.read_csv(csv_path)

    # X: prefer a headline column, fall back to slug-from-URL
    headline_col = _find_col(df, HEADLINE_CANDIDATES)
    if headline_col is not None:
        X = [_normalise(v) for v in df[headline_col].fillna("").astype(str)]
    else:
        url_col = _find_col(df, URL_CANDIDATES)
        if url_col is None:
            raise ValueError(
                f"CSV has no headline or url column. Got columns: {df.columns.tolist()}"
            )
        X = [_slug_to_text(v) for v in df[url_col].fillna("").astype(str)]

    # y: optional; return [] if absent. Normalise common spellings.
    label_col = _find_col(df, LABEL_CANDIDATES)
    if label_col is not None:
        y = []
        for v in df[label_col].fillna("").astype(str):
            v = v.strip()
            v_lower = v.lower()
            if v_lower in ("foxnews", "fox", "fox news", "1", "true"):
                y.append("FoxNews")
            elif v_lower in ("nbc", "nbc news", "nbcnews", "0", "false"):
                y.append("NBC")
            else:
                y.append(v)  # pass through; the grader's mapping will handle it
    else:
        y = [""] * len(X)

    return X, y
