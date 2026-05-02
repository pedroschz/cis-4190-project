"""Preprocessing for the news-headline classifier."""

import re
import unicodedata

import pandas as pd


HEADLINE_COLS = ("headline", "scraped_headline", "alternative_headline", "title", "text", "h1")
LABEL_COLS = ("source", "label", "class", "target", "y")
URL_COLS = ("url", "link", "article_url")

SUFFIX_PATTERNS = [
    r"\s*[\|\-]\s*Fox\s*News.*$",
    r"\s*[\|\-]\s*NBC\s*News.*$",
    r"\s*[\|\-]\s*NBCNews\.com.*$",
    r"\s*[\|\-]\s*foxnews\.com.*$",
]
WS_RE = re.compile(r"\s+")


def _find_col(df, candidates):
    cols = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in cols:
            return cols[c.lower()]
    return None


def _clean(text):
    text = unicodedata.normalize("NFKC", str(text))
    text = WS_RE.sub(" ", text).strip()
    for pat in SUFFIX_PATTERNS:
        text = re.sub(pat, "", text, flags=re.IGNORECASE)
    return text


def _slug(url):
    if not url:
        return ""
    last = str(url).rstrip("/").rsplit("/", 1)[-1]
    return last.replace("-", " ").replace("_", " ")


def prepare_data(csv_path):
    df = pd.read_csv(csv_path)

    h_col = _find_col(df, HEADLINE_COLS)
    if h_col is not None:
        X = [_clean(v) for v in df[h_col].fillna("").astype(str)]
    else:
        u_col = _find_col(df, URL_COLS)
        if u_col is None:
            raise ValueError(f"No headline or url column in CSV. Columns: {df.columns.tolist()}")
        X = [_slug(v) for v in df[u_col].fillna("").astype(str)]

    y_col = _find_col(df, LABEL_COLS)
    if y_col is None:
        return X, [""] * len(X)

    y = []
    for v in df[y_col].fillna("").astype(str):
        s = v.strip().lower()
        if s in ("foxnews", "fox", "fox news", "1", "true"):
            y.append("FoxNews")
        elif s in ("nbc", "nbc news", "nbcnews", "0", "false"):
            y.append("NBC")
        else:
            y.append(v)
    return X, y
