"""Preprocessing for the news-headline classifier."""

import re
import unicodedata
from urllib.parse import urlparse

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


_ARTIFACT_RES = [
    re.compile(r"\brcna\d+\b", re.IGNORECASE),     # NBC CMS article IDs
    re.compile(r"\bncna\d+\b", re.IGNORECASE),     # older NBC CMS IDs
    re.compile(r"\.(print|html|amp)\b", re.IGNORECASE),
    re.compile(r"\bn\d{6,}\b"),                    # generic long numeric IDs
]


def _strip_artifacts(text):
    out = text
    for r in _ARTIFACT_RES:
        out = r.sub(" ", out)
    return WS_RE.sub(" ", out).strip()


def _slug(url):
    if not url:
        return ""
    s = str(url).split("?", 1)[0].split("#", 1)[0].rstrip("/")
    parts = [p for p in s.split("/") if p and "-" in p and any(c.isalpha() for c in p)]
    if parts:
        best = max(parts, key=lambda p: p.count("-"))
    else:
        best = s.rsplit("/", 1)[-1]
    text = best.replace("-", " ").replace("_", " ")
    return _strip_artifacts(text)


def _category_tokens(url):
    """Extract URL path category segments (the parts before the final slug).

    NBC uses two-level paths (`news/world/...`), Fox single-level (`politics/...`).
    Path depth + vocabulary differ by source, providing strong signal even when
    the headline body is short or content-thin.
    """
    if not url:
        return ""
    p = urlparse(str(url)).path.strip("/").split("/")
    cats = [seg for seg in p[:-1] if seg and not any(c.isdigit() for c in seg[:3])]
    return " ".join(cats[:3])


def _enrich(text, url):
    cat = _category_tokens(url)
    if cat:
        return f"{cat} | {text}".strip(" |")
    return text


def _label_from_url(url):
    u = str(url).lower()
    if "foxnews" in u:
        return "FoxNews"
    if "nbcnews" in u:
        return "NBC"
    return ""


def prepare_data(csv_path):
    df = pd.read_csv(csv_path)

    h_col = _find_col(df, HEADLINE_COLS)
    u_col = _find_col(df, URL_COLS)

    if h_col is not None and u_col is not None:
        X = [
            _enrich(_clean(h), u)
            for h, u in zip(df[h_col].fillna("").astype(str), df[u_col].fillna("").astype(str))
        ]
    elif h_col is not None:
        X = [_clean(v) for v in df[h_col].fillna("").astype(str)]
    elif u_col is not None:
        X = [
            _enrich(_slug(u), u)
            for u in df[u_col].fillna("").astype(str)
        ]
    else:
        raise ValueError(f"No headline or url column in CSV. Columns: {df.columns.tolist()}")

    y_col = _find_col(df, LABEL_COLS)
    if y_col is not None:
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

    if u_col is not None:
        return X, [_label_from_url(v) for v in df[u_col].fillna("").astype(str)]

    return X, [""] * len(X)
