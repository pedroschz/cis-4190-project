"""Train V2 and V3 on category-enriched text (URL category prefix + cleaned headline).

The enriched X mirrors what `submission/preprocess.prepare_data` now produces at
inference time, so the train and inference distributions match. This closes the
URL-structure gap that the prior submission (cleaned headlines only) couldn't see.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "submission"))
from preprocess import _clean, _enrich  # noqa: E402


def build_v2(c: float = 1.0) -> Pipeline:
    union = FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    max_features=50_000,
                    min_df=2,
                    sublinear_tf=True,
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    max_features=50_000,
                    min_df=2,
                    sublinear_tf=True,
                ),
            ),
        ]
    )
    return Pipeline(
        [
            ("features", union),
            ("lr", LogisticRegression(max_iter=2000, class_weight="balanced", C=c)),
        ]
    )


def build_v3(c: float = 2.0) -> Pipeline:
    union = FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    max_features=200_000,
                    min_df=2,
                    sublinear_tf=True,
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    max_features=200_000,
                    min_df=2,
                    sublinear_tf=True,
                ),
            ),
        ]
    )
    return Pipeline(
        [
            ("features", union),
            ("lr", LogisticRegression(max_iter=2000, class_weight="balanced", C=c)),
        ]
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--data", type=Path, default=Path("data/processed/headlines.csv"))
    p.add_argument("--model-dir", type=Path, default=Path("models"))
    p.add_argument("--variants", nargs="+", default=["v2", "v3"])
    args = p.parse_args()

    df = pd.read_csv(args.data)
    print(f"Loaded {len(df)} rows from {args.data}")

    X = [_enrich(_clean(h), u) for h, u in zip(df["headline"], df["url"])]
    y = [1 if s == "FoxNews" else 0 for s in df["source"]]

    print(f"Sample enriched X[0]: {X[0][:120]}")
    print(f"Sample enriched X[1]: {X[1][:120]}")

    args.model_dir.mkdir(parents=True, exist_ok=True)
    for v in args.variants:
        builder = build_v2 if v == "v2" else build_v3
        pipe = builder()
        print(f"Fitting {v}_cat_all (n={len(X)})...")
        pipe.fit(X, y)
        out = args.model_dir / f"classical_{v}_cat_all.joblib"
        joblib.dump(pipe, out)
        print(f"  → {out} ({out.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
