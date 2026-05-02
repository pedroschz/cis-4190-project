"""Retrain V3 on (headline ∪ url-slug) pairs and re-encode submission/model.py.

V3 recipe (matches reports/figures/results_table.csv:5):
  TF-IDF word 1-2gram, 200k features, min_df=2, sublinear_tf
  TF-IDF char_wb 3-5gram, 200k features, min_df=2, sublinear_tf
  LogisticRegression(C=2.0, class_weight="balanced", max_iter=2000)

Training augmentation: each row in headlines.csv contributes TWO examples,
the cleaned headline AND the URL-slug derived via the same _slug() the
inference-time preprocess.py uses. This closes the headline↔slug domain gap
and lets the model pick up source-marker tokens (e.g. NBC's `rcna######`).
"""

from __future__ import annotations

import argparse
import base64
import gzip
import pickle
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "submission"))
from preprocess import _clean, _slug  # noqa: E402


def build_v3(c: float = 4.0) -> Pipeline:
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
    lr = LogisticRegression(max_iter=2000, class_weight="balanced", C=c)
    return Pipeline([("features", union), ("lr", lr)])


def build_view(df: pd.DataFrame, mode: str) -> tuple[list[str], list[int]]:
    """mode='slug' (recommended): each row → URL slug only.
       mode='head': each row → cleaned headline only.
       mode='dual': each row → both (slug + headline)."""
    X: list[str] = []
    y: list[int] = []
    for _, row in df.iterrows():
        label = 1 if str(row["source"]) == "FoxNews" else 0
        if mode in ("head", "dual"):
            h = _clean(row["headline"])
            if h:
                X.append(h)
                y.append(label)
        if mode in ("slug", "dual"):
            s = _slug(row["url"])
            if s:
                X.append(s)
                y.append(label)
    return X, y


MODEL_TEMPLATE = '''"""TF-IDF + Logistic Regression classifier for Fox vs NBC headlines."""

import base64
import gzip
import pickle


_WEIGHTS_B64 = """{b64_payload}"""


def _load():
    raw = base64.b64decode(_WEIGHTS_B64.strip())
    return pickle.loads(gzip.decompress(raw))


_LABELS = {{0: "NBC", 1: "FoxNews"}}


class Model:
    def __init__(self):
        self.pipe = _load()

    def predict(self, batch):
        if hasattr(batch, "tolist"):
            batch = batch.tolist()
        if isinstance(batch, str):
            batch = [batch]
        else:
            batch = [str(x) for x in batch]
        return [_LABELS[int(p)] for p in self.pipe.predict(batch)]

    def __call__(self, batch):
        return self.predict(batch)


def get_model():
    return Model()
'''


def encode_pipeline_to_model_py(pipe: Pipeline, out_path: Path) -> int:
    raw = pickle.dumps(pipe, protocol=pickle.HIGHEST_PROTOCOL)
    gz = gzip.compress(raw, compresslevel=9)
    b64 = base64.b64encode(gz).decode("ascii")
    out_path.write_text(MODEL_TEMPLATE.format(b64_payload=b64))
    return len(b64)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument(
        "--data",
        type=Path,
        default=Path("data/processed/headlines.csv"),
        help="cleaned canonical dataset",
    )
    p.add_argument(
        "--mode",
        choices=("slug", "head", "dual"),
        default="slug",
        help="slug = train on URL slugs only (matches inference exactly, recommended)",
    )
    p.add_argument("--c", type=float, default=4.0, help="LogisticRegression C")
    p.add_argument(
        "--joblib-out",
        type=Path,
        default=Path("models/classical_v3_slug_all.joblib"),
        help="where to write the fitted pipeline",
    )
    p.add_argument(
        "--model-py-out",
        type=Path,
        default=Path("submission/model.py"),
        help="where to write the leaderboard model.py with embedded weights",
    )
    args = p.parse_args()

    df = pd.read_csv(args.data)
    print(f"Loaded {len(df)} rows from {args.data}")
    print(f"  by source: {df['source'].value_counts().to_dict()}")
    print(f"  by year:   {df['year'].value_counts().sort_index().to_dict()}")

    X, y = build_view(df, args.mode)
    print(f"Training set ({args.mode}): {len(X)} examples")
    pos = sum(y)
    print(f"  FoxNews: {pos}  NBC: {len(y) - pos}")

    pipe = build_v3(c=args.c)
    print(f"Fitting V3 (mode={args.mode}, C={args.c}) ...")
    pipe.fit(X, y)

    args.joblib_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, args.joblib_out)
    print(f"Wrote pipeline → {args.joblib_out} ({args.joblib_out.stat().st_size:,} bytes)")

    nb64 = encode_pipeline_to_model_py(pipe, args.model_py_out)
    print(f"Wrote model.py → {args.model_py_out} (base64 payload = {nb64:,} chars)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
