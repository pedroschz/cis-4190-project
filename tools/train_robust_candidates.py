"""Train candidate robust models on the full dataset and select the best by slug-form eval.

Variants:
  v1_all              V1 (word 1-2gram, balanced LR, C=1) on full headlines
  v1_slugaug_all      V1 trained on union of cleaned headlines + slug-form transforms
  v3_slugaug_all      V3 (word + char_wb, C=2) trained on the same augmented set

Slug-form transform: lowercase, strip punctuation, strip CMS artifacts, collapse
whitespace. Different from `tools/retrain_v3_dual.py` which augments with REAL
URL slugs (those carry `rcna<digits>`/`.print` artifacts that overfit). This
script simulates slug-form ON the cleaned headlines so the augmented data has
no URL-derived artifacts.

After training each variant, evaluates on a held-out slice (random test) under
both head-form and slug-form, then selects the variant with highest slug-form
accuracy and (optionally) repackages it into submission/model.py.
"""

from __future__ import annotations

import argparse
import base64
import gzip
import logging
import pickle
import re
import sys
import unicodedata
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "submission"))

from src.eval.metrics import LABEL_TO_INT, evaluate_predictions  # noqa: E402
from preprocess import _clean, _strip_artifacts  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_robust")


_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WS_RE = re.compile(r"\s+")


def to_slug_form(headline: str) -> str:
    text = unicodedata.normalize("NFKC", str(headline))
    text = text.lower()
    text = _PUNCT_RE.sub(" ", text)
    text = _strip_artifacts(text)
    text = _WS_RE.sub(" ", text).strip()
    return text


def build_v1(c: float = 1.0) -> Pipeline:
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=50_000, min_df=2)),
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


def make_examples(df: pd.DataFrame, slug_aug: bool) -> tuple[list[str], list[int]]:
    X: list[str] = []
    y: list[int] = []
    for _, row in df.iterrows():
        label = LABEL_TO_INT[str(row["source"])]
        head = _clean(row["headline"])
        if not head:
            continue
        X.append(head)
        y.append(label)
        if slug_aug:
            slug = to_slug_form(row["headline"])
            if slug and slug != head.lower():
                X.append(slug)
                y.append(label)
    return X, y


def evaluate_on_test(name: str, pipe: Pipeline, test_df: pd.DataFrame) -> dict:
    X_head = [_clean(h) for h in test_df["headline"]]
    X_slug = [to_slug_form(h) for h in test_df["headline"]]
    y = [LABEL_TO_INT[s] for s in test_df["source"]]
    head_b = evaluate_predictions(f"{name}_head", y, pipe.predict(X_head))
    slug_b = evaluate_predictions(f"{name}_slug", y, pipe.predict(X_slug))
    return {
        "name": name,
        "head_acc": head_b.accuracy,
        "head_f1": head_b.f1_macro,
        "slug_acc": slug_b.accuracy,
        "slug_f1": slug_b.f1_macro,
    }


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


VARIANTS = {
    "v1_all": ("v1", False),
    "v1_slugaug_all": ("v1", True),
    "v3_slugaug_all": ("v3", True),
}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument(
        "--train-csvs",
        nargs="+",
        type=Path,
        default=[
            Path("data/processed/splits_random/train.csv"),
            Path("data/processed/splits_random/val.csv"),
        ],
        help="One or more CSVs concatenated for training.",
    )
    p.add_argument(
        "--test-csv",
        type=Path,
        default=Path("data/processed/splits_random/test.csv"),
        help="Held-out eval set. Must be disjoint from --train-csvs for honest numbers.",
    )
    p.add_argument("--variants", nargs="+", default=list(VARIANTS.keys()))
    p.add_argument("--model-dir", type=Path, default=Path("models"))
    p.add_argument(
        "--repackage",
        action="store_true",
        help="If set, write the best variant to submission/model.py.",
    )
    p.add_argument(
        "--model-py-out",
        type=Path,
        default=Path("submission/model.py"),
    )
    p.add_argument(
        "--name-suffix",
        default="",
        help="Suffix to append to joblib filenames (e.g. '_trainval').",
    )
    args = p.parse_args()

    df = pd.concat([pd.read_csv(x) for x in args.train_csvs], ignore_index=True)
    test_df = pd.read_csv(args.test_csv)
    logger.info(
        "Train: %d rows from %s  Test: %d rows from %s",
        len(df),
        [str(x) for x in args.train_csvs],
        len(test_df),
        args.test_csv,
    )

    args.model_dir.mkdir(parents=True, exist_ok=True)
    results = []
    pipes: dict[str, Pipeline] = {}

    for v in args.variants:
        if v not in VARIANTS:
            logger.warning("Unknown variant %s, skipping", v)
            continue
        kind, slug_aug = VARIANTS[v]
        X, y = make_examples(df, slug_aug=slug_aug)
        builder = build_v1 if kind == "v1" else build_v3
        c = 1.0 if kind == "v1" else 2.0
        pipe = builder(c=c)
        logger.info("Fitting %s (kind=%s, slug_aug=%s, n=%d)...", v, kind, slug_aug, len(X))
        pipe.fit(X, y)
        joblib_path = args.model_dir / f"classical_{v}{args.name_suffix}.joblib"
        joblib.dump(pipe, joblib_path)
        logger.info("  → %s (%d bytes)", joblib_path, joblib_path.stat().st_size)
        pipes[v] = pipe
        result = evaluate_on_test(v, pipe, test_df)
        result["joblib_path"] = str(joblib_path)
        results.append(result)

    print(f"\n=== Held-out test eval (n={len(test_df)}, NB: test rows are in training): ===")
    header = f"{'variant':<22}  {'head_acc':>9}  {'slug_acc':>9}  {'slug_f1':>9}"
    print(header)
    print("-" * len(header))
    for r in sorted(results, key=lambda r: r["slug_acc"], reverse=True):
        print(
            f"{r['name']:<22}  {r['head_acc']:>9.4f}  {r['slug_acc']:>9.4f}  {r['slug_f1']:>9.4f}"
        )

    results.sort(key=lambda r: r["slug_acc"], reverse=True)
    best = results[0]
    print(f"\nBest by slug_acc: {best['name']} (slug_acc={best['slug_acc']:.4f})")

    if args.repackage and best["name"] in pipes:
        nb64 = encode_pipeline_to_model_py(pipes[best["name"]], args.model_py_out)
        print(f"Repackaged {best['name']} → {args.model_py_out} (b64 payload {nb64:,} chars)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
