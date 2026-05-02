"""Classical TF-IDF + Logistic Regression baselines.

Three variants:
  - v0: course-baseline replica (TfidfVectorizer(stop_words='english', max_features=100) + LogisticRegression(max_iter=100)).
        Should land near 66.5% on a stratified split, matching the spec PDF.
  - v1: word 1-2gram TF-IDF, no max_features cap, class_weight='balanced'.
  - v2: FeatureUnion of word 1-2gram + char 3-5gram TF-IDF, class_weight='balanced'.

Outputs:
  - models/classical_<variant>.joblib  (sklearn Pipeline)
  - reports/figures/results_table.csv  (appended)
  - prints classification_report to stdout

Usage:
    python -m src.models.classical --variant v0 --split random
    python -m src.models.classical --variant v2 --split temporal
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from src.eval.metrics import (
    append_results_row,
    evaluate_predictions,
    labels_to_ints,
    print_report,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("models.classical")


def build_pipeline(variant: str) -> Pipeline:
    if variant == "v0":
        return Pipeline(
            [
                ("tfidf", TfidfVectorizer(stop_words="english", max_features=100)),
                ("lr", LogisticRegression(max_iter=100)),
            ]
        )
    if variant == "v1":
        return Pipeline(
            [
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=50000, min_df=2)),
                ("lr", LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)),
            ]
        )
    if variant == "v2":
        union = FeatureUnion(
            [
                (
                    "word",
                    TfidfVectorizer(
                        analyzer="word",
                        ngram_range=(1, 2),
                        max_features=50000,
                        min_df=2,
                        sublinear_tf=True,
                    ),
                ),
                (
                    "char",
                    TfidfVectorizer(
                        analyzer="char_wb",
                        ngram_range=(3, 5),
                        max_features=50000,
                        min_df=2,
                        sublinear_tf=True,
                    ),
                ),
            ]
        )
        return Pipeline(
            [
                ("features", union),
                ("lr", LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)),
            ]
        )
    raise ValueError(f"Unknown variant: {variant}")


def load_split(split: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = Path("data/processed") / f"splits_{split}"
    if not base.exists():
        raise SystemExit(f"Split dir not found: {base}. Run src.data.splits first.")
    train = pd.read_csv(base / "train.csv")
    val = pd.read_csv(base / "val.csv")
    test = pd.read_csv(base / "test.csv")
    return train, val, test


def fit_eval(variant: str, split: str, results_csv: Path, model_dir: Path) -> dict:
    train, val, test = load_split(split)
    logger.info(
        "Train: %d  Val: %d  Test: %d  (%s split)", len(train), len(val), len(test), split
    )

    pipe = build_pipeline(variant)
    X_train = train["headline"].astype(str).tolist()
    y_train = labels_to_ints(train["source"].tolist())
    X_val = val["headline"].astype(str).tolist()
    y_val = labels_to_ints(val["source"].tolist())
    X_test = test["headline"].astype(str).tolist()
    y_test = labels_to_ints(test["source"].tolist())

    pipe.fit(X_train, y_train)

    out: dict = {}
    for name, X, y in [("val", X_val, y_val), ("test", X_test, y_test)]:
        if len(X) == 0:
            continue
        y_pred = pipe.predict(X)
        bundle = evaluate_predictions(f"classical_{variant}_{split}_{name}", y, y_pred)
        print_report(bundle.name, y, y_pred)
        append_results_row(
            results_csv,
            bundle,
            extras={"model": f"classical_{variant}", "split": split, "stage": name},
        )
        out[name] = bundle.asdict()

    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"classical_{variant}_{split}.joblib"
    joblib.dump(pipe, model_path)
    logger.info("Saved model -> %s", model_path)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--variant", choices=["v0", "v1", "v2"], default="v0")
    p.add_argument("--split", choices=["random", "temporal"], default="random")
    p.add_argument("--results-csv", type=Path, default=Path("reports/figures/results_table.csv"))
    p.add_argument("--model-dir", type=Path, default=Path("models"))
    args = p.parse_args(argv)

    fit_eval(args.variant, args.split, args.results_csv, args.model_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
