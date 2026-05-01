"""Temporal robustness experiments — the exploratory component of the project.

Two experiments:

1. **Decay curve**: train each model class on a single year Y of headlines, then
   evaluate on each year >= Y. Plot accuracy as a function of (test_year - train_year).
   Compares the classical V2 model (TF-IDF word+char) against a transformer.

2. **Fixed train, sliding test**: train on all data <= 2022, evaluate per-year
   on 2023 and beyond. This is the "what does the leaderboard test look like?" view.

This module orchestrates the classical experiments directly. For the transformer,
we shell out to `src.models.transformer` and read back its metrics.json — keeps
the analysis runnable on CPU even when the transformer training has to happen on
Colab.

Usage:
    python -m src.eval.temporal --do-classical
    python -m src.eval.temporal --do-classical --plot-only   # if results already on disk
    python -m src.eval.temporal --do-transformer --model distilbert-base-uncased  # Colab only
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.eval.metrics import evaluate_predictions, labels_to_ints

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("eval.temporal")

DEFAULT_RESULTS = Path("reports/figures/temporal_results.csv")


def _build_classical_v2():
    from src.models.classical import build_pipeline

    return build_pipeline("v2")


def per_year_train_eval_classical(
    df: pd.DataFrame, train_years: list[int], eval_years: list[int]
) -> pd.DataFrame:
    """For each train_year, train on rows with year==train_year, eval on rows with year in eval_years."""
    rows = []
    for ty in train_years:
        train_df = df[df["year"] == ty]
        if len(train_df) < 50:
            logger.warning("Skipping train_year=%d (only %d rows)", ty, len(train_df))
            continue
        pipe = _build_classical_v2()
        pipe.fit(train_df["headline"].astype(str).tolist(), labels_to_ints(train_df["source"].tolist()))

        for ey in eval_years:
            test_df = df[df["year"] == ey]
            if len(test_df) < 30:
                continue
            y_true = labels_to_ints(test_df["source"].tolist())
            y_pred = pipe.predict(test_df["headline"].astype(str).tolist())
            b = evaluate_predictions(f"classical_v2_{ty}_to_{ey}", y_true, y_pred)
            rows.append(
                dict(
                    model="classical_v2",
                    train_year=ty,
                    eval_year=ey,
                    gap=ey - ty,
                    n_train=len(train_df),
                    n_eval=len(test_df),
                    **b.asdict(),
                )
            )
            logger.info(
                "classical_v2 train=%d eval=%d (gap=%d) acc=%.4f", ty, ey, ey - ty, b.accuracy
            )
    return pd.DataFrame(rows)


def fixed_train_sliding_eval_classical(
    df: pd.DataFrame, train_until: int = 2022, eval_years: list[int] | None = None
) -> pd.DataFrame:
    """Train on years <= train_until, evaluate per year for each year > train_until."""
    train_df = df[df["year"] <= train_until]
    if not len(train_df):
        return pd.DataFrame()
    eval_years = eval_years or sorted(set(int(y) for y in df["year"].unique() if y > train_until))

    pipe = _build_classical_v2()
    pipe.fit(train_df["headline"].astype(str).tolist(), labels_to_ints(train_df["source"].tolist()))

    rows = []
    for ey in eval_years:
        test_df = df[df["year"] == ey]
        if not len(test_df):
            continue
        y_true = labels_to_ints(test_df["source"].tolist())
        y_pred = pipe.predict(test_df["headline"].astype(str).tolist())
        b = evaluate_predictions(f"classical_v2_le{train_until}_eval{ey}", y_true, y_pred)
        rows.append(
            dict(
                model="classical_v2",
                experiment="fixed_train_sliding_eval",
                train_until=train_until,
                eval_year=ey,
                **b.asdict(),
            )
        )
    return pd.DataFrame(rows)


def plot_decay(decay_df: pd.DataFrame, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))

    for model, sub in decay_df.groupby("model"):
        # Average accuracy across train_years for each gap
        gap_acc = sub.groupby("gap")["accuracy"].mean().reset_index()
        ax.plot(gap_acc["gap"], gap_acc["accuracy"], marker="o", label=model)

    ax.set_xlabel("Test year − Train year (gap)")
    ax.set_ylabel("Accuracy")
    ax.set_title("News-source classification accuracy vs. temporal gap")
    ax.set_ylim(0.5, 1.0)
    ax.axhline(0.6649, ls="--", color="grey", alpha=0.6, label="course baseline (0.6649)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    logger.info("Saved %s", out_path)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--input", type=Path, default=Path("data/processed/headlines.csv"))
    p.add_argument("--results-csv", type=Path, default=DEFAULT_RESULTS)
    p.add_argument("--plot", type=Path, default=Path("reports/figures/temporal_decay.png"))
    p.add_argument("--do-classical", action="store_true")
    p.add_argument("--plot-only", action="store_true")
    p.add_argument("--train-years", default="2019,2020,2021,2022")
    p.add_argument("--eval-years", default="2019,2020,2021,2022,2023,2024")
    args = p.parse_args(argv)

    if args.plot_only:
        df = pd.read_csv(args.results_csv)
        plot_decay(df, args.plot)
        return 0

    if not args.input.exists():
        raise SystemExit(f"Input not found: {args.input}")

    df = pd.read_csv(args.input)
    if "year" not in df.columns:
        df["year"] = pd.to_datetime(df["publish_date"]).dt.year
    df = df.dropna(subset=["headline", "source", "year"])
    df["year"] = df["year"].astype(int)

    train_years = [int(y) for y in args.train_years.split(",")]
    eval_years = [int(y) for y in args.eval_years.split(",")]

    if args.do_classical:
        decay = per_year_train_eval_classical(df, train_years, eval_years)
        slide = fixed_train_sliding_eval_classical(df)

        args.results_csv.parent.mkdir(parents=True, exist_ok=True)
        if args.results_csv.exists():
            existing = pd.read_csv(args.results_csv)
            out = pd.concat([existing, decay, slide], ignore_index=True)
        else:
            out = pd.concat([decay, slide], ignore_index=True)
        out.to_csv(args.results_csv, index=False)
        logger.info("Wrote %d rows to %s", len(out), args.results_csv)

        plot_decay(decay, args.plot)

    return 0


if __name__ == "__main__":
    sys.exit(main())
