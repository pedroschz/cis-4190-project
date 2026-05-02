"""Generate report figures from saved results CSVs."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("report.figures")

sns.set_theme(style="whitegrid")


def fig_results_bar(results_csv: Path, out: Path) -> None:
    if not results_csv.exists():
        logger.warning("No results CSV at %s", results_csv)
        return
    df = pd.read_csv(results_csv)
    if not len(df):
        return
    if "stage" in df.columns and "split" in df.columns:
        sub = df[(df["stage"] == "val") & (df["split"] == "random")].copy()
    else:
        sub = df
    if not len(sub):
        return

    sub = sub.sort_values("accuracy")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(sub["model"], sub["accuracy"], color="#3b6fb6")
    ax.axvline(0.6649, color="grey", ls="--", label="course baseline (0.6649)")
    for i, (m, a) in enumerate(zip(sub["model"], sub["accuracy"])):
        ax.text(a + 0.005, i, f"{a:.3f}", va="center")
    ax.set_xlim(0.55, 1.0)
    ax.set_xlabel("Accuracy (random val split)")
    ax.set_title("Model comparison")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    logger.info("Saved %s", out)


def fig_year_histogram(headlines_csv: Path, out: Path) -> None:
    if not headlines_csv.exists():
        return
    df = pd.read_csv(headlines_csv)
    if "year" not in df.columns or not len(df):
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    counts = df.groupby(["year", "source"]).size().unstack(fill_value=0)
    counts.plot(kind="bar", stacked=False, ax=ax, color={"FoxNews": "#c0392b", "NBC": "#2c3e50"})
    ax.set_xlabel("Year")
    ax.set_ylabel("Headlines")
    ax.set_title("Headlines per year, by source")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    logger.info("Saved %s", out)


def fig_headline_length(headlines_csv: Path, out: Path) -> None:
    if not headlines_csv.exists():
        return
    df = pd.read_csv(headlines_csv)
    if not len(df):
        return
    df = df.copy()
    df["len"] = df["headline"].astype(str).map(len)
    fig, ax = plt.subplots(figsize=(7, 4))
    for src, sub in df.groupby("source"):
        ax.hist(sub["len"], bins=40, alpha=0.5, label=src)
    ax.set_xlabel("Headline length (characters)")
    ax.set_ylabel("Count")
    ax.set_title("Headline-length distribution by source")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    logger.info("Saved %s", out)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--out-dir", type=Path, default=Path("reports/figures"))
    p.add_argument("--headlines", type=Path, default=Path("data/processed/headlines.csv"))
    p.add_argument("--results", type=Path, default=Path("reports/figures/results_table.csv"))
    args = p.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig_results_bar(args.results, args.out_dir / "results_bar.png")
    fig_year_histogram(args.headlines, args.out_dir / "dataset_year_hist.png")
    fig_headline_length(args.headlines, args.out_dir / "headline_length.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
