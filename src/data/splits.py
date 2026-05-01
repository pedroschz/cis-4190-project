"""Build the random and temporal train/val/test splits.

Random split (for the leaderboard model): stratified 80/10/10 on `source`, seed=42.
Temporal split (for the exploratory analysis):
    train: publish_date <= 2022-12-31
    val:   publish_date in 2023
    test:  publish_date >= 2024-01-01

Both are written to data/processed/splits_random/ and splits_temporal/ as
{train,val,test}.csv with the same columns as headlines.csv.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("data.splits")

SEED = 42


def random_split(df: pd.DataFrame, val_size: float = 0.1, test_size: float = 0.1) -> dict:
    """Stratified 80/10/10 by source. Returns {'train', 'val', 'test'} -> df."""
    train_val, test = train_test_split(
        df, test_size=test_size, stratify=df["source"], random_state=SEED
    )
    val_relative = val_size / (1.0 - test_size)
    train, val = train_test_split(
        train_val, test_size=val_relative, stratify=train_val["source"], random_state=SEED
    )
    return {"train": train, "val": val, "test": test}


def temporal_split(df: pd.DataFrame) -> dict:
    """train ≤ 2022, val = 2023, test ≥ 2024. Source-balanced where possible."""
    df = df.copy()
    df["publish_date"] = pd.to_datetime(df["publish_date"])
    train = df[df["publish_date"] <= "2022-12-31"]
    val = df[(df["publish_date"] >= "2023-01-01") & (df["publish_date"] <= "2023-12-31")]
    test = df[df["publish_date"] >= "2024-01-01"]
    return {"train": train, "val": val, "test": test}


def _save(split_dict: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, sdf in split_dict.items():
        path = out_dir / f"{name}.csv"
        sdf.to_csv(path, index=False)
        bal = sdf["source"].value_counts().to_dict() if len(sdf) else {}
        logger.info("%s -> %d rows | %s", path, len(sdf), bal)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--input", type=Path, default=Path("data/processed/headlines.csv"))
    p.add_argument("--out-random", type=Path, default=Path("data/processed/splits_random"))
    p.add_argument("--out-temporal", type=Path, default=Path("data/processed/splits_temporal"))
    args = p.parse_args(argv)

    if not args.input.exists():
        raise SystemExit(f"Input not found: {args.input}. Run src.data.clean first.")

    df = pd.read_csv(args.input)
    logger.info("Loaded %d rows", len(df))

    rs = random_split(df)
    _save(rs, args.out_random)

    ts = temporal_split(df)
    _save(ts, args.out_temporal)

    return 0


if __name__ == "__main__":
    sys.exit(main())
