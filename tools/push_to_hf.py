"""Push the cleaned headlines dataset to the Hugging Face Hub.

Required by the project spec (must include the HF Dataset link in the report).

Usage (one-time setup on your Mac):
    uv pip install huggingface_hub datasets
    huggingface-cli login                       # paste a write-scope token
    uv run python tools/push_to_hf.py --repo-id <your-username>/cis5190-fox-vs-nbc

The script will create the dataset repo if it doesn't exist (private by default;
pass --public to make it visible on the leaderboard / report link).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--repo-id",
        required=True,
        help="Hugging Face repo id, e.g. pedroschz/cis5190-fox-vs-nbc",
    )
    p.add_argument(
        "--csv",
        type=Path,
        default=Path("data/processed/headlines.csv"),
        help="path to the cleaned headlines CSV",
    )
    p.add_argument("--public", action="store_true", help="make the dataset public")
    p.add_argument(
        "--include-splits",
        action="store_true",
        help="also push train/val/test (random + temporal) as separate config files",
    )
    args = p.parse_args()

    if not args.csv.exists():
        sys.exit(f"CSV not found: {args.csv}")

    try:
        from datasets import Dataset, DatasetDict, load_dataset
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        sys.exit("Install dependencies: uv pip install huggingface_hub datasets")

    import pandas as pd

    api = HfApi()
    create_repo(args.repo_id, repo_type="dataset", private=not args.public, exist_ok=True)
    print(f"Repo ready: https://huggingface.co/datasets/{args.repo_id}")

    # Build the canonical dataset
    df = pd.read_csv(args.csv)
    df = df[["headline", "source", "publish_date", "year", "url", "source_split"]]
    print(f"Loaded {len(df)} rows from {args.csv}")

    ds_main = Dataset.from_pandas(df, preserve_index=False)
    ds_main.push_to_hub(args.repo_id, config_name="headlines", split="all")
    print("Pushed config 'headlines' (all rows).")

    if args.include_splits:
        for split_dir, cfg in [
            (Path("data/processed/splits_random"), "random"),
            (Path("data/processed/splits_temporal"), "temporal"),
        ]:
            if not split_dir.exists():
                print(f"(skipping {cfg}: {split_dir} missing)")
                continue
            split_dict = DatasetDict()
            for name in ("train", "val", "test"):
                p_csv = split_dir / f"{name}.csv"
                if p_csv.exists():
                    split_dict[name] = Dataset.from_pandas(
                        pd.read_csv(p_csv), preserve_index=False
                    )
            split_dict.push_to_hub(args.repo_id, config_name=cfg)
            print(f"Pushed config '{cfg}'.")

    # Push a brief README so the dataset card explains itself
    card = f"""---
license: cc-by-nc-4.0
language:
  - en
size_categories:
  - 1K<n<10K
task_categories:
  - text-classification
tags:
  - news
  - source-classification
  - fox-news
  - nbc-news
  - cis5190
configs:
  - config_name: headlines
    data_files:
      - split: all
        path: headlines/all-*
{'  - config_name: random' if args.include_splits else ''}
{'    data_files:' if args.include_splits else ''}
{chr(10).join([f"      - split: {s}" + chr(10) + f"        path: random/{s}-*" for s in ('train','val','test')]) if args.include_splits else ''}
{'  - config_name: temporal' if args.include_splits else ''}
{'    data_files:' if args.include_splits else ''}
{chr(10).join([f"      - split: {s}" + chr(10) + f"        path: temporal/{s}-*" for s in ('train','val','test')]) if args.include_splits else ''}
---

# Fox News vs NBC News headlines (CIS 4190/5190 Spring 2026)

A binary text-classification corpus of {len(df):,} news headlines, scraped from FoxNews.com and NBCNews.com.

## Schema
| Column         | Type   | Description |
|----------------|--------|-------------|
| `headline`     | string | The article headline (suffixes like " | Fox News" stripped). |
| `source`       | string | `FoxNews` or `NBC`. |
| `publish_date` | string | ISO date (YYYY-MM-DD) from the article's structured metadata. |
| `year`         | int    | Year extracted from `publish_date`. |
| `url`          | string | Source article URL. |
| `source_split` | string | Which scrape this row came from (provenance). |

## Splits
- `random`: stratified 80/10/10 train/val/test on `source` (seed=42).
- `temporal`: train ≤ 2022, val = 2023, test ≥ 2024 — used for our temporal-robustness analysis.

## Class balance & date span
- Sources: ~52.5% FoxNews / ~47.5% NBC.
- Date range: 2019-06 to 2026-04.

## Cleaning
Site suffixes stripped, Unicode-normalised, deduplicated by lowercased exact match,
filtered to headlines in [10, 300] chars, leakage filter for any remaining mention of
"Fox News" / "NBC News" in the headline body.

## Citation
Collected for the CIS 4190/5190 Applied Machine Learning final project, University of Pennsylvania, Spring 2026.
"""
    api.upload_file(
        path_or_fileobj=card.encode(),
        path_in_repo="README.md",
        repo_id=args.repo_id,
        repo_type="dataset",
    )
    print("Pushed README.md.")
    print()
    print(f"Done. Link to put in your report:  https://huggingface.co/datasets/{args.repo_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
