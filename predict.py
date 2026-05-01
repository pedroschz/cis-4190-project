"""Inference entrypoint for the final submitted model.

Loads either a saved sklearn pipeline (joblib) or a Hugging Face model directory
based on the --model-path argument, and predicts FoxNews vs NBC for each headline
in --input (CSV with a `headline` column) or from stdin.

Usage:
    python predict.py --model-path models/classical_v2_random.joblib --input headlines.csv
    python predict.py --model-path models/transformer_run/final --input headlines.csv
    echo "Trump rallies in Iowa" | python predict.py --model-path models/classical_v2_random.joblib
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from src.eval.metrics import ints_to_labels

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("predict")


def _is_hf_dir(p: Path) -> bool:
    if not p.is_dir():
        return False
    return (p / "config.json").exists() or (p / "tokenizer_config.json").exists()


def _load_classical(path: Path):
    import joblib

    return joblib.load(path)


def _predict_classical(pipe, headlines: list[str]) -> list[str]:
    preds = pipe.predict(headlines)
    return ints_to_labels(preds)


def _predict_transformer(model_dir: Path, headlines: list[str], batch: int = 32) -> list[str]:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    import torch

    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    out: list[str] = []
    with torch.no_grad():
        for i in range(0, len(headlines), batch):
            chunk = headlines[i : i + batch]
            enc = tok(chunk, return_tensors="pt", padding=True, truncation=True, max_length=128)
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = model(**enc).logits
            preds = logits.argmax(dim=-1).cpu().tolist()
            out.extend(ints_to_labels(preds))
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--model-path", required=True, type=Path)
    p.add_argument("--input", type=Path, default=None, help="CSV with a 'headline' column")
    p.add_argument("--output", type=Path, default=None, help="where to write predictions (CSV)")
    args = p.parse_args(argv)

    # Read headlines
    if args.input:
        df = pd.read_csv(args.input)
        if "headline" not in df.columns:
            raise SystemExit(f"Input CSV must have a 'headline' column. Got {df.columns.tolist()}")
        headlines = df["headline"].astype(str).tolist()
    else:
        headlines = [line.strip() for line in sys.stdin if line.strip()]
        df = pd.DataFrame({"headline": headlines})

    if not headlines:
        logger.error("No headlines to predict.")
        return 2

    if _is_hf_dir(args.model_path):
        preds = _predict_transformer(args.model_path, headlines)
    else:
        pipe = _load_classical(args.model_path)
        preds = _predict_classical(pipe, headlines)

    df["prediction"] = preds
    if args.output:
        df.to_csv(args.output, index=False)
        logger.info("Wrote %d predictions to %s", len(df), args.output)
    else:
        # Stream to stdout
        for h, pr in zip(headlines, preds):
            print(f"{pr}\t{h}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
