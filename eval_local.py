"""Local sanity check that mimics the leaderboard backend.

Usage:
    python eval_local.py --csv data/processed/splits_random/test.csv
"""

import argparse
import importlib
import sys
import time
from pathlib import Path

import numpy as np


def _norm(s):
    s = str(s).strip().lower()
    if s in ("foxnews", "fox", "fox news", "1"):
        return "FoxNews"
    if s in ("nbc", "nbcnews", "nbc news", "0"):
        return "NBC"
    return str(s)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--batch-size", type=int, default=64)
    args = p.parse_args()

    sys.path.insert(0, str(Path("submission").resolve()))
    preprocess = importlib.import_module("preprocess")
    model_mod = importlib.import_module("model")

    X, y = preprocess.prepare_data(args.csv)
    print(f"loaded {len(X)} examples")

    model = model_mod.get_model() if hasattr(model_mod, "get_model") else model_mod.Model()

    preds = []
    t0 = time.perf_counter()
    for i in range(0, len(X), args.batch_size):
        batch = X[i : i + args.batch_size]
        out = model.predict(batch) if hasattr(model, "predict") else model(batch)
        if hasattr(out, "tolist"):
            out = out.tolist()
        preds.extend(list(out))
    elapsed = time.perf_counter() - t0

    yn = [_norm(v) for v in y]
    pn = [_norm(v) for v in preds]
    if all(v == "" for v in yn):
        print("(no labels)")
        return

    correct = sum(int(a == b) for a, b in zip(yn, pn))
    acc = correct / len(yn)
    print(f"accuracy: {acc:.4f}  ({correct}/{len(yn)})")
    print(f"time:     {elapsed*1000:.1f} ms total, {elapsed*1000/len(X):.3f} ms/ex")

    labels = sorted({*yn, *pn})
    if len(labels) == 2:
        cm = np.zeros((2, 2), dtype=int)
        idx = {l: i for i, l in enumerate(labels)}
        for a, b in zip(yn, pn):
            cm[idx[a], idx[b]] += 1
        print(f"confusion (rows=true, cols=pred), labels={labels}:\n{cm}")


if __name__ == "__main__":
    main()
