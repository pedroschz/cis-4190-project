"""Mirror the backend evaluator for Project B.

Per the spec:
  1. Read CSV.
  2. preprocess.prepare_data(csv) returns (X, y).
  3. Instantiate model from model.py (Model() or get_model()).
  4. Load model.pt (if provided).
  5. Run inference via model.predict(batch) or model(batch).
  6. Compare predictions to y; compute accuracy.

This file lives in the same dir as model.py + preprocess.py so they
import cleanly.

Usage:
  python -m submission._local_eval --csv data/processed/splits_random/test.csv
"""

from __future__ import annotations

import argparse
import importlib
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def _normalise_label(s):
    s = str(s).strip().lower()
    if s in ("foxnews", "fox", "fox news", "1"):
        return "FoxNews"
    if s in ("nbc", "nbcnews", "nbc news", "0"):
        return "NBC"
    return str(s)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="path to a CSV with headlines + labels")
    p.add_argument(
        "--checkpoint",
        default=None,
        help="path to model.pt (optional; only used if your Model has load_state_dict)",
    )
    p.add_argument("--batch-size", type=int, default=64)
    args = p.parse_args()

    # Step 1+2: preprocess
    preprocess = importlib.import_module("preprocess")
    X, y = preprocess.prepare_data(args.csv)
    print(f"prepare_data returned: {len(X)} examples, sample X[0]={X[0]!r}, sample y[0]={y[0]!r}")

    # Step 3: instantiate
    model_module = importlib.import_module("model")
    if hasattr(model_module, "get_model"):
        model = model_module.get_model()
    elif hasattr(model_module, "Model"):
        model = model_module.Model()
    elif hasattr(model_module, "NewsClassifier"):
        model = model_module.NewsClassifier()
    else:
        raise SystemExit("model.py exposes no Model / NewsClassifier / get_model")

    # Step 4: optional weights load
    if args.checkpoint:
        import torch

        state = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(state)

    # Step 5: batched inference
    preds: list = []
    t0 = time.perf_counter()
    for i in range(0, len(X), args.batch_size):
        batch = X[i : i + args.batch_size]
        out = model.predict(batch) if hasattr(model, "predict") else model(batch)
        # Backend falls back to argmax if a tensor is returned
        try:
            import torch as _t

            if isinstance(out, _t.Tensor) and out.ndim == 2:
                out = out.argmax(dim=-1).cpu().tolist()
        except ImportError:
            pass
        if hasattr(out, "tolist"):
            out = out.tolist()
        preds.extend(list(out))
    elapsed = time.perf_counter() - t0

    # Step 6: accuracy (if labels are real)
    y_norm = [_normalise_label(v) for v in y]
    p_norm = [_normalise_label(v) for v in preds]

    if all(v == "" for v in y_norm):
        print("(no labels in CSV — skipping accuracy)")
    else:
        correct = sum(int(a == b) for a, b in zip(y_norm, p_norm))
        acc = correct / len(y_norm) if y_norm else 0.0
        print()
        print(f"Accuracy:           {acc:.4f}  ({correct}/{len(y_norm)})")
        print(f"Total inference:    {elapsed*1000:.1f} ms")
        print(f"Per-example avg:    {elapsed*1000/len(X):.3f} ms")
        print()
        # Confusion
        labels = sorted({*y_norm, *p_norm})
        if len(labels) == 2:
            cm = np.zeros((2, 2), dtype=int)
            idx = {l: i for i, l in enumerate(labels)}
            for a, b in zip(y_norm, p_norm):
                cm[idx[a], idx[b]] += 1
            print(f"Confusion (rows=true, cols=pred), labels={labels}:")
            print(cm)


if __name__ == "__main__":
    main()
