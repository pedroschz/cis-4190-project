"""Package an ensemble of V2 + V3 (both trained on full data) into submission/model.py.

The two pipelines are pickled as a list. At inference, each pipeline produces
predict_proba, the probabilities are averaged, and the argmax decides the label.

Usage:
    python tools/build_ensemble_submission.py
"""

from __future__ import annotations

import argparse
import base64
import gzip
import pickle
import shutil
import sys
from pathlib import Path

import joblib


MODEL_TEMPLATE = '''"""Ensemble (V2 + V3) for Fox vs NBC headlines: averaged TF-IDF + LR pipelines."""

import base64
import gzip
import pickle

import numpy as np


_WEIGHTS_B64 = """{b64_payload}"""


def _load():
    raw = base64.b64decode(_WEIGHTS_B64.strip())
    return pickle.loads(gzip.decompress(raw))


_LABELS = {{0: "NBC", 1: "FoxNews"}}


class Model:
    def __init__(self):
        self.pipes = _load()

    def predict(self, batch):
        if hasattr(batch, "tolist"):
            batch = batch.tolist()
        if isinstance(batch, str):
            batch = [batch]
        else:
            batch = [str(x) for x in batch]
        probs = np.mean([p.predict_proba(batch) for p in self.pipes], axis=0)
        ids = probs.argmax(axis=1)
        return [_LABELS[int(i)] for i in ids]

    def __call__(self, batch):
        return self.predict(batch)


def get_model():
    return Model()
'''


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument(
        "--joblibs",
        nargs="+",
        type=Path,
        default=[
            Path("models/classical_v2_all.joblib"),
            Path("models/classical_v3_all.joblib"),
        ],
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("submission/model.py"),
    )
    p.add_argument("--no-backup", action="store_true")
    args = p.parse_args()

    pipes = []
    for jp in args.joblibs:
        if not jp.exists():
            print(f"error: {jp} not found", file=sys.stderr)
            return 2
        pipes.append(joblib.load(jp))
        print(f"loaded {jp} ({jp.stat().st_size:,} bytes)")

    raw = pickle.dumps(pipes, protocol=pickle.HIGHEST_PROTOCOL)
    gz = gzip.compress(raw, compresslevel=9)
    b64 = base64.b64encode(gz).decode("ascii")

    if not args.no_backup and args.out.exists():
        backup = args.out.with_suffix(args.out.suffix + ".bak2")
        shutil.copy2(args.out, backup)
        print(f"backed up {args.out} → {backup}")

    args.out.write_text(MODEL_TEMPLATE.format(b64_payload=b64))
    print(f"wrote {args.out} (b64 payload {len(b64):,} chars)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
