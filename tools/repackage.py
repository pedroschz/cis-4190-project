"""Encode a joblib pipeline into submission/model.py as base64-gzipped pickle.

Usage:
    python tools/repackage.py models/classical_v1_all.joblib

Backs up the existing submission/model.py to submission/model.py.bak before
overwriting.
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


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("joblib_path", type=Path, help="Path to a fitted joblib Pipeline.")
    p.add_argument(
        "--out",
        type=Path,
        default=Path("submission/model.py"),
        help="Output model.py path.",
    )
    p.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip writing model.py.bak.",
    )
    args = p.parse_args()

    if not args.joblib_path.exists():
        print(f"error: {args.joblib_path} not found", file=sys.stderr)
        return 2

    pipe = joblib.load(args.joblib_path)
    raw = pickle.dumps(pipe, protocol=pickle.HIGHEST_PROTOCOL)
    gz = gzip.compress(raw, compresslevel=9)
    b64 = base64.b64encode(gz).decode("ascii")

    if not args.no_backup and args.out.exists():
        backup = args.out.with_suffix(args.out.suffix + ".bak")
        shutil.copy2(args.out, backup)
        print(f"backed up {args.out} → {backup}")

    args.out.write_text(MODEL_TEMPLATE.format(b64_payload=b64))
    print(f"wrote {args.out} (b64 payload {len(b64):,} chars, src joblib {args.joblib_path.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
