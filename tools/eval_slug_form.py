"""Evaluate candidate models on slug-form input vs cleaned-headline input.

The leaderboard backend feeds our preprocess.prepare_data a URL-only CSV. Our
preprocess.py derives a 'slug' (the dashed segment of the URL path) and passes
it as text to the model. The model was trained on cleaned headlines, so this
script measures the gap by transforming test headlines into a slug-form proxy
(lowercase, punctuation stripped, whitespace collapsed) and comparing accuracy.

Caveat: any model trained on the full dataset has seen the test rows in
training. Treat head_acc on those rows as an upper bound; the slug_acc
column is still informative because slug-form is OOD relative to training.
Models trained only on the random train split (v1_random, v2_random,
v3_random_trainval) are honest baselines.
"""

from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "submission"))

from eval.metrics import LABEL_TO_INT, evaluate_predictions  # noqa: E402
from preprocess import _clean, _strip_artifacts  # noqa: E402


_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WS_RE = re.compile(r"\s+")


def to_slug_form(headline: str) -> str:
    """Headline → slug-form proxy: NFKC, lowercase, strip punctuation, collapse whitespace."""
    text = unicodedata.normalize("NFKC", str(headline))
    text = text.lower()
    text = _PUNCT_RE.sub(" ", text)
    text = _strip_artifacts(text)
    text = _WS_RE.sub(" ", text).strip()
    return text


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--csv", default="data/processed/splits_random/test.csv")
    p.add_argument(
        "--models",
        nargs="+",
        default=[
            "models/classical_v1_random.joblib",
            "models/classical_v2_random.joblib",
            "models/classical_v3_random_trainval.joblib",
            "models/classical_v3_all.joblib",
            "models/classical_v3_dual_all.joblib",
            "models/classical_v3_slug_all.joblib",
        ],
    )
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    print(f"loaded {len(df)} rows from {args.csv}")
    print(f"  source counts: {df['source'].value_counts().to_dict()}")

    X_head = [_clean(h) for h in df["headline"]]
    X_slug = [to_slug_form(h) for h in df["headline"]]
    y = [LABEL_TO_INT[s] for s in df["source"]]

    print("\nSample slug-form transforms:")
    for i in range(3):
        print(f"  HEAD: {X_head[i][:90]}")
        print(f"  SLUG: {X_slug[i][:90]}")
        print()

    rows = []
    for path in args.models:
        full = ROOT / path if not Path(path).is_absolute() else Path(path)
        if not full.exists():
            print(f"  [skip] {path} not found")
            continue
        pipe = joblib.load(full)
        head_preds = pipe.predict(X_head)
        slug_preds = pipe.predict(X_slug)
        head_b = evaluate_predictions(path, y, head_preds)
        slug_b = evaluate_predictions(path, y, slug_preds)
        rows.append(
            {
                "model": full.name,
                "head_acc": head_b.accuracy,
                "head_f1": head_b.f1_macro,
                "slug_acc": slug_b.accuracy,
                "slug_f1": slug_b.f1_macro,
                "delta": slug_b.accuracy - head_b.accuracy,
            }
        )

    print(f"\n=== Results (n={len(y)}): ===")
    header = f"{'model':<42}  {'head_acc':>9}  {'slug_acc':>9}  {'slug_f1':>9}  {'delta':>9}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['model']:<42}  {r['head_acc']:>9.4f}  {r['slug_acc']:>9.4f}  "
            f"{r['slug_f1']:>9.4f}  {r['delta']:>+9.4f}"
        )

    if rows:
        rows.sort(key=lambda r: r["slug_acc"], reverse=True)
        print(f"\nBest on slug-form: {rows[0]['model']} (slug_acc={rows[0]['slug_acc']:.4f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
