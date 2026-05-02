# Leaderboard submission — Project B

Files to upload to the [News Headline Classifier Leaderboard](https://huggingface.co/spaces/...):

- `model.py` — `Model` class with the trained TF-IDF + LR pipeline embedded as a base64 gzip
- `preprocess.py` — `prepare_data(csv_path)` returning `(X, y)`
- **No `model.pt` needed** — weights are embedded in `model.py`

## What the model is

- Classical pipeline: TF-IDF FeatureUnion of word 1-2grams (200k features) + char_wb 3-5grams (200k features)
- Logistic Regression with `class_weight='balanced'`, `C=2.0`, `sublinear_tf=True`
- Trained on **all 3,734 cleaned headlines** (Fox + NBC, 2019-2026)
- Predictions: returns "FoxNews" / "NBC" string labels

## Local eval (mirror of the backend)

```bash
cd submission
uv run python _local_eval.py --csv ../data/processed/splits_random/test.csv
# ~0.86 on a random held-out test (model not trained on this in the train-only variant)
# ~0.98 on the full-data model (it has seen the test set; not a real metric)
```

## Why not DistilBERT?

The backend env has only `numpy, pandas, torch==2.9.1, torchvision, scikit-learn, opencv-python` — **no `transformers` library**. We'd need a pure-PyTorch reimplementation + WordPiece tokenizer (~3-4 hr work) and the gain over classical is marginal (DistilBERT got 0.821 val / 0.840 test vs classical V2 0.813 val / 0.858 test).

If `transformers` gets approved via Ed post, we can swap in DistilBERT without changing `preprocess.py`.

## How to submit (per Project_submission.pdf)

1. Open the Hugging Face leaderboard: [News Headline Classifier](https://huggingface.co/spaces/...)
2. Click **Student Submissions** tab
3. Fill in **Group ID** (your number on the sheet) and **Alias** (your team name)
4. Upload `model.py` and `preprocess.py`
5. Skip `model.pt` (not needed)
6. Click submit; results appear in **Submission Status** then **Leaderboard**.
