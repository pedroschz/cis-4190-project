"""Hugging Face Trainer wrapper for headline classification."""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.eval.metrics import (
    append_results_row,
    evaluate_predictions,
    labels_to_ints,
    print_report,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("models.transformer")

SEED = 42


def _set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _load_split_csv(split: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = Path("data/processed") / f"splits_{split}"
    if not base.exists():
        raise SystemExit(f"Split dir not found: {base}")
    return (
        pd.read_csv(base / "train.csv"),
        pd.read_csv(base / "val.csv"),
        pd.read_csv(base / "test.csv"),
    )


def _df_to_dataset(df: pd.DataFrame, tokenizer, max_length: int):
    """Materialize a HF datasets.Dataset from a pandas frame."""
    from datasets import Dataset

    ds = Dataset.from_pandas(
        pd.DataFrame(
            {
                "text": df["headline"].astype(str).tolist(),
                "label": labels_to_ints(df["source"].tolist()).tolist(),
            }
        )
    )

    def _tok(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_length, padding=False)

    return ds.map(_tok, batched=True, remove_columns=["text"])


def _compute_metrics(eval_pred):
    import numpy as np

    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return evaluate_predictions("trainer_eval", labels, preds).asdict()


def train(
    model_name: str,
    split: str,
    epochs: int,
    batch_size: int,
    lr: float,
    max_length: int,
    output_dir: Path,
    max_samples: int | None,
    eval_only: bool,
    fp16: bool,
) -> dict:
    """Fine-tune model_name on the requested split. Returns final metrics dict."""
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    _set_seed()
    set_seed(SEED)

    train_df, val_df, test_df = _load_split_csv(split)
    if max_samples:
        train_df = train_df.head(max_samples)
        val_df = val_df.head(max(20, max_samples // 5))

    logger.info("Loaded split=%s | train=%d val=%d test=%d", split, len(train_df), len(val_df), len(test_df))
    logger.info("Class balance (train): %s", train_df["source"].value_counts().to_dict())

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label={0: "NBC", 1: "FoxNews"},
        label2id={"NBC": 0, "FoxNews": 1},
    )

    train_ds = _df_to_dataset(train_df, tokenizer, max_length)
    val_ds = _df_to_dataset(val_df, tokenizer, max_length) if len(val_df) else None
    test_ds = _df_to_dataset(test_df, tokenizer, max_length) if len(test_df) else None

    output_dir.mkdir(parents=True, exist_ok=True)
    args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        learning_rate=lr,
        warmup_ratio=0.06,
        weight_decay=0.01,
        eval_strategy="epoch" if val_ds is not None else "no",
        save_strategy="epoch",
        load_best_model_at_end=val_ds is not None,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        save_total_limit=2,
        fp16=fp16,
        seed=SEED,
        logging_steps=50,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=_compute_metrics,
    )

    if not eval_only:
        trainer.train()

    results: dict = {}
    results_csv = Path("reports/figures/results_table.csv")
    safe_model_id = model_name.replace("/", "_")

    for stage_name, ds, df in (("val", val_ds, val_df), ("test", test_ds, test_df)):
        if ds is None or len(df) == 0:
            continue
        preds_out = trainer.predict(ds)
        preds = np.argmax(preds_out.predictions, axis=-1)
        y_true = labels_to_ints(df["source"].tolist())
        bundle = evaluate_predictions(f"{safe_model_id}_{split}_{stage_name}", y_true, preds)
        print_report(bundle.name, y_true, preds)
        append_results_row(
            results_csv,
            bundle,
            extras={
                "model": safe_model_id,
                "split": split,
                "stage": stage_name,
                "epochs": epochs,
                "lr": lr,
                "batch_size": batch_size,
                "max_length": max_length,
            },
        )
        results[stage_name] = bundle.asdict()

    # Save final tokenizer + model in a clean subdir
    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(final_dir)
    (output_dir / "metrics.json").write_text(json.dumps(results, indent=2, default=str))
    logger.info("Final metrics:\n%s", json.dumps(results, indent=2))
    return results


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--model", default="distilbert-base-uncased")
    p.add_argument("--split", choices=["random", "temporal"], default="random")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--output-dir", type=Path, default=Path("models/transformer_run"))
    p.add_argument("--max-samples", type=int, default=None, help="cap train rows (smoke test)")
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--no-fp16", action="store_true")
    args = p.parse_args(argv)

    train(
        model_name=args.model,
        split=args.split,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_length=args.max_length,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        eval_only=args.eval_only,
        fp16=not args.no_fp16,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
