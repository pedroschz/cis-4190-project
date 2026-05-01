"""Shared metrics + reporting helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

LABEL_TO_INT = {"FoxNews": 1, "NBC": 0}
INT_TO_LABEL = {v: k for k, v in LABEL_TO_INT.items()}


def labels_to_ints(labels: Iterable[str]) -> np.ndarray:
    return np.array([LABEL_TO_INT[s] for s in labels], dtype=np.int64)


def ints_to_labels(ints: Iterable[int]) -> list[str]:
    return [INT_TO_LABEL[int(i)] for i in ints]


@dataclass
class MetricBundle:
    name: str
    accuracy: float
    f1_macro: float
    f1_fox: float
    f1_nbc: float
    precision_fox: float
    recall_fox: float
    precision_nbc: float
    recall_nbc: float
    n: int

    def asdict(self) -> dict:
        return asdict(self)


def evaluate_predictions(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> MetricBundle:
    return MetricBundle(
        name=name,
        accuracy=float(accuracy_score(y_true, y_pred)),
        f1_macro=float(f1_score(y_true, y_pred, average="macro")),
        f1_fox=float(f1_score(y_true, y_pred, pos_label=LABEL_TO_INT["FoxNews"])),
        f1_nbc=float(f1_score(y_true, y_pred, pos_label=LABEL_TO_INT["NBC"])),
        precision_fox=float(precision_score(y_true, y_pred, pos_label=LABEL_TO_INT["FoxNews"], zero_division=0)),
        recall_fox=float(recall_score(y_true, y_pred, pos_label=LABEL_TO_INT["FoxNews"], zero_division=0)),
        precision_nbc=float(precision_score(y_true, y_pred, pos_label=LABEL_TO_INT["NBC"], zero_division=0)),
        recall_nbc=float(recall_score(y_true, y_pred, pos_label=LABEL_TO_INT["NBC"], zero_division=0)),
        n=int(len(y_true)),
    )


def print_report(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    print(f"\n=== {name} ===")
    print(classification_report(y_true, y_pred, target_names=["NBC", "FoxNews"], digits=4))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    print("Confusion matrix (rows=true, cols=pred), order [NBC, FoxNews]:")
    print(cm)


def append_results_row(out_csv: Path, bundle: MetricBundle, extras: dict | None = None) -> None:
    """Append one row to the running results CSV used by the report."""
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    row = bundle.asdict()
    if extras:
        row.update(extras)
    if out_csv.exists():
        import pandas as pd

        existing = pd.read_csv(out_csv)
        all_rows = list(existing.to_dict(orient="records")) + [row]
    else:
        all_rows = [row]

    import pandas as pd

    pd.DataFrame(all_rows).to_csv(out_csv, index=False)


def save_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=2, default=str)
