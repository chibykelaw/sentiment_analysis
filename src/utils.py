"""
utils.py

Helper functions: metrics + saving JSON.
"""

import json
import os
from typing import Any, Dict

from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def classification_metrics(y_true, y_pred) -> Dict[str, float]:
    """
    Compute accuracy, precision, recall and F1 (weighted average).
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def save_json(data: Dict[str, Any], path: str) -> None:
    """
    Save a dictionary to a JSON file with pretty formatting.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
