"""
preprocess.py

Functions for loading and cleaning the Android app reviews dataset.
"""

import os
import re
from typing import Tuple

import pandas as pd


def load_datasets(data_dir: str = "data") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the training and test datasets from the given data directory.

    Expects two tab-separated files:
      - reviews_Apps_for_Android_5.training.txt
      - reviews_Apps_for_Android_5.test.txt

    Returns:
        train_df, test_df  (pandas DataFrames)
    """
    train_path = os.path.join(data_dir, "reviews_Apps_for_Android_5.training.txt")
    test_path = os.path.join(data_dir, "reviews_Apps_for_Android_5.test.txt")

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training file not found: {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test file not found: {test_path}")

    train_df = pd.read_csv(
        train_path,
        sep="\t",
        header=None,
        names=["Class Label", "ID", "Reviews_text"],
    )

    test_df = pd.read_csv(
        test_path,
        sep="\t",
        header=None,
        names=["Class Label", "ID", "Reviews_text"],
    )

    return train_df, test_df


def preprocess_text(text: str) -> str:
    """
    Basic text cleaning:
      - lowercasing
      - remove non-letters
      - collapse extra spaces
    """
    if not isinstance(text, str):
        text = str(text)

    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)  # keep only letters and spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_series(text_series: pd.Series) -> pd.Series:
    """
    Apply preprocess_text() to a pandas Series of review texts.
    """
    return text_series.astype(str).apply(preprocess_text)


if __name__ == "__main__":
    # Quick test: load data and show a few cleaned examples
    train_df, test_df = load_datasets()
    print(train_df.head())
    print(preprocess_series(train_df["Reviews_text"]).head())
