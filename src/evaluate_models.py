"""
evaluate_models.py

Evaluate trained models on the test set and:
  - compute metrics (accuracy, precision, recall, F1)
  - compute average sentiment score per AAD company using SVM predictions
"""

import argparse
import os
import sys
from typing import Dict, Tuple

import pandas as pd
from joblib import load

# ensure we can import from this folder when running from project root
CURRENT_DIR = os.path.dirname(__file__)
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from preprocess import load_datasets, preprocess_series  # noqa: E402
from utils import classification_metrics, save_json  # noqa: E402


# Mapping from app IDs to companies (from the coursework)
APP_COMPANIES: Dict[str, str] = {
    # AAD_1
    "B004NWLM8K": "AAD_1",
    "B004Q1NH4U": "AAD_1",
    "B004LPBTAA": "AAD_1",
    # AAD_2
    "B004S6NAOU": "AAD_2",
    "B004R6HTWU": "AAD_2",
    "B004N8KDNY": "AAD_2",
    # AAD_3
    "B004KA0RBS": "AAD_3",
    "B004NPELDA": "AAD_3",
    "B004L26XXQ": "AAD_3",
}


def load_models(models_dir: str = "models") -> Tuple[object, Dict[str, object]]:
    """
    Load the TF-IDF vectorizer and all available models from models_dir.
    """
    vectorizer_path = os.path.join(models_dir, "tfidf_vectorizer.joblib")
    if not os.path.exists(vectorizer_path):
        raise FileNotFoundError(
            f"TF-IDF vectorizer not found at {vectorizer_path}. "
            f"Run train_models.py first."
        )

    vectorizer = load(vectorizer_path)

    models: Dict[str, object] = {}
    for name in ["naive_bayes", "svm", "knn", "decision_tree"]:
        model_path = os.path.join(models_dir, f"{name}.joblib")
        if os.path.exists(model_path):
            models[name] = load(model_path)
        else:
            print(f"Warning: model file not found: {model_path}")

    return vectorizer, models


def evaluate_models(
    data_dir: str = "data",
    models_dir: str = "models",
    metrics_output: str = "reports/model_metrics.json",
) -> None:
    """
    Evaluate each model on the test set and save metrics.
    Also compute average sentiment per company using the SVM model.
    """
    print(f"Loading datasets from {data_dir}...")
    _, test_df = load_datasets(data_dir)

    print("Preprocessing test text...")
    X_test_raw = preprocess_series(test_df["Reviews_text"])
    y_test = test_df["Class Label"]

    print(f"Loading models from {models_dir}...")
    vectorizer, models = load_models(models_dir)
    X_test = vectorizer.transform(X_test_raw)

    all_metrics: Dict[str, Dict[str, float]] = {}

    for name, model in models.items():
        print(f"Evaluating model: {name}...")
        y_pred = model.predict(X_test)
        metrics = classification_metrics(y_test, y_pred)
        all_metrics[name] = metrics
        print(
            f"{name} -> "
            f"accuracy={metrics['accuracy']:.3f}, "
            f"precision={metrics['precision']:.3f}, "
            f"recall={metrics['recall']:.3f}, "
            f"f1={metrics['f1']:.3f}"
        )

    save_json(all_metrics, metrics_output)
    print(f"\nSaved metrics to {metrics_output}")

    # Use SVM predictions to compute company-level sentiment
    if "svm" in models:
        print("\nComputing average sentiment score per company using SVM predictions...")
        svm_model = models["svm"]
        svm_preds = svm_model.predict(X_test)

        test_with_preds: pd.DataFrame = test_df.copy()
        test_with_preds["Predicted_Label"] = svm_preds
        test_with_preds["Company"] = test_with_preds["ID"].map(APP_COMPANIES)

        # drop any rows where company mapping is missing
        test_with_preds = test_with_preds.dropna(subset=["Company"])

        company_scores = (
            test_with_preds.groupby("Company")["Predicted_Label"].mean().sort_values(ascending=False)
        )

        print("\nAverage sentiment scores (higher = more positive):")
        for company, score in company_scores.items():
            print(f"{company}: {score:.4f}")
    else:
        print("SVM model not found; skipping company sentiment analysis.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained sentiment analysis models.")
    parser.add_argument("--data-dir", default="data", help="Directory containing training/test data.")
    parser.add_argument("--models-dir", default="models", help="Directory where models are saved.")
    parser.add_argument(
        "--metrics-output",
        default="reports/model_metrics.json",
        help="Path to save evaluation metrics JSON.",
    )

    args = parser.parse_args()
    evaluate_models(
        data_dir=args.data_dir,
        models_dir=args.models_dir,
        metrics_output=args.metrics_output,
    )
