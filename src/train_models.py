"""
train_models.py

Train multiple machine learning models for sentiment analysis and save them.

Models:
  - Naive Bayes (MultinomialNB)
  - Support Vector Machine (LinearSVC)
  - K-Nearest Neighbors
  - Decision Tree
"""

import argparse
import os
import sys
from typing import Dict

from joblib import dump
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

# Allow "from preprocess import ..." when running from project root
CURRENT_DIR = os.path.dirname(__file__)
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from preprocess import load_datasets, preprocess_series  # noqa: E402


def get_models() -> Dict[str, object]:
    """
    Define the models to be trained.
    """
    return {
        "naive_bayes": MultinomialNB(),
        "svm": LinearSVC(),
        "knn": KNeighborsClassifier(n_neighbors=5),
        "decision_tree": DecisionTreeClassifier(random_state=42),
    }


def train_and_save_models(
    data_dir: str = "data",
    models_dir: str = "models",
    max_features: int = 20000,
) -> None:
    """
    Train models on the training dataset and save them along with the TF-IDF vectorizer.
    """
    print(f"Loading datasets from {data_dir}...")
    train_df, _ = load_datasets(data_dir)

    print("Preprocessing training text...")
    X_train_raw = preprocess_series(train_df["Reviews_text"])
    y_train = train_df["Class Label"]

    print("Fitting TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(X_train_raw)

    os.makedirs(models_dir, exist_ok=True)

    print(f"Saving TF-IDF vectorizer to {models_dir}...")
    dump(vectorizer, os.path.join(models_dir, "tfidf_vectorizer.joblib"))

    models = get_models()

    for name, model in models.items():
        print(f"Training model: {name}...")
        model.fit(X_train, y_train)
        model_path = os.path.join(models_dir, f"{name}.joblib")
        dump(model, model_path)
        print(f"Saved {name} to {model_path}")

    print("All models trained and saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train sentiment analysis models.")
    parser.add_argument("--data-dir", default="data", help="Directory containing training/test data.")
    parser.add_argument("--models-dir", default="models", help="Directory to save trained models.")

    args = parser.parse_args()
    train_and_save_models(data_dir=args.data_dir, models_dir=args.models_dir)
