# Sentiment Analysis of Android App Reviews

This repository contains my **MSc Data Science & Artificial Intelligence** coursework for the module **CIS4515 – Practical Data Analysis (Coursework 2)**.

The project builds a **sentiment analysis model** for Amazon reviews of Android applications and uses the results to compare three Android Application Development (AAD) companies based on user satisfaction.

---

## Project Overview


The dataset contains reviews for **nine Android apps**, grouped into **three AAD companies**:

- Each review has:
  - a sentiment label:
    - `1` = negative
    - `2` = neutral
    - `3` = positive
  - an app ID (e.g. `B004NWLM8K`)
  - the review text

The goal is to:

1. Train machine learning models to classify review sentiment.
2. Evaluate and compare different algorithms.
3. Use the best model’s predictions to estimate which AAD company has the highest average sentiment score, and therefore appears most successful.

A full academic report describing the methodology, experiments, and results is included in this repository.

---

## Repository Structure

```text
sentiment_analysis/
├── data/                    # Raw data files (not tracked in detail)
│   ├── reviews_Apps_for_Android_5.training.txt
│   └── reviews_Apps_for_Android_5.test.txt
├── notebooks/
│   ├── pda_sentiment_analysis.ipynb         # Original coursework notebook
│   └── pda_sentiment_analysis_clean.ipynb   # Clean, portfolio-ready notebook
│
├── reports/
│   ├── sentiment_analysis_report.pdf        # Formal coursework report
│   └── model_metrics.json                   # Saved evaluation metrics for all models
│
├── src/
│   ├── preprocess.py                        # Data loading and text cleaning
│   ├── utils.py                             # Helper functions & metrics
│   ├── train_models.py                      # Train sentiment models and save them
│   └── evaluate_models.py                   # Evaluate models & company sentiment
│
├── README.md
├── requirements.txt                         # Python dependencies
└── .gitignore


---

## Model Performance (from `model_metrics.json`)

| Model          | Accuracy | Precision | Recall | F1-Score |
|----------------|----------|-----------|--------|----------|
| Naive Bayes    | ~0.76    | ~0.76     | ~0.76  | ~0.68    |
| SVM            | ~0.81    | ~0.79     | ~0.81  | ~0.80    |
| KNN            | ~0.75    | ~0.69     | ~0.75  | ~0.70    |
| Decision Tree  | ~0.69    | ~0.69     | ~0.69  | ~0.69    |

SVM achieved the strongest overall performance and is used for company sentiment prediction.

---

## Company Sentiment Analysis (Using SVM Predictions)

Average predicted sentiment score per company (1 = negative, 3 = positive):

| Company | Avg Score |
|---------|-----------|
| AAD_1   | 2.94      |
| AAD_2   | 2.27      |
| AAD_3   | 1.95      |

Interpretation:

- AAD_1 has the most positive user sentiment  
- AAD_3 has the lowest sentiment  

---

## How to Run the Project Locally

### 1. Install dependencies
