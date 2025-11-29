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
sentiment-analysis/
├── README.md                # Project overview (this file)
│
├── data/                    # Raw data files (not tracked in detail)
│   ├── reviews_Apps_for_Android_5.training.txt
│   └── reviews_Apps_for_Android_5.test.txt
│
├── notebooks/               # Jupyter notebooks
│   └── pda_sentiment_analysis.ipynb
│
└── reports/                 # Written report(s) and generated outputs
    └── sentiment_analysis_report.pdf
