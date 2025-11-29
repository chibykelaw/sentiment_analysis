# Sentiment Analysis of Android App Reviews  
### CIS4515 – Practical Data Analysis (Coursework 2)  
### MSc Data Science & Artificial Intelligence  
#### Edge Hill University
**Author:** Chibuike Lawrence Orji-Oko

---

## 📌 Project Overview

This repository contains my **MSc Data Science & Artificial Intelligence** coursework for the module **CIS4515 – Practical Data Analysis (Coursework 2)**.

The goal of this project is to build a **sentiment analysis model** using Amazon reviews of Android applications, and use the predictions to compare **three Android Application Development (AAD) companies** based on user satisfaction.

---

## 📃 Dataset Description

The dataset contains reviews for **nine Android apps**, grouped into **three AAD companies**.

Each review includes:

- **Sentiment label**  
  - `1` = negative  
  - `2` = neutral  
  - `3` = positive  
- **App ID** (e.g., `B004WNLW8K`)  
- **Review text**

Files used:

- `reviews_Apps_for_Android_5.training.txt`  
- `reviews_Apps_for_Android_5.test.txt`

---

## 🧠 Project Tasks

1. **Preprocess** the text data  
2. **Train multiple ML models**:
   - Naive Bayes  
   - SVM  
   - kNN  
   - Decision Tree  
3. **Evaluate all models**  
4. **Select the best model** (SVM)  
5. **Predict sentiment per company**  
6. **Compute average sentiment scores**  
7. **Identify the best-performing AAD company**

A full academic report is included in this repository (`reports/sentiment_analysis_report.pdf`).

---

## 📊 Model Performance Summary

Performance metrics (saved in `reports/model_metrics.json`):

| Model          | Accuracy | Precision | Recall | F1-Score |
|----------------|----------|-----------|--------|----------|
| Naive Bayes    | 0.757    | 0.759     | 0.757  | 0.679    |
| SVM            | **0.814** | 0.789     | 0.814  | **0.797** |
| kNN            | 0.751    | 0.694     | 0.751  | 0.704    |
| Decision Tree  | 0.698    | 0.689     | 0.698  | 0.693    |

✔ **SVM achieved the highest performance** and was selected as the final model.

---

## 🏆 Average Sentiment Scores (Using SVM Predictions)

Higher = more positive user sentiment.

| Company | Avg Sentiment |
|---------|----------------|
| **AAD_1** | **2.94** |
| AAD_2 | 2.27 |
| AAD_3 | 1.95 |

**Conclusion:**  
**AAD_1** demonstrates the highest overall user satisfaction among the three companies.

---

## 📁 Repository Structure

```text
sentiment_analysis/
│
├── README.md                  # Project overview and documentation
├── .gitignore                 # Ignore rules
├── requirements.txt           # Python dependencies
│
├── data/                      # Raw dataset files
│   ├── reviews_Apps_for_Android_5.training.txt
│   └── reviews_Apps_for_Android_5.test.txt
│
├── notebooks/                 # Jupyter notebooks
│   ├── pda_sentiment_analysis.ipynb
│   └── pda_sentiment_analysis_clean.ipynb
│
├── reports/                   # Written report and results
│   ├── sentiment_analysis_report.pdf
│   └── model_metrics.json
│
└── src/                       # Source code
    ├── preprocess.py          # Data loading & preprocessing functions
    ├── train_models.py        # Model training & saving
    └── evaluate_models.py     # Model evaluation & company scoring
```
---
## ▶️ Running the Project
### 1. Create & activate a virtual environment

python -m venv venv

venv\Scripts\activate   # Windows
