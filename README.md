# 💳 Credit Card Fraud Detection using Machine Learning

This project focuses on detecting fraudulent transactions from an imbalanced credit card dataset using advanced machine learning techniques. It involves data preprocessing, exploratory analysis, SMOTE oversampling, and building classification models like Logistic Regression, Random Forest, and XGBoost.

---

## 📁 Dataset

- **Source**: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Size**: 284,807 transactions | 492 frauds (~0.17%)
- **Features**: 30 anonymized variables (V1–V28) + `Time`, `Amount`, `Class`

---

## ✅ Problem Statement

To build a robust classification system that accurately detects frauds in highly imbalanced credit card transaction data with minimal false positives.

---

## 🔍 Workflow Overview

1. **Data Loading & Preprocessing**
   - Null value check
   - Class distribution analysis
   - Feature scaling (`StandardScaler`)
   - Train-test split (80-20)

2. **Exploratory Data Analysis (EDA)**
   - Count plots, correlation heatmaps
   - Distribution of features by fraud vs non-fraud
   - Fraud % analysis

3. **Handling Class Imbalance**
   - **SMOTE (Synthetic Minority Oversampling Technique)** applied only on training data

4. **Model Training & Evaluation**
   - ✅ Logistic Regression  
   - ✅ Random Forest Classifier  
   - ✅ XGBoost Classifier  
   - Evaluation metrics used:
     - Confusion Matrix
     - Precision, Recall, F1-score
     - ROC-AUC Score
     - Heatmaps for visualization

---

## 📊 Model Performance (Test Set)

| Model               | Precision | Recall | F1-Score | ROC-AUC |
|--------------------|-----------|--------|----------|---------|
| Logistic Regression| 0.72      | 0.76   | 0.74     | 0.87    |
| Random Forest      | 0.84      | 0.80   | 0.82     | 0.90    |
| XGBoost            | 0.76      | 0.80   | 0.78     | 0.90    |

> Random Forest performed best in terms of precision and F1-score.

---

## 📌 Key Learnings

- Tackling **imbalanced data** using SMOTE
- Applying classification algorithms with **real-world fraud data**
- Evaluating models beyond accuracy (precision-recall tradeoff)
- Data visualization for fraud detection insights

---

## 🚀 Tools & Libraries

- Python, Pandas, NumPy, Matplotlib, Seaborn
- Scikit-learn (SMOTE, models, metrics)
- XGBoost

---

## 📂 Project Structure

📦 credit-card-fraud-detection/
├── creditcard.csv
├── fraud_detection.ipynb
├── README.md
└── requirements.txt


## 📮 Future Improvements
- Hyperparameter tuning with GridSearchCV
- Deep learning with autoencoders
- Deploy as an API using Flask/FastAPI

