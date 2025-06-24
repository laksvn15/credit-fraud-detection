# Credit Card Fraud Detection 🔍💳

Detect fraudulent credit card transactions using machine learning and oversampling techniques.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen)

---

## 📌 Overview

This project uses **Logistic Regression** and **XGBoost** to detect fraudulent credit card transactions. We address the **class imbalance** problem using **SMOTE (Synthetic Minority Oversampling Technique)** and evaluate performance using metrics such as **ROC-AUC**, **precision-recall**, and **confusion matrix**.

---

## 📁 Dataset

- **Source**: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Size**: 284,807 transactions | 492 frauds (0.172%)

---

## 🛠️ Techniques Used

- Data preprocessing and feature scaling
- SMOTE for balancing classes
- Model training with:
  - Logistic Regression
  - XGBoost Classifier
- ROC-AUC, classification reports
- Precision-Recall curve visualization

---

## 📊 Model Evaluation

![Precision-Recall Curve](outputs/precision_recall_curve.png)

Classification report and confusion matrices for thresholds `0.5` and `0.8` are saved in [`outputs/classification_report.txt`](outputs/classification_report.txt)

---

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/credit-fraud-detection.git
cd credit-fraud-detection
