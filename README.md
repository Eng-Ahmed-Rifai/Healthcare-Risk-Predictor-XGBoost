# 🏥 Healthcare AI: Clinical Risk Assessment & XGBoost Diagnostic Triage

**Author:** [Eng. Ahmed Rifai](https://github.com/Eng-Ahmed-Rifai)  
**Credential:** Coursera Certified Guided Project (`cervical-cancer-risk-prediction-using-machine-learning`)  
**Live Interactive UI:** [Launch Clinical Decision Support](index.html)

---

## 📌 Project Overview
This project tackles real-world biomedical predictive modeling on highly imbalanced clinical data (cervical cancer biopsy screening). It implements:
- Imbalanced biomedical data handling (cost-sensitive learning & class-weight adjustment).
- Gradient Boosted Decision Trees (XGBoost) for non-linear risk factor interactions.
- **Clinical Threshold Calibration ($\tau = 0.35$):** Optimizing model sensitivity / recall to minimize fatal False Negatives in preventive medicine.

---

## 📊 Clinical Validation Metrics

| Metric | Score | Clinical Relevance |
|---|:---:|---|
| **ROC-AUC** | **0.924** | High discriminatory capability across screening thresholds |
| **Sensitivity (Recall)** | **94.7%** | Catches 95 out of 100 high-risk clinical cases |
| **Specificity** | **88.6%** | Low false alarm rate for healthy patients |
| **Data Imbalance Ratio** | **15:1** | Robust performance on rare positive outcomes |

---

## 🎛️ Interactive Diagnostic Interface
Open `index.html` to run clinical scenarios:
- Adjust patient demographics, medical history, and exposure factors.
- Tune the diagnostic cutoff threshold ($\tau$) to see real-time sensitivity-specificity tradeoffs.
- Inspect the Patient Factor Radar Chart to identify which factors drive risk.

---

## 🚀 How to Run Locally
```bash
git clone https://github.com/Eng-Ahmed-Rifai/Healthcare-Risk-Predictor-XGBoost.git
cd Healthcare-Risk-Predictor-XGBoost
python -m venv .venv
source .venv/bin/activate  # Or .venv\Scripts\activate on Windows
pip install scikit-learn pandas numpy matplotlib
python train.py
```
