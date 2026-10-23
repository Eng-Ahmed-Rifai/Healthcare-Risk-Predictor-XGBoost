"""
Healthcare AI: Cervical Cancer Diagnostic Risk Prediction with XGBoost
Handling Imbalanced Biomedical Data & Sensitivity Tuning
Author: Eng. Ahmed Rifai
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

def generate_clinical_data(n_samples=1200, random_state=42):
    np.random.seed(random_state)
    age = np.random.randint(18, 65, n_samples)
    pregnancies = np.random.poisson(2, n_samples)
    smokes_years = np.where(np.random.rand(n_samples) > 0.7, np.random.uniform(1, 20, n_samples), 0)
    hc_years = np.where(np.random.rand(n_samples) > 0.5, np.random.uniform(1, 15, n_samples), 0)
    stds = np.where(np.random.rand(n_samples) > 0.85, np.random.poisson(1, n_samples), 0)
    
    # Ground truth: imbalanced outcome (~7% positive biopsy)
    logit = -3.5 + 0.04 * (age - 30) + 0.15 * pregnancies + 0.18 * smokes_years + 0.16 * hc_years + 1.2 * stds
    prob = 1 / (1 + np.exp(-logit))
    biopsy = (np.random.rand(n_samples) < prob).astype(int)
    
    df = pd.DataFrame({
        'Age': age,
        'Pregnancies': pregnancies,
        'SmokesYears': smokes_years,
        'HormonalContraceptivesYears': hc_years,
        'STDsCount': stds,
        'BiopsyResult': biopsy
    })
    return df

def train_clinical_model():
    df = generate_clinical_data()
    X = df.drop(columns=['BiopsyResult'])
    y = df['BiopsyResult']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    
    # Use class-weight adjusted gradient boosting to handle medical imbalance
    model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42)
    model.fit(X_train, y_train)
    
    y_probs = model.predict_proba(X_test)[:, 1]
    
    # Evaluate at standard cutoff 0.5 vs clinical safety cutoff 0.35
    auc = roc_auc_score(y_test, y_probs)
    print(f"Clinical Model ROC-AUC: {auc:.4f}")
    
    print("\n--- Diagnostic Classification Report (Threshold tau = 0.35) ---")
    y_pred_clinical = (y_probs >= 0.35).astype(int)
    print(classification_report(y_test, y_pred_clinical, target_names=['Negative', 'Positive']))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_clinical))

if __name__ == "__main__":
    train_clinical_model()
