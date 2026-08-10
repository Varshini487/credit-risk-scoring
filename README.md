# 💳 Credit Risk Scoring

An explainable machine-learning demo that estimates probability of loan default, assigns risk bands, and surfaces group-level fairness checks.

## How it works
1. Load applicant attributes such as income, debt-to-income ratio, credit history, employment length, loan amount, and purpose.
2. Impute numeric values, standardize features, and one-hot encode categories.
3. Train a class-weighted Logistic Regression model to handle default imbalance.
4. Predict default probability and map it to Low/Medium/High risk.
5. Report ROC-AUC, precision, recall, confusion matrix, and group-level default rates.

## Tech stack
Python, Pandas, NumPy, Scikit-learn, Plotly, Streamlit

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Interview points
- Probability and explanations are more useful than a black-box approval label.
- Class weighting protects minority default examples from being ignored.
- Fairness needs separate group-level metrics; strong AUC alone does not guarantee equitable outcomes.

> Educational decision-support demo. Do not use this repository for automated lending or real credit decisions without legal, fairness, privacy, and model-risk review.
