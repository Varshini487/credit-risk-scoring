# 💳 Credit Risk Scoring

An explainable machine-learning project that estimates loan default risk and audits model behavior across applicant groups. It is designed as an educational decision-support demo, not an autonomous lending system.

## What it does
- Cleans applicant data and handles categorical features
- Trains a class-weighted Logistic Regression baseline
- Reports ROC-AUC, precision, recall, and confusion matrix
- Produces a probability of default and a risk band
- Shows simple coefficient-based explanations
- Compares approval/default rates across groups for a fairness check

## How it works
1. Applicant features such as income, debt-to-income ratio, employment length, credit history, loan amount, and age are collected.
2. Numeric features are imputed and standardized; categorical features are one-hot encoded.
3. A class-weighted model learns from historical repayment outcomes.
4. The probability is calibrated into Low, Medium, or High risk bands.
5. Performance and group-level metrics are reviewed before any human decision.

## Tech stack
Python, Pandas, NumPy, Scikit-learn, Plotly, Streamlit

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Responsible-use notes
Credit decisions affect people’s access to finance. Do not use protected attributes as decision features, validate on representative data, monitor drift, provide explanations and appeals, and keep a qualified human in the loop.
