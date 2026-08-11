# 💳 Credit Risk Scoring

An educational, explainable ML demo that estimates loan default probability, assigns risk bands, and compares outcomes across applicant groups. **It is not an automated lending system.**

## Pipeline
1. Generate or upload applicant data.
2. Impute missing values, scale numeric fields, and one-hot encode categories.
3. Train class-weighted Logistic Regression.
4. Predict probability of default and map it to Low/Medium/High risk.
5. Evaluate ROC-AUC, precision, recall, and confusion matrix.
6. Compare group-level default rates as an initial fairness check.

## Features
- Income, loan amount, debt-to-income ratio
- Credit history, employment length, loan purpose
- Risk probability and interpretable risk band
- Group-level fairness summary

## Interview talking points
- Class weighting prevents majority-class repayment examples from hiding defaults.
- Explainability and human review are essential for high-impact financial decisions.
- A strong AUC does not guarantee fairness; subgroup metrics and monitoring are required.

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```
