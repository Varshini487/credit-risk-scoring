import streamlit as st
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix

st.set_page_config(page_title="Credit Risk Scoring", layout="wide")
st.title("💳 Explainable Credit Risk Scoring")
st.caption("Educational demo — not a lending decision")

@st.cache_data
def data(n=1200):
    rng=np.random.default_rng(7)
    d=pd.DataFrame({"income":rng.normal(65000,22000,n).clip(18000),"debt_to_income":rng.beta(2,5,n),"credit_history_years":rng.integers(1,30,n),"employment_years":rng.integers(0,25,n),"loan_amount":rng.normal(18000,9000,n).clip(1000),"purpose":rng.choice(["home","education","auto","personal"],n),"group":rng.choice(["Group A","Group B"],n,p=[.7,.3])})
    risk=.8*d.debt_to_income-.025*d.credit_history_years-.03*d.employment_years+.000012*d.loan_amount+rng.normal(0,.15,n)
    d["default"]=(risk>np.quantile(risk,.78)).astype(int)
    return d

df=data(); num=["income","debt_to_income","credit_history_years","employment_years","loan_amount"]; cat=["purpose"]
pre=ColumnTransformer([("num",Pipeline([("impute",SimpleImputer(strategy="median")),("scale",StandardScaler())]),num),("cat",OneHotEncoder(handle_unknown="ignore"),cat)])
model=Pipeline([("pre",pre),("clf",LogisticRegression(class_weight="balanced",max_iter=1000))])
Xtr,Xte,ytr,yte=train_test_split(df[num+cat],df.default,test_size=.25,random_state=42,stratify=df.default)
model.fit(Xtr,ytr); proba=model.predict_proba(Xte)[:,1]

c1,c2,c3=st.columns(3); c1.metric("Applicants",len(df)); c2.metric("Default rate",f"{df.default.mean():.1%}"); c3.metric("ROC-AUC",f"{roc_auc_score(yte,proba):.3f}")
st.subheader("Applicant scoring")
with st.form("score"):
    a,b=st.columns(2); income=a.number_input("Annual income",18000.,250000.,65000.); dti=b.slider("Debt-to-income",0.,1.,.25); hist=a.slider("Credit history (years)",0,40,8); emp=b.slider("Employment (years)",0,40,5); loan=a.number_input("Loan amount",1000.,250000.,18000.); purpose=b.selectbox("Purpose",["home","education","auto","personal"]); go=st.form_submit_button("Score applicant")
if go:
    p=float(model.predict_proba(pd.DataFrame([{"income":income,"debt_to_income":dti,"credit_history_years":hist,"employment_years":emp,"loan_amount":loan,"purpose":purpose}]))[0,1]); band="High" if p>=.65 else "Medium" if p>=.35 else "Low"
    st.metric("Estimated probability of default",f"{p:.1%}"); st.warning(f"Risk band: {band}") if band!="Low" else st.success(f"Risk band: {band}")

st.subheader("Fairness check")
st.dataframe(df.groupby("group").default.agg(["count","mean"]).rename(columns={"mean":"default_rate"}),use_container_width=True)
st.info("For production: add calibration, adverse-action explanations, consent/privacy controls, subgroup validation, and human review.")
