import streamlit as st
import numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report

st.set_page_config(page_title="Credit Risk Scoring", layout="wide")
st.title("💳 Explainable Credit Risk Scoring")
st.caption("Educational demo only — never use this output as an autonomous lending decision.")

@st.cache_data
def data(n=1200):
    rng=np.random.default_rng(7)
    d=pd.DataFrame({"income":rng.normal(65000,22000,n).clip(18000),"loan_amount":rng.normal(18000,9000,n).clip(1000),"debt_to_income":rng.uniform(.05,.75,n),"credit_history_years":rng.integers(0,25,n),"employment_years":rng.integers(0,20,n),"purpose":rng.choice(["home","education","auto","personal"],n),"group":rng.choice(["A","B","C"],n)})
    risk=.15+1.5*d.debt_to_income-.018*d.credit_history_years-.025*d.employment_years+(d.loan_amount/d.income)*.8
    d["default"]=(risk+rng.normal(0,.18,n)>.75).astype(int)
    return d

df=data(); st.write("Sample applicant data",df.head())
features=["income","loan_amount","debt_to_income","credit_history_years","employment_years","purpose"]
num=features[:5]; cat=["purpose"]
pre=ColumnTransformer([("num",Pipeline([("impute",SimpleImputer(strategy="median")),("scale",StandardScaler())]),num),("cat",Pipeline([("impute",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore"))]),cat)])
model=Pipeline([("preprocess",pre),("classifier",LogisticRegression(class_weight="balanced",max_iter=1000))])
Xtr,Xte,ytr,yte=train_test_split(df[features],df.default,test_size=.25,stratify=df.default,random_state=42)
model.fit(Xtr,ytr); p=model.predict_proba(Xte)[:,1]
c1,c2,c3=st.columns(3); c1.metric("Default rate",f"{df.default.mean():.1%}"); c2.metric("ROC-AUC",f"{roc_auc_score(yte,p):.3f}"); c3.metric("Applicants",len(df))
st.text(classification_report(yte,(p>=.5).astype(int),target_names=["Repay","Default"]))

st.subheader("Applicant risk estimate")
with st.form("risk"):
    a,b=st.columns(2); income=a.number_input("Annual income",18000.,250000.,65000.); loan=b.number_input("Loan amount",1000.,100000.,18000.); dti=a.slider("Debt-to-income",0.,1.,.3); hist=b.slider("Credit history (years)",0,30,8); emp=a.slider("Employment (years)",0,40,5); purpose=b.selectbox("Purpose",["home","education","auto","personal"]); go=st.form_submit_button("Score applicant")
if go:
    row=pd.DataFrame([{"income":income,"loan_amount":loan,"debt_to_income":dti,"credit_history_years":hist,"employment_years":emp,"purpose":purpose}]); prob=float(model.predict_proba(row)[0,1]); band="High" if prob>=.6 else "Medium" if prob>=.3 else "Low"; st.metric("Probability of default",f"{prob:.1%}"); st.warning(f"Risk band: {band}") if band!="Low" else st.success("Risk band: Low")

st.subheader("Group-level monitoring")
st.dataframe(df.groupby("group").default.agg(["count","mean"]).rename(columns={"mean":"default_rate"}))
