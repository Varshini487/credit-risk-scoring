import streamlit as st
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import plotly.express as px

st.set_page_config(page_title="Credit Risk Scoring", layout="wide")
st.title("💳 Explainable Credit Risk Scoring")
st.caption("Educational demo only — not a lending decision system.")

@st.cache_data
def sample_data(n=1600):
    rng=np.random.default_rng(7)
    d=pd.DataFrame({
      "income":rng.normal(65000,22000,n).clip(18000,180000),
      "debt_to_income":rng.uniform(.05,.75,n),
      "credit_history_years":rng.integers(1,30,n),
      "loan_amount":rng.normal(18000,10000,n).clip(1000,60000),
      "employment_years":rng.integers(0,25,n),
      "purpose":rng.choice(["home","education","car","personal"],n),
      "group":rng.choice(["Group A","Group B"],n,p=[.65,.35])})
    risk=(d.debt_to_income*1.8-d.credit_history_years*.035-d.employment_years*.025+d.loan_amount/d.income*.7+rng.normal(0,.25,n))
    d["default"]=(risk>0.68).astype(int)
    return d

df=sample_data()
num=["income","debt_to_income","credit_history_years","loan_amount","employment_years"]
cat=["purpose"]
X=df[num+cat]; y=df.default
pre=ColumnTransformer([("num",Pipeline([("imp",SimpleImputer(strategy="median")),("scale",StandardScaler())]),num),("cat",OneHotEncoder(handle_unknown="ignore"),cat)])
model=Pipeline([("pre",pre),("clf",LogisticRegression(class_weight="balanced",max_iter=1000))])
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.25,stratify=y,random_state=42)
model.fit(Xtr,ytr); probs=model.predict_proba(Xte)[:,1]

c1,c2,c3=st.columns(3); c1.metric("Applicants",len(df)); c2.metric("Default rate",f"{y.mean():.1%}"); c3.metric("ROC-AUC",f"{roc_auc_score(yte,probs):.3f}")

tab1,tab2,tab3=st.tabs(["📊 Model evaluation","🔮 Score an applicant","⚖️ Fairness check"])
with tab1:
    st.dataframe(df.head(15),use_container_width=True)
    st.text(classification_report(yte,(probs>=.5).astype(int),target_names=["Repay","Default"]))
    cm=confusion_matrix(yte,(probs>=.5).astype(int))
    st.plotly_chart(px.imshow(cm,text_auto=True,x=['Predicted repay','Predicted default'],y=['Actual repay','Actual default'],title='Confusion matrix'),use_container_width=True)
with tab2:
    a,b=st.columns(2)
    income=a.number_input("Annual income ($)",18000,250000,65000,1000)
    dti=b.slider("Debt-to-income ratio",0.01,.95,.30,.01)
    history=a.slider("Credit history (years)",0,40,8)
    loan=b.number_input("Loan amount ($)",1000,100000,18000,1000)
    emp=a.slider("Employment years",0,40,5)
    purpose=b.selectbox("Loan purpose",["home","education","car","personal"])
    row=pd.DataFrame([[income,dti,history,loan,emp,purpose]],columns=num+cat)
    p=float(model.predict_proba(row)[0,1]); band='High' if p>=.65 else ('Medium' if p>=.35 else 'Low')
    st.metric("Estimated probability of default",f"{p:.1%}"); st.info(f"Risk band: **{band}**")
    st.write("Use this result as one input for human review; investigate adverse-action explanations separately.")
with tab3:
    rates=df.groupby('group')['default'].agg(['mean','count']).reset_index().rename(columns={'mean':'default_rate'})
    st.dataframe(rates,use_container_width=True)
    st.plotly_chart(px.bar(rates,x='group',y='default_rate',text_auto='.1%',title='Observed default rate by group'),use_container_width=True)
    st.warning("Fairness review should include additional legally appropriate metrics, confidence intervals, and domain governance.")
