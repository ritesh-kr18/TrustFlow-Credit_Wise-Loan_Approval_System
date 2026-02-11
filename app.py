import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. Load the model and scaler
model = joblib.load('loan_model.pkl')
scaler = joblib.load('scaler.pkl')

st.set_page_config(page_title="TrustFlow Loan System", layout="wide")
st.title("🏦 TrustFlow: Loan Approval Prediction System")
st.write("Enter the applicant's details below to check for loan eligibility.")

# 2. Create the User Interface
with st.form("loan_form"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        income = st.number_input("Applicant Income", min_value=0, value=5000)
        co_income = st.number_input("Coapplicant Income", min_value=0, value=0)
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, value=0)
        edu = st.selectbox("Education Level", ["Graduate", "Not Graduate"])
        gender = st.selectbox("Gender", ["Male", "Female"])

    with col2:
        loan_amt = st.number_input("Loan Amount Requested", min_value=0, value=15000)
        loan_term = st.number_input("Loan Term (Months)", min_value=1, value=360)
        credit_score = st.slider("Credit Score", 300, 900, 700)
        exist_loans = st.number_input("Existing Loans count", min_value=0, value=0)
        marital = st.selectbox("Marital Status", ["Single", "Married"])

    with col3:
        savings = st.number_input("Savings Balance", min_value=0, value=1000)
        collateral = st.number_input("Collateral Value", min_value=0, value=10000)
        property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
        emp_status = st.selectbox("Employment Status", ["Salaried", "Self-employed", "Unemployed"])
        purpose = st.selectbox("Loan Purpose", ["Home", "Personal", "Education", "Car"])
        emp_cat = st.selectbox("Employer Category", ["Private", "Government", "MNC", "Unemployed"])

    submit = st.form_submit_button("Predict Loan Approval")

if submit:
    dti = loan_amt / (income + co_income + 1) # +1 to avoid division by zero
    
    data = {
        'Applicant_Income': income,
        'Coapplicant_Income': co_income,
        'Age': age,
        'Dependents': dependents,
        'Existing_Loans': exist_loans,
        'Savings': savings,
        'Collateral_Value': collateral,
        'Loan_Amount': loan_amt,
        'Loan_Term': loan_term,
        'Education_Level': 1 if edu == "Graduate" else 0,
        'Employment_Status_Salaried': 1 if emp_status == "Salaried" else 0,
        'Employment_Status_Self-employed': 1 if emp_status == "Self-employed" else 0,
        'Employment_Status_Unemployed': 1 if emp_status == "Unemployed" else 0,
        'Marital_Status_Single': 1 if marital == "Single" else 0,
        'Loan_Purpose_Car': 1 if purpose == "Car" else 0,
        'Loan_Purpose_Education': 1 if purpose == "Education" else 0,
        'Loan_Purpose_Home': 1 if purpose == "Home" else 0,
        'Loan_Purpose_Personal': 1 if purpose == "Personal" else 0,
        'Property_Area_Semiurban': 1 if property_area == "Semiurban" else 0,
        'Property_Area_Urban': 1 if property_area == "Urban" else 0,
        'Gender_Male': 1 if gender == "Male" else 0,
        'Employer_Category_Government': 1 if emp_cat == "Government" else 0,
        'Employer_Category_MNC': 1 if emp_cat == "MNC" else 0,
        'Employer_Category_Private': 1 if emp_cat == "Private" else 0,
        'Employer_Category_Unemployed': 1 if emp_cat == "Unemployed" else 0,
        'DTI_Ratio_sq': dti ** 2,
        'Credit_Score_sq': credit_score ** 2
    }
    
    # Create DataFrame and ensure the columns match the scaler's order
    df_input = pd.DataFrame([data])
    feature_order = scaler.feature_names_in_
    df_input = df_input[feature_order]
    
    # 4. Scale and Predict
    scaled_data = scaler.transform(df_input)
    prediction = model.predict(scaled_data)
    
    st.divider()
    if prediction[0] == 1:
        st.success("🎉 **Result: LOAN APPROVED**")
        st.balloons()
    else:
        st.error("❌ **Result: LOAN REJECTED**")