import streamlit as st
import pickle
import numpy as np

# Load the trained model
with open('smote_rf_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Page config
st.set_page_config(page_title="Loan Approval Predictor", page_icon="💰", layout="centered")

# Custom dark theme
st.markdown("""
    <style>
        body {
            background-color: #0d0d0d;
            color: #e6ccff;
        }
        .stApp {
            background-color: #1a1a1a;
        }
        h1, h2, h3, h4 {
            color: #cc99ff;
        }
        .stButton>button {
            background-color: #9933ff;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("💰 Loan Approval Predictor")
st.subheader("Model: Random Forest + SMOTE")

# Input form
with st.form("loan_form"):
    Married = st.selectbox("Married", ["Yes", "No"])
    Dependents = st.selectbox("Dependents", [0, 1, 2, 3])
    Education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    Self_Employed = st.selectbox("Self Employed", ["Yes", "No"])
    ApplicantIncome = st.number_input("Applicant Income", min_value=0)
    CoapplicantIncome = st.number_input("Coapplicant Income", min_value=0)
    LoanAmount = st.number_input("Loan Amount (in thousands)", min_value=1)
    Loan_Amount_Term = st.selectbox("Loan Term (in months)", [360, 120, 180, 240, 300, 60, 84])
    Credit_History = st.selectbox("Credit History (1 = Good, 0 = Bad)", [1, 0])
    Property_Area = st.selectbox("Property Area", ["Urban", "Rural"])

    submit = st.form_submit_button("Check Loan Approval")

if submit:
    # Encode categorical features
    Married = 1 if Married == "Yes" else 0
    Education = 1 if Education == "Graduate" else 0
    Self_Employed = 1 if Self_Employed == "Yes" else 0
    Property_Urban = 1 if Property_Area == "Urban" else 0

    # Prepare feature array (10 features)
    features = np.array([[
        Married, Dependents, Education, Self_Employed,
        ApplicantIncome, CoapplicantIncome, LoanAmount,
        Loan_Amount_Term, Credit_History, Property_Urban
    ]])

    # Predict
    prediction = model.predict(features)[0]
    prob = model.predict_proba(features)[0][prediction]

    # Display result
    if prediction == 1:
        st.success(f"✅ Loan is likely to be **Approved**!\n\nConfidence: {prob*100:.2f}%")
    else:
        st.error(f"❌ Loan is likely to be **Rejected**.\n\nConfidence: {prob*100:.2f}%")