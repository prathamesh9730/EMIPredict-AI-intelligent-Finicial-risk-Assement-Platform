import os

import joblib
import pandas as pd
import streamlit as st

# Project root = one folder above app/
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(BASE, "models")

st.set_page_config(page_title="EMIPredict AI", layout="centered")


@st.cache_resource
def load_assets():
    clf = joblib.load(os.path.join(MODELS, "emi_classifier.joblib"))
    reg = joblib.load(os.path.join(MODELS, "emi_regressor.joblib"))
    cols = joblib.load(os.path.join(MODELS, "feature_columns.joblib"))
    le = joblib.load(os.path.join(MODELS, "label_encoder.joblib"))
    return clf, reg, cols, le


clf, reg, cols, le = load_assets()

# Categorical inputs: the one-hot column names come from these values.
# Values not present in feature_columns are ignored safely.
CAT_CHOICES = {
    "gender": ["Male", "Female"],
    "marital_status": ["Single", "Married"],
    "education": ["High School", "Graduate", "Post Graduate", "Professional"],
    "employment_type": ["Private", "Government", "Self-employed"],
    "company_type": ["MNC", "Mid-size", "Small", "Startup", "Large"],
    "house_type": ["Own", "Rented", "Family"],
    "emi_scenario": ["E-commerce Shopping EMI", "Home Appliances EMI",
                     "Vehicle EMI", "Personal Loan EMI", "Education EMI"],
    "existing_loans": ["Yes", "No"],
}

st.title("EMIPredict AI")
st.caption("EMI eligibility and maximum safe monthly EMI")

with st.form("applicant"):
    st.subheader("Personal details")
    c1, c2 = st.columns(2)
    with c1:
        age = st.number_input("Age", 25, 60, 30)
        gender = st.selectbox("Gender", CAT_CHOICES["gender"])
        marital_status = st.selectbox("Marital status", CAT_CHOICES["marital_status"])
        education = st.selectbox("Education", CAT_CHOICES["education"])
        family_size = st.number_input("Family size", 1, 10, 3)
        dependents = st.number_input("Dependents", 0, 10, 1)
    with c2:
        years_of_employment = st.number_input("Years of employment", 0, 40, 5)
        employment_type = st.selectbox("Employment type", CAT_CHOICES["employment_type"])
        company_type = st.selectbox("Company type", CAT_CHOICES["company_type"])
        house_type = st.selectbox("House type", CAT_CHOICES["house_type"])
        credit_score = st.number_input("Credit score", 300, 850, 700)
        existing_loans = st.selectbox("Existing loans", CAT_CHOICES["existing_loans"])

    st.subheader("Income and expenses (INR per month)")
    c3, c4 = st.columns(2)
    with c3:
        monthly_salary = st.number_input("Monthly salary", 15000, 200000, 50000, step=1000)
        monthly_rent = st.number_input("Monthly rent", 0, 100000, 10000, step=500)
        school_fees = st.number_input("School fees", 0, 100000, 0, step=500)
        college_fees = st.number_input("College fees", 0, 200000, 0, step=500)
        travel_expenses = st.number_input("Travel expenses", 0, 100000, 3000, step=500)
    with c4:
        groceries_utilities = st.number_input("Groceries and utilities", 0, 200000, 10000, step=500)
        other_monthly_expenses = st.number_input("Other monthly expenses", 0, 100000, 2000, step=500)
        current_emi_amount = st.number_input("Current EMI amount", 0, 200000, 0, step=500)
        bank_balance = st.number_input("Bank balance", 0, 10000000, 100000, step=5000)
        emergency_fund = st.number_input("Emergency fund", 0, 5000000, 50000, step=5000)

    st.subheader("Loan request")
    c5, c6 = st.columns(2)
    with c5:
        emi_scenario = st.selectbox("EMI scenario", CAT_CHOICES["emi_scenario"])
    with c6:
        requested_amount = st.number_input("Requested loan amount", 10000, 1500000, 200000, step=5000)
        requested_tenure = st.number_input("Tenure (months)", 3, 84, 24)

    submitted = st.form_submit_button("Check EMI eligibility", type="primary")


def build_row(v):
    """Rebuild the training features for one applicant, then align to the saved columns."""
    eps = 1
    sal = v["monthly_salary"]
    total_exp = (v["monthly_rent"] + v["school_fees"] + v["college_fees"]
                 + v["travel_expenses"] + v["groceries_utilities"]
                 + v["other_monthly_expenses"])

    feats = dict(v)
    feats.update({
        "total_monthly_expenses": total_exp,
        "debt_to_income": v["current_emi_amount"] / (sal + eps),
        "expense_to_income": total_exp / (sal + eps),
        "affordability_ratio": (sal - total_exp - v["current_emi_amount"]) / (sal + eps),
        "loan_to_salary": v["requested_amount"] / (sal + eps),
        "monthly_loan_burden": v["requested_amount"] / (v["requested_tenure"] + eps),
        "savings_ratio": (v["bank_balance"] + v["emergency_fund"]) / (sal + eps),
        "credit_risk_score": v["credit_score"] / 850,
    })

    # One-hot columns: set the matching "<column>_<value>" to 1
    for cat in CAT_CHOICES:
        feats[f"{cat}_{v[cat]}"] = 1

    row = {c: feats.get(c, 0) for c in cols}
    return pd.DataFrame([row], columns=cols)


if submitted:
    applicant = {
        "age": age, "gender": gender, "marital_status": marital_status,
        "education": education, "family_size": family_size, "dependents": dependents,
        "years_of_employment": years_of_employment, "employment_type": employment_type,
        "company_type": company_type, "house_type": house_type,
        "credit_score": credit_score, "existing_loans": existing_loans,
        "monthly_salary": monthly_salary, "monthly_rent": monthly_rent,
        "school_fees": school_fees, "college_fees": college_fees,
        "travel_expenses": travel_expenses, "groceries_utilities": groceries_utilities,
        "other_monthly_expenses": other_monthly_expenses,
        "current_emi_amount": current_emi_amount, "bank_balance": bank_balance,
        "emergency_fund": emergency_fund, "emi_scenario": emi_scenario,
        "requested_amount": requested_amount, "requested_tenure": requested_tenure,
    }
    X_new = build_row(applicant)

    label = le.inverse_transform(clf.predict(X_new))[0]
    proba = clf.predict_proba(X_new)[0]
    max_emi = reg.predict(X_new)[0]

    st.subheader("Result")
    st.metric("EMI eligibility", label)
    st.metric("Maximum safe monthly EMI (INR)", f"{max_emi:,.0f}")

    st.write("Class probabilities:")
    st.bar_chart(pd.Series(proba, index=le.classes_))