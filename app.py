import streamlit as st
import pandas as pd
import pickle
import numpy as np

# ───────────────────── Load Model & Encoders ─────────────────────
@st.cache_resource
def load_model():
    with open("employee_turnover_hr.pkl", "rb") as f:
        model = pickle.load(f)
    with open("encoders.pkl", "rb") as f:
        encoders = pickle.load(f)
    return model, encoders

model, encoders = load_model()

# ───────────────────── App Title ─────────────────────
st.title("💼 Employee Attrition Prediction")
st.markdown("Predict the likelihood of an employee leaving the company.")

# ───────────────────── Input Form ─────────────────────
st.subheader("Employee Information")

# Numerical inputs
age = st.slider("Age", 18, 65, 35)
monthly_income = st.number_input("Monthly Income ($)", min_value=1000, max_value=200000, value=5000)
years_at_company = st.number_input("Years at Company", min_value=0, max_value=50, value=5)
education = st.selectbox("Education Level", 
                         options=[1, 2, 3, 4, 5], 
                         format_func=lambda x: {1: 'Below College', 2: 'College', 3: 'Bachelor', 4: 'Master', 5: 'Doctor'}[x])

# Categorical inputs
gender = st.radio("Gender", ["Male", "Female"])
over_time = st.radio("Works Overtime?", ["Yes", "No"])

job_role = st.selectbox("Job Role", [
    "Sales Executive", "Research Scientist", "Laboratory Technician",
    "Manufacturing Director", "Healthcare Representative", "Manager",
    "Sales Representative", "Research Director", "Human Resources"
])

marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
department = st.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])
business_travel = st.selectbox("Business Travel", ["No Travel", "Travel Rarely", "Travel Frequently"])

# ───────────────────── Encode Input ─────────────────────
def preprocess_input():
    # Create DataFrame
    data = {
        "Age": age,
        "Gender": gender,
        "JobRole": job_role,
        "MaritalStatus": marital_status,
        "BusinessTravel": business_travel,
        "Department": department,
        "Education": education,
        "YearsAtCompany": years_at_company,
        "MonthlyIncome": monthly_income,
        "OverTime": over_time
    }
    df = pd.DataFrame([data])

    # Apply label encoding for binary variables
    for col in ["Gender", "OverTime"]:
        if col in encoders:
            try:
                df[col] = encoders[col].transform(df[col])
            except ValueError:
                st.error(f"Invalid value for {col}.")
                return None

    # One-hot encode categorical variables (same as training)
    categorical_cols = ["JobRole", "MaritalStatus", "BusinessTravel", "Department"]
    df = pd.get_dummies(df, columns=categorical_cols)

    # Add missing dummy columns (if any) to match training data
    required_columns = pickle.load(open("employee_turnover_hr.pkl", "rb")).feature_names_in_
    for col in required_columns:
        if col not in df.columns:
            df[col] = 0  # Add missing column with 0

    # Ensure column order matches training
    df = df[required_columns]

    # Feature Engineering (same as training)
    df["YearlyIncome"] = monthly_income * 12
    df["Tenure_Age_Ratio"] = years_at_company / (age + 1)
    df["Income_Age_Ratio"] = monthly_income / (age + 1)
    df["AgeGroup"] = pd.cut([age], bins=[18,25,35,45,60,100], labels=[0,1,2,3,4]).astype(int)[0]
    
    return df

# ───────────────────── Predict Button ─────────────────────
if st.button("🔮 Predict Attrition Risk"):
    input_df = preprocess_input()
    if input_df is not None:
        try:
            proba = model.predict_proba(input_df)[0]
            prediction = model.predict(input_df)[0]
            risk = "High" if prediction == 1 else "Low"
            confidence = proba[1] if prediction == 1 else proba[0]

            # Display result
            st.success(f"Attrition Risk: **{risk}**")
            st.info(f"Confidence: **{confidence:.2%}**")
            
            # Show probabilities
            st.write(f"Probability of staying: {proba[0]:.2%}")
            st.write(f"Probability of leaving: {proba[1]:.2%}")
            
            # Visualize
            st.progress(float(proba[1]))
        except Exception as e:
            st.error(f"Prediction error: {e}")
    else:
        st.warning("Please check inputs and try again.")
