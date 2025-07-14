import pickle
import pandas as pd
import streamlit as st

# ────────────────────────── Model Loader ──────────────────────────
st.set_page_config(page_title="Turnover Predictor", layout="centered")
st.title("🔍 Employee Turnover Prediction")

model = None
model_filename = "employee_turnover_optimized.pkl"

try:
    with open(model_filename, "rb") as f:
        model = pickle.load(f)
    st.success(f"Model loaded from '{model_filename}'!")
except Exception as e:
    st.error(f"Error loading model: {e}")

# ────────────────────────── Input UI ──────────────────────────
if model:
    st.header("Enter employee details")

    # Raw input fields
    age          = st.number_input("Age", 18, 70, 30)
    stag         = st.number_input("Tenure in months (stag)", 0, 600, 24)
    coach        = st.selectbox("Coach received?", ["yes", "no"])
    extraversion = st.slider("Extraversion", 0.0, 10.0, 5.0)
    independ     = st.slider("Independence", 0.0, 10.0, 5.0)
    selfcontrol  = st.slider("Self‑control", 0.0, 10.0, 5.0)
    anxiety      = st.slider("Anxiety", 0.0, 10.0, 5.0)
    novator      = st.slider("Novator", 0.0, 10.0, 5.0)

    profession   = st.selectbox("Profession", ["worker", "manager", "clerk", "technician"])
    greywage     = st.number_input("Gross Monthly Wage (greywage)", 1000, 100000, 30000)
    gender       = st.selectbox("Gender", ["male", "female"])
    industry     = st.selectbox("Industry", ["IT", "HR", "Finance", "Manufacturing"])
    way          = st.selectbox("Commute Method (way)", ["car", "bike", "walk", "public"])
    traffic      = st.selectbox("Traffic Level", ["low", "medium", "high"])
    head_gender  = st.selectbox("Supervisor Gender (head_gender)", ["male", "female"])

    # ───────────── Raw Feature DataFrame ─────────────
    input_df = pd.DataFrame({
        "age":          [age],
        "stag":         [stag],
        "coach":        [coach],
        "extraversion": [extraversion],
        "independ":     [independ],
        "selfcontrol":  [selfcontrol],
        "anxiety":      [anxiety],
        "novator":      [novator],
        "profession":   [profession],
        "greywage":     [greywage],
        "gender":       [gender],
        "industry":     [industry],
        "way":          [way],
        "traffic":      [traffic],
        "head_gender":  [head_gender],
    })

    # ───────────── Add required missing columns with default values ─────────────
    missing_columns = {
        "low_risk": 0,
        "high_risk": 0,
        "stress_level": 0.0,
        "coaching_impact": 0,
        "personality_score": 0.0,
        "adaptability": 0.0,
        "tenure_years": 0.0,
        "tenure_age_ratio": 0.0,
        "age_group": "26-35",  # use any category your model recognizes
    }
    for col, default_val in missing_columns.items():
        input_df[col] = default_val

    # ───────────── Prediction ─────────────
    if st.button("Predict Turnover"):
        try:
            # Ensure categorical columns are strings
            for col in input_df.select_dtypes(include=["object", "category"]).columns:
                input_df[col] = input_df[col].astype(str)

            prediction = model.predict(input_df)
            result = "leave" if prediction[0] == 1 else "stay"
            st.subheader("🔎 Prediction Result:")
            st.success(f"**Employee is likely to {result}.**")

        except Exception as e:
            st.error(f"⚠️ Prediction failed:\n{e}")
else:
    st.info("Please ensure the model file is available and properly loaded.")
