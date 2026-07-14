import streamlit as st
import joblib
import pandas as pd
from pathlib import Path

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(
    page_title="Titanic Survival Prediction",
    page_icon="🚢",
    layout="centered"
)

# ----------------------------
# Load Model
# ----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
model = joblib.load(BASE_DIR / "models" / "random_forest.pkl")

# ----------------------------
# Title
# ----------------------------
st.title("🚢 Titanic Survival Prediction")
st.markdown(
    "Predict whether a passenger would survive based on passenger details."
)

st.divider()

# ----------------------------
# Input Form
# ----------------------------
col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox("Passenger Class", [1, 2, 3])
    age = st.slider("Age", 0, 80, 25)
    fare = st.number_input("Fare", 0.0, 600.0, 32.0)

with col2:
    gender = st.selectbox("Gender", ["Male", "Female"])
    sibsp = st.number_input("Siblings / Spouses", 0, 10, 0)
    parch = st.number_input("Parents / Children", 0, 10, 0)

embarked = st.selectbox("Embarked", ["S", "C", "Q"])

# Encoding
sex = 0 if gender == "Male" else 1
embarked = {"S": 0, "C": 1, "Q": 2}[embarked]

st.divider()

if st.button("🔍 Predict Survival", use_container_width=True):

    input_data = pd.DataFrame(
        [[pclass, sex, age, sibsp, parch, fare, embarked]],
        columns=[
            "Pclass",
            "Sex",
            "Age",
            "SibSp",
            "Parch",
            "Fare",
            "Embarked"
        ]
    )

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]

    if prediction == 1:
        st.success("✅ Passenger is likely to Survive")
        st.metric("Confidence", f"{probability[1]*100:.2f}%")
    else:
        st.error("❌ Passenger is NOT likely to Survive")
        st.metric("Confidence", f"{probability[0]*100:.2f}%")

st.divider()

st.caption("Developed by Uday Verma | Machine Learning Project")