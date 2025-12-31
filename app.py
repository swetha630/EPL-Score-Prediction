import streamlit as st
import pandas as pd
import joblib

# Load models
reg_model = joblib.load("rf_regression_model.pkl")
clf_model = joblib.load("gb_classification_model.pkl")
scaler = joblib.load("scaler.pkl")

# Get feature names used during training
FEATURES = reg_model.feature_names_in_.tolist()

st.title("⚽ Football Performance Predictor")

menu = st.sidebar.selectbox(
    "Select Prediction Type",
    ["Player Performance", "Match Outcome"]
)

# ---------------- PLAYER PERFORMANCE ----------------
if menu == "Player Performance":
    st.header("Player Performance Prediction")

    inputs = []
    for f in FEATURES:
        val = st.number_input(f, value=0.0)
        inputs.append(val)

    if st.button("Predict Goals"):
        df = pd.DataFrame([inputs], columns=FEATURES)
        scaled = scaler.transform(df)
        prediction = reg_model.predict(scaled)
        st.success(f"🎯 Predicted Goals: {round(prediction[0], 2)}")

# ---------------- MATCH OUTCOME ----------------
else:
    st.header("Match Outcome Prediction")

    inputs = []
    for f in FEATURES:
        val = st.number_input(f, value=0.0)
        inputs.append(val)

    if st.button("Predict Outcome"):
        df = pd.DataFrame([inputs], columns=FEATURES)
        prediction = clf_model.predict(df)
        label_map = {0: "Loss", 1: "Draw", 2: "Win"}
        st.success(f"🏆 Predicted Result: {label_map[int(prediction[0])]}")





























