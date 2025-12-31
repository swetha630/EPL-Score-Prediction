import streamlit as st
import pandas as pd
import joblib

# Load trained model and scaler
model = joblib.load("rf_regression_model.pkl")
scaler = joblib.load("scaler.pkl")

# Feature order (must match training)
FEATURES = ["Goals", "Shots", "Passes", "Appearances"]

# -------------------------------------------------
# APP CONFIG
# -------------------------------------------------
st.set_page_config(page_title="Football Predictor", layout="centered")
st.title("⚽ Football Prediction System")

# Sidebar menu
menu = st.sidebar.radio(
    "Select Option",
    ["Player Performance", "Match Outcome"]
)

# -------------------------------------------------
# SCREEN 1 — PLAYER PERFORMANCE
# -------------------------------------------------
if menu == "Player Performance":

    st.header("📊 Player Performance Prediction")

    inputs = []
    for feature in FEATURES:
        value = st.number_input(feature, value=0.0)
        inputs.append(value)

    if st.button("Predict Player Performance"):
        input_df = pd.DataFrame([inputs], columns=FEATURES)
        scaled = scaler.transform(input_df)
        prediction = model.predict(scaled)

        st.success(f"🎯 Predicted Goals: {round(prediction[0], 2)}")

# -------------------------------------------------
# SCREEN 2 — MATCH OUTCOME
# -------------------------------------------------
elif menu == "Match Outcome":

    st.header("🏟 Match Outcome Prediction")

    inputs = []
    for feature in FEATURES:
        value = st.number_input(feature, value=0.0)
        inputs.append(value)

    if st.button("Predict Match Outcome"):
        input_df = pd.DataFrame([inputs], columns=FEATURES)
        prediction = model.predict(input_df)

        # Simple interpretation
        if prediction[0] < 1:
            result = "Loss"
        elif prediction[0] < 2:
            result = "Draw"
        else:
            result = "Win"

        st.success(f"🏆 Predicted Match Outcome: {result}")





































