import streamlit as st
import pandas as pd
import joblib

# -------------------------------
# Load trained models & metadata
# -------------------------------
reg_model = joblib.load("rf_regression_model.pkl")
clf_model = joblib.load("gb_classification_model.pkl")
scaler = joblib.load("scaler.pkl")

# IMPORTANT: load correct feature order
feature_names = reg_model.feature_names_in_

# -------------------------------
# App UI
# -------------------------------
st.set_page_config(page_title="Football Prediction App", layout="centered")
st.title("⚽ Football Performance Prediction")

menu = st.sidebar.radio(
    "Select Prediction Type",
    ["Player Performance", "Match Outcome"]
)

# =====================================================
# PLAYER PERFORMANCE
# =====================================================
if menu == "Player Performance":
    st.header("📊 Player Performance Prediction")

    goals = st.number_input("Goals", 0, 50, 5)
    shots = st.number_input("Shots", 0, 200, 40)
    passes = st.number_input("Passes", 0, 3000, 800)
    appearances = st.number_input("Appearances", 0, 50, 20)

    if st.button("Predict Player Performance"):
        input_df = pd.DataFrame(
            [[goals, shots, passes, appearances]],
            columns=feature_names
        )

        scaled = scaler.transform(input_df)
        prediction = reg_model.predict(scaled)

        st.success(f"🎯 Predicted Goals: {round(prediction[0], 2)}")

# =====================================================
# MATCH OUTCOME
# =====================================================
elif menu == "Match Outcome":
    st.header("🏟 Match Outcome Prediction")

    goals = st.number_input("Goals", 0, 50, 5)
    shots = st.number_input("Shots", 0, 200, 40)
    passes = st.number_input("Passes", 0, 3000, 800)
    appearances = st.number_input("Appearances", 0, 50, 20)

    if st.button("Predict Match Outcome"):
        input_df = pd.DataFrame(
            [[goals, shots, passes, appearances]],
            columns=feature_names
        )

        prediction = clf_model.predict(input_df)
        label_map = {0: "Loss", 1: "Draw", 2: "Win"}

        st.success(f"🏆 Predicted Result: {label_map[int(prediction[0])]}")































