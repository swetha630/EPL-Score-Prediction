import streamlit as st
import pandas as pd
import joblib

# Load models
reg_model = joblib.load("rf_regression_model.pkl")
clf_model = joblib.load("gb_classification_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Football Predictor", layout="centered")

st.title("⚽ Football Performance Prediction App")

# Sidebar navigation
page = st.sidebar.radio(
    "Select Prediction Type",
    ["Match Outcome Prediction", "Player Performance Prediction"]
)

# =====================================================
# 🟦 SCREEN 1 — MATCH OUTCOME PREDICTION
# =====================================================
if page == "Match Outcome Prediction":
    st.header("🏟 Match Outcome Prediction")

    st.write("Predict match result (Win / Draw / Loss)")

    home_form = st.slider("Home Team Form (1–10)", 1, 10, 5)
    away_form = st.slider("Away Team Form (1–10)", 1, 10, 5)
    home_goals = st.number_input("Average Home Goals", 0.0, 5.0, 1.5)
    away_goals = st.number_input("Average Away Goals", 0.0, 5.0, 1.2)

    if st.button("Predict Match Result"):
        X = pd.DataFrame([[home_form, away_form, home_goals, away_goals]],
                         columns=["Home_Form", "Away_Form", "Home_Goals", "Away_Goals"])

        prediction = clf_model.predict(X)[0]

        outcome_map = {0: "Loss", 1: "Draw", 2: "Win"}
        st.success(f"🏆 Predicted Match Result: **{outcome_map[prediction]}**")

# =====================================================
# 🟩 SCREEN 2 — PLAYER PERFORMANCE PREDICTION
# =====================================================
elif page == "Player Performance Prediction":
    st.header("⚽ Player Performance Prediction")

    goals = st.number_input("Goals", 0, 50, 5)
    shots = st.number_input("Shots", 0, 200, 40)
    passes = st.number_input("Passes", 0, 3000, 800)
    appearances = st.number_input("Appearances", 0, 50, 20)

    if st.button("Predict Performance"):
        data = pd.DataFrame(
            [[goals, shots, passes, appearances]],
            columns=["Goals", "Shots", "Passes", "Appearances"]
        )

        scaled = scaler.transform(data)
        prediction = reg_model.predict(scaled)

        st.success(f"⭐ Predicted Goals: {round(prediction[0], 2)}")


























