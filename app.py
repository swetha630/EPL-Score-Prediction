import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ----------------------------------
# LOAD MODELS & FEATURE LISTS
# ----------------------------------
reg_model = joblib.load("player_performance_model.pkl")
clf_model = joblib.load("match_outcome_model.pkl")

reg_features = joblib.load("feature_columns.pkl")   # regression features
clf_features = joblib.load("feature_columns.pkl")   # classification features (same format)

st.title("⚽ Football Analytics & Prediction System")

menu = st.sidebar.selectbox(
    "Select Module",
    ["Player Performance Prediction", "Match Outcome Prediction"]
)

# ======================================================
# PLAYER PERFORMANCE PREDICTION
# ======================================================
if menu == "Player Performance Prediction":

    st.header("📊 Player Goal Prediction")

    age = st.number_input("Age", min_value=15, max_value=45, value=25)
    appearances = st.number_input("Appearances", min_value=0, max_value=60, value=25)
    shots = st.number_input("Shots", min_value=0, max_value=200, value=50)
    passes = st.number_input("Passes", min_value=0, max_value=3000, value=1200)

    if st.button("Predict Goals"):
        # Create empty dataframe with trained columns
        input_df = pd.DataFrame(columns=reg_features)
        input_df.loc[0] = 0

        # Fill known values
        input_map = {
            "Age": age,
            "Appearances": appearances,
            "Shots": shots,
            "Passes": passes
        }

        for col, val in input_map.items():
            if col in input_df.columns:
                input_df.at[0, col] = val

        prediction = reg_model.predict(input_df)

        st.success(f"⚽ Predicted Goals: **{round(prediction[0], 2)}**")

# ======================================================
# MATCH OUTCOME PREDICTION
# ======================================================
if menu == "Match Outcome Prediction":

    st.header("🏟 Match Outcome Prediction")

    home_form = st.slider("Home Team Form (0–5)", 0, 5, 3)
    away_form = st.slider("Away Team Form (0–5)", 0, 5, 3)
    home_goals = st.number_input("Home Avg Goals", 0.0, 5.0, 1.5)
    away_goals = st.number_input("Away Avg Goals", 0.0, 5.0, 1.2)

    if st.button("Predict Match Result"):
        input_df = pd.DataFrame(columns=clf_features)
        input_df.loc[0] = 0

        input_values = {
            "Home_Form": home_form,
            "Away_Form": away_form,
            "Home_Goals": home_goals,
            "Away_Goals": away_goals
        }

        for col, val in input_values.items():
            if col in input_df.columns:
                input_df.at[0, col] = val

        prediction = clf_model.predict(input_df)[0]

        outcome_map = {
            0: "❌ Loss",
            1: "➖ Draw",
            2: "✅ Win"
        }

        st.success(f"Predicted Match Outcome: **{outcome_map[prediction]}**")








