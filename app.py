import os
import joblib
import streamlit as st
import pandas as pd

# -------------------------------------------------
# APP CONFIG
# -------------------------------------------------
st.set_page_config(page_title="EPL Predictor", layout="centered")
st.title("⚽ EPL Match & Player Prediction")

# -------------------------------------------------
# SAFE MODEL LOADING
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def safe_load(filename):
    path = os.path.join(BASE_DIR, filename)
    if not os.path.exists(path):
        st.error(f"Missing file: {filename}")
        st.stop()
    return joblib.load(path)

reg_model = safe_load("player_performance_model.pkl")
clf_model = safe_load("match_outcome_model.pkl")

st.success("✅ Models loaded successfully")

# -------------------------------------------------
# PLAYER PERFORMANCE PREDICTION
# -------------------------------------------------
st.header("🎯 Player Performance Prediction")

with st.form("player_form"):
    shots = st.number_input("Shots", 0, 200, 50)
    passes = st.number_input("Passes", 0, 3000, 1200)
    age = st.number_input("Age", 16, 45, 25)
    appearances = st.number_input("Appearances", 0, 60, 25)

    submit_player = st.form_submit_button("Predict Goals")

if submit_player:
    player_input = pd.DataFrame([{
        "Shots": shots,
        "Passes": passes,
        "Age": age,
        "Appearances": appearances
    }])

    prediction = reg_model.predict(player_input)
    st.success(f"⚽ Predicted Goals: **{round(prediction[0], 2)}**")

# -------------------------------------------------
# MATCH OUTCOME PREDICTION
# -------------------------------------------------
st.header("🏟 Match Outcome Prediction")

with st.form("match_form"):
    home_form = st.slider("Home Team Form", 0, 5, 3)
    away_form = st.slider("Away Team Form", 0, 5, 3)
    home_goals = st.number_input("Home Goals Avg", 0.0, 5.0, 1.5)
    away_goals = st.number_input("Away Goals Avg", 0.0, 5.0, 1.2)

    submit_match = st.form_submit_button("Predict Match Outcome")

if submit_match:
    match_input = pd.DataFrame([{
        "Home_Form": home_form,
        "Away_Form": away_form,
        "Home_Goals": home_goals,
        "Away_Goals": away_goals
    }])

    result = clf_model.predict(match_input)[0]
    label_map = {0: "❌ Loss", 1: "➖ Draw", 2: "✅ Win"}

    st.success(f"🏆 Predicted Result: **{label_map[result]}**")























