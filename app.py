import os
import joblib
import streamlit as st
import pandas as pd

st.set_page_config(page_title="EPL Predictor", layout="centered")

st.title("⚽ EPL Score Prediction App")
st.write("App is loading...")

# --------------------------------------------------
# SAFE MODEL LOADING
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def safe_load(filename):
    path = os.path.join(BASE_DIR, filename)
    if not os.path.exists(path):
        st.error(f"❌ Missing file: {filename}")
        st.stop()
    return joblib.load(path)

try:
    reg_model = safe_load("player_performance_model.pkl")
    clf_model = safe_load("match_outcome_model.pkl")
    model_features = safe_load("feature_columns.pkl")
    st.success("✅ Models loaded successfully!")
except Exception as e:
    st.error("❌ Failed to load model files")
    st.exception(e)
    st.stop()

# --------------------------------------------------
# SIMPLE UI TEST
# --------------------------------------------------
st.header("Model Ready ✅")
st.write("If you see this, your models loaded correctly!")
# ===============================
# PLAYER PERFORMANCE PREDICTION
# ===============================
st.subheader("⚽ Player Performance Prediction")

with st.form("player_form"):
    age = st.number_input("Age", 16, 45, 25)
    appearances = st.number_input("Appearances", 0, 60, 20)
    shots = st.number_input("Shots", 0, 200, 50)
    passes = st.number_input("Passes", 0, 3000, 800)

    submit_player = st.form_submit_button("Predict Goals")

if submit_player:
    # Create input row with all features
    input_df = pd.DataFrame(columns=model_features)
    input_df.loc[0] = 0

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

    st.success(f"🎯 Predicted Goals: **{round(prediction[0], 2)}**")

# ===============================
# MATCH OUTCOME PREDICTION
# ===============================

st.markdown("---")
st.subheader("🏟 Match Outcome Prediction")

with st.form("match_form"):
    home_form = st.slider("Home Team Form", 0, 5, 3)
    away_form = st.slider("Away Team Form", 0, 5, 3)
    home_goals = st.number_input("Home Avg Goals", 0.0, 5.0, 1.2)
    away_goals = st.number_input("Away Avg Goals", 0.0, 5.0, 1.0)

    submit_match = st.form_submit_button("Predict Match Outcome")

if submit_match:
    match_df = pd.DataFrame(columns=feature_columns)
    match_df.loc[0] = 0

    values = {
        "Home_Form": home_form,
        "Away_Form": away_form,
        "Home_Goals": home_goals,
        "Away_Goals": away_goals
    }

    for col, val in values.items():
        if col in match_df.columns:
            match_df.at[0, col] = val

    result = clf_model.predict(match_df)[0]

    label_map = {0: "❌ Loss", 1: "➖ Draw", 2: "✅ Win"}
    st.success(f"Predicted Result: **{label_map[result]}**")










