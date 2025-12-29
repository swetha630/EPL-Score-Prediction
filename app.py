import os
import joblib
import streamlit as st
import pandas as pd

# -------------------------------
# App Setup
# -------------------------------
st.set_page_config(page_title="EPL Predictor", layout="centered")
st.title("⚽ EPL Match & Player Performance Predictor")

# -------------------------------
# Load Models
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_model(file):
    path = os.path.join(BASE_DIR, file)
    if not os.path.exists(path):
        st.error(f"Missing file: {file}")
        st.stop()
    return joblib.load(path)

reg_model = load_model("player_performance_model.pkl")
clf_model = load_model("match_outcome_model.pkl")

FEATURES = ["Goals", "Shots", "Passes", "Appearances"]

# -------------------------------
# USER INPUT
# -------------------------------
st.header("Enter Player & Match Information")

col1, col2 = st.columns(2)

with col1:
    goals = st.number_input("Goals", 0, 50, 10)
    shots = st.number_input("Shots", 0, 200, 40)

with col2:
    passes = st.number_input("Passes", 0, 3000, 800)
    appearances = st.number_input("Appearances", 0, 60, 20)

input_df = pd.DataFrame([[goals, shots, passes, appearances]], columns=FEATURES)

# -------------------------------
# PLAYER PERFORMANCE PREDICTION
# -------------------------------
if st.button("Predict Player Performance"):
    pred_goals = reg_model.predict(input_df)
    st.success(f"⚽ Predicted Goals: **{round(pred_goals[0], 2)}**")

# -------------------------------
# MATCH OUTCOME PREDICTION
# -------------------------------
if st.button("Predict Match Outcome"):
    result = clf_model.predict(input_df)[0]

    label_map = {
        0: "❌ Loss",
        1: "➖ Draw",
        2: "✅ Win"
    }

    st.success(f"🏆 Predicted Match Result: **{label_map[result]}**")
























