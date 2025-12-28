import streamlit as st
import pandas as pd
import joblib

# Load models
clf_model = joblib.load("classification_pipeline.pkl")
reg_model = joblib.load("regression_pipeline.pkl")

# Get feature names directly from model
feature_columns = clf_model.feature_names_in_

st.title("⚽ Football Match Prediction")

menu = st.sidebar.selectbox(
    "Select Prediction Type",
    ["Match Outcome", "Player Performance"]
)

# ---------- MATCH OUTCOME ----------
if menu == "Match Outcome":
    st.header("🏆 Match Outcome Prediction")

    user_input = {}
    for col in feature_columns:
        user_input[col] = st.number_input(col, value=0)

    input_df = pd.DataFrame([user_input])

    # IMPORTANT FIX
    prediction = clf_model.predict(input_df.values)

    label_map = {0: "Loss", 1: "Draw", 2: "Win"}
    st.success(f"🏁 Predicted Outcome: {label_map[prediction[0]]}")

# ---------- PLAYER PERFORMANCE ----------
elif menu == "Player Performance":
    st.header("⚽ Player Performance Prediction")

    user_input = {}
    for col in feature_columns:
        user_input[col] = st.number_input(col, value=0)

    input_df = pd.DataFrame([user_input])

    goals = reg_model.predict(input_df.values)
    st.success(f"🎯 Predicted Goals: {round(goals[0], 2)}")






