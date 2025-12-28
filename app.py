import streamlit as st
import pandas as pd
import joblib

# Load trained assets
reg_model = joblib.load("regression_pipeline.pkl")
clf_model = joblib.load("classification_pipeline.pkl")
feature_columns = joblib.load("feature_columns.pkl")

st.set_page_config(page_title="Football Prediction App", layout="centered")

st.title("⚽ Football Match Prediction")

menu = st.sidebar.selectbox(
    "Choose Prediction Type",
    ["Match Outcome Prediction", "Player Performance Prediction"]
)

# ---------------- MATCH OUTCOME ----------------
if menu == "Match Outcome Prediction":
    st.header("🏆 Match Outcome Prediction")

    user_input = {}
    for col in feature_columns:
        user_input[col] = st.number_input(col, value=0.0)

    if st.button("Predict Match Outcome"):
        input_df = pd.DataFrame([user_input])

        # CRITICAL LINE (DO NOT REMOVE)
        input_df = input_df[feature_columns]

        prediction = clf_model.predict(input_df)
        label_map = {0: "Loss", 1: "Draw", 2: "Win"}

        st.success(f"🏁 Predicted Outcome: {label_map[prediction[0]]}")

# ---------------- PLAYER PERFORMANCE ----------------
elif menu == "Player Performance Prediction":
    st.header("⚽ Player Performance Prediction")

    user_input = {}
    for col in feature_columns:
        user_input[col] = st.number_input(col, value=0.0)

    if st.button("Predict Goals"):
        input_df = pd.DataFrame([user_input])
        input_df = input_df[feature_columns]

        prediction = reg_model.predict(input_df)
        st.success(f"🎯 Predicted Goals: {round(prediction[0], 2)}")

