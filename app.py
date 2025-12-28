import streamlit as st
import pandas as pd
import joblib

# Load models and feature list
clf_model = joblib.load("classification_pipeline.pkl")
reg_model = joblib.load("regression_pipeline.pkl")
feature_columns = joblib.load("feature_columns.pkl")

st.title("⚽ Football Match Prediction")

menu = st.sidebar.selectbox(
    "Select Prediction Type",
    ["Match Outcome", "Player Performance"]
)

if menu == "Match Outcome":
    user_input = {}
    for col in feature_columns:
        user_input[col] = st.number_input(col, value=0.0)

    input_df = pd.DataFrame([user_input])
    input_df = input_df[feature_columns]  # 🔥 THIS LINE FIXES EVERYTHING

    prediction = clf_model.predict(input_df)
    label_map = {0: "Loss", 1: "Draw", 2: "Win"}
    st.success(f"Predicted Result: {label_map[prediction[0]]}")

elif menu == "Player Performance":
    user_input = {}
    for col in feature_columns:
        user_input[col] = st.number_input(col, value=0.0)

    input_df = pd.DataFrame([user_input])
    input_df = input_df[feature_columns]

    goals = reg_model.predict(input_df)
    st.success(f"Predicted Goals: {round(goals[0], 2)}")


