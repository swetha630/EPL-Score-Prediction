import streamlit as st
import pandas as pd
import joblib

# Load models
clf_model = joblib.load("classification_pipeline.pkl")
reg_model = joblib.load("regression_pipeline.pkl")
feature_columns = joblib.load("feature_columns.pkl")

st.title("⚽ Football Match Prediction")

menu = st.sidebar.selectbox(
    "Select Prediction Type",
    ["Match Outcome", "Player Performance"]
)

# ---------------- MATCH OUTCOME ----------------
if menu == "Match Outcome":
    st.header("🏆 Match Outcome Prediction")

    user_input = {}
    for col in feature_columns:
        user_input[col] = st.number_input(col, value=0)

    input_df = pd.DataFrame([user_input])
    input_df = input_df[feature_columns]

    if st.button("Predict Match Outcome"):
        prediction = clf_model.predict(input_df)
        label_map = {0: "Loss", 1: "Draw", 2: "Win"}
        st.success(f"🏁 Predicted Outcome: {label_map[prediction[0]]}")

# ---------------- PLAYER PERFORMANCE ----------------
elif menu == "Player Performance":
    st.header("⚽ Player Performance Prediction")

    user_input = {}
    for col in feature_columns:
        user_input[col] = st.number_input(col, value=0)

    input_df = pd.DataFrame([user_input])
    input_df = input_df[feature_columns]

    if st.button("Predict Goals"):
        goals = reg_model.predict(input_df)
        st.success(f"🎯 Predicted Goals: {round(goals[0], 2)}")



