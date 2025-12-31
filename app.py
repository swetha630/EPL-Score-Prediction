import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Load trained models & assets
# -----------------------------
reg_model = joblib.load("rf_regression_model.pkl")
clf_model = joblib.load("gb_classification_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")  # IMPORTANT

# -----------------------------
# App UI
# -----------------------------
st.set_page_config(page_title="Football Prediction App", layout="centered")
st.title("⚽ Football Performance Predictor")

menu = st.sidebar.radio(
    "Select Prediction Type",
    ["Player Performance", "Match Outcome"]
)

# =====================================================
# PLAYER PERFORMANCE PREDICTION
# =====================================================
if menu == "Player Performance":
    st.subheader("📊 Player Performance Prediction")

    inputs = []
    for feature in feature_names:
        value = st.number_input(f"{feature}", value=0.0)
        inputs.append(value)

    if st.button("Predict Goals"):
        input_df = pd.DataFrame([inputs], columns=feature_names)
        scaled_input = scaler.transform(input_df)
        prediction = reg_model.predict(scaled_input)

        st.success(f"🎯 Predicted Goals: {round(prediction[0], 2)}")

# =====================================================
# MATCH OUTCOME PREDICTION
# =====================================================
elif menu == "Match Outcome":
    st.subheader("🏟 Match Outcome Prediction")

    inputs = []
    for feature in feature_names:
        value = st.number_input(f"{feature}", value=0.0)
        inputs.append(value)

    if st.button("Predict Match Outcome"):
        input_df = pd.DataFrame([inputs], columns=feature_names)
        prediction = clf_model.predict(input_df)

        label_map = {0: "Loss", 1: "Draw", 2: "Win"}
        st.success(f"🏆 Predicted Result: {label_map[int(prediction[0])]}")



































