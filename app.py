import streamlit as st
import pandas as pd
import joblib

# Load trained assets
model = joblib.load("rf_regression_model.pkl")
scaler = joblib.load("scaler.pkl")
features = joblib.load("feature_names.pkl")

st.title("⚽ Football Performance Predictor")

inputs = []
for f in features:
    value = st.number_input(f, value=0.0)
    inputs.append(value)

if st.button("Predict"):
    df = pd.DataFrame([inputs], columns=features)
    scaled = scaler.transform(df)
    prediction = model.predict(scaled)
    st.success(f"🎯 Predicted Goals: {round(prediction[0], 2)}")



































