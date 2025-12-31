import streamlit as st
import pandas as pd
import joblib

# Load trained artifacts
model = joblib.load("rf_regression_model.pkl")
scaler = joblib.load("scaler.pkl")
features = joblib.load("feature_names.pkl")

st.title("⚽ Football Performance Predictor")

# User inputs
goals = st.number_input("Goals", 0, 50)
shots = st.number_input("Shots", 0, 200)
passes = st.number_input("Passes", 0, 3000)
appearances = st.number_input("Appearances", 0, 50)

if st.button("Predict"):
    input_df = pd.DataFrame(
        [[goals, shots, passes, appearances]],
        columns=features
    )

    scaled = scaler.transform(input_df)
    prediction = model.predict(scaled)

    st.success(f"🎯 Predicted Goals: {round(prediction[0], 2)}")

































