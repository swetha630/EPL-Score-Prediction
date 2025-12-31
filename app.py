import streamlit as st
import joblib
import pandas as pd

# Load models
model = joblib.load("rf_regression_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("⚽ Football Performance Predictor")

goals = st.number_input("Goals", 0)
shots = st.number_input("Shots", 0)
passes = st.number_input("Passes", 0)
appearances = st.number_input("Appearances", 0)

if st.button("Predict"):
    data = pd.DataFrame([[goals, shots, passes, appearances]],
                        columns=["Goals", "Shots", "Passes", "Appearances"])
    data_scaled = scaler.transform(data)
    prediction = model.predict(data_scaled)
    st.success(f"Predicted Goals: {prediction[0]:.2f}")


























