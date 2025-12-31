import streamlit as st
import joblib
import pandas as pd

# Load models
reg_model = joblib.load("rf_regression_model.pkl")
clf_model = joblib.load("gb_classification_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("⚽ Football Performance Prediction")

menu = st.sidebar.radio("Select Option", ["Player Performance", "Match Outcome"])

if menu == "Player Performance":
    st.header("Predict Player Goals")

    goals = st.number_input("Goals", 0)
    shots = st.number_input("Shots", 0)
    passes = st.number_input("Passes", 0)
    appearances = st.number_input("Appearances", 0)

    if st.button("Predict Goals"):
        data = pd.DataFrame([[goals, shots, passes, appearances]],
                            columns=["Goals", "Shots", "Passes", "Appearances"])
        scaled = scaler.transform(data)
        prediction = reg_model.predict(scaled)
        st.success(f"Predicted Goals: {round(prediction[0],2)}")

elif menu == "Match Outcome":
    st.header("Match Result Prediction")

    home = st.number_input("Home Team Strength")
    away = st.number_input("Away Team Strength")

    if st.button("Predict Result"):
        result = clf_model.predict([[home, away]])
        st.success(f"Predicted Outcome: {result[0]}")

























