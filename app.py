import streamlit as st
import pandas as pd
import joblib

# Load trained pipelines
reg_model = joblib.load("regression_pipeline.pkl")
clf_model = joblib.load("classification_pipeline.pkl")

st.title("⚽ Football Match Prediction")

menu = st.sidebar.selectbox(
    "Select Prediction Type",
    ["Match Outcome", "Player Performance"]
)

# ---------------- MATCH OUTCOME ----------------
if menu == "Match Outcome":
    st.header("🏆 Match Outcome Prediction")

    # Input fields
    user_input = {
        "Age": st.number_input("Age", 0),
        "Appearances": st.number_input("Appearances", 0),
        "Goals": st.number_input("Goals", 0),
        "Assists": st.number_input("Assists", 0),
        "Shots": st.number_input("Shots", 0),
        "Passes": st.number_input("Passes", 0),
        "Tackles": st.number_input("Tackles", 0),
        "Interceptions": st.number_input("Interceptions", 0),
    }

    input_df = pd.DataFrame([user_input])

    if st.button("Predict Match Outcome"):
        prediction = clf_model.predict(input_df)
        label_map = {0: "Loss", 1: "Draw", 2: "Win"}
        st.success(f"🏁 Predicted Outcome: {label_map[prediction[0]]}")

# ---------------- PLAYER PERFORMANCE ----------------
elif menu == "Player Performance":
    st.header("⚽ Player Performance Prediction")

    user_input = {
        "Age": st.number_input("Age", 0),
        "Appearances": st.number_input("Appearances", 0),
        "Goals": st.number_input("Goals", 0),
        "Assists": st.number_input("Assists", 0),
        "Shots": st.number_input("Shots", 0),
        "Passes": st.number_input("Passes", 0),
        "Tackles": st.number_input("Tackles", 0),
        "Interceptions": st.number_input("Interceptions", 0),
    }

    input_df = pd.DataFrame([user_input])

    prediction = reg_model.predict(input_df)
    st.success(f"🎯 Predicted Goals: {round(prediction[0], 2)}")







