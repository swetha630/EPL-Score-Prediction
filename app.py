import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression

st.set_page_config(page_title="Football Prediction App", layout="centered")

st.title("⚽ Football Prediction System")

# Sidebar navigation
page = st.sidebar.selectbox(
    "Select Feature",
    ["Match Outcome Prediction", "Player Performance Prediction"]
)

# =========================
# SCREEN 1: MATCH OUTCOME
# =========================
if page == "Match Outcome Prediction":

    st.header("🏆 Match Outcome Prediction (Win / Draw / Loss)")

    uploaded_file = st.file_uploader("Upload Match Dataset (CSV)", type=["csv"], key="match")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        # Assume last column is match result (0=Loss,1=Draw,2=Win)
        target = df.columns[-1]
        X = df.drop(columns=[target])
        y = df[target]

        model = LogisticRegression(max_iter=1000)
        model.fit(X, y)

        st.subheader("Enter Match Statistics")

        user_input = {}
        for col in X.columns:
            user_input[col] = st.number_input(f"{col}", value=float(X[col].mean()))

        if st.button("Predict Match Result"):
            input_df = pd.DataFrame([user_input])
            prediction = model.predict(input_df)[0]

            result_map = {0: "Loss", 1: "Draw", 2: "Win"}
            st.success(f"🏟️ Predicted Match Result: **{result_map[prediction]}**")

# =========================
# SCREEN 2: PLAYER PERFORMANCE
# =========================
elif page == "Player Performance Prediction":

    st.header("⚽ Player Performance Prediction")

    uploaded_file = st.file_uploader("Upload Player Dataset (CSV)", type=["csv"], key="player")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        target = df.columns[-1]
        X = df.drop(columns=[target])
        y = df[target]

        model = LinearRegression()
        model.fit(X, y)

        st.subheader("Enter Player Statistics")

        user_input = {}
        for col in X.columns:
            user_input[col] = st.number_input(f"{col}", value=float(X[col].mean()))

        if st.button("Predict Player Performance"):
            input_df = pd.DataFrame([user_input])
            prediction = model.predict(input_df)
            st.success(f"⚽ Predicted Value: {prediction[0]:.2f}")


















