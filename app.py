import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression

st.set_page_config(page_title="Football Prediction System", layout="centered")

st.title("⚽ Football Prediction System")

# Sidebar navigation
page = st.sidebar.radio("Choose Prediction Type", 
                        ["Match Outcome Prediction", "Player Performance Prediction"])

# ======================================
# COMMON FUNCTION
# ======================================
def load_and_prepare_data(file):
    df = pd.read_csv(file)

    # Keep only numeric columns
    df = df.select_dtypes(include=["int64", "float64"])

    # Remove missing values
    df = df.dropna()

    return df

# ======================================
# SCREEN 1 – MATCH OUTCOME (CLASSIFICATION)
# ======================================
if page == "Match Outcome Prediction":

    st.header("🏆 Match Outcome Prediction (Win / Draw / Loss)")

    file = st.file_uploader("Upload Match Dataset", type=["csv"])

    if file:
        df = load_and_prepare_data(file)

        if df.shape[1] < 2:
            st.error("Dataset must contain at least 2 numeric columns.")
            st.stop()

        target = df.columns[-1]
        X = df.drop(columns=[target])
        y = df[target]

        # Check classification validity
        if y.nunique() < 2:
            st.error("Target must have at least two classes (e.g., Win/Draw/Loss).")
            st.stop()

        model = LogisticRegression(max_iter=1000)
        model.fit(X, y)

        st.subheader("Enter Match Statistics")

        user_input = {}
        for col in X.columns:
            user_input[col] = st.number_input(col, value=float(X[col].mean()))

        if st.button("Predict Match Result"):
            input_df = pd.DataFrame([user_input])
            prediction = model.predict(input_df)[0]

            st.success(f"🏆 Predicted Match Outcome: **{prediction}**")

# ======================================
# SCREEN 2 – PLAYER PERFORMANCE (REGRESSION)
# ======================================
elif page == "Player Performance Prediction":

    st.header("⚽ Player Performance Prediction")

    file = st.file_uploader("Upload Player Dataset", type=["csv"])

    if file:
        df = load_and_prepare_data(file)

        if df.shape[1] < 2:
            st.error("Dataset must contain at least 2 numeric columns.")
            st.stop()

        target = df.columns[-1]
        X = df.drop(columns=[target])
        y = df[target]

        model = LinearRegression()
        model.fit(X, y)

        st.subheader("Enter Player Statistics")

        user_input = {}
        for col in X.columns:
            user_input[col] = st.number_input(col, value=float(X[col].mean()))

        if st.button("Predict Performance"):
            input_df = pd.DataFrame([user_input])
            prediction = model.predict(input_df)
            st.success(f"⚽ Predicted Value: {prediction[0]:.2f}")



















