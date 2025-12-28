import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression

st.set_page_config(page_title="Football Prediction System", layout="centered")

st.title("⚽ Football Prediction System")

# ==========================
# SIDEBAR NAVIGATION
# ==========================
page = st.sidebar.radio(
    "Select Prediction Type",
    ["Match Outcome Prediction", "Player Performance Prediction"]
)

# ======================================
# FUNCTION: LOAD & CLEAN DATA
# ======================================
def load_and_clean_data(file):
    df = pd.read_csv(file)

    # Convert all columns to numeric
    df = df.apply(pd.to_numeric, errors="coerce")

    # Drop rows with missing values
    df = df.dropna()

    return df

# ======================================
# MATCH OUTCOME PREDICTION
# ======================================
if page == "Match Outcome Prediction":

    st.header("🏆 Match Outcome Prediction")

    file = st.file_uploader("Upload Match Dataset", type=["csv"], key="match")

    if file:
        df = load_and_clean_data(file)

        if df.shape[1] < 2:
            st.error("Dataset must have at least 2 numeric columns.")
            st.stop()

        # Assume last column = goal difference or match result
        target_col = df.columns[-1]

        # Convert continuous values to classes
        def classify_result(x):
            if x > 0:
                return 2   # Win
            elif x == 0:
                return 1   # Draw
            else:
                return 0   # Loss

        df["result"] = df[target_col].apply(classify_result)

        X = df.drop(columns=[target_col, "result"])
        y = df["result"]

        model = LogisticRegression(max_iter=1000)
        model.fit(X, y)

        st.subheader("Enter Match Statistics")

        user_input = {}
        for col in X.columns:
            user_input[col] = st.number_input(col, value=float(X[col].mean()))

        if st.button("Predict Match Outcome"):
            input_df = pd.DataFrame([user_input])
            prediction = model.predict(input_df)[0]

            result_map = {0: "Loss", 1: "Draw", 2: "Win"}
            st.success(f"🏆 Match Result: **{result_map[prediction]}**")

# ======================================
# PLAYER PERFORMANCE PREDICTION
# ======================================
elif page == "Player Performance Prediction":

    st.header("⚽ Player Performance Prediction")

    file = st.file_uploader("Upload Player Dataset", type=["csv"], key="player")

    if file:
        df = load_and_clean_data(file)

        if df.shape[1] < 2:
            st.error("Dataset must contain at least 2 numeric columns.")
            st.stop()

        target = df.columns[-1]
        X = df.drop(columns=[target])
        y = df[target]

        model = LinearRegression()
        model.fit(X, y)

        st.subheader("Enter Player Stats")

        user_input = {}
        for col in X.columns:
            user_input[col] = st.number_input(col, value=float(X[col].mean()))

        if st.button("Predict Performance"):
            input_df = pd.DataFrame([user_input])
            prediction = model.predict(input_df)
            st.success(f"⭐ Predicted Value: {prediction[0]:.2f}")



















