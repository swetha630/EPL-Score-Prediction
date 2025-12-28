import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression

st.set_page_config(page_title="Football Prediction App", layout="centered")

st.title("⚽ Football Prediction System")

# -------------------------------
# SIDEBAR NAVIGATION
# -------------------------------
page = st.sidebar.radio(
    "Select Prediction Type",
    ["Match Outcome Prediction", "Player Performance Prediction"]
)

# ------------------------------------------------
# HELPER FUNCTION
# ------------------------------------------------
def clean_dataset(df):
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna()
    return df


# =================================================
# MATCH OUTCOME PREDICTION (CLASSIFICATION)
# =================================================
if page == "Match Outcome Prediction":

    st.header("🏆 Match Outcome Prediction")

    uploaded_file = st.file_uploader("Upload Match Dataset (CSV)", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        df = clean_dataset(df)

        if df.shape[1] < 2:
            st.error("Dataset must contain at least two numeric columns.")
            st.stop()

        # Last column assumed to be score difference
        target_col = df.columns[-1]

        # Convert numeric target to categorical
        def classify(val):
            if val > 0:
                return 2   # Win
            elif val == 0:
                return 1   # Draw
            else:
                return 0   # Loss

        df["result"] = df[target_col].apply(classify)

        X = df.drop(columns=[target_col, "result"])
        y = df["result"]

        model = LogisticRegression(max_iter=1000)
        model.fit(X, y)

        st.subheader("Enter Match Statistics")

        user_input = {}
        for col in X.columns:
            user_input[col] = st.number_input(
                f"{col}",
                value=float(X[col].mean())
            )

        if st.button("Predict Match Outcome"):
            input_df = pd.DataFrame([user_input])
            pred = model.predict(input_df)[0]

            label_map = {0: "Loss", 1: "Draw", 2: "Win"}
            st.success(f"🏆 Predicted Result: **{label_map[pred]}**")

# =================================================
# PLAYER PERFORMANCE PREDICTION
# =================================================
elif page == "Player Performance Prediction":

    st.header("⚽ Player Performance Prediction")

    uploaded_file = st.file_uploader("Upload Player Dataset (CSV)", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        df = clean_dataset(df)

        if df.shape[1] < 2:
            st.error("Dataset must contain at least two numeric columns.")
            st.stop()

        target = df.columns[-1]
        X = df.drop(columns=[target])
        y = df[target]

        model = LinearRegression()
        model.fit(X, y)

        st.subheader("Enter Player Statistics")

        user_input = {}
        for col in X.columns:
            user_input[col] = st.number_input(
                f"{col}",
                value=float(X[col].mean())
            )

        if st.button("Predict Performance"):
            input_df = pd.DataFrame([user_input])
            prediction = model.predict(input_df)
            st.success(f"⭐ Predicted Value: {prediction[0]:.2f}")





















