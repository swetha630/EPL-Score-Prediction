import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression

st.set_page_config(page_title="Football Prediction System", layout="centered")

st.title("⚽ Football Prediction System")

page = st.sidebar.selectbox(
    "Choose Prediction Type",
    ["Match Outcome Prediction", "Player Performance Prediction"]
)

# -----------------------------------------------------
# HELPER FUNCTION
# -----------------------------------------------------
def load_data(file):
    df = pd.read_csv(file)
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna()
    return df

# =====================================================
# MATCH OUTCOME PREDICTION
# =====================================================
if page == "Match Outcome Prediction":

    st.header("🏆 Match Outcome Prediction")
    st.info("Enter match statistics such as goals, shots, possession etc.")

    file = st.file_uploader("Upload Match Dataset", type=["csv"])

    if file:
        df = load_data(file)

        # Last column = match result (0 = Loss, 1 = Draw, 2 = Win)
        target = df.columns[-1]
        X = df.drop(columns=[target])
        y = df[target]

        model = LogisticRegression(max_iter=1000)
        model.fit(X, y)

        st.subheader("Enter Match Statistics")

        user_input = {}
        for col in X.columns:
            user_input[col] = st.text_input(
                f"Enter {col}",
                placeholder=f"Example: {round(X[col].mean(),2)}"
            )

        if st.button("Predict Match Outcome"):
            if "" in user_input.values():
                st.warning("⚠️ Please fill all input fields.")
            else:
                input_df = pd.DataFrame([{k: float(v) for k, v in user_input.items()}])
                prediction = model.predict(input_df)[0]

                label_map = {0: "Loss", 1: "Draw", 2: "Win"}
                st.success(f"🏆 Predicted Match Result: **{label_map[prediction]}**")

# =====================================================
# PLAYER PERFORMANCE PREDICTION
# =====================================================
elif page == "Player Performance Prediction":

    st.header("⚽ Player Performance Prediction")
    st.info("Enter player statistics to predict performance")

    file = st.file_uploader("Upload Player Dataset", type=["csv"])

    if file:
        df = load_data(file)

        target = df.columns[-1]
        X = df.drop(columns=[target])
        y = df[target]

        model = LinearRegression()
        model.fit(X, y)

        st.subheader("Enter Player Statistics")

        user_input = {}
        for col in X.columns:
            user_input[col] = st.text_input(
                f"Enter {col}",
                placeholder=f"Example: {round(X[col].mean(),2)}"
            )

        if st.button("Predict Performance"):
            if "" in user_input.values():
                st.warning("⚠️ Please fill all fields.")
            else:
                input_df = pd.DataFrame([{k: float(v) for k, v in user_input.items()}])
                prediction = model.predict(input_df)
                st.success(f"⭐ Predicted Performance Value: {prediction[0]:.2f}")




















