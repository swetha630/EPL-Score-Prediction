import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression

st.set_page_config(page_title="Football Prediction System", layout="centered")

st.title("⚽ Football Prediction System")

page = st.sidebar.selectbox(
    "Choose Prediction Type",
    ["Match Outcome Prediction", "Player Performance Prediction"]
)

# -----------------------------------------
# Helper function
# -----------------------------------------
def load_clean_data(file):
    df = pd.read_csv(file)
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna()
    return df

# ========================================
# MATCH OUTCOME PREDICTION
# ========================================
if page == "Match Outcome Prediction":

    st.header("🏆 Match Outcome Prediction")
    st.write("Enter match statistics manually to predict the result.")

    file = st.file_uploader("Upload match dataset (CSV)", type=["csv"])

    if file:
        df = load_clean_data(file)

        # Assume last column is goal difference
        target_col = df.columns[-1]

        # Convert to classification
        def classify(x):
            if x > 0:
                return 2   # Win
            elif x == 0:
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
            value = st.text_input(f"{col}", placeholder="Enter numeric value")
            if value != "":
                user_input[col] = float(value)

        if st.button("Predict Match Outcome"):
            if len(user_input) != len(X.columns):
                st.warning("⚠️ Please fill all fields.")
            else:
                input_df = pd.DataFrame([user_input])
                pred = model.predict(input_df)[0]
                label_map = {0: "Loss", 1: "Draw", 2: "Win"}
                st.success(f"🏆 Predicted Result: **{label_map[pred]}**")

# ========================================
# PLAYER PERFORMANCE PREDICTION
# ========================================
elif page == "Player Performance Prediction":

    st.header("⚽ Player Performance Prediction")
    st.write("Enter player statistics to predict performance.")

    file = st.file_uploader("Upload player dataset", type=["csv"])

    if file:
        df = load_clean_data(file)

        target = df.columns[-1]
        X = df.drop(columns=[target])
        y = df[target]

        model = LinearRegression()
        model.fit(X, y)

        st.subheader("Enter Player Stats")

        user_input = {}
        for col in X.columns:
            value = st.text_input(f"{col}", placeholder="Enter numeric value")
            if value != "":
                user_input[col] = float(value)

        if st.button("Predict Performance"):
            if len(user_input) != len(X.columns):
                st.warning("⚠️ Please fill all fields.")
            else:
                input_df = pd.DataFrame([user_input])
                prediction = model.predict(input_df)
                st.success(f"⭐ Predicted Performance: {prediction[0]:.2f}")






















