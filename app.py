import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Player Performance Prediction", layout="centered")

st.title("⚽ Player Performance Prediction")

# Upload cleaned dataset
uploaded_file = st.file_uploader("Upload cleaned dataset (CSV)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.success("Dataset loaded successfully!")

    # Select target column (last column)
    target_col = df.columns[-1]
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Train model
    model = LinearRegression()
    model.fit(X, y)

    st.subheader("Enter Player Statistics")

    user_input = {}

    for col in X.columns:
        value = st.text_input(f"{col}", placeholder=f"Enter {col}")
        if value != "":
            try:
                user_input[col] = float(value)
            except ValueError:
                st.error(f"Invalid value for {col}")

    if st.button("Predict"):
        if len(user_input) != len(X.columns):
            st.warning("⚠️ Please fill all fields before predicting.")
        else:
            input_df = pd.DataFrame([user_input])
            prediction = model.predict(input_df)
            st.success(f"🎯 Predicted Value: {prediction[0]:.2f}")

else:
    st.info("Please upload a cleaned CSV file to continue.")

















