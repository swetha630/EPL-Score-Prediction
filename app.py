import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

st.title("⚽ Player Performance Prediction")

uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.success("Dataset loaded successfully!")

    # Keep only numeric columns
    numeric_df = df.select_dtypes(include=["int64", "float64"])

    if numeric_df.shape[1] < 2:
        st.error("Dataset must contain at least one feature and one target column.")
        st.stop()

    target_column = numeric_df.columns[-1]
    X = numeric_df.drop(columns=[target_column])
    y = numeric_df[target_column]

    model = LinearRegression()
    model.fit(X, y)

    st.subheader("Enter Player Stats")

    user_input = {}
    for col in X.columns:
        user_input[col] = st.number_input(
            col,
            float(X[col].min()),
            float(X[col].max()),
            float(X[col].mean())
        )

    if st.button("Predict"):
        input_df = pd.DataFrame([user_input])
        prediction = model.predict(input_df)
        st.success(f"🎯 Predicted Value: {prediction[0]:.2f}")













