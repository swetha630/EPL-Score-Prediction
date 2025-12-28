import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Player Performance Prediction")

st.title("⚽ Player Performance Prediction")

uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("Dataset loaded successfully!")

    # Convert all columns to numeric (force errors to NaN)
    df = df.apply(pd.to_numeric, errors="coerce")

    # Remove infinite values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Drop rows with missing values
    df = df.dropna()

    # Check if enough data exists
    if df.shape[0] < 2 or df.shape[1] < 2:
        st.error("Dataset does not contain enough clean numeric data.")
        st.stop()

    # Split features and target
    target_column = df.columns[-1]
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Train model
    model = LinearRegression()
    model.fit(X, y)

    st.subheader("Enter Input Values")

    user_input = {}
    for col in X.columns:
        user_input[col] = st.number_input(
            col,
            value=float(X[col].mean())
        )

    if st.button("Predict"):
        input_df = pd.DataFrame([user_input])
        prediction = model.predict(input_df)
        st.success(f"🎯 Predicted Value: {prediction[0]:.2f}")

    with st.expander("📊 Cleaned Dataset Preview"):
        st.dataframe(df.head())

else:
    st.info("Please upload a CSV file to begin.")















