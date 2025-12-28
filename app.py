import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Player Performance Prediction", layout="centered")

st.title("⚽ Player Performance Prediction")

# Upload dataset
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("Dataset loaded successfully!")

    # Keep only numeric columns
    df_numeric = df.select_dtypes(include=["int64", "float64"])

    if df_numeric.shape[1] < 2:
        st.error("Dataset must contain at least two numeric columns.")
        st.stop()

    # Remove rows with missing values
    df_numeric = df_numeric.dropna()

    # Select target (last column)
    target_column = df_numeric.columns[-1]
    X = df_numeric.drop(columns=[target_column])
    y = df_numeric[target_column]

    # Train model
    model = LinearRegression()
    model.fit(X, y)

    st.subheader("Enter Player Data")

    user_input = {}
    for col in X.columns:
        user_input[col] = st.number_input(
            label=col,
            min_value=float(X[col].min()),
            max_value=float(X[col].max()),
            value=float(X[col].mean())
        )

    if st.button("Predict"):
        input_df = pd.DataFrame([user_input])
        prediction = model.predict(input_df)

        st.success(f"🎯 Predicted Value: {prediction[0]:.2f}")

    with st.expander("🔍 View cleaned dataset"):
        st.dataframe(df_numeric.head())

else:
    st.info("Please upload a CSV file to get started.")














