import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

st.title("⚽ Player Performance Prediction")

uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.success("Dataset loaded successfully!")

    # Separate features & target
    target_column = df.columns[-1]
    feature_columns = df.columns[:-1]

    X = df[feature_columns]
    y = df[target_column]

    # Train model
    model = LinearRegression()
    model.fit(X, y)

    st.subheader("Enter Player Data")

    user_input = {}
    for col in feature_columns:
        user_input[col] = st.number_input(
            col,
            float(X[col].min()),
            float(X[col].max()),
            float(X[col].mean())
        )

    if st.button("Predict Goals"):
        input_df = pd.DataFrame([user_input])
        prediction = model.predict(input_df)
        st.success(f"🎯 Predicted Goals: {prediction[0]:.2f}")

else:
    st.warning("Please upload a CSV file to continue.")












