import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

st.title("⚽ Player Performance Prediction")

uploaded_file = st.file_uploader("Upload cleaned dataset", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.success("Dataset loaded successfully!")

    # Select target (last column)
    target = df.columns[-1]
    X = df.drop(columns=[target])
    y = df[target]

    # Train model
    model = LinearRegression()
    model.fit(X, y)

    st.subheader("Enter Player Stats")

    user_input = {}
    for col in X.columns:
        user_input[col] = st.number_input(col, value=float(X[col].mean()))

    if st.button("Predict"):
        input_df = pd.DataFrame([user_input])
        prediction = model.predict(input_df)
        st.success(f"🎯 Predicted Value: {prediction[0]:.2f}")
















