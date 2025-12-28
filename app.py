import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# ------------------------------
# Load Dataset
# ------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("player_data.csv")  # change filename if needed
    return df

df = load_data()

# ------------------------------
# Prepare Data
# ------------------------------
target_column = df.columns[-1]  # last column as target
feature_columns = df.columns[:-1]

X = df[feature_columns]
y = df[target_column]

# ------------------------------
# Train Model
# ------------------------------
model = LinearRegression()
model.fit(X, y)

# ------------------------------
# UI
# ------------------------------
st.title("⚽ Player Performance Prediction")

st.markdown("Enter player statistics to predict performance:")

user_input = {}

for col in feature_columns:
    user_input[col] = st.number_input(
        label=col,
        min_value=float(X[col].min()),
        max_value=float(X[col].max()),
        value=float(X[col].mean())
    )

# Convert input to array
input_data = np.array([list(user_input.values())])

# ------------------------------
# Prediction
# ------------------------------
if st.button("Predict Goals"):
    prediction = model.predict(input_data)
    st.success(f"🎯 Predicted Goals: **{prediction[0]:.2f}**")











