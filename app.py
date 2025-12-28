import os
import joblib
import streamlit as st
import pandas as pd

st.set_page_config(page_title="EPL Predictor", layout="centered")

st.title("⚽ EPL Score Prediction App")
st.write("App is loading...")

# --------------------------------------------------
# SAFE MODEL LOADING
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def safe_load(filename):
    path = os.path.join(BASE_DIR, filename)
    if not os.path.exists(path):
        st.error(f"❌ Missing file: {filename}")
        st.stop()
    return joblib.load(path)

try:
    reg_model = safe_load("player_performance_model.pkl")
    clf_model = safe_load("match_outcome_model.pkl")
    feature_columns = safe_load("feature_columns.pkl")
    st.success("✅ Models loaded successfully!")
except Exception as e:
    st.error("❌ Failed to load model files")
    st.exception(e)
    st.stop()

# --------------------------------------------------
# SIMPLE UI TEST
# --------------------------------------------------
st.header("Model Ready ✅")
st.write("If you see this, your models loaded correctly!")









