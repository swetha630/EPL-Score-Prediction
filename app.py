import os
import joblib
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_file(filename):
    path = os.path.join(BASE_DIR, filename)
    if not os.path.exists(path):
        st.error(f"Missing file: {filename}")
        st.stop()
    return joblib.load(path)

# Load models safely
reg_model = load_file("player_performance_model.pkl")
clf_model = load_file("match_outcome_model.pkl")
feature_columns = load_file("feature_columns.pkl")








