import joblib
import pandas as pd

model = joblib.load("rf_regression_model.pkl")
scaler = joblib.load("scaler.pkl")

sample = pd.DataFrame([[10, 50, 800, 30]],
                      columns=["Goals", "Shots", "Passes", "Appearances"])

sample_scaled = scaler.transform(sample)
prediction = model.predict(sample_scaled)

print("Predicted Goals:", prediction)
