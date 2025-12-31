import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("infosys_project(2).csv")


# ⚠️ CHOOSE FEATURES ONCE AND NEVER CHANGE
FEATURES = ["Goals", "Shots", "Passes", "Appearances"]

X = df[FEATURES]
y = df["Goals"]

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = RandomForestRegressor(random_state=42)
model.fit(X_scaled, y)

# Save everything
joblib.dump(model, "rf_regression_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(FEATURES, "feature_names.pkl")

print("Training complete. Models saved.")

