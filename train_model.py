import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

# Load data
df = pd.read_csv("dataset.csv")

# Feature selection
features = ["Goals", "Shots", "Passes", "Appearances"]
X = df[features]
y_reg = df["Goals"]

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_reg, test_size=0.2, random_state=42
)

# Train model
rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)

# Save model & scaler
joblib.dump(rf, "rf_regression_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Model saved successfully!")
