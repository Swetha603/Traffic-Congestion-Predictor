import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


# --------------------------------
# Load dataset
# --------------------------------

df = pd.read_csv("Data/traffic_data.csv")

df["Timestamp"] = pd.to_datetime(df["Timestamp"])


# --------------------------------
# Feature Engineering
# --------------------------------

df["Hour"] = df["Timestamp"].dt.hour
df["Day"] = df["Timestamp"].dt.day
df["Month"] = df["Timestamp"].dt.month
df["Day_of_Week"] = df["Timestamp"].dt.dayofweek

df["Weekend"] = df["Timestamp"].dt.dayofweek >= 5

df["Peak_Hour"] = (
    ((df["Hour"] >= 7) & (df["Hour"] <= 9)) |
    ((df["Hour"] >= 17) & (df["Hour"] <= 20))
)


# --------------------------------
# Select Features
# --------------------------------

features = [
    "Vehicle_Count",
    "Vehicle_Speed",
    "Hour",
    "Day",
    "Month",
    "Day_of_Week",
    "Weekend",
    "Peak_Hour"
]

X = df[features]

y = df["Congestion_Level"]


# --------------------------------
# Create Model
# --------------------------------

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


# --------------------------------
# Train Model
# --------------------------------

model.fit(X, y)


# --------------------------------
# Save Model
# --------------------------------

joblib.dump(
    model,
    "models/traffic_congestion_model.pkl"
)

print("Model trained successfully!")
print("Model saved successfully!")
print("Location:")
print("models/traffic_congestion_model.pkl")