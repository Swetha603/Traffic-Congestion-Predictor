import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


# --------------------------------
# Load dataset
# --------------------------------

df = pd.read_csv("Data/traffic_data.csv")


# --------------------------------
# Convert Timestamp
# --------------------------------

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
# Split data
# --------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# --------------------------------
# Train Random Forest
# --------------------------------

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)


# --------------------------------
# Create new traffic data
# --------------------------------

new_traffic = pd.DataFrame({
    "Vehicle_Count": [150],
    "Vehicle_Speed": [30],
    "Hour": [18],
    "Day": [5],
    "Month": [9],
    "Day_of_Week": [5],
    "Weekend": [False],
    "Peak_Hour": [True]
})


# --------------------------------
# Make prediction
# --------------------------------

prediction = model.predict(new_traffic)


# --------------------------------
# Display result
# --------------------------------

print("Traffic Information:")
print(new_traffic)

print("\nPredicted Congestion Level:")
print(prediction[0])