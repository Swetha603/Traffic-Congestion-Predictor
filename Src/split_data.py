import pandas as pd

from sklearn.model_selection import train_test_split


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
# Display results
# --------------------------------

print("Total records:", len(df))

print("Training records:", len(X_train))

print("Testing records:", len(X_test))

print("\nTraining data:")
print(X_train.head())

print("\nTesting data:")
print(X_test.head())

print("\nTraining target distribution:")
print(y_train.value_counts())

print("\nTesting target distribution:")
print(y_test.value_counts())