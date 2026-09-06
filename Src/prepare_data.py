import pandas as pd

# Load dataset
df = pd.read_csv("Data/traffic_data.csv")

# Convert Timestamp
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

# Target
y = df["Congestion_Level"]

# --------------------------------
# Display information
# --------------------------------

print("Features (X):")
print(X.head())

print("\nTarget (y):")
print(y.head())

print("\nX shape:")
print(X.shape)

print("\ny shape:")
print(y.shape)

print("\nTarget values:")
print(y.value_counts())