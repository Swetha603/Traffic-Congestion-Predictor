import pandas as pd

# Load dataset
df = pd.read_csv("Data/traffic_data.csv")

# Convert Timestamp to datetime
df["Timestamp"] = pd.to_datetime(df["Timestamp"])

# --------------------------------
# Create time-based features
# --------------------------------

df["Hour"] = df["Timestamp"].dt.hour

df["Day"] = df["Timestamp"].dt.day

df["Month"] = df["Timestamp"].dt.month

df["Day_of_Week"] = df["Timestamp"].dt.day_name()

# --------------------------------
# Create Weekend feature
# --------------------------------

df["Weekend"] = df["Timestamp"].dt.dayofweek >= 5

# --------------------------------
# Create Peak Hour feature
# --------------------------------

df["Peak_Hour"] = (
    ((df["Hour"] >= 7) & (df["Hour"] <= 9)) |
    ((df["Hour"] >= 17) & (df["Hour"] <= 20))
)

# Display the new data
print("Dataset with new features:")
print(df.head())

print("\nColumns:")
print(df.columns)

print("\nSample of time features:")
print(
    df[
        [
            "Timestamp",
            "Hour",
            "Day",
            "Month",
            "Day_of_Week",
            "Weekend",
            "Peak_Hour"
        ]
    ].head(10)
)