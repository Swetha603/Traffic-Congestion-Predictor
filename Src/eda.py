import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("Data/traffic_data.csv")

# Convert Timestamp
df["Timestamp"] = pd.to_datetime(df["Timestamp"])

# Create Hour column
df["Hour"] = df["Timestamp"].dt.hour

# -------------------------------
# 1. Basic information
# -------------------------------

print("Dataset shape:")
print(df.shape)

print("\nColumns:")
print(df.columns)

print("\nStatistics:")
print(df.describe())


# -------------------------------
# 2. Congestion Level Count
# -------------------------------

plt.figure(figsize=(8, 5))

sns.countplot(data=df, x="Congestion_Level")

plt.title("Traffic Congestion Levels")
plt.xlabel("Congestion Level")
plt.ylabel("Number of Records")

plt.show()


# -------------------------------
# 3. Vehicle Count Distribution
# -------------------------------

plt.figure(figsize=(8, 5))

sns.histplot(data=df, x="Vehicle_Count", bins=20)

plt.title("Vehicle Count Distribution")
plt.xlabel("Number of Vehicles")
plt.ylabel("Frequency")

plt.show()


# -------------------------------
# 4. Vehicle Count vs Speed
# -------------------------------

plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x="Vehicle_Count",
    y="Vehicle_Speed",
    hue="Congestion_Level"
)

plt.title("Vehicle Count vs Vehicle Speed")
plt.xlabel("Vehicle Count")
plt.ylabel("Vehicle Speed")

plt.show()


# -------------------------------
# 5. Traffic by Hour
# -------------------------------

hourly_traffic = df.groupby("Hour")["Vehicle_Count"].mean()

plt.figure(figsize=(10, 5))

hourly_traffic.plot()

plt.title("Average Vehicle Count by Hour")
plt.xlabel("Hour of Day")
plt.ylabel("Average Vehicle Count")

plt.grid()

plt.show()
print("\nAverage vehicle count by congestion level:")

print(
    df.groupby("Congestion_Level")["Vehicle_Count"]
    .mean()
)