import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
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
# Train-Test Split
# --------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# --------------------------------
# Create Models
# --------------------------------

logistic_model = LogisticRegression(max_iter=1000)

decision_tree_model = DecisionTreeClassifier(
    random_state=42
)

random_forest_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


# --------------------------------
# Train Models
# --------------------------------

logistic_model.fit(X_train, y_train)

decision_tree_model.fit(X_train, y_train)

random_forest_model.fit(X_train, y_train)


# --------------------------------
# Make Predictions
# --------------------------------

logistic_predictions = logistic_model.predict(X_test)

decision_tree_predictions = decision_tree_model.predict(X_test)

random_forest_predictions = random_forest_model.predict(X_test)


# --------------------------------
# Display Predictions
# --------------------------------

print("Logistic Regression Predictions:")
print(logistic_predictions)

print("\nDecision Tree Predictions:")
print(decision_tree_predictions)

print("\nRandom Forest Predictions:")
print(random_forest_predictions)


print("\nActual Values:")
print(y_test.values)