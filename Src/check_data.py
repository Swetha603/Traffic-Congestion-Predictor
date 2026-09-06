import pandas as pd

# Load the traffic dataset
df = pd.read_csv("Data/traffic_data.csv")

# Remove duplicate rows
df = df.drop_duplicates()

# Convert Timestamp into date/time format
df["Timestamp"] = pd.to_datetime(df["Timestamp"])

# Check missing values
print("Missing values:")
print(df.isnull().sum())

# Fill missing numeric values with the median
numeric_columns = df.select_dtypes(include=["number"]).columns

for column in numeric_columns:
    df[column] = df[column].fillna(df[column].median())

# Fill missing text values with the most common value
text_columns = df.select_dtypes(include=["object"]).columns

for column in text_columns:
    df[column] = df[column].fillna(df[column].mode()[0])

# Display cleaned data
print("\nCleaned dataset:")
print(df.head())

print("\nDataset information:")
print(df.info())

print("\nFinal shape:")
print(df.shape)