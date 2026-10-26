import pandas as pd

file_path = "data/fertilizer_recommendation.csv"

df = pd.read_csv(file_path)

print("\n--- FIRST 5 ROWS ---")
print(df.head())

print("\n--- COLUMN NAMES ---")
print(df.columns.tolist())

print("\n--- DATASET SIZE ---")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

print("\n--- MISSING VALUES ---")
print(df.isnull().sum())

print("\n--- DATA TYPES ---")
print(df.dtypes)