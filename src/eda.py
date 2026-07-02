from pathlib import Path
import pandas as pd

# Dataset Path
BASE_DIR = Path(__file__).resolve().parent.parent
csv_file = BASE_DIR / "data" / "Titanic-Dataset.csv"

# Load Datasetgit add .
df = pd.read_csv(csv_file)

print("="*50)
print("Dataset Shape")
print("="*50)
print(df.shape)

print("\n")

print("="*50)
print("Columns")
print("="*50)
print(df.columns)

print("\n")

print("="*50)
print("Dataset Information")
print("="*50)
print(df.info())

print("\n")

print("="*50)
print("Missing Values")
print("="*50)
print(df.isnull().sum())

print("\n")

print("="*50)
print("Summary Statistics")
print("="*50)
print(df.describe())

print("\n")

print("="*50)
print("Unique Values")
print("="*50)

for column in df.columns:
    print(f"{column}: {df[column].nunique()} unique values")