from pathlib import Path
import pandas as pd
BASE_DIR = Path(__file__).resolve().parent.parent
csv_path = BASE_DIR / "data" / "Titanic-Dataset.csv"
df = pd.read_csv(csv_path)

print("=" * 50)
print("Before Cleaning")
print("=" * 50)

print(df.info())
print("\nMissing Values:")
print(df.isnull().sum())

df.drop(columns=["Cabin"], inplace=True)

df["Age"].fillna(df["Age"].median(), inplace=True)

# ----------------------------
# Fill Embarked with Mode
# ----------------------------
df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)

# ----------------------------
# Convert Gender to Numbers
# ----------------------------
df["Sex"] = df["Sex"].map({
    "male": 0,
    "female": 1
})

# ----------------------------
# Convert Embarked
# ----------------------------
df["Embarked"] = df["Embarked"].map({
    "S": 0,
    "C": 1,
    "Q": 2
})

print("\n")
print("=" * 50)
print("After Cleaning")
print("=" * 50)

print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())
clean_path = BASE_DIR / "data" / "clean_titanic.csv"
df.to_csv(clean_path, index=False)

print("\nCleaned dataset saved as:")
print(clean_path)
