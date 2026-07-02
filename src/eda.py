from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Project root
BASE_DIR = Path(__file__).resolve().parent.parent

# Dataset
csv_path = BASE_DIR / "data" / "Titanic-Dataset.csv"

# Images folder
images_path = BASE_DIR / "images"
images_path.mkdir(exist_ok=True)

# Load dataset
df = pd.read_csv(csv_path)

print("Dataset Shape:", df.shape)
print("\nMissing Values:")
print(df.isnull().sum())

# ----------------------------
# Survival Count
# ----------------------------
plt.figure(figsize=(6,4))
df["Survived"].value_counts().plot(kind="bar")
plt.title("Survival Count")
plt.xlabel("Survived")
plt.ylabel("Passengers")
plt.xticks([0,1],["No","Yes"])
plt.tight_layout()
plt.savefig(images_path/"survival_count.png")
plt.close()

# ----------------------------
# Gender Distribution
# ----------------------------
plt.figure(figsize=(6,4))
df["Sex"].value_counts().plot(kind="bar")
plt.title("Gender Distribution")
plt.tight_layout()
plt.savefig(images_path/"gender_distribution.png")
plt.close()

# ----------------------------
# Passenger Class
# ----------------------------
plt.figure(figsize=(6,4))
df["Pclass"].value_counts().sort_index().plot(kind="bar")
plt.title("Passenger Class")
plt.xlabel("Class")
plt.tight_layout()
plt.savefig(images_path/"passenger_class.png")
plt.close()

# ----------------------------
# Age Distribution
# ----------------------------
plt.figure(figsize=(6,4))
df["Age"].plot(kind="hist", bins=20)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.tight_layout()
plt.savefig(images_path/"age_distribution.png")
plt.close()

print("\nGraphs saved successfully!")