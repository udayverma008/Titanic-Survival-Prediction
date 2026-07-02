from pathlib import Path
import pandas as pd

# Get the project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Build the dataset path
csv_file = BASE_DIR / "data" / "Titanic-Dataset.csv"

print("Dataset Path:", csv_file)

# Load the dataset
df = pd.read_csv(csv_file)

# Display first 5 rows
print("\nFirst 5 Rows:")
print(df.head())