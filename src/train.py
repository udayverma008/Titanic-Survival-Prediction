import joblib
from pathlib import Path
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# ----------------------------
# Load Dataset
# ----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
df = pd.read_csv(BASE_DIR / "data" / "clean_titanic.csv")

# Remove unnecessary columns
df = df.drop(["PassengerId", "Name", "Ticket"], axis=1)

# Features and Target
X = df.drop("Survived", axis=1)
y = df["Survived"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "KNN": KNeighborsClassifier(),
    "SVM": SVC()
}

print("=" * 50)
print("Model Comparison")
print("=" * 50)

results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    accuracy = accuracy_score(y_test, prediction)

    results.append((name, accuracy))
    print(f"{name:<25} {accuracy:.4f}")

best_model = max(results, key=lambda x: x[1])

print("\nBest Model:")
print(f"{best_model[0]} ({best_model[1]:.4f})")
# ----------------------------
# Save Random Forest Model
# ----------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

best_model = RandomForestClassifier(random_state=42)
best_model.fit(X_train, y_train)

joblib.dump(best_model, BASE_DIR / "models" / "random_forest.pkl")

print("\nModel saved successfully!")