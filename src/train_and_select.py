import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, recall_score

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# -----------------------------
# Paths
# -----------------------------
DATA_PATH = "data/heart.csv"
MODEL_PATH = "models/pipeline.pkl"
REPORT_PATH = "reports/model_comparison.xlsx"

# -----------------------------
# Load Data
# -----------------------------
df = pd.read_csv(DATA_PATH)

df["thal"] = df["thal"].replace(0, pd.NA)
df["thal"] = df["thal"].fillna(df["thal"].median())

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# -----------------------------
# Candidate Models
# -----------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Naive Bayes": GaussianNB(),
    "SVM": SVC(probability=True),
    "KNN": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        class_weight="balanced",
        random_state=42
    )
}

results = []
best_auc = 0
best_pipeline = None
best_model_name = ""

# -----------------------------
# Train + Evaluate
# -----------------------------
for name, model in models.items():
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_proba)
    recall = recall_score(y_test, y_pred)

    results.append({
        "Model": name,
        "ROC-AUC": round(auc, 4),
        "Recall": round(recall, 4)
    })

    # Select best model
    if auc > best_auc:
        best_auc = auc
        best_pipeline = pipeline
        best_model_name = name

# -----------------------------
# Save Results
# -----------------------------
os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)

joblib.dump(best_pipeline, MODEL_PATH)
pd.DataFrame(results).to_excel(REPORT_PATH, index=False)

print("✅ Best Model Selected:", best_model_name)
print("📈 Best ROC-AUC:", round(best_auc, 4))
print("💾 Saved pipeline to:", MODEL_PATH)
