# src/train.py

import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)

# -----------------------------
# Paths
# -----------------------------
DATA_PATH = "data/heart.csv"
MODEL_PATH = "models/pipeline.pkl"

# -----------------------------
# Training Function
# -----------------------------
def train():
    # Load dataset
    df = pd.read_csv(DATA_PATH)

    # Handle invalid thal values identified during EDA
    df["thal"] = df["thal"].replace(0, pd.NA)
    df["thal"] = df["thal"].fillna(df["thal"].median())

    X = df.drop("target", axis=1)
    df["target"] = df["target"].apply(lambda x: 1 if x > 0 else 0)
    y = df["target"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # ML pipeline (Scaling + Model)
    pipeline = Pipeline([
        ("model", RandomForestClassifier(
            n_estimators=400,
            max_depth=10,
            min_samples_leaf=3,
            random_state=42,
            class_weight={0:1, 1:2}
        ))
    ])

    # Train
    pipeline.fit(X_train, y_train)

    # Evaluate
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    print("\n📊 Model Evaluation")
    print("Accuracy :", round(accuracy_score(y_test, y_pred), 4))
    print("Recall   :", round(recall_score(y_test, y_pred), 4))
    print("F1 Score :", round(f1_score(y_test, y_pred), 4))
    print("ROC-AUC  :", round(roc_auc_score(y_test, y_proba), 4))
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    # Ensure model directory exists
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    # Save pipeline
    joblib.dump(pipeline, MODEL_PATH)
    print(f"\n✅ Model pipeline saved at: {MODEL_PATH}")

# -----------------------------
# Entry Point
# -----------------------------
if __name__ == "__main__":
    train()
