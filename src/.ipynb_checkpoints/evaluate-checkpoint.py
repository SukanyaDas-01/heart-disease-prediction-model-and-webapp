# src/evaluate.py

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


def evaluate_model(pipeline, X_test, y_test, model_name="Model"):
    """
    Evaluates a trained pipeline on test data.

    Returns a dictionary of evaluation metrics.
    """

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    return {
        "Model": model_name,
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "Precision": round(
            precision_score(y_test, y_pred, zero_division=0), 4
        ),
        "Recall": round(recall_score(y_test, y_pred), 4),
        "F1 Score": round(f1_score(y_test, y_pred), 4),
        "ROC-AUC": round(roc_auc_score(y_test, y_proba), 4)
    }


def save_results(results: list, path="reports/model_comparison.xlsx"):
    """
    Saves evaluation results to an Excel file.
    """

    df = pd.DataFrame(results)
    df.to_excel(path, index=False)
    print(f"📊 Evaluation results saved to {path}")
