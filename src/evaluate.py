"""Evaluate the best trained cab cancellation model.

This script loads the saved model, re-creates the train/test split, generates
common classification evaluation plots, and saves them into the reports folder.
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_curve
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import TARGET_COLUMN, get_model_input_columns

matplotlib.use("Agg")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "cab_bookings.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pkl"
REPORTS_DIR = PROJECT_ROOT / "reports"


def load_model() -> object:
    """Load the saved best pipeline model."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Saved model not found: {MODEL_PATH}. Train the model first with 'python src/train.py'.")
    return joblib.load(MODEL_PATH)


def generate_confusion_matrix(model: object, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    """Save a confusion matrix plot."""
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_labels = ["Not Cancelled", "Cancelled"]
    plt.xticks([0, 1], tick_labels)
    plt.yticks([0, 1], tick_labels)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")

    plt.xlabel("Predicted label")
    plt.ylabel("Actual label")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "confusion_matrix.png", dpi=300)
    plt.close()


def save_classification_report(model: object, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    """Write a classification report to a text file."""
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=["Not Cancelled", "Cancelled"])
    with open(REPORTS_DIR / "classification_report.txt", "w", encoding="utf-8") as file:
        file.write(report)
    print(report)


def save_roc_curve(model: object, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    """Save the ROC curve plot."""
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label="ROC curve")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Baseline")
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "roc_curve.png", dpi=300)
    plt.close()


def save_feature_importance(model: object) -> None:
    """Save feature importance for tree-based models or coefficient magnitude for linear models."""
    pipeline = model
    preprocessor = pipeline.named_steps["preprocessor"]
    model_step = pipeline.named_steps["model"]
    feature_names = preprocessor.get_feature_names_out()

    if hasattr(model_step, "feature_importances_"):
        importances = model_step.feature_importances_
        importance_label = "Feature Importance"
    elif hasattr(model_step, "coef_"):
        importances = np.abs(model_step.coef_[0])
        importance_label = "Absolute Coefficient Magnitude"
    else:
        print("This model does not provide a native feature importance metric. A coefficient or tree importance is unavailable.")
        return

    importances_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=False).head(20)

    plt.figure(figsize=(10, 6))
    plt.barh(importances_df["feature"][::-1], importances_df["importance"][::-1])
    plt.title(f"Top 20 Features by {importance_label}")
    plt.xlabel(importance_label)
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "feature_importance.png", dpi=300)
    plt.close()


def main() -> None:
    """Run evaluation generation for the saved best model."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at: {DATA_PATH}. Run 'python src/data_generation.py' first.")

    df = pd.read_csv(DATA_PATH)
    X = df[get_model_input_columns()]
    y = df[TARGET_COLUMN].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = load_model()
    generate_confusion_matrix(model, X_test, y_test)
    save_classification_report(model, X_test, y_test)
    save_roc_curve(model, X_test, y_test)
    save_feature_importance(model)

    print(f"Saved confusion matrix to: {REPORTS_DIR / 'confusion_matrix.png'}")
    print(f"Saved roc curve to: {REPORTS_DIR / 'roc_curve.png'}")
    print(f"Saved feature importance to: {REPORTS_DIR / 'feature_importance.png'}")
    print(f"Saved classification report to: {REPORTS_DIR / 'classification_report.txt'}")


if __name__ == "__main__":
    main()
