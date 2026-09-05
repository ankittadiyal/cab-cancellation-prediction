"""Train multiple ML models for cab cancellation prediction.

This script:
1. loads the synthetic cab booking dataset
2. splits data into train/test sets with stratification
3. trains several models inside Scikit-learn pipelines
4. compares their performance using multiple metrics
5. saves the best model and a model comparison table
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import TARGET_COLUMN, build_preprocessor, get_model_input_columns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "cab_bookings.csv"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"


def load_and_prepare_data() -> tuple[pd.DataFrame, pd.Series]:
    """Load the dataset and select feature/target columns."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at: {DATA_PATH}. Run 'python src/data_generation.py' first.")

    df = pd.read_csv(DATA_PATH)
    required_columns = get_model_input_columns() + [TARGET_COLUMN]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {missing}")

    X = df[get_model_input_columns()]
    y = df[TARGET_COLUMN].astype(int)
    return X, y


def build_model_candidates() -> dict[str, object]:
    """Create a dictionary of models to compare."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        ),
        "Decision Tree": DecisionTreeClassifier(
            random_state=42,
            class_weight="balanced",
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            random_state=42,
        ),
    }


def evaluate_model(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    """Evaluate a model using the metrics required by the project."""
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob),
    }


def train_models() -> tuple[pd.DataFrame, Pipeline, str]:
    """Train each model and return a comparison table, best pipeline, and best model name."""
    X, y = load_and_prepare_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    results = []
    best_pipeline = None
    best_name = ""

    for model_name, estimator in build_model_candidates().items():
        preprocessor = build_preprocessor()
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", estimator),
        ])

        pipeline.fit(X_train, y_train)
        metrics = evaluate_model(pipeline, X_test, y_test)
        metrics["model_name"] = model_name

        results.append(metrics)

        if best_pipeline is None:
            best_pipeline = pipeline
            best_name = model_name
            best_score = (metrics["f1_score"], metrics["roc_auc"])
        else:
            current_score = (metrics["f1_score"], metrics["roc_auc"])
            if current_score > best_score:
                best_pipeline = pipeline
                best_name = model_name
                best_score = current_score

    comparison_df = pd.DataFrame(results).sort_values(
        by=["f1_score", "roc_auc"],
        ascending=False,
    ).reset_index(drop=True)

    return comparison_df, best_pipeline, best_name


def save_artifacts(comparison_df: pd.DataFrame, best_pipeline: Pipeline, best_model_name: str) -> None:
    """Save model comparison results and trained pipeline artifacts."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    comparison_df.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)

    joblib.dump(best_pipeline, MODELS_DIR / "best_model.pkl")
    joblib.dump(best_pipeline.named_steps["preprocessor"], MODELS_DIR / "preprocessor.pkl")

    print(f"Saved comparison table to: {REPORTS_DIR / 'model_comparison.csv'}")
    print(f"Saved best model ({best_model_name}) to: {MODELS_DIR / 'best_model.pkl'}")
    print(f"Saved preprocessor to: {MODELS_DIR / 'preprocessor.pkl'}")


def main() -> None:
    """Train all models, select the best model, and save artifacts."""
    comparison_df, best_pipeline, best_name = train_models()
    if best_pipeline is None:
        raise RuntimeError("No model was trained successfully.")

    save_artifacts(comparison_df, best_pipeline, best_name)

    print("\nModel comparison results:")
    print(comparison_df[["model_name", "accuracy", "precision", "recall", "f1_score", "roc_auc"]].to_string(index=False))

    print(f"\nBest model selected: {best_name}")


if __name__ == "__main__":
    main()
