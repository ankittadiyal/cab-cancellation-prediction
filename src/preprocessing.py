"""Preprocessing utilities for the cab cancellation project.

This module builds a reusable Scikit-learn preprocessing pipeline that handles:
- missing numerical values via SimpleImputer
- scaling numerical features with StandardScaler
- missing categorical values via SimpleImputer
- one-hot encoding for categorical variables

The pipeline is designed to avoid data leakage by fitting only on the training
set and then reusing the fitted pipeline for validation and prediction.
"""

from __future__ import annotations

from typing import List

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET_COLUMN = "cancellation"

NUMERIC_FEATURES = [
    "distance_km",
    "estimated_fare",
    "booking_hour",
    "booking_day",
    "booking_day_of_week",
    "passenger_count",
    "driver_rating",
    "customer_rating",
    "driver_experience_years",
    "driver_acceptance_rate",
    "customer_previous_cancellations",
    "customer_total_bookings",
    "driver_previous_cancellations",
    "surge_multiplier",
    "estimated_pickup_time_minutes",
    "is_weekend",
]

CATEGORICAL_FEATURES = [
    "pickup_location",
    "drop_location",
    "vehicle_type",
    "payment_method",
    "weather",
    "traffic_condition",
    "booking_source",
]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def build_preprocessor() -> ColumnTransformer:
    """Create a column transformer with separate pipelines for numeric and categorical features."""
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    return preprocessor


def get_model_input_columns() -> List[str]:
    """Return the feature columns used for model training and prediction."""
    return FEATURE_COLUMNS


if __name__ == "__main__":
    preprocessor = build_preprocessor()
    print("Preprocessor created successfully.")
    print(preprocessor)
