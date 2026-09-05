"""Prediction utilities for the cab cancellation app.

This module loads the saved trained model and converts a dictionary of user
inputs into a DataFrame with the exact feature order expected by the model.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import FEATURE_COLUMNS, NUMERIC_FEATURES, TARGET_COLUMN

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pkl"


def load_model() -> Any:
    """Load the best trained pipeline from disk."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model file not found. Please train the model first with: python src/train.py"
        )
    return joblib.load(MODEL_PATH)


def prepare_input_data(raw_input: Dict[str, Any]) -> pd.DataFrame:
    """Convert a dictionary of inputs into a DataFrame matching training features."""
    clean_data = {}
    for feature in FEATURE_COLUMNS:
        if feature in raw_input:
            value = raw_input[feature]
            if feature in NUMERIC_FEATURES and value is not None:
                clean_data[feature] = float(value)
            else:
                clean_data[feature] = value
        else:
            clean_data[feature] = None

    df = pd.DataFrame([clean_data], columns=FEATURE_COLUMNS)
    return df


def predict_cancellation(raw_input: Dict[str, Any]) -> Tuple[str, float, float]:
    """Return the prediction label, cancellation probability, and success probability."""
    model = load_model()
    features = prepare_input_data(raw_input)
    probability = float(model.predict_proba(features)[0, 1])
    prediction = "Likely to be Cancelled" if probability >= 0.5 else "Likely to be Successful"
    cancellation_probability = probability * 100
    success_probability = (1 - probability) * 100
    return prediction, cancellation_probability, success_probability


if __name__ == "__main__":
    sample = {
        "pickup_location": "Downtown",
        "drop_location": "Airport",
        "distance_km": 12.5,
        "estimated_fare": 260,
        "booking_hour": 18,
        "booking_day": 7,
        "booking_day_of_week": 6,
        "passenger_count": 2,
        "vehicle_type": "SUV",
        "payment_method": "Card",
        "driver_rating": 4.4,
        "customer_rating": 4.6,
        "driver_experience_years": 7,
        "driver_acceptance_rate": 80,
        "customer_previous_cancellations": 2,
        "customer_total_bookings": 30,
        "driver_previous_cancellations": 1,
        "surge_multiplier": 1.4,
        "estimated_pickup_time_minutes": 18,
        "weather": "Rainy",
        "traffic_condition": "Heavy",
        "booking_source": "App",
        "is_weekend": 1,
    }

    prediction, cancellation_prob, success_prob = predict_cancellation(sample)
    print(f"Prediction: {prediction}")
    print(f"Cancellation probability: {cancellation_prob:.1f}%")
    print(f"Success probability: {success_prob:.1f}%")
