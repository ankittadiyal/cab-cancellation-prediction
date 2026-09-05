"""Generate a realistic synthetic cab-booking dataset.

This script creates a synthetic dataset for a cab cancellation prediction model.
The data is not completely random. Instead, it uses controlled rules so that
features such as long pickup times, bad weather, traffic, and low driver
acceptance rates increase the chance of a cancellation.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DATA_PATH = DATA_DIR / "cab_bookings.csv"


def generate_synthetic_cab_data(num_rows: int = 18000, random_seed: int = 42) -> pd.DataFrame:
    """Create a cab-booking dataset with realistic relationships.

    The cancellation probability is influenced by a risk score that depends on:
    - high estimated pickup time
    - heavy traffic
    - bad weather
    - higher surge multiplier
    - low driver acceptance rate
    - low driver rating
    - customer cancellation history
    - longer trip distances
    - certain late-night booking hours
    """
    rng = np.random.default_rng(random_seed)

    vehicle_types = ["Economy", "Premium", "SUV", "Auto"]
    weather_types = ["Clear", "Cloudy", "Rainy", "Storm", "Snow"]
    traffic_levels = ["Low", "Moderate", "Heavy", "Severe"]
    payment_methods = ["Cash", "Card", "Wallet", "UPI"]
    booking_sources = ["App", "Website", "Call Center", "Partner"]
    pickup_zones = ["Downtown", "Airport", "Old Town", "Business District", "Residential", "Market", "University", "Harbor"]

    rows = []

    for i in range(num_rows):
        booking_datetime = pd.Timestamp("2024-01-01") + pd.to_timedelta(rng.integers(0, 365 * 3), unit="D")
        booking_datetime += pd.to_timedelta(rng.integers(0, 24), unit="h")
        booking_datetime += pd.to_timedelta(rng.integers(0, 60), unit="min")

        booking_hour = booking_datetime.hour
        booking_day = booking_datetime.day
        booking_day_of_week = booking_datetime.dayofweek
        is_weekend = int(booking_day_of_week >= 5)

        vehicle_type = rng.choice(vehicle_types, p=[0.45, 0.25, 0.20, 0.10])
        weather = rng.choice(weather_types, p=[0.35, 0.25, 0.23, 0.12, 0.05])
        traffic_condition = rng.choice(traffic_levels, p=[0.50, 0.25, 0.18, 0.07])
        payment_method = rng.choice(payment_methods, p=[0.30, 0.25, 0.25, 0.20])
        booking_source = rng.choice(booking_sources, p=[0.55, 0.20, 0.15, 0.10])

        pickup_location = rng.choice(pickup_zones)
        drop_location = rng.choice(pickup_zones)

        distance_km = max(1.0, float(rng.normal(8.5, 4.5)))
        estimated_fare = max(120.0, distance_km * 22 + rng.normal(0, 25))

        passenger_count = int(min(5, max(1, rng.integers(1, 5))))

        driver_rating = float(np.clip(rng.normal(4.5, 0.45), 3.0, 5.0))
        customer_rating = float(np.clip(rng.normal(4.4, 0.5), 2.5, 5.0))
        driver_experience_years = max(0.5, float(rng.normal(5.5, 3.2)))

        driver_acceptance_rate = float(np.clip(rng.normal(83, 15), 45, 99))
        customer_previous_cancellations = int(np.clip(rng.poisson(1.4), 0, 10))
        customer_total_bookings = int(max(5, rng.poisson(22) + 12))
        driver_previous_cancellations = int(np.clip(rng.poisson(1.0), 0, 8))

        # Surge multiplier usually rises in bad weather or peak hours
        surge_multiplier = 1.0
        if weather in ["Rainy", "Storm", "Snow"]:
            surge_multiplier += rng.uniform(0.2, 0.9)
        if booking_hour in [8, 9, 17, 18, 19]:
            surge_multiplier += rng.uniform(0.1, 0.4)
        if traffic_condition in ["Heavy", "Severe"]:
            surge_multiplier += rng.uniform(0.15, 0.5)
        surge_multiplier = float(np.clip(surge_multiplier, 1.0, 3.5))

        traffic_weight = {"Low": 0, "Moderate": 1, "Heavy": 2, "Severe": 3}[traffic_condition]
        weather_weight = {"Clear": 0, "Cloudy": 1, "Rainy": 2, "Storm": 3, "Snow": 3}[weather]

        # Estimated pickup time increases with distance, traffic, and poor weather.
        base_pickup_time = 5 + distance_km * 2.5 + traffic_weight * 8 + weather_weight * 4
        estimated_pickup_time_minutes = max(5.0, float(base_pickup_time + rng.normal(0, 8)))

        # Risk score influences cancellation probability.
        # Higher scores mean greater cancellation likelihood.
        # The coefficients are tuned to create a realistic overall rate of roughly
        # 20-35% rather than making cancellations nearly inevitable.
        risk_score = (
            -6.4
            + 0.06 * estimated_pickup_time_minutes
            + 0.55 * traffic_weight
            + 0.52 * weather_weight
            + 0.9 * max(0, surge_multiplier - 1.0)
            + 0.05 * (100 - driver_acceptance_rate)
            + 0.85 * max(0, 5.0 - driver_rating)
            + 0.24 * customer_previous_cancellations
            + 0.05 * distance_km
            + 0.30 * is_weekend
            + (0.7 if 0 <= booking_hour <= 5 else 0)
            + (0.45 if 20 <= booking_hour <= 23 else 0)
            + (0.35 if vehicle_type == "Auto" else 0)
            + (0.20 if payment_method == "Cash" else 0)
        )

        # Convert the score to a probability between 0 and 1.
        cancellation_probability = 1 / (1 + math.exp(-risk_score))
        cancellation = int(rng.random() < cancellation_probability)

        booking = {
            "booking_id": f"CAB_{i:05d}",
            "booking_datetime": booking_datetime,
            "pickup_location": pickup_location,
            "drop_location": drop_location,
            "distance_km": round(distance_km, 2),
            "estimated_fare": round(estimated_fare, 2),
            "booking_hour": booking_hour,
            "booking_day": booking_day,
            "booking_day_of_week": booking_day_of_week,
            "passenger_count": passenger_count,
            "vehicle_type": vehicle_type,
            "payment_method": payment_method,
            "driver_rating": round(driver_rating, 2),
            "customer_rating": round(customer_rating, 2),
            "driver_experience_years": round(driver_experience_years, 2),
            "driver_acceptance_rate": round(driver_acceptance_rate, 2),
            "customer_previous_cancellations": customer_previous_cancellations,
            "customer_total_bookings": customer_total_bookings,
            "driver_previous_cancellations": driver_previous_cancellations,
            "surge_multiplier": round(surge_multiplier, 2),
            "estimated_pickup_time_minutes": round(estimated_pickup_time_minutes, 2),
            "weather": weather,
            "traffic_condition": traffic_condition,
            "booking_source": booking_source,
            "is_weekend": is_weekend,
            "cancellation": cancellation,
        }
        rows.append(booking)

    df = pd.DataFrame(rows)
    return df


def main() -> None:
    """Generate and save the synthetic cab-booking dataset."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = generate_synthetic_cab_data(num_rows=18000, random_seed=42)
    df.to_csv(DATA_PATH, index=False)

    print(f"Dataset saved successfully to: {DATA_PATH}")
    print(f"Rows: {len(df)}")
    print(f"Cancellation rate: {df['cancellation'].mean():.2%}")


if __name__ == "__main__":
    main()
