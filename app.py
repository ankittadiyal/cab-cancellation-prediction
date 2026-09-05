"""Streamlit web application for cab cancellation prediction.

This app loads the trained model, accepts user inputs, and predicts the chance
that a cab booking will be cancelled.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.predict import predict_cancellation

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "cab_bookings.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pkl"


def load_dataset() -> pd.DataFrame | None:
    """Load the dataset for the dashboard if it exists."""
    if not DATA_PATH.exists():
        return None
    return pd.read_csv(DATA_PATH)


def safe_prediction(raw_input: dict) -> tuple[str, float, float]:
    """Return prediction results, showing friendly errors if anything fails."""
    try:
        return predict_cancellation(raw_input)
    except FileNotFoundError as exc:
        st.error(f"Model file not found. Please train the model before running the app. Details: {exc}")
        raise
    except Exception as exc:  # pragma: no cover - UI safety net
        st.error(f"Prediction failed because of a problem with the input or saved model. Please check your values and try again. {exc}")
        raise


st.set_page_config(
    page_title="Cab Cancellation Prediction",
    page_icon="🚕",
    layout="wide",
)

st.title("🚕 Cab Cancellation Prediction")
st.caption("Predict the probability of a cab booking being cancelled using Machine Learning.")

with st.sidebar:
    st.header("About Project")
    st.write(
        "This project predicts whether a cab booking is likely to be cancelled using a synthetic, realistic dataset and a trained ML model."
    )

    st.header("Model Used")
    st.write("Best model selected from Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting.")

    st.header("Dataset Information")
    if DATA_PATH.exists():
        df = load_dataset()
        if df is not None:
            st.write(f"Total bookings: {len(df):,}")
            st.write(f"Cancellation rate: {df['cancellation'].mean():.2%}")
    else:
        st.write("Dataset not yet generated.")

    st.header("Evaluation Metrics")
    st.write("Accuracy, Precision, Recall, F1-score, and ROC-AUC are tracked for model comparison.")


if not DATA_PATH.exists():
    st.warning("The dataset is missing. Please generate it by running: python src/data_generation.py")

if not MODEL_PATH.exists():
    st.warning("The trained model is missing. Please train the model by running: python src/train.py")


# Dashboard summary
with st.container():
    df = load_dataset()
    if df is not None:
        total_bookings = len(df)
        cancelled = int(df["cancellation"].sum())
        successful = total_bookings - cancelled
        cancellation_rate = df["cancellation"].mean()

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total bookings", f"{total_bookings:,}")
        col2.metric("Cancellation rate", f"{cancellation_rate:.2%}")
        col3.metric("Cancelled", f"{cancelled:,}")
        col4.metric("Successful", f"{successful:,}")

        st.subheader("Dataset Overview")
        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            weather_cancellation = df.groupby("weather")["cancellation"].mean().sort_values(ascending=False)
            st.bar_chart(weather_cancellation)
        with chart_col2:
            traffic_cancellation = df.groupby("traffic_condition")["cancellation"].mean().sort_values(ascending=False)
            st.bar_chart(traffic_cancellation)


st.subheader("Booking Details")
with st.form("cab_prediction_form"):
    input_values = {}

    col1, col2 = st.columns(2)
    with col1:
        input_values["pickup_location"] = st.selectbox("Pickup Location", [
            "Downtown", "Airport", "Old Town", "Business District", "Residential", "Market", "University", "Harbor"
        ])
        input_values["drop_location"] = st.selectbox("Drop Location", [
            "Downtown", "Airport", "Old Town", "Business District", "Residential", "Market", "University", "Harbor"
        ])
        input_values["distance_km"] = st.number_input("Distance (km)", min_value=1.0, max_value=60.0, value=12.5, step=0.5)
        input_values["passenger_count"] = st.number_input("Passenger Count", min_value=1, max_value=5, value=2)
        input_values["vehicle_type"] = st.selectbox("Vehicle Type", ["Economy", "Premium", "SUV", "Auto"])
        input_values["booking_hour"] = st.slider("Booking Hour", 0, 23, 18)
        input_values["booking_day_of_week"] = st.slider("Day of Week", 0, 6, 5)
        input_values["is_weekend"] = int(st.checkbox("Weekend", value=True))

    with col2:
        input_values["driver_rating"] = st.slider("Driver Rating", 3.0, 5.0, 4.5, step=0.1)
        input_values["driver_experience_years"] = st.number_input("Driver Experience (years)", min_value=0.5, max_value=30.0, value=5.0, step=0.5)
        input_values["driver_acceptance_rate"] = st.slider("Driver Acceptance Rate (%)", 45, 99, 82)
        input_values["driver_previous_cancellations"] = st.number_input("Driver Previous Cancellations", min_value=0, max_value=10, value=1)

        input_values["customer_rating"] = st.slider("Customer Rating", 2.5, 5.0, 4.6, step=0.1)
        input_values["customer_previous_cancellations"] = st.number_input("Customer Previous Cancellations", min_value=0, max_value=10, value=1)
        input_values["customer_total_bookings"] = st.number_input("Customer Total Bookings", min_value=1, max_value=200, value=25)

    st.subheader("Trip Conditions")
    col3, col4 = st.columns(2)
    with col3:
        input_values["estimated_fare"] = st.number_input("Estimated Fare", min_value=50.0, max_value=1200.0, value=240.0, step=10.0)
        input_values["surge_multiplier"] = st.slider("Surge Multiplier", 1.0, 3.5, 1.4, step=0.1)
        input_values["estimated_pickup_time_minutes"] = st.number_input("Estimated Pickup Time (minutes)", min_value=5.0, max_value=120.0, value=20.0, step=1.0)
        input_values["weather"] = st.selectbox("Weather", ["Clear", "Cloudy", "Rainy", "Storm", "Snow"])
        input_values["traffic_condition"] = st.selectbox("Traffic Condition", ["Low", "Moderate", "Heavy", "Severe"])

    with col4:
        input_values["payment_method"] = st.selectbox("Payment Method", ["Cash", "Card", "Wallet", "UPI"])
        input_values["booking_source"] = st.selectbox("Booking Source", ["App", "Website", "Call Center", "Partner"])
        input_values["booking_day"] = st.number_input("Booking Day", min_value=1, max_value=31, value=15)

    submit_button = st.form_submit_button("Predict Cancellation")

if submit_button:
    if not MODEL_PATH.exists():
        st.error("The trained model file is missing. Please run the training script first.")
    else:
        try:
            prediction, cancellation_probability, success_probability = safe_prediction(input_values)
            st.subheader("Prediction Result")

            if cancellation_probability >= 50:
                st.error("⚠️ High Cancellation Risk")
            else:
                st.success("✅ Low Cancellation Risk")

            st.metric("Cancellation probability", f"{cancellation_probability:.1f}%")
            st.metric("Successful booking probability", f"{success_probability:.1f}%")
            st.progress(min(100.0, max(0.0, cancellation_probability / 100)))

            if cancellation_probability >= 50:
                st.warning("This booking has a fairly high chance of being cancelled. Consider checking travel conditions, surge pricing, and driver availability.")
            else:
                st.info("This booking appears likely to go through successfully. The estimated risk is relatively low.")

            st.write(f"Prediction: {prediction}")
        except Exception:
            st.stop()
