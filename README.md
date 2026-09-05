# Cab Cancellation Prediction

## Overview
This project builds a beginner-friendly machine learning app that predicts whether a cab booking is likely to be cancelled. It uses a synthetic but realistic cab-booking dataset, performs data exploration, trains multiple classic ML models, compares them, and saves the best one for deployment in a Streamlit application.

## Problem Statement
Cab cancellations can hurt driver availability, reduce customer satisfaction, and affect business operations. Being able to estimate cancellation risk helps ride-hailing platforms reduce wasted trips, plan driver allocation, and improve customer experience.

## Why this is useful
Businesses can use a cancellation prediction model to:
- identify bookings at risk of being cancelled
- adjust pricing or driver assignment in advance
- reduce operational inefficiencies
- improve customer support and service quality

## Machine Learning Approach
This project uses traditional machine learning rather than deep learning. The workflow is:
1. generate realistic synthetic data
2. explore the dataset and understand patterns
3. clean and preprocess data
4. train several models
5. compare metrics
6. select the best model based on F1-score and ROC-AUC
7. save the pipeline for prediction
8. deploy it through a user-friendly Streamlit app

## Dataset Description
The dataset is generated programmatically and saved in `data/cab_bookings.csv`.

It contains realistic fields such as:
- booking ID and timestamp
- pickup and drop locations
- trip distance and fare
- booking hour and day information
- passenger count and payment method
- driver and customer ratings
- driver experience and acceptance rate
- cancellation history
- surge pricing and estimated pickup time
- weather and traffic conditions
- booking source
- target variable: `cancellation`

The `cancellation` target is:
- `0` = Not Cancelled
- `1` = Cancelled

## Features Used
The model uses features such as:
- `distance_km`
- `estimated_fare`
- `booking_hour`
- `booking_day`
- `booking_day_of_week`
- `passenger_count`
- `vehicle_type`
- `payment_method`
- `driver_rating`
- `customer_rating`
- `driver_experience_years`
- `driver_acceptance_rate`
- `customer_previous_cancellations`
- `customer_total_bookings`
- `driver_previous_cancellations`
- `surge_multiplier`
- `estimated_pickup_time_minutes`
- `weather`
- `traffic_condition`
- `booking_source`
- `is_weekend`

## Data Generation
Synthetic data is created in `src/data_generation.py`.

The data is not completely random. It follows logical patterns so that cancellations become more likely when:
- estimated pickup time is high
- traffic is heavy
- weather is poor
- surge multiplier is high
- driver acceptance rate is low
- driver rating is low
- customer has many previous cancellations
- trip distance is longer
- booking occurs during certain busy hours

This makes the data realistic enough for a beginner-friendly ML demo.

## EDA
The notebook at `notebooks/01_eda_and_model_training.ipynb` covers:
- importing libraries
- loading the dataset
- showing the first rows
- checking shape and column information
- finding missing values and duplicates
- reviewing basic statistics
- plotting target distribution
- analysing numerical and categorical patterns
- checking cancellation rates by vehicle type, weather, traffic, payment, hour, and surge
- generating useful visualizations

## Preprocessing
The preprocessing pipeline lives in `src/preprocessing.py`.

It uses `ColumnTransformer` with:
- `SimpleImputer` for missing numerical values
- `StandardScaler` for numeric columns
- `SimpleImputer` for missing categorical values
- `OneHotEncoder` for categorical variables

This is important because machine learning models usually expect numeric values. We also avoid data leakage by fitting the preprocessing only on the training data and then applying it to the test set and prediction inputs.

## Models Used
The project trains and compares:
1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier
4. Gradient Boosting Classifier

### Why each model is used
- Logistic Regression: easy to understand, fast, and provides probability estimates.
- Decision Tree: simple and interpretable, can detect clear decision rules.
- Random Forest: robust and often performs well on tabular data.
- Gradient Boosting: strong predictive performance on structured data.

## Evaluation Metrics
The models are evaluated using:
- Accuracy: how often the model is correct overall
- Precision: how many predicted cancellations were actually cancellations
- Recall: how many actual cancellations were correctly identified
- F1-score: balance between precision and recall
- ROC-AUC: ability to separate classes across thresholds

### Why F1-score and ROC-AUC matter
Accuracy alone can be misleading when class distribution is imbalanced. F1-score focuses on the positive class (cancellations), while ROC-AUC measures ranking quality. This is especially helpful for a problem where the minority class is important.

## Class Imbalance Handling
The dataset is checked for imbalance, and the chosen approach is not blind oversampling.

For the models, class weights are used where appropriate (`class_weight="balanced"`) because this helps the model pay more attention to the minority class without creating artificial duplicates in the dataset.

## Final Model Selection
The best model is selected primarily using:
- F1-score
- ROC-AUC

The best model is saved as `models/best_model.pkl`.

## Project Architecture
```text
cab-cancellation-prediction/
├── data/
│   └── cab_bookings.csv
├── notebooks/
│   └── 01_eda_and_model_training.ipynb
├── src/
│   ├── __init__.py
│   ├── data_generation.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── models/
│   ├── best_model.pkl
│   └── preprocessor.pkl
├── reports/
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   └── model_comparison.csv
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── .venv/
```

## Installation
Create a virtual environment and install dependencies.

### macOS/Linux
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Windows
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Generate Data
```bash
python src/data_generation.py
```

## Train the Model
```bash
python src/train.py
```

## Evaluate the Model
```bash
python src/evaluate.py
```

## Run the Streamlit App
```bash
streamlit run app.py
```

## Example Prediction
The app accepts trip and customer details and gives a result such as:
- `Likely to be Cancelled`
- `Likely to be Successful`

It also displays the probability of cancellation and the probability of a successful booking.

## How the Streamlit App Works
The app loads the saved trained pipeline, collects user input, transforms the data into the correct format, and passes it to the model for prediction. The model output is then converted into a human-friendly prediction and probability summary.

## Train/Test Split
The code uses `train_test_split` with `stratify=y` and `random_state=42`.

This means:
- data is split into training and testing sets
- the class distribution is preserved in both sets
- results are reproducible because the random state is fixed

This helps evaluate the model fairly and reduces chance-based variation.

## Avoiding Data Leakage
Data leakage is when information from the test set accidentally influences the training process. In this project, preprocessing is done inside a Scikit-learn pipeline, which ensures that imputation and scaling are learned only from the training data before being applied to the test set. This is a best practice in real machine learning projects.

## Future Improvements
Possible next steps:
- use a bigger real-world dataset
- add feature selection and hyperparameter tuning
- compare more models such as XGBoost or LightGBM
- add a back-end API for serving predictions
- deploy the app to a hosting platform

## Useful Commands Summary
```bash
python src/data_generation.py
python src/train.py
python src/evaluate.py
streamlit run app.py
```
