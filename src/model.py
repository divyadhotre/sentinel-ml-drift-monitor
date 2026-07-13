"""
model.py

Trains the "production" trip-duration prediction model using ONLY the
training-era (January 2020, pre-pandemic) NYC Yellow Taxi data -- exactly
like a real company that trained a model months ago and hasn't retrained
it since.

We then evaluate this exact same trained model on current-era (April 2020,
COVID-19 lockdown) data to show how much its accuracy degrades -- that
comparison is the core proof of this project.

NOTE: This module previously used simulated food-delivery data
(distance_km, is_raining, etc. -- see data_simulation.py, kept in the repo
as the "v1 controlled validation" of the methodology). It now points at the
real NYC TLC taxi data produced by data_loader.py.
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

FEATURE_COLUMNS = [
    "trip_distance",
    "passenger_count",
    "PULocationID",
    "DOLocationID",
    "fare_amount",
    "hour_of_day",
    "is_weekend",
]
TARGET_COLUMN = "trip_duration_min"


def load_training_data(path="data/processed/taxi_training_era.csv"):
    return pd.read_csv(path)


def train_baseline_model(df, model_type="random_forest", random_state=42):
    """
    Trains a model on training-era data, holding out a test split from the
    SAME era to report an honest "before drift" accuracy baseline.
    """
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    if model_type == "linear":
        model = LinearRegression()
    elif model_type == "random_forest":
        model = RandomForestRegressor(
            n_estimators=200, max_depth=8, random_state=random_state
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print(f"[{model_type}] Training-era hold-out performance:")
    print(f"  MAE : {mae:.2f} minutes")
    print(f"  R^2 : {r2:.3f}")

    return model, {"mae": mae, "r2": r2}


def evaluate_on_new_data(model, df, label=""):
    """
    Runs an already-trained model on a NEW dataset (e.g. current-era/drifted data)
    and reports how accurate it still is. This is what we'll use to PROVE
    performance decay caused by drift.
    """
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    preds = model.predict(X)
    mae = mean_absolute_error(y, preds)
    r2 = r2_score(y, preds)

    print(f"[Evaluation on {label}]")
    print(f"  MAE : {mae:.2f} minutes")
    print(f"  R^2 : {r2:.3f}")

    return {"mae": mae, "r2": r2}


if __name__ == "__main__":
    train_df = load_training_data()
    model, train_metrics = train_baseline_model(train_df, model_type="random_forest")

    joblib.dump(model, "data/processed/baseline_model.pkl")
    print("\nSaved trained model to data/processed/baseline_model.pkl")

    # Quick sanity check: evaluate on current-era data right now, so you can
    # SEE the performance drop immediately (we'll formalize this in a notebook).
    current_df = pd.read_csv("data/processed/taxi_current_era.csv")
    print()
    evaluate_on_new_data(model, current_df, label="current_era (April 2020 lockdown) data")