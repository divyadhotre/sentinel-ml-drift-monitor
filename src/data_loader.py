"""
data_loader.py

Loads and cleans REAL NYC Taxi & Limousine Commission (TLC) Yellow Taxi
trip data, replacing the synthetic data_simulation.py pipeline for the
"real-world dataset" phase of this project.

Source: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
Files used:
    yellow_tripdata_2020-01.parquet   -> TRAINING ERA (pre-pandemic, "normal")
    yellow_tripdata_2020-04.parquet   -> CURRENT ERA  (peak COVID-19 lockdown)

This is a genuine regime change that we did NOT design ourselves --
real behavioral shift in a real city, not synthetic drift.

Cleaning steps applied (documented so this is defensible in an interview):
    - Drop rows with nulls in required columns
    - Remove negative or zero fares (data entry errors)
    - Remove trips with 0 or absurd distance (>100 miles) -- GPS/meter errors
    - Remove trips with 0 or absurd duration (<1 min or >180 min)
    - Remove passenger_count of 0 or >6 (invalid/commercial van misclassification)
    - Cap to only VendorID 1 and 2 (valid vendors)

Engineered features:
    - trip_duration_min   : (dropoff - pickup) in minutes  [THIS IS OUR TARGET]
    - hour_of_day         : hour trip started (0-23)
    - is_weekend          : 1 if Saturday/Sunday
    - day_of_month        : for potential time-series analysis later
"""

import pandas as pd

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

REQUIRED_RAW_COLUMNS = [
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
    "trip_distance",
    "passenger_count",
    "PULocationID",
    "DOLocationID",
    "fare_amount",
    "VendorID",
]


def load_and_clean(path, sample_size=None, random_state=42):
    """
    Loads a raw NYC TLC parquet file and returns a cleaned DataFrame with
    engineered features, ready for modeling and drift detection.

    Parameters:
        path (str): path to the .parquet file
        sample_size (int, optional): if given, randomly samples this many
            rows AFTER cleaning (useful to keep runtime fast on large files)
    """
    df = pd.read_parquet(path)

    missing_cols = [c for c in REQUIRED_RAW_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing expected columns in {path}: {missing_cols}")

    df = df[REQUIRED_RAW_COLUMNS].copy()

    n_before = len(df)

    # 1. Drop nulls in required columns
    df = df.dropna(subset=REQUIRED_RAW_COLUMNS)

    # 2. Valid vendors only
    df = df[df["VendorID"].isin([1, 2])]

    # 3. Remove invalid fares
    df = df[(df["fare_amount"] > 0) & (df["fare_amount"] < 300)]

    # 4. Remove invalid distances
    df = df[(df["trip_distance"] > 0) & (df["trip_distance"] < 100)]

    # 5. Remove invalid passenger counts
    df = df[(df["passenger_count"] > 0) & (df["passenger_count"] <= 6)]

    # 6. Engineer trip duration (the target) and filter invalid durations
    df["trip_duration_min"] = (
        df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]
    ).dt.total_seconds() / 60.0
    df = df[(df["trip_duration_min"] >= 1) & (df["trip_duration_min"] <= 180)]

    # 7. Engineer time-based features
    df["hour_of_day"] = df["tpep_pickup_datetime"].dt.hour
    df["is_weekend"] = (df["tpep_pickup_datetime"].dt.dayofweek >= 5).astype(int)
    df["day_of_month"] = df["tpep_pickup_datetime"].dt.day

    n_after = len(df)
    pct_kept = 100 * n_after / n_before if n_before else 0
    print(f"[{path}] Loaded {n_before:,} rows -> kept {n_after:,} rows after cleaning ({pct_kept:.1f}%)")

    final_cols = FEATURE_COLUMNS + [TARGET_COLUMN, "day_of_month"]
    df = df[final_cols].reset_index(drop=True)

    if sample_size is not None and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=random_state).reset_index(drop=True)
        print(f"  Sampled down to {sample_size:,} rows for faster iteration")

    return df


if __name__ == "__main__":
    # NOTE: update these paths once you've downloaded the real files.
    train_path = "data/raw/yellow_tripdata_2020-01.parquet"
    current_path = "data/raw/yellow_tripdata_2020-04.parquet"

    train_df = load_and_clean(train_path, sample_size=50000)
    current_df = load_and_clean(current_path, sample_size=50000)

    train_df.to_csv("data/processed/taxi_training_era.csv", index=False)
    current_df.to_csv("data/processed/taxi_current_era.csv", index=False)

    print("\nTraining era summary:")
    print(train_df.describe().round(2))
    print("\nCurrent era summary:")
    print(current_df.describe().round(2))
    print("\nSaved cleaned data to data/processed/taxi_training_era.csv and taxi_current_era.csv")