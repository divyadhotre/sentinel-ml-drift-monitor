"""
data_simulation.py

Generates two synthetic datasets representing a food-delivery-time prediction problem:

1. TRAINING ERA  ("January") - normal conditions, used to train the baseline model.
2. CURRENT ERA   ("July / Monsoon") - same underlying business, but real-world
   conditions have shifted: more rain, worse traffic, some new far-away restaurants.

The goal is to inject REALISTIC, DOCUMENTED drift so that later, in drift_metrics.py,
we can prove our detector actually catches it.

Features generated per order:
- distance_km          : delivery distance in kilometers
- is_raining           : 1 if raining, 0 otherwise
- traffic_level        : 0 (low) to 10 (severe)
- restaurant_prep_min  : minutes the restaurant takes to prepare food
- hour_of_day          : hour the order was placed (0-23)
- is_weekend           : 1 if Saturday/Sunday

Target:
- delivery_time_min    : total delivery time in minutes (what we want to predict)
"""

import numpy as np
import pandas as pd

RANDOM_SEED = 42


def _base_delivery_time(distance_km, is_raining, traffic_level, restaurant_prep_min, rng):
    """
    The TRUE underlying relationship between features and delivery time.
    We keep this function identical across both eras -- what changes is the
    DISTRIBUTION of inputs, not the physics of delivery itself.
    """
    time = (
        10                              # fixed base (packing, handoff, etc.)
        + distance_km * 2.5             # ~2.5 min per km under normal conditions
        + restaurant_prep_min * 0.6     # restaurant prep spills into total time
        + traffic_level * 1.8           # traffic adds delay
        + is_raining * 12               # rain adds a flat delay
        + is_raining * traffic_level * 0.8  # rain WORSENS traffic delay (interaction)
    )
    noise = rng.normal(0, 4, size=len(distance_km))
    return np.clip(time + noise, 5, None)  # delivery can't be negative/near-zero


def generate_training_era(n=5000, seed=RANDOM_SEED):
    """
    'January' data -- normal conditions.
    - Rain is rare (~8% of orders)
    - Traffic is moderate
    - Distances are mostly short-to-medium (local restaurants)
    """
    rng = np.random.default_rng(seed)

    distance_km = rng.gamma(shape=2.0, scale=1.5, size=n).clip(0.5, 15)
    is_raining = rng.binomial(1, p=0.08, size=n)
    traffic_level = rng.normal(loc=4, scale=1.5, size=n).clip(0, 10)
    restaurant_prep_min = rng.normal(loc=15, scale=5, size=n).clip(5, 40)
    hour_of_day = rng.integers(0, 24, size=n)
    is_weekend = rng.binomial(1, p=2 / 7, size=n)

    delivery_time_min = _base_delivery_time(
        distance_km, is_raining, traffic_level, restaurant_prep_min, rng
    )

    df = pd.DataFrame({
        "distance_km": distance_km,
        "is_raining": is_raining,
        "traffic_level": traffic_level,
        "restaurant_prep_min": restaurant_prep_min,
        "hour_of_day": hour_of_day,
        "is_weekend": is_weekend,
        "delivery_time_min": delivery_time_min,
    })
    return df


def generate_current_era(n=2000, seed=RANDOM_SEED + 1):
    """
    'July / Monsoon' data -- DRIFTED conditions.
    Deliberate, documented shifts vs training era:
    - Rain is now common (~55% of orders)      -> feature drift
    - Traffic is generally higher              -> feature drift
    - New farther-away restaurants added        -> feature drift (distance shifts right)
    - The underlying physics (_base_delivery_time) is UNCHANGED --
      this is pure DATA drift, not concept drift, by design (v1 of the project).
    """
    rng = np.random.default_rng(seed)

    # distance now has a second, farther cluster mixed in (new restaurants onboarded)
    short_dist = rng.gamma(shape=2.0, scale=1.5, size=int(n * 0.7)).clip(0.5, 15)
    long_dist = rng.gamma(shape=3.0, scale=3.0, size=n - int(n * 0.7)).clip(8, 25)
    distance_km = np.concatenate([short_dist, long_dist])
    rng.shuffle(distance_km)

    is_raining = rng.binomial(1, p=0.55, size=n)                 # monsoon
    traffic_level = rng.normal(loc=6.2, scale=1.8, size=n).clip(0, 10)  # worse traffic
    restaurant_prep_min = rng.normal(loc=16, scale=6, size=n).clip(5, 45)
    hour_of_day = rng.integers(0, 24, size=n)
    is_weekend = rng.binomial(1, p=2 / 7, size=n)

    delivery_time_min = _base_delivery_time(
        distance_km, is_raining, traffic_level, restaurant_prep_min, rng
    )

    df = pd.DataFrame({
        "distance_km": distance_km,
        "is_raining": is_raining,
        "traffic_level": traffic_level,
        "restaurant_prep_min": restaurant_prep_min,
        "hour_of_day": hour_of_day,
        "is_weekend": is_weekend,
        "delivery_time_min": delivery_time_min,
    })
    return df


if __name__ == "__main__":
    train_df = generate_training_era()
    current_df = generate_current_era()

    train_df.to_csv("data/raw/training_era.csv", index=False)
    current_df.to_csv("data/raw/current_era.csv", index=False)

    print("Training era shape:", train_df.shape)
    print(train_df.describe().round(2))
    print("\nCurrent era shape:", current_df.shape)
    print(current_df.describe().round(2))
    print("\nSaved to data/raw/training_era.csv and data/raw/current_era.csv")