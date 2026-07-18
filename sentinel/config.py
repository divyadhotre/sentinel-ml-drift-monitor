"""
sentinel/config.py

Loads a YAML config file describing multiple models/datasets to monitor
in one run -- a lightweight step toward multi-tenancy: one command that
checks several models at once and produces a single consolidated report,
instead of running Sentinel separately for each one.

Example config file (sentinel_config.yaml):

    jobs:
      - name: taxi-duration-model
        reference: data/processed/taxi_training_era.csv
        current: data/processed/taxi_current_era.csv
        features: [trip_distance, passenger_count, PULocationID, DOLocationID, fare_amount, hour_of_day, is_weekend]
        target: trip_duration_min

      - name: house-price-model
        reference: data/houses_2023.csv
        current: data/houses_2024.csv
        features: [sqft, bedrooms, age_years]
        target: price
"""

import yaml


class ConfigError(ValueError):
    pass


def load_config(path):
    """
    Loads and validates a Sentinel batch-monitoring config file.
    Returns a list of job dicts, each with: name, reference, current,
    features, and optionally target.
    """
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    if not raw or "jobs" not in raw:
        raise ConfigError(f"Config file '{path}' must have a top-level 'jobs' list.")

    jobs = raw["jobs"]
    if not isinstance(jobs, list) or len(jobs) == 0:
        raise ConfigError("'jobs' must be a non-empty list.")

    validated = []
    for i, job in enumerate(jobs):
        missing = [k for k in ("name", "reference", "current", "features") if k not in job]
        if missing:
            raise ConfigError(f"Job #{i} is missing required fields: {missing}")
        if not isinstance(job["features"], list) or len(job["features"]) == 0:
            raise ConfigError(f"Job '{job['name']}': 'features' must be a non-empty list.")

        validated.append({
            "name": job["name"],
            "reference": job["reference"],
            "current": job["current"],
            "features": job["features"],
            "target": job.get("target"),
        })

    return validated
