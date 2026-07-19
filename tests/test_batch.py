"""
test_batch.py

Tests for multi-model batch monitoring. Verifies:
    - Multiple valid jobs all run and appear in the summary
    - A job with a bad file path is isolated as an ERROR without
      crashing the other jobs in the same batch
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import yaml
import pytest

from sentinel.batch import run_batch


@pytest.fixture
def two_job_config(tmp_path):
    rng = np.random.default_rng(20)

    ref1 = pd.DataFrame({"a": rng.normal(0, 1, 500), "y": rng.normal(0, 1, 500)})
    cur1 = pd.DataFrame({"a": rng.normal(5, 1, 500), "y": rng.normal(0, 1, 500)})  # drifted
    ref1_path, cur1_path = tmp_path / "ref1.csv", tmp_path / "cur1.csv"
    ref1.to_csv(ref1_path, index=False)
    cur1.to_csv(cur1_path, index=False)

    ref2 = pd.DataFrame({"b": rng.normal(0, 1, 500), "y": rng.normal(0, 1, 500)})
    cur2 = pd.DataFrame({"b": rng.normal(0, 1, 500), "y": rng.normal(0, 1, 500)})  # stable
    ref2_path, cur2_path = tmp_path / "ref2.csv", tmp_path / "cur2.csv"
    ref2.to_csv(ref2_path, index=False)
    cur2.to_csv(cur2_path, index=False)

    config = {
        "jobs": [
            {"name": "drifted-job", "reference": str(ref1_path), "current": str(cur1_path), "features": ["a"], "target": "y"},
            {"name": "stable-job", "reference": str(ref2_path), "current": str(cur2_path), "features": ["b"], "target": "y"},
            {"name": "broken-job", "reference": "does_not_exist.csv", "current": str(cur2_path), "features": ["b"], "target": "y"},
        ]
    }
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    return str(config_path)


class TestRunBatch:
    def test_all_jobs_appear_in_summary(self, two_job_config):
        summary_df, results = run_batch(two_job_config)
        assert len(summary_df) == 3
        assert set(summary_df["job"]) == {"drifted-job", "stable-job", "broken-job"}

    def test_drifted_job_shows_higher_psi_than_stable_job(self, two_job_config):
        summary_df, results = run_batch(two_job_config)
        drifted_psi = summary_df.loc[summary_df["job"] == "drifted-job", "max_psi"].values[0]
        stable_psi = summary_df.loc[summary_df["job"] == "stable-job", "max_psi"].values[0]
        assert drifted_psi > stable_psi

    def test_broken_job_isolated_as_error_without_crashing_others(self, two_job_config):
        summary_df, results = run_batch(two_job_config)
        broken_row = summary_df[summary_df["job"] == "broken-job"].iloc[0]
        assert broken_row["status"] == "ERROR"
        assert broken_row["error"] is not None

        # the other two jobs should have run successfully despite the broken one
        assert results["drifted-job"] is not None
        assert results["stable-job"] is not None
