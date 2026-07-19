"""
test_config.py

Tests for the batch-monitoring config loader. Verifies:
    - A valid config file loads correctly with all fields
    - Missing required fields raise a clear ConfigError
    - An empty jobs list raises a clear ConfigError
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import pytest
import yaml

from sentinel.config import load_config, ConfigError


def _write_config(tmp_path, content):
    path = tmp_path / "config.yaml"
    with open(path, "w") as f:
        yaml.dump(content, f)
    return str(path)


class TestLoadConfig:
    def test_valid_config_loads_correctly(self, tmp_path):
        content = {
            "jobs": [
                {
                    "name": "test-job",
                    "reference": "ref.csv",
                    "current": "cur.csv",
                    "features": ["a", "b"],
                    "target": "y",
                }
            ]
        }
        path = _write_config(tmp_path, content)
        jobs = load_config(path)
        assert len(jobs) == 1
        assert jobs[0]["name"] == "test-job"
        assert jobs[0]["target"] == "y"

    def test_missing_target_defaults_to_none(self, tmp_path):
        content = {
            "jobs": [
                {"name": "test-job", "reference": "ref.csv", "current": "cur.csv", "features": ["a"]}
            ]
        }
        path = _write_config(tmp_path, content)
        jobs = load_config(path)
        assert jobs[0]["target"] is None

    def test_missing_required_field_raises_config_error(self, tmp_path):
        content = {"jobs": [{"name": "test-job", "reference": "ref.csv", "features": ["a"]}]}
        path = _write_config(tmp_path, content)
        with pytest.raises(ConfigError, match="missing required fields"):
            load_config(path)

    def test_empty_jobs_list_raises_config_error(self, tmp_path):
        content = {"jobs": []}
        path = _write_config(tmp_path, content)
        with pytest.raises(ConfigError, match="non-empty list"):
            load_config(path)

    def test_no_jobs_key_raises_config_error(self, tmp_path):
        content = {"not_jobs": []}
        path = _write_config(tmp_path, content)
        with pytest.raises(ConfigError, match="top-level 'jobs'"):
            load_config(path)
