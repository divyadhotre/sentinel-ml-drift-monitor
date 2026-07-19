"""
test_cli.py

Tests for the command-line interface. Calls the command functions
directly with constructed argparse.Namespace objects (rather than
spawning a real subprocess), catching the expected SystemExit that
signals success/failure via exit code -- the same convention a CI
pipeline would rely on.
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import argparse
import numpy as np
import pandas as pd
import pytest

from sentinel.cli import monitor_command, monitor_all_command


@pytest.fixture
def drifted_csvs(tmp_path):
    rng = np.random.default_rng(42)
    ref = pd.DataFrame({"a": rng.normal(0, 1, 500), "y": rng.normal(0, 1, 500)})
    cur = pd.DataFrame({"a": rng.normal(5, 1, 500), "y": rng.normal(0, 1, 500)})
    ref_path, cur_path = tmp_path / "ref.csv", tmp_path / "cur.csv"
    ref.to_csv(ref_path, index=False)
    cur.to_csv(cur_path, index=False)
    return str(ref_path), str(cur_path)


class TestMonitorCommand:
    def test_alert_status_exits_with_code_1(self, drifted_csvs, tmp_path):
        ref_path, cur_path = drifted_csvs
        args = argparse.Namespace(
            reference=ref_path, current=cur_path, features="a", target="y",
            output=str(tmp_path / "out.csv"),
        )
        with pytest.raises(SystemExit) as exc_info:
            monitor_command(args)
        assert exc_info.value.code == 1
        assert os.path.exists(tmp_path / "out.csv")

    def test_stable_data_exits_with_code_0(self, tmp_path):
        rng = np.random.default_rng(1)
        ref = pd.DataFrame({"a": rng.normal(0, 1, 500)})
        cur = pd.DataFrame({"a": rng.normal(0, 1, 500)})
        ref_path, cur_path = tmp_path / "ref.csv", tmp_path / "cur.csv"
        ref.to_csv(ref_path, index=False)
        cur.to_csv(cur_path, index=False)

        args = argparse.Namespace(
            reference=str(ref_path), current=str(cur_path), features="a", target=None, output=None,
        )
        with pytest.raises(SystemExit) as exc_info:
            monitor_command(args)
        assert exc_info.value.code == 0


class TestMonitorAllCommand:
    def test_batch_command_runs_and_saves_output(self, drifted_csvs, tmp_path):
        import yaml
        ref_path, cur_path = drifted_csvs
        config = {"jobs": [{"name": "job1", "reference": ref_path, "current": cur_path, "features": ["a"], "target": "y"}]}
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        args = argparse.Namespace(config=str(config_path), output=str(tmp_path / "summary.csv"))
        with pytest.raises(SystemExit):
            monitor_all_command(args)
        assert os.path.exists(tmp_path / "summary.csv")
