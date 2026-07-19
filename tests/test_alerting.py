"""
test_alerting.py

Tests for Slack alerting. Uses a mock HTTP server (not a real Slack
webhook) to verify:
    - An ALERT/WATCH status actually triggers a POST request with a
      correctly formatted payload
    - An OK status correctly skips sending anything
    - A missing webhook URL is handled gracefully, without crashing
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import json
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler

import pandas as pd
import pytest

from sentinel.alerting import send_slack_alert
from sentinel.core import SentinelResult


@pytest.fixture
def mock_slack_server():
    received = {}

    class MockHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers["Content-Length"])
            body = self.rfile.read(length)
            received["payload"] = json.loads(body)
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")

        def log_message(self, format, *args):
            pass

    server = HTTPServer(("localhost", 0), MockHandler)  # port 0 = pick a free port
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    yield f"http://localhost:{port}/mock-webhook", received

    server.shutdown()


def _make_result(status, max_psi=0.5):
    drift_report = pd.DataFrame([
        {"feature": "x", "psi": max_psi, "psi_verdict": "test", "ref_mean": 1.0, "cur_mean": 2.0}
    ])
    return SentinelResult(
        status=status,
        max_psi=max_psi,
        drift_report=drift_report,
        summary=f"{status}: test summary",
        performance=None,
        concept_drift=None,
    )


class TestSendSlackAlert:
    def test_alert_status_sends_real_post_request(self, mock_slack_server):
        webhook_url, received = mock_slack_server
        result = _make_result("ALERT", max_psi=1.5)

        sent = send_slack_alert(result, webhook_url=webhook_url)

        assert sent is True
        assert "payload" in received
        assert "ALERT" in received["payload"]["text"]

    def test_ok_status_does_not_send(self, mock_slack_server):
        webhook_url, received = mock_slack_server
        result = _make_result("OK", max_psi=0.02)

        sent = send_slack_alert(result, webhook_url=webhook_url, min_status="WATCH")

        assert sent is False
        assert "payload" not in received

    def test_missing_webhook_url_does_not_crash(self, monkeypatch):
        monkeypatch.delenv("SENTINEL_SLACK_WEBHOOK", raising=False)
        result = _make_result("ALERT", max_psi=1.5)

        sent = send_slack_alert(result, webhook_url=None)

        assert sent is False
