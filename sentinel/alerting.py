"""
sentinel/alerting.py

Sends REAL Slack notifications via an incoming webhook -- not a simulation.
Requires a Slack webhook URL, which should NEVER be hardcoded or committed
to version control. Pass it via the SENTINEL_SLACK_WEBHOOK environment
variable, or explicitly to send_slack_alert().

Setup (one-time, free):
    1. Go to https://api.slack.com/apps -> "Create New App" -> "From scratch"
    2. Enable "Incoming Webhooks" for the app
    3. "Add New Webhook to Workspace" -> choose a channel
    4. Copy the webhook URL (looks like https://hooks.slack.com/services/...)
    5. Set it as an environment variable:
         Windows (PowerShell): $env:SENTINEL_SLACK_WEBHOOK = "https://hooks.slack.com/..."
         Mac/Linux:             export SENTINEL_SLACK_WEBHOOK="https://hooks.slack.com/..."
"""

import os
import requests


def _build_payload(result, source_label="Sentinel"):
    """Builds a Slack Block Kit message from a SentinelResult."""
    status_emoji = {"OK": "✅", "WATCH": "⚠️", "ALERT": "🚨"}.get(result.status, "ℹ️")

    drifted = result.drift_report[result.drift_report["psi"] >= 0.10]
    if len(drifted) > 0:
        drift_lines = "\n".join(
            f"• `{row['feature']}`: PSI {row['psi']:.3f} ({row['ref_mean']:.2f} → {row['cur_mean']:.2f})"
            for _, row in drifted.iterrows()
        )
    else:
        drift_lines = "No features above drift threshold."

    perf_line = ""
    if result.performance is not None:
        perf_line = f"\nModel MAE: {result.performance['mae']:.3f} · R²: {result.performance['r2']:.3f}"

    text = (
        f"{status_emoji} *{source_label} drift check: {result.status}*\n"
        f"Max PSI: {result.max_psi:.3f}{perf_line}\n\n"
        f"*Drifted features:*\n{drift_lines}\n\n"
        f"_{result.summary}_"
    )

    return {"text": text}


def send_slack_alert(result, webhook_url=None, source_label="Sentinel", min_status="WATCH"):
    """
    Sends a real Slack message via incoming webhook if result.status is at
    least `min_status` severity ("WATCH" or "ALERT"). Returns True if sent,
    False if skipped (status below threshold) or failed (network/config error).

    webhook_url: if not provided, reads from SENTINEL_SLACK_WEBHOOK env var.
    """
    severity = {"OK": 0, "WATCH": 1, "ALERT": 2}
    if severity.get(result.status, 0) < severity.get(min_status, 1):
        print(f"[sentinel.alerting] Status is {result.status}, below '{min_status}' threshold -- no alert sent.")
        return False

    webhook_url = webhook_url or os.environ.get("SENTINEL_SLACK_WEBHOOK")
    if not webhook_url:
        print("[sentinel.alerting] No webhook URL configured (set SENTINEL_SLACK_WEBHOOK) -- alert not sent.")
        return False

    payload = _build_payload(result, source_label=source_label)

    try:
        response = requests.post(webhook_url, json=payload, timeout=10)
        response.raise_for_status()
        print(f"[sentinel.alerting] Slack alert sent successfully (status {response.status_code}).")
        return True
    except requests.RequestException as e:
        print(f"[sentinel.alerting] Failed to send Slack alert: {e}")
        return False
