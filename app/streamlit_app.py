"""
streamlit_app.py

"Sentinel" -- an ML Model Drift Monitoring dashboard.

Detects when a production model's input data has drifted away from its
training distribution, quantifies drift per feature (PSI / KS-test /
KL-divergence), tracks real model performance decay, and renders an
actionable alert -- including a simulated Slack notification, the way a
real MLOps monitoring tool would surface this to an engineering team.

Demonstrated on real NYC Yellow Taxi trip data: January 2020 (pre-pandemic)
vs April 2020 (COVID-19 lockdown) -- a genuine, documented regime change.

Run with:
    streamlit run app/streamlit_app.py
"""

import sys
import os
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from monitor import run_monitor
from model import FEATURE_COLUMNS

st.set_page_config(
    page_title="Sentinel · ML Drift Monitor",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# CUSTOM THEME
# ============================================================================
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .stApp { background-color: #0b0f14; }

    section[data-testid="stSidebar"] {
        background-color: #10151c;
        border-right: 1px solid #1f2733;
    }

    .sentinel-header {
        display: flex;
        align-items: center;
        gap: 14px;
        margin-bottom: 4px;
    }
    .sentinel-mark {
        width: 40px; height: 40px;
        border-radius: 10px;
        background: linear-gradient(135deg, #00d4b8 0%, #0092ff 100%);
        display: flex; align-items: center; justify-content: center;
        font-weight: 800; font-size: 18px; color: #061019;
        flex-shrink: 0;
    }
    .sentinel-title { font-size: 26px; font-weight: 800; color: #eef2f6; letter-spacing: -0.5px; }
    .sentinel-subtitle { color: #7d8a9c; font-size: 14px; margin-top: 2px; }

    .status-pill {
        display: inline-block;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 13px;
        letter-spacing: 0.5px;
        font-family: 'JetBrains Mono', monospace;
    }
    .status-OK   { background: rgba(0, 200, 120, 0.15); color: #00e08c; border: 1px solid rgba(0,200,120,0.4); }
    .status-WATCH{ background: rgba(255, 176, 0, 0.15); color: #ffb000; border: 1px solid rgba(255,176,0,0.4); }
    .status-ALERT{ background: rgba(255, 60, 60, 0.15); color: #ff5c5c; border: 1px solid rgba(255,60,60,0.4); }

    .metric-card {
        background: #10151c;
        border: 1px solid #1f2733;
        border-radius: 12px;
        padding: 18px 20px;
        height: 100%;
    }
    .metric-label { color: #7d8a9c; font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; }
    .metric-value { color: #eef2f6; font-size: 28px; font-weight: 700; font-family: 'JetBrains Mono', monospace; margin-top: 4px; }
    .metric-delta-up { color: #ff5c5c; font-size: 13px; font-weight: 600; }
    .metric-delta-down { color: #00e08c; font-size: 13px; font-weight: 600; }

    .slack-window {
        background: #1a1d21;
        border-radius: 10px;
        border: 1px solid #2c2f34;
        padding: 0;
        overflow: hidden;
        font-family: 'Inter', sans-serif;
    }
    .slack-header {
        background: #1a1d21;
        padding: 10px 16px;
        border-bottom: 1px solid #2c2f34;
        color: #d1d2d3;
        font-weight: 700;
        font-size: 14px;
    }
    .slack-msg {
        padding: 14px 16px;
        display: flex;
        gap: 12px;
    }
    .slack-avatar {
        width: 36px; height: 36px;
        border-radius: 6px;
        background: linear-gradient(135deg, #00d4b8, #0092ff);
        flex-shrink: 0;
    }
    .slack-body { color: #d1d2d3; font-size: 14px; line-height: 1.5; }
    .slack-botname { font-weight: 700; color: #fff; }
    .slack-tag { background:#2c2f34; color:#9aa; font-size:10px; padding:1px 5px; border-radius:3px; margin-left:6px; }
    .slack-time { color: #8a8d91; font-size: 11px; margin-left: 6px; }
    .slack-alert-block {
        border-left: 3px solid #ff5c5c;
        background: rgba(255,92,92,0.08);
        padding: 10px 12px;
        margin-top: 8px;
        border-radius: 4px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 12.5px;
    }
    .slack-alert-block.watch { border-left-color: #ffb000; background: rgba(255,176,0,0.08); }
    .slack-alert-block.ok { border-left-color: #00e08c; background: rgba(0,224,140,0.08); }

    section[data-testid="stSidebar"] .stRadio label { color: #d1d5db; }

    div[data-testid="stDataFrame"] { border: 1px solid #1f2733; border-radius: 10px; }

    .footer-note { color: #4a5568; font-size: 12px; text-align: center; margin-top: 40px; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ============================================================================
# HEADER
# ============================================================================
st.markdown(
    """
    <div class="sentinel-header">
        <div class="sentinel-mark">S</div>
        <div>
            <div class="sentinel-title">Sentinel — ML Drift Monitor</div>
            <div class="sentinel-subtitle">
                Statistical drift detection for production ML models · PSI · KS-test · KL-divergence
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.write("")

# ============================================================================
# SIDEBAR — data source
# ============================================================================
st.sidebar.markdown("### Data Source")

data_source = st.sidebar.radio(
    "",
    ["NYC Taxi: Jan 2020 vs Apr 2020 (COVID lockdown)", "Upload my own CSVs"],
    label_visibility="collapsed",
)

if data_source.startswith("NYC Taxi"):
    train_df = pd.read_csv("data/processed/taxi_training_era.csv")
    current_df = pd.read_csv("data/processed/taxi_current_era.csv")
    st.sidebar.markdown(
        """
        <div style="background:#10151c; border:1px solid #1f2733; border-radius:8px; padding:12px; margin-top:8px;">
        <span style="color:#7d8a9c; font-size:12px;">SCENARIO</span><br>
        <span style="color:#eef2f6; font-size:13px;">
        Reference window: <b>Jan 2020</b> (pre-pandemic, ~6.1M trips)<br>
        Current window: <b>Apr 2020</b> (lockdown, ~203K trips)<br>
        <span style="color:#ff5c5c;">96% ridership collapse</span> — real, documented regime change.
        </span>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    train_file = st.sidebar.file_uploader("Reference (training) CSV", type="csv")
    current_file = st.sidebar.file_uploader("Current CSV", type="csv")
    if train_file is not None and current_file is not None:
        train_df = pd.read_csv(train_file)
        current_df = pd.read_csv(current_file)
    else:
        st.info("Upload both CSVs in the sidebar, or switch to the built-in demo.")
        st.stop()

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    <div style="color:#4a5568; font-size:12px; line-height:1.6;">
    <b style="color:#7d8a9c;">PSI thresholds</b><br>
    &lt; 0.10 &nbsp;→&nbsp; stable<br>
    0.10–0.25 → moderate drift<br>
    &gt; 0.25 &nbsp;→&nbsp; major drift, retrain
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================================
# RUN MONITOR
# ============================================================================
model = None
model_path = "data/processed/baseline_model.pkl"
if os.path.exists(model_path):
    try:
        model = joblib.load(model_path)
    except Exception:
        model = None

available_features = [c for c in FEATURE_COLUMNS if c in train_df.columns and c in current_df.columns]
result = run_monitor(train_df, current_df, model=model, feature_columns=available_features)
status = result["status"]
drift_report = result["drift_report"]
n_drifted = (drift_report["psi"] >= 0.10).sum()

# ============================================================================
# TOP ROW — status + KPIs
# ============================================================================
top_col1, top_col2, top_col3, top_col4 = st.columns([1.3, 1, 1, 1])

with top_col1:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">System Status</div>
            <div style="margin-top:8px;">
                <span class="status-pill status-{status}">● {status}</span>
            </div>
            <div style="color:#7d8a9c; font-size:12px; margin-top:10px;">
                Max PSI: <span style="color:#eef2f6; font-family:'JetBrains Mono',monospace;">{result['max_psi']:.3f}</span>
                &nbsp;·&nbsp; {n_drifted} of {len(available_features)} features drifted
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

if result["performance"] is not None:
    mae = result["performance"]["mae"]
    r2 = result["performance"]["r2"]
    with top_col2:
        st.markdown(
            f"""<div class="metric-card"><div class="metric-label">Model MAE (current)</div>
            <div class="metric-value">{mae:.2f}<span style="font-size:14px; color:#7d8a9c;"> min</span></div></div>""",
            unsafe_allow_html=True,
        )
    with top_col3:
        st.markdown(
            f"""<div class="metric-card"><div class="metric-label">Model R² (current)</div>
            <div class="metric-value">{r2:.3f}</div></div>""",
            unsafe_allow_html=True,
        )
with top_col4:
    st.markdown(
        f"""<div class="metric-card"><div class="metric-label">Rows Compared</div>
        <div class="metric-value" style="font-size:22px;">{len(train_df):,} <span style="color:#7d8a9c; font-size:14px;">vs</span> {len(current_df):,}</div></div>""",
        unsafe_allow_html=True,
    )

st.write("")

# ============================================================================
# TABS
# ============================================================================
tab_overview, tab_drift, tab_distributions, tab_alert = st.tabs(
    ["Overview", "Drift Report", "Distribution Explorer", "Alert Simulation"]
)

# ---------------------------------------------------------------------------
# TAB 1: Overview
# ---------------------------------------------------------------------------
with tab_overview:
    st.markdown("#### Executive Summary")
    st.markdown(
        f"""
        <div style="background:#10151c; border:1px solid #1f2733; border-radius:10px; padding:18px 22px; color:#d1d5db; font-size:15px; line-height:1.7;">
        {result['summary']}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    st.markdown("#### Top Drift Drivers")
    top3 = drift_report.sort_values("psi", ascending=False).head(3)
    cols = st.columns(3)
    for i, (_, row) in enumerate(top3.iterrows()):
        with cols[i]:
            verdict_color = "#ff5c5c" if "Major" in row["psi_verdict"] else ("#ffb000" if "Moderate" in row["psi_verdict"] else "#00e08c")
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">{row['feature']}</div>
                    <div class="metric-value" style="color:{verdict_color};">PSI {row['psi']:.3f}</div>
                    <div style="color:#7d8a9c; font-size:12px; margin-top:6px;">
                        mean {row['ref_mean']:.2f} → {row['cur_mean']:.2f}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.write("")
    st.markdown("#### Key Finding")
    st.markdown(
        """
        <div style="background:rgba(0,146,255,0.08); border:1px solid rgba(0,146,255,0.3); border-radius:10px; padding:16px 20px; color:#c8d6e5; font-size:14px; line-height:1.6;">
        Even during a historic 96% collapse in ridership, core trip mechanics — <b>distance</b> and <b>fare</b> —
        remained statistically stable. The real drift signal was <b>behavioral</b>: <i>when</i> and <i>where</i>
        people traveled shifted, while <i>how far and how much</i> they paid did not. This distinction — data
        drift in some features but not others — is exactly what a naive "did accuracy drop?" check would miss.
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# TAB 2: Drift Report
# ---------------------------------------------------------------------------
with tab_drift:
    st.markdown("#### Per-Feature Statistical Drift")

    def highlight_verdict(row):
        if "Major" in row["psi_verdict"]:
            return ["background-color: rgba(255,92,92,0.12); color: #ffb3b3"] * len(row)
        elif "Moderate" in row["psi_verdict"]:
            return ["background-color: rgba(255,176,0,0.12); color: #ffdb99"] * len(row)
        else:
            return ["background-color: rgba(0,224,140,0.08); color: #b3f0d9"] * len(row)

    styled = drift_report.style.apply(highlight_verdict, axis=1)
    st.dataframe(styled, use_container_width=True, height=280)

    st.caption(
        "PSI: Population Stability Index (bank/fintech standard). "
        "KS-statistic: max distance between cumulative distributions (0=identical, 1=disjoint). "
        "KL-divergence: information-theoretic distance from reference to current distribution."
    )

    csv_bytes = drift_report.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download drift report (CSV)",
        data=csv_bytes,
        file_name=f"drift_report_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
    )

# ---------------------------------------------------------------------------
# TAB 3: Distribution Explorer
# ---------------------------------------------------------------------------
with tab_distributions:
    st.markdown("#### Distribution Shift — All Features At A Glance")

    plt.rcParams.update({
        "figure.facecolor": "#0b0f14",
        "axes.facecolor": "#10151c",
        "axes.edgecolor": "#1f2733",
        "axes.labelcolor": "#d1d5db",
        "xtick.color": "#7d8a9c",
        "ytick.color": "#7d8a9c",
        "text.color": "#eef2f6",
        "font.size": 9,
    })

    n_feat = len(available_features)
    n_cols = 3
    n_rows = (n_feat + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(13, 3.2 * n_rows))
    axes = axes.flatten() if n_feat > 1 else [axes]

    for i, feat in enumerate(available_features):
        ax = axes[i]
        ax.hist(train_df[feat], bins=30, alpha=0.55, label="Reference", density=True, color="#0092ff")
        ax.hist(current_df[feat], bins=30, alpha=0.55, label="Current", density=True, color="#ff8a00")
        psi_val = drift_report.loc[drift_report["feature"] == feat, "psi"].values[0]
        badge_color = "#ff5c5c" if psi_val >= 0.25 else ("#ffb000" if psi_val >= 0.10 else "#00e08c")
        ax.set_title(f"{feat}  (PSI {psi_val:.3f})", fontsize=10, color=badge_color, fontweight="bold")
        ax.legend(fontsize=7, frameon=False)

    for j in range(n_feat, len(axes)):
        axes[j].axis("off")

    fig.tight_layout()
    st.pyplot(fig)

    st.caption("Blue = reference (training) window · Orange = current window. Colored titles indicate drift severity.")

# ---------------------------------------------------------------------------
# TAB 4: Alert Simulation
# ---------------------------------------------------------------------------
with tab_alert:
    st.markdown("#### What This Would Post To Your Team's Slack")
    st.caption("A real drift-monitoring tool doesn't just compute numbers — it has to reach an engineer. Here's the notification this run would trigger.")

    now_str = datetime.now().strftime("%I:%M %p")
    alert_class = "ok" if status == "OK" else ("watch" if status == "WATCH" else "")
    icon = "✅" if status == "OK" else ("⚠️" if status == "WATCH" else "🚨")

    top_features = drift_report[drift_report["psi"] >= 0.10]["feature"].tolist()
    feature_lines = "\n".join([
        f"  • {row['feature']}: PSI {row['psi']:.2f} ({row['ref_mean']:.2f} → {row['cur_mean']:.2f})"
        for _, row in drift_report[drift_report["psi"] >= 0.10].iterrows()
    ]) or "  • none above threshold"

    perf_line = ""
    if result["performance"] is not None:
        perf_line = f"Model MAE: {result['performance']['mae']:.2f} min · R²: {result['performance']['r2']:.3f}"

    st.markdown(
        f"""
        <div class="slack-window">
            <div class="slack-header">#ml-platform-alerts</div>
            <div class="slack-msg">
                <div class="slack-avatar"></div>
                <div class="slack-body">
                    <span class="slack-botname">Sentinel Bot</span>
                    <span class="slack-tag">APP</span>
                    <span class="slack-time">{now_str}</span><br>
                    {icon} <b>Drift check complete for trip-duration model</b><br>
                    Status: <b>{status}</b> · Max PSI: {result['max_psi']:.3f} · {perf_line}
                    <div class="slack-alert-block {alert_class}">
Drifted features (PSI ≥ 0.10):
{feature_lines}

Recommendation: {"No action needed." if status == "OK" else ("Monitor closely; schedule retrain within 2 weeks." if status == "WATCH" else "Retrain before next deploy — model reliability is degrading.")}
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    st.markdown(
        """
        <div style="color:#7d8a9c; font-size:13px;">
        In a production setup, <code style="color:#0092ff;">monitor.py</code> would run on a schedule
        (e.g. a daily Airflow/cron job) and post this message via a Slack webhook whenever status
        moves to WATCH or ALERT — rather than requiring anyone to open a dashboard.
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    '<div class="footer-note">Sentinel · Built to demonstrate PSI / KS-test / KL-divergence drift detection on real NYC TLC trip data</div>',
    unsafe_allow_html=True,
)
