# Sentinel — ML Model Drift Monitor

**Detects when a production ML model's input data has silently drifted away from its training distribution, quantifies *what kind* of drift occurred, and flags whether retraining is needed — before it costs the business money.**

Demonstrated on real NYC Yellow Taxi trip data: **January 2020 (pre-pandemic) vs April 2020 (COVID-19 lockdown)** — a genuine, documented 96% collapse in ridership.

---

## The Problem

Machine learning models deployed in production silently degrade as real-world data drifts away from the distribution they were trained on. Most teams discover this only after business metrics have already suffered, because drift monitoring is not standard practice outside large tech companies. This project builds an automated drift-detection and alerting system — the same core function performed by MLOps monitoring tools like **Evidently AI** and **WhyLabs** — and validates it three independent ways: on real historical data, via a relationship-level (concept drift) check, and via a controlled robustness study against a naive baseline.

## The Headline Finding

Between January and April 2020, NYC Yellow Taxi ridership collapsed **96%** (6,405,008 → 238,073 trips) — one of the most extreme, well-documented disruptions to urban transportation in recent history. Even so:

> **Core trip mechanics — distance and fare — remained statistically stable. The real drift signal was *behavioral*: when (`hour_of_day`) and where (`PULocationID`) people traveled shifted moderately, while how far and how much they paid did not.**

This distinction — drift concentrated in some features and absent in others — is exactly what a naive "did accuracy drop?" check would have missed, and is only visible through feature-level statistical monitoring.

---

## Three Layers of Validation

Most student drift-detection projects stop at "I ran PSI on some data and it worked." This project validates the methodology three separate ways:

| Validation | What it proves | Where |
|---|---|---|
| **1. Real-world data drift** | The pipeline correctly detects a real, historic regime change (COVID-19) using PSI, KS-test, and KL-divergence — computed from their statistical formulas, not a black-box library call | `src/drift_metrics.py`, `notebooks/01_eda_and_results.ipynb` |
| **2. Concept drift check** | Beyond "did the inputs change," this checks whether the *relationship* between inputs and the target changed — training separate models per era and comparing feature importances and correlations | `src/concept_drift.py` |
| **3. Robustness & baseline comparison** | A controlled synthetic study proving (a) detection reliability scales predictably with drift magnitude across 20+ random seeds, and (b) a naive mean/z-score check misses shape-only distribution shifts that PSI correctly catches | `notebooks/05_robustness_analysis.ipynb` |

### Result of validation #2 — Concept Drift

```
CONCEPT DRIFT VERDICT: NO CONCEPT DRIFT
```
`fare_amount`'s correlation with trip duration barely moved (0.852 → 0.860), and `trip_distance` stayed similarly stable (0.800 → 0.835). This **confirms** the data-drift finding above: the physics of "how distance/fare relate to duration" held steady — only *when and where* people traveled changed. Three independent analyses telling the same coherent story is a much stronger claim than any one of them alone.

### Result of validation #3 — Naive Baseline vs PSI

Two distributions with the **same mean** but 3x the spread:
```
Naive z-score: 0.0525   -> NOT flagged (looks fine)
PSI:           0.6086   -> FLAGGED (major drift)
```
A mean-only check would let this drift through completely silently. Across a 20-seed x 8-magnitude sensitivity sweep, PSI's detection rate climbs from 0% to a consistent 100% once injected drift crosses roughly half the original scenario's strength — proving detection is reliable, not a lucky one-off.

---

## Architecture

```
data/raw (NYC TLC parquet files)
        │
        ▼
 data_loader.py  ──────► cleans real data (nulls, negative fares, GPS errors, invalid durations)
        │
        ▼
   model.py      ──────► trains Random Forest on Jan 2020 only, evaluates on Apr 2020
        │
        ▼
drift_metrics.py ──────► PSI / KS-test / KL-divergence / naive z-score, per feature
        │
        ▼
concept_drift.py ──────► per-era model comparison: did the relationship itself change?
        │
        ▼
   monitor.py    ──────► combines drift + performance into one OK/WATCH/ALERT verdict
        │
        ▼
streamlit_app.py ──────► "Sentinel" dashboard: KPIs, drift table, distribution charts,
                          simulated Slack alert
```

`data_simulation.py` is a second, independent data source (synthetic food-delivery data) kept in the repo as a **controlled testbed** — used in the robustness notebook to stress-test the detector under known, dialable drift conditions where real-world ground truth isn't available.

---

## Key Results

| Metric | Jan 2020 (reference) | Apr 2020 (current) | Change |
|---|---|---|---|
| Total trips (raw) | 6,405,008 | 238,073 | **−96%** |
| Model MAE | 1.10 min (hold-out) | 1.25 min | +14% |
| Model R² | 0.910 | 0.828 | −9% |
| `hour_of_day` PSI | — | 0.158 | Moderate drift |
| `PULocationID` PSI | — | 0.102 | Moderate drift |
| `trip_distance` PSI | — | 0.020 | Stable |
| `fare_amount` PSI | — | 0.063 | Stable |

*(Full per-feature table with KS-test, KL-divergence, and naive z-score comparison in `reports/drift_report.csv`.)*

---

## Dashboard

An interactive Streamlit dashboard ("Sentinel") with four views:
- **Overview** — status banner, KPIs, top drift drivers, key finding
- **Drift Report** — full statistical table, color-coded by severity, downloadable as CSV
- **Distribution Explorer** — all 7 features' before/after distributions in one grid
- **Alert Simulation** — a simulated Slack notification showing what this tool would post to an engineering channel in a real production setup

```bash
streamlit run app/streamlit_app.py
```
![Overview Tab](reports/screenshots/overview.png)

<br><br>

![Distribution Explorer Tab](reports/screenshots/Exploror.png)

---

## Tech Stack

Python · pandas · NumPy · scikit-learn · SciPy (statistical tests) · Matplotlib · Streamlit · Jupyter

## Repository Structure

```
ml-drift-monitor/
├── data/
│   ├── raw/               # NYC TLC parquet files (not committed — see setup)
│   └── processed/         # cleaned CSVs + trained model (generated by pipeline)
├── notebooks/
│   ├── 01_eda_and_results.ipynb      # real-data analysis, charts, key finding
│   └── 05_robustness_analysis.ipynb  # naive-vs-PSI demo + sensitivity sweep
├── src/
│   ├── data_loader.py      # real NYC TLC data cleaning + feature engineering
│   ├── data_simulation.py  # synthetic controlled testbed (used in robustness notebook)
│   ├── model.py            # baseline Random Forest training/evaluation
│   ├── drift_metrics.py    # PSI, KS-test, KL-divergence, naive z-score
│   ├── concept_drift.py    # relationship-level (concept) drift detection
│   └── monitor.py          # combines everything into one OK/WATCH/ALERT verdict
├── app/
│   └── streamlit_app.py    # "Sentinel" interactive dashboard
├── reports/
│   ├── figures/            # saved charts from notebooks
│   └── drift_report.csv    # latest generated drift report
└── requirements.txt
```

---

## How to Run

**1. Set up the environment** (Python 3.10 recommended for library compatibility):
```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

**2. Download the real data** from [NYC TLC Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page) — under **2020**, get the **Yellow Taxi Trip Records (Parquet)** for **January** and **April**. Place them at:
```
data/raw/yellow_tripdata_2020-01.parquet
data/raw/yellow_tripdata_2020-04.parquet
```

**3. Run the pipeline in order:**
```bash
python src/data_simulation.py    # optional: generates the synthetic testbed
python src/data_loader.py        # cleans real taxi data
python src/model.py              # trains + evaluates the baseline model
python src/drift_metrics.py      # generates the statistical drift report
python src/concept_drift.py      # checks for relationship-level drift
python src/monitor.py            # combined OK/WATCH/ALERT verdict
```

**4. Launch the dashboard:**
```bash
streamlit run app/streamlit_app.py
```

**5. Explore the notebooks** in `notebooks/` (select the project's `venv` as the Jupyter kernel).

---

## Limitations

- **Sampling:** Data is sampled to 50,000 rows per era for iteration speed; a production pipeline would monitor the full stream.
- **Extreme case by design:** COVID-19 was chosen deliberately as a large, well-documented event to validate methodology under known ground truth. Smaller, gradual drift in a live system is a harder detection problem — the robustness notebook's sensitivity sweep addresses this directly by testing detection reliability across a *range* of drift magnitudes, not just the extreme case.
- **Two-point comparison:** This compares two fixed windows (Jan vs Apr). A production version would monitor drift continuously over a rolling window.
- **Single model family:** Only Random Forest is evaluated. Comparing against a simpler linear baseline (already scaffolded in `model.py` via `model_type="linear"`) is a natural extension.

## Data Source

[NYC Taxi & Limousine Commission — Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page) (public, free, official government source)
