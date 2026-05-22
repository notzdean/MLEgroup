# CS611 MLE Group Project — Context Handoff
*Last updated: 13 May 2026*

---

## 1. Course & Assessment Context

- **Course:** CS611 — Machine Learning Engineering, SMU SCIS
- **Lecturer:** Ulysses Chong — Lead Data Scientist at Grab, Adjunct at SMU. Skills: GenAI, MLOps, **Propensity Modelling**, **Uplift Modeling**, Recommender Systems.
- **Key rubric signal (from class transcript):** *"Don't focus on model complexity. Focus on how you solve a complex problem... explain design choices, that's where the marks are."*
- **His formula for a good project:**
  1. One real anchor dataset
  2. Add a time column if missing
  3. Find real auxiliary datasets
  4. Synthesise only what genuinely doesn't exist publicly
  5. Show ingest → clean → join → store → prepare pipeline
  6. Run everything locally in Docker, no cloud spend

### Deliverables
| # | Deliverable | Weight | Due |
|---|---|---|---|
| 1 | Proposal deck (5 min presentation) | 20% | Week 6 |
| 2 | Code + Report (8 pages max, A4, font 11 Arial) | 30% | End of term |
| 3 | Final presentation (10 min) | 30% | End of term |
| 4 | Peer evaluation | 20% | End of term |

---

## 2. Project Decision — Two Options Being Evaluated

The team has two fully developed proposals. A comparison slide (slide 7 of `dengue_proposal.pptx`) with a voting grid was created for the team to decide.

### Option A — Dengue Outbreak Risk Prediction ⭐ (current focus)
**Business problem:** Predict which Singapore subzones will have active dengue clusters in the next 14 days — enabling NEA or pest control operators to pre-deploy fogging teams proactively before clusters form.

**Users:**
- NEA Operations Team — views weekly risk map (330 subzones scored Low/Med/High), deploys fogging teams
- Pest Control Operators (Rentokil, PestBusters) — secondary user, allocates teams to hotspots
- Data/Ops Team — monitors model performance, triggers retraining

**ML task:** Binary classification per subzone per week. Label = cluster present in subzone in next 14 days (1/0).

**Model:** XGBoost or LightGBM with time-series cross-validation (not standard k-fold — prevents data leakage).

**Deployment:** Weekly batch inference every Monday, Airflow orchestrated, outputs risk tier per subzone to Postgres.

**Monitoring story:** Dengue serotype shift in 2020 (DENV-3 replaced DENV-1) caused record 35,000+ cases — a historically documented concept drift event that justifies the entire monitoring and retraining pipeline. This is the strongest monitoring narrative of any topic considered.

### Option B — Motor Insurance Claim Propensity (Porto Seguro)
**Business problem:** Predict at renewal whether a policyholder will file a claim next year — enabling actuarial pricing and proactive intervention.

**Anchor:** Porto Seguro Safe Driver Prediction (Kaggle) — 595k rows, 57 features, binary label, real data from Brazil's largest motor insurer.

**Auxiliary:** World Bank Open Data API — Brazil GDP growth, inflation, unemployment, lending rate by year. Joined to Porto Seguro on policy year.

**Users:** Actuary/underwriter (views risk tier at renewal), Data/Ops team (weekly monitoring).

**Model:** LightGBM, Optuna tuning, SMOTE for 3.6% class imbalance.

**Deployment:** Monthly batch scoring at renewal cycle.

---

## 3. Data Sources — Dengue Project

| Source | Role | Status | Location |
|---|---|---|---|
| SGCharts dengue cluster CSV | Anchor | To download | `outbreak.sgcharts.com/data` → Google Drive ZIP |
| NEA Historical Daily Weather (CSV) | Auxiliary 1 | ✅ Downloaded | `HistoricalDailyWeatherRecords.csv` — Admiralty station, 2013–2017 |
| NEA v2 API (rainfall, temp, humidity) | Auxiliary 1 extension | 🔄 In progress | `api-open.data.gov.sg/v2/real-time/api` — covers 2018–2020 |
| LTA MRT tap volumes | Auxiliary 2 | Pending API key | LTA DataMall — requires free registration |
| SingStat subzone population | Auxiliary 3 | To download | `tablebuilder.singstat.gov.sg` → "Resident Population by Planning Area/Subzone, Age Group and Sex (Census of Population 2020)" |
| URA subzone boundaries | Spatial join | To download | `data.gov.sg` → "Master Plan 2019 Subzone Boundary GeoJSON" |

### NEA Weather — Important Details
- `HistoricalDailyWeatherRecords.csv` = **Admiralty station only**, 2009–2017, columns: `date, station, daily_rainfall_total, mean_temperature, maximum_temperature, minimum_temperature, mean_wind_speed`
- Missing values encoded as `na`
- Only one station available via CSV download — Admiralty (North Singapore)
- For 2018–2020: use NEA v2 API (`api-open.data.gov.sg/v2/real-time/api`) — returns multiple stations
- Old v1 API (`api.data.gov.sg/v1/environment/...`) returns 500 errors for historical dates — do not use
- v2 API rate limits: use 3s delay between calls. On 429 error, wait 30s and retry.

### NEA v2 API Endpoints
```
Base: https://api-open.data.gov.sg/v2/real-time/api
Rainfall:          /rainfall          (unit: mm, 5-min intervals)
Air Temperature:   /air-temperature   (unit: °C, 1-min intervals)
Relative Humidity: /relative-humidity (unit: %, 1-min intervals)
Wind Speed:        /wind-speed        (unit: knots)
Wind Direction:    /wind-direction    (unit: degrees)

Query param: date=YYYY-MM-DD
No API key needed (optional for higher rate limits)
Response: data.stations[], data.readings[].data[{stationId, value}]
Pagination: data.paginationToken (loop until null)
Coverage: Rainfall from Dec 2016, Temperature from May 2016
```

### Data Directory Structure
```
C:\Users\ADMIN\Desktop\CS611 - MLE\Group Project\
├── data\
│   └── dengue\
│       ├── HistoricalDailyWeatherRecords.csv   ← Admiralty station 2013-2017
│       ├── nea_weather.csv                     ← output of ingest_nea_weather.py
│       ├── sgcharts_clusters.csv               ← to be created
│       └── (other sources TBD)
├── ingest_nea_weather.py                       ← written, in progress
├── ingest_worldbank.py                         ← written, tested, works
└── (other scripts TBD)
```

---

## 4. Scripts Written So Far

### `ingest_worldbank.py` (Porto Seguro project)
- Fetches Brazil macro indicators from World Bank API
- Output: `worldbank_macro.xlsx` with GDP growth, inflation, unemployment, lending rate 2012–2017
- Status: ✅ **Tested and working on student's machine**
- Output path: `C:\Users\ADMIN\Desktop\CS611 - MLE\Group Project\data\dengue\worldbank_macro.xlsx`

### `ingest_nea_weather.py` (Dengue project)
- Part A: Loads `HistoricalDailyWeatherRecords.csv` → filters 2013–2017 → saves to `nea_weather.csv`
- Part B: Calls NEA v2 API for 2018–2020 with 3s delays, retry on 429, checkpoints every ~10 days
- Status: 🔄 **Part A complete, Part B running (takes ~2-3 hours)**
- Resume-safe: yes — skips already fetched dates on rerun

### Scripts still needed
- `ingest_sgcharts.py` — load and concatenate 250+ SGCharts cluster CSV files
- `ingest_lta_mrt.py` — pull LTA monthly MRT tap volumes (needs API key first)
- `preprocess.py` — clean, spatial join, feature engineering
- `train.py` — XGBoost/LightGBM, Optuna, MLflow tracking
- `inference.py` — weekly batch scoring
- `monitor.py` — drift detection, retraining trigger

---

## 5. Architecture — Dengue Project

Follows the course ML lifecycle diagram style (phase banner: Process Data → Develop Model → Deploy → Monitor & Retrain).

### Pipeline summary
```
Data Sources
  ├── SGCharts cluster CSV (anchor, 2013-2020)
  ├── NEA weather API (auxiliary 1)
  ├── LTA MRT volumes (auxiliary 2)
  └── SingStat population (auxiliary 3)
        ↓ Airflow DAG (weekly)
Ingestion → Raw Storage (Postgres)
        ↓
Feature Engineering
  - Rainfall lag 1/2/4 weeks (mosquito breeding cycle)
  - Temperature rolling mean
  - Cluster proximity per subzone
  - Seasonality features (week of year, month)
  - Population density weighting
        ↓ Store to Feature Store (Postgres)
Training
  - XGBoost binary classifier
  - Optuna hyperparameter tuning
  - Time-series cross-validation (no standard k-fold — prevents leakage)
  - SMOTE for class imbalance
  - MLflow experiment tracking
        ↓ Model promoted to MLflow Model Registry
Batch Inference (weekly, Monday)
  - Score all 330 Singapore subzones
  - Output: risk tier (Low/Med/High) per subzone
  - Stored in Postgres → risk map dashboard
        ↓
Monitor
  - Weekly Gini/Precision/Recall tracking
  - Feature drift (weather seasonality)
  - Concept drift (serotype change → triggers retraining)
  - Alarm Manager + Scheduler
  - Retrain trigger → fires Airflow retraining DAG
```

### Tech stack
Docker + docker-compose, Apache Airflow, MLflow, XGBoost/LightGBM, Optuna, Postgres, Python, GeoPandas (spatial join)

---

## 6. Proposal Decks Created

Two PPTX files were built:

### `proposal_v2.pptx` — Porto Seguro (navy/teal palette)
6 slides: Title, Business Problem, Dataset, Users, Architecture, Design Choices & Open Questions

### `dengue_proposal.pptx` — Dengue (forest green palette)
7 slides: Title, Business Problem, Dataset, Users, Architecture, Design Choices & Open Questions, **Topic Comparison (slide 7 — for team vote)**

Slide 7 has a side-by-side scoring table (9 criteria, star ratings) and a voting grid (D/P boxes for 5 team members).

---

## 7. Key Design Decisions to Document in Report

| Decision | Choice | Rationale |
|---|---|---|
| Inference cadence | Weekly batch | Mosquito breeding cycle is 7–14 days; NEA plans fogging weekly |
| Model | XGBoost | Tabular + spatial data; interpretable for public health reporting |
| Cross-validation | Time-series CV | Standard k-fold leaks future cluster data into training |
| Subzone granularity | 330 subzones | Fine-grained enough for operational fogging deployment |
| Labels | Spatial join of cluster coordinates to subzone boundaries | Only feasible approach given no direct subzone-level labels exist |
| Population weighting | 2020 Census for 2018–2020, 2010 Census for 2013–2014 | Standard practice; assumption documented |
| Data gap post-2020 | Train on 2013–2020; demo serotype drift | SGCharts stopped Nov 2020; 2020 outbreak is the key drift event |
| No cloud spend | Local Docker only | Prof explicitly said AWS spend ≠ better grade |

---

## 8. Open Questions for Proposal Presentation

1. SGCharts data ends 2020 — train on 2013–2020 and demo drift, or synthesise 2021–2025?
2. Subzone vs planning area granularity (330 vs 55)?
3. Optimise Precision vs Recall — false negatives (missed outbreaks) are costly
4. Google Popular Times as auxiliary footfall signal — ToS risk?
5. SMOTE vs threshold tuning for class imbalance?

---

## 9. Python Environment

- Python 3.13 at `C:/Users/ADMIN/AppData/Local/Programs/Python/Python313/python.exe`
- Always run scripts from project directory:
  ```powershell
  cd "C:\Users\ADMIN\Desktop\CS611 - MLE\Group Project"
  & C:/Users/ADMIN/AppData/Local/Programs/Python/Python313/python.exe script_name.py
  ```
- Installed packages: `requests`, `pandas`, `openpyxl`
- Still needed: `geopandas`, `pyspark`, `scikit-learn`, `xgboost`, `lightgbm`, `mlflow`, `optuna`
