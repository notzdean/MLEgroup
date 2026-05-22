# Dengue Outbreak Risk Prediction

> *"The cost of a false negative in the batch model is a missed fogging run. The cost of a false negative in the real-time model is a preventable death."*

---

## Problem Statement

Singapore's 2020 dengue outbreak recorded over 35,000 cases — the worst in history — driven by a DENV-3 serotype shift. Existing response tools were reactive: NEA and CDA had no system to predict which neighbourhoods were at risk **before** clusters formed.

We build a subzone-level dengue outbreak risk prediction system that answers two operational questions:

| User | Question | Cadence |
|---|---|---|
| NEA Operations | Where should fogging be deployed next week? | Weekly batch |
| CDA Officer at MOH | Who needs same-day outreach right now? | Per confirmed case |

---

## Architecture — One Model, Two Inference Paths

Dengue control is a two-speed problem. We serve both use cases from **one trained model artifact**:

```
                    ┌─────────────────────────────┐
                    │     HISTORICAL DATA          │
                    │  SGCharts + MSS + SingStat   │
                    └──────────┬──────────────────┘
                               │
                    ┌──────────▼──────────────────┐
                    │   MEDALLION PIPELINE         │
                    │  Bronze → Silver → Gold      │
                    └──────────┬──────────────────┘
                               │
                    ┌──────────▼──────────────────┐
                    │   LIGHTGBM MODEL             │
                    │  P(active cluster in subzone │
                    │  S over next 14 days)        │
                    └──────┬───────────────┬───────┘
                           │               │
              ┌────────────▼───┐     ┌─────▼──────────────┐
              │  BATCH PATH    │     │  REAL-TIME PATH     │
              │  Airflow DAG   │     │  FastAPI + Redis    │
              │  Mon 06:00 SGT │     │  < 500ms latency    │
              │  NEA Ops       │     │  CDA Officer @ MOH  │
              └────────────────┘     └─────────────────────┘
```

*Same predictive logic. Different freshness contracts. Different thresholds. Different actions.*

---

## Model Performance

**Production model:** LightGBM  
**Optimised for:** Recall ≥ 0.70 (false negative = missed outbreak)

| Split | Period | Recall | Precision | F1 | AUC-ROC |
|---|---|---|---|---|---|
| Validation | Jan–Jun 2019 | 0.9923 | 0.2047 | 0.3393 | 0.8509 |
| **Test** | **Jul–Dec 2019** | **0.9960** | **0.3205** | **0.4849** | **0.8271** |
| OOT | Jan–Nov 2020 | 0.9826 | 0.2756 | 0.4305 | 0.7871 |

**OOT note:** Jan–Nov 2020 covers the DENV-3 serotype shift outbreak (35,000+ cases). The 1.3 percentage point recall drop from Test to OOT is within our 10pp tolerance gate. Performance degradation on OOT is expected and documented as a concept drift case study, not a model failure.

**Score PSI (Test → OOT):** 0.0179 — stable. Score distribution barely shifted despite the outbreak severity.

### Promotion gate results
| Gate | Threshold | Result |
|---|---|---|
| Recall on test set | ≥ 0.70 | ✓ 0.9960 |
| OOT recall drop | ≤ 10pp | ✓ 1.3pp |
| AUC-ROC on test set | ≥ 0.75 | ✓ 0.8271 |
| SHAP values logged | Required | ✓ |

### Note on precision
Precision is 0.32 on test — roughly 2 in 3 alerts are false positives. This is intentional. The cost function is asymmetric: a false positive means an unnecessary fogging run; a false negative means a missed outbreak. We optimise for recall.

---

## Feature Importance (SHAP)

Mean absolute SHAP values on test set:

| Rank | Feature | Mean \|SHAP\| | Interpretation |
|---|---|---|---|
| 1 | population | 0.5313 | Larger subzones have more exposure |
| 2 | cluster_count_rolling2w | 0.3483 | Cluster momentum — recent activity |
| 3 | cluster_count_rolling4w | 0.2316 | Longer-window cluster trend |
| 4 | population_density | 0.1684 | Dense areas spread faster |
| 5 | elderly_pct | 0.1114 | Vulnerability proxy |
| 6 | area_km2 | 0.0873 | Subzone size |
| 7 | vulnerability_index | 0.0765 | PCA composite score |
| 8 | rainfall_lag1w | 0.0293 | Mosquito breeding conditions |
| 9 | rainfall_lag2w | 0.0247 | |
| 10 | rainfall_lag4w | 0.0056 | |

SHAP order makes domain sense: population and cluster momentum dominate, rainfall lags contribute at the margin (consistent with the 10–14 day mosquito incubation cycle).

---

## Vulnerability Index

The real-time alert rule uses a composite vulnerability index per subzone:

```
alert_score = model_score × vulnerability_index × log(case_count + 1)
```

The vulnerability index is a weighted combination of `elderly_pct` and `population_density`. Rather than manually assigning weights (e.g. 60/40), we use **PCA on the 2019 census data** across all 274 populated subzones:

| Parameter | Value |
|---|---|
| Method | PCA PC1 loadings, abs normalised |
| elderly_pct weight | 0.50 |
| population_density weight | 0.50 |
| PC1 variance explained | 64.8% |
| Subzones used | 274 |

The equal 50/50 weights are not an assumption — they are the empirical result. PC1 explains 64.8% of variance across subzones, confirming that age profile and density contribute equally to subzone differentiation. Weights are saved to `data/gold/vulnerability_pca_weights.json` and reused at inference time.

---

## Concept Drift Analysis

The monitoring report (run against OOT vs training baseline) shows:

| Feature | CSI | Status |
|---|---|---|
| rainfall_lag4w | 0.686 | ⚠ Significant drift |
| rainfall_lag2w | 0.545 | ⚠ Significant drift |
| rainfall_lag1w | 0.376 | ⚠ Significant drift |
| cluster_count_rolling4w | 11.95 | ⚠ Significant drift |
| cluster_count_rolling2w | 10.18 | ⚠ Significant drift |
| population | 0.000 | ✓ Stable |
| elderly_pct | 0.000 | ✓ Stable |
| area_km2 | 0.000 | ✓ Stable |
| population_density | 0.000 | ✓ Stable |
| vulnerability_index | 0.000 | ✓ Stable |

**Interpretation:** Time-varying features (weather lags, cluster rolling counts) show significant drift in 2020 — expected given the record outbreak. Static demographic features are perfectly stable (CSI=0.0000) as they come from the 2019 census snapshot. This is the DENV-3 concept drift event documented in our report.

---

## Data Sources

| Source | File | Coverage | Role |
|---|---|---|---|
| SGCharts | `raw_sgcharts_dengue.csv` | May 2013 – Nov 2020 | Anchor — cluster labels |
| MSS Weather | `raw_mss_weather_2013_2020.csv` | 2013 – 2020, 63 stations | Rainfall lag features |
| MSS Stations | `raw_mss_stations_list_mss.csv` | 63 stations | Station metadata join |
| SingStat | `raw_singstat_pop_17560.csv` | Census 2020 | Population + age breakdown |
| URA GeoJSON | `raw_MasterPlan2019Subzone...geojson` | 332 subzones | Subzone polygons + spatial join |

**Why MSS over NEA API:** MSS station data extends back to 2013. The NEA real-time API only starts December 2016 — using it would cost three years of training data.

**Data files are not committed to git.** Place them in `data/bronze/` locally. They are mounted as Docker volumes and shared via the team Google Drive.

---

## Data Pipeline

### Medallion architecture

```
data/bronze/   ← raw ingested files (schema enforced, no cleaning)
data/silver/   ← cleaned files (outliers, nulls, bad rows handled)
data/gold/     ← feature store (one row per subzone per snapshot date)
```

### Key preprocessing decisions

**SGCharts:** The `incorrect_latitude_longitude` folder name is a scraper artefact — coordinates are valid. Inspection confirmed 16,969 of 17,511 flagged rows have valid Singapore coordinates, recovering May 2013 – Jul 2015 data. Only 542 genuinely bad rows (blank or out-of-bbox lat/lng) are dropped.

**MSS Weather:** Temperature (28.7% coverage) and wind (34.2% coverage) columns dropped — too sparse for reliable lag features. Rainfall (96.6% coverage) retained. Aggregated to daily Singapore-wide mean. Forward-fill gaps ≤ 3 days, linear interpolation for longer gaps.

**SingStat:** 46 zero-population subzones dropped. 56 subzones with missing age breakdowns imputed from planning area average. Loyang West (90.9% elderly — retirement home) retained and flagged.

**Label creation:** Forward-looking 14-day window — for each subzone at snapshot date T, label = 1 if any active cluster overlaps the subzone between T+1 and T+14. Features use only data available at T. No temporal leakage.

### Class imbalance
- 320 subzones × 256 snapshots = 81,920 total rows
- Positive labels: 10,128 (12.4%)
- Negative labels: 71,792 (87.6%)
- SMOTE applied to training folds only — never to validation, test, or OOT

### Data split

| Split | Period | Rows | Positive Rate | Purpose |
|---|---|---|---|---|
| Train | Jul 2015 – Dec 2018 | 51,238 | 10.4% | Model training |
| Validation | Jan – Jun 2019 | 4,658 | 16.7% | Optuna HPT |
| Test | Jul – Dec 2019 | 6,850 | 29.2% | Final evaluation (touched once) |
| OOT | Jan – Nov 2020 | 7,398 | 25.7% | Concept drift case study |

5-fold time-series sliding-window CV within training window. Earlier dates always train on later dates — never reversed.

---

## Project Structure

```
cs611-dengue/
├── pipeline/
│   ├── ingest_sgcharts.py          # Bronze: SGCharts cluster snapshots
│   ├── ingest_mss_weather.py       # Bronze: MSS weather station records
│   ├── ingest_population.py        # Bronze: SingStat census data
│   ├── ingest_geodata.py           # Bronze: URA subzone GeoJSON
│   ├── preprocess.py               # Bronze → Silver: cleaning per source
│   └── feature_engineering.py     # Silver → Gold: features + label
├── model/
│   ├── train.py                    # XGBoost + LightGBM + Optuna HPT
│   └── evaluate.py                 # Test + OOT + promotion gate
├── inference/
│   ├── realtime_inference.py       # FastAPI: LISTEN/NOTIFY → Redis → alert
│   └── airflow/
│       ├── batch_dag.py            # Airflow: Mon 06:00 SGT batch scoring
│       └── retrain_dag.py          # Airflow: drift-triggered retraining
├── monitoring/
│   └── monitor.py                  # PSI/CSI drift detection + retrain trigger
├── infrastructure/
│   └── init_db.sql                 # Postgres schema
├── data/                           # gitignored — mount as Docker volumes
│   ├── bronze/
│   ├── silver/
│   └── gold/
├── notebooks/
│   └── compare_weather_sources.ipynb
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── .env.example
```

---

## Quickstart — Teammate Setup (Windows)

> Follow these steps exactly. Takes about 20-30 minutes on first run.

### Prerequisites
- Python 3.11+ — [python.org](https://www.python.org/downloads/)
- Docker Desktop — [docker.com](https://www.docker.com/products/docker-desktop/) — **must be running before Step 5**
- Git — [git-scm.com](https://git-scm.com/)

---

### Step 1 — Clone the repo

Open CMD and run:
```cmd
git clone https://github.com/notzdean/MLEgroup.git
cd MLEgroup
```

---

### Step 2 — Place data files

Data files are NOT in the repo (gitignored). Get them from the shared Google Drive and place them in `data\bronze\`:

```
data\bronze\raw_sgcharts_dengue.csv
data\bronze\raw_mss_weather_2013_2020.csv
data\bronze\raw_mss_stations_list_mss.csv
data\bronze\raw_singstat_pop_17560.csv
data\bronze\raw_MasterPlan2019SubzoneBoundaryNoSeaGEOJSON.geojson
```

Create the folders if they don't exist:
```cmd
mkdir data\bronze
mkdir data\silver
mkdir data\gold
```

---

### Step 3 — Install Python dependencies

```cmd
pip install -r requirements.txt
```

If geopandas fails on Windows, install it separately first:
```cmd
pip install geopandas pyarrow
pip install -r requirements.txt
```

---

### Step 4 — Run the pipeline

Run in this exact order — each step depends on the previous:

```cmd
:: Bronze ingestion (loads raw files, schema enforcement only)
python pipeline\ingest_sgcharts.py
python pipeline\ingest_mss_weather.py
python pipeline\ingest_population.py
python pipeline\ingest_geodata.py

:: Silver cleaning (drops bad rows, fills gaps, derives columns)
python pipeline\preprocess.py

:: Gold features + label (spatial join, weather lags, vulnerability index)
python pipeline\feature_engineering.py

:: Train model (XGBoost + LightGBM, Optuna HPT, ~5 minutes)
python model\train.py

:: Evaluate + promote to Production
python model\evaluate.py
```

Expected output at the end of `evaluate.py`:
```
[gate] Promotion gate
  Recall ≥ 0.7 on test    : ✓
  OOT drop ≤ 10pp         : ✓
  AUC-ROC ≥ 0.75 on test  : ✓
  SHAP values logged       : ✓
  Overall: ✓ PASS — promote to Production
```

---

### Step 5 — Create `.env` file

The `.env` file is gitignored (contains passwords). Create it by running this in CMD:

```cmd
(
echo POSTGRES_USER=dengue
echo POSTGRES_PASSWORD=dengue
echo POSTGRES_DB=dengue
echo POSTGRES_HOST=postgres
echo POSTGRES_PORT=5432
echo DATABASE_URL=postgresql://dengue:dengue@postgres:5432/dengue
echo REDIS_HOST=redis
echo REDIS_PORT=6379
echo REDIS_URL=redis://redis:6379/0
echo MLFLOW_TRACKING_URI=http://mlflow:5000
echo MLFLOW_BACKEND_STORE_URI=postgresql://dengue:dengue@postgres:5432/mlflow
echo MLFLOW_ARTIFACT_ROOT=/mlflow/artifacts
echo AIRFLOW__CORE__EXECUTOR=LocalExecutor
echo AIRFLOW__DATABASE__SQL_ALCHEMY_CONN=postgresql://dengue:dengue@postgres:5432/airflow
echo AIRFLOW__CORE__FERNET_KEY=yLNRaItTnErXtgko8KX3w7l2ZceMT-O571vgP2OJAQk=
echo AIRFLOW__WEBSERVER__SECRET_KEY=3cb72cfa6f6c28fd3fc2784ea934583e0a5aed5ce61d8bb85cfa109d5dc37d4c
echo AIRFLOW_ADMIN_USER=airflow
echo AIRFLOW_ADMIN_PASSWORD=airflow
echo FASTAPI_HOST=0.0.0.0
echo FASTAPI_PORT=8000
echo ALERT_THRESHOLD=0.6
echo MODEL_NAME=dengue_cluster_model
echo MODEL_STAGE=Production
) > .env
```

---

### Step 6 — Start Docker stack

Make sure Docker Desktop is open and running, then:

```cmd
docker-compose up -d
```

First run downloads images and takes ~5 minutes. You'll see:
```
✔ Container dengue_postgres    Healthy
✔ Container dengue_redis       Healthy
✔ Container dengue_airflow     Started
✔ Container dengue_mlflow      Started
✔ Container dengue_fastapi     Started
```

**Important:** On first boot, MLflow and Airflow need their databases created. Run:
```cmd
docker exec dengue_postgres psql -U dengue -d dengue -c "CREATE DATABASE mlflow;"
docker exec dengue_postgres psql -U dengue -d dengue -c "CREATE DATABASE airflow;"
```

Then restart the services:
```cmd
docker-compose restart mlflow airflow
```

Wait 2-3 minutes for MLflow to finish installing, then restart FastAPI:
```cmd
docker-compose restart fastapi
```

---

### Step 7 — Verify everything is running

```cmd
curl http://localhost:8000/health
```

Expected: `{"status":"ok","model":"lightgbm","redis":true,"postgres":true}`

---

### Step 8 — Open the dashboards

| Dashboard | URL | What it shows |
|---|---|---|
| NEA Risk Map | http://localhost:8000/dashboard | Singapore choropleth map, subzone risk tiers, click for details |
| CDA Mobile Emulator | http://localhost:8000/mobile | Real-time alert simulation for CDC Officer |
| FastAPI docs | http://localhost:8000/docs | All API endpoints |
| MLflow | http://localhost:5000 | Model runs, metrics, registry |
| Airflow | http://localhost:8080 | DAG schedule (airflow/airflow) |

---

### Test the real-time alert

Submit a confirmed case and watch the alert fire:

**Windows CMD:**
```cmd
curl -X POST http://localhost:8000/cases/confirmed -H "Content-Type: application/json" -d "{\"subzone_name\": \"TAMPINES EAST\", \"case_count\": 5, \"source\": \"hospital\"}"
```

**Mac/Linux:**
```bash
curl -X POST http://localhost:8000/cases/confirmed \
  -H "Content-Type: application/json" \
  -d '{"subzone_name": "TAMPINES EAST", "case_count": 5, "source": "hospital"}'
```

Run it 3 times — the alert triggers on the 3rd call when `alert_score > 0.6`.

---

### Troubleshooting

| Problem | Fix |
|---|---|
| `geopandas` install fails | `pip install geopandas pyarrow` separately first |
| `Empty reply from server` on port 5000/8080 | Services still starting — wait 2-3 min and retry |
| MLflow `database does not exist` | Run the `CREATE DATABASE mlflow` command in Step 6 |
| Airflow `Fernet key` error | Check `.env` has the full key from Step 5 |
| FastAPI `Not Found` on /dashboard | Rebuild: `docker-compose up -d --no-deps --build fastapi` |
| `git push` rejected | Run `git pull origin main --allow-unrelated-histories` first |

---

### Stopping the stack

```cmd
docker-compose down
```

This stops containers but **keeps data** (Postgres volumes persist). To also wipe data:
```cmd
docker-compose down -v
```

> ⚠️ `down -v` deletes the Postgres databases. You'll need to recreate them (Step 6) on next start.

---

## Quickstart — Teammate Setup (Mac)

> Follow these steps exactly. Takes about 20-30 minutes on first run.

### Prerequisites
- Python 3.11+ — install via [Homebrew](https://brew.sh): `brew install python@3.11`
- Docker Desktop — [docker.com](https://www.docker.com/products/docker-desktop/) — **must be running before Step 5**
- Git — comes with Xcode Command Line Tools: `xcode-select --install`

---

### Step 1 — Clone the repo

Open Terminal and run:
```bash
git clone https://github.com/notzdean/MLEgroup.git
cd MLEgroup
```

---

### Step 2 — Place data files

Data files are NOT in the repo (gitignored). Get them from the shared Google Drive and place them in `data/bronze/`:

```
data/bronze/raw_sgcharts_dengue.csv
data/bronze/raw_mss_weather_2013_2020.csv
data/bronze/raw_mss_stations_list_mss.csv
data/bronze/raw_singstat_pop_17560.csv
data/bronze/raw_MasterPlan2019SubzoneBoundaryNoSeaGEOJSON.geojson
```

Create the folders if they don't exist:
```bash
mkdir -p data/bronze data/silver data/gold
```

---

### Step 3 — Create a virtual environment (recommended)

```bash
python3 -m venv venv
source venv/bin/activate
```

You'll need to run `source venv/bin/activate` each time you open a new Terminal session.

---

### Step 4 — Install Python dependencies

```bash
pip install -r requirements.txt
```

If you're on an **Apple Silicon Mac (M1/M2/M3)**, geopandas may need extra steps:
```bash
brew install gdal
pip install geopandas pyarrow
pip install -r requirements.txt
```

---

### Step 5 — Run the pipeline

Run in this exact order:

```bash
# Bronze ingestion
python3 pipeline/ingest_sgcharts.py
python3 pipeline/ingest_mss_weather.py
python3 pipeline/ingest_population.py
python3 pipeline/ingest_geodata.py

# Silver cleaning
python3 pipeline/preprocess.py

# Gold features + label
python3 pipeline/feature_engineering.py

# Train model (~5 minutes)
python3 model/train.py

# Evaluate + promote
python3 model/evaluate.py
```

---

### Step 6 — Create `.env` file

```bash
cat > .env << 'EOF'
POSTGRES_USER=dengue
POSTGRES_PASSWORD=dengue
POSTGRES_DB=dengue
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
DATABASE_URL=postgresql://dengue:dengue@postgres:5432/dengue
REDIS_HOST=redis
REDIS_PORT=6379
REDIS_URL=redis://redis:6379/0
MLFLOW_TRACKING_URI=http://mlflow:5000
MLFLOW_BACKEND_STORE_URI=postgresql://dengue:dengue@postgres:5432/mlflow
MLFLOW_ARTIFACT_ROOT=/mlflow/artifacts
AIRFLOW__CORE__EXECUTOR=LocalExecutor
AIRFLOW__DATABASE__SQL_ALCHEMY_CONN=postgresql://dengue:dengue@postgres:5432/airflow
AIRFLOW__CORE__FERNET_KEY=yLNRaItTnErXtgko8KX3w7l2ZceMT-O571vgP2OJAQk=
AIRFLOW__WEBSERVER__SECRET_KEY=3cb72cfa6f6c28fd3fc2784ea934583e0a5aed5ce61d8bb85cfa109d5dc37d4c
AIRFLOW_ADMIN_USER=airflow
AIRFLOW_ADMIN_PASSWORD=airflow
FASTAPI_HOST=0.0.0.0
FASTAPI_PORT=8000
ALERT_THRESHOLD=0.6
MODEL_NAME=dengue_cluster_model
MODEL_STAGE=Production
EOF
```

---

### Step 7 — Start Docker stack

Make sure Docker Desktop is open and running, then:

```bash
docker-compose up -d
```

First run downloads images and takes ~5 minutes.

**Important — on first boot only**, create the MLflow and Airflow databases:
```bash
docker exec dengue_postgres psql -U dengue -d dengue -c "CREATE DATABASE mlflow;"
docker exec dengue_postgres psql -U dengue -d dengue -c "CREATE DATABASE airflow;"
docker-compose restart mlflow airflow
```

Wait 2-3 minutes, then restart FastAPI:
```bash
docker-compose restart fastapi
```

---

### Step 8 — Verify

```bash
curl http://localhost:8000/health
```

Expected: `{"status":"ok","model":"lightgbm","redis":true,"postgres":true}`

---

### Step 9 — Open the dashboards

| Dashboard | URL |
|---|---|
| NEA Risk Map + Analytics | http://localhost:8000/dashboard |
| CDA Mobile Emulator | http://localhost:8000/mobile |
| FastAPI docs | http://localhost:8000/docs |
| MLflow | http://localhost:5000 |
| Airflow | http://localhost:8080 (airflow/airflow) |

---

### Troubleshooting (Mac)

| Problem | Fix |
|---|---|
| `geopandas` install fails on M1/M2/M3 | `brew install gdal` first |
| `Empty reply from server` | Services still starting — wait 2-3 min and retry |
| MLflow `database does not exist` | Run the `CREATE DATABASE` commands in Step 7 |
| Airflow crash loop | Check `.env` Fernet key is the full value from Step 6 |
| FastAPI `Not Found` on /dashboard | `docker-compose up -d --no-deps --build fastapi` |
| `git push` rejected | `git pull origin main --allow-unrelated-histories` first |
| Permission denied on `venv` | `chmod +x venv/bin/activate` |

---

### Stopping the stack

```bash
docker-compose down          # stops containers, keeps data
docker-compose down -v       # stops containers AND wipes databases
```

> ⚠️ `down -v` deletes the Postgres databases. Recreate them (Step 7) on next start.

---

## Design Decisions

| Decision | Rationale |
|---|---|
| One model, two inference paths | Same cost function, different serving contracts. "One trained artifact — served two ways." |
| LightGBM (winner) + XGBoost | Tabular spatial+temporal data. SHAP interpretable. Trains in minutes locally. Both tuned, best promoted. |
| Recall ≥ 0.70 target | Asymmetric cost: false negative = missed outbreak. We optimise for the more expensive error. |
| Time-series sliding-window CV | Standard k-fold leaks future cluster data. Temporal CV is the only valid choice. |
| 320 subzones not 55 planning areas | Operational precision for fogging deployment. Planning areas lose deployment granularity. |
| MSS weather not NEA API | MSS extends to 2013. NEA API floor is Dec 2016 — switching costs 3 years of training data. |
| Alert rule not a second model | No public outcome labels for individual cases. Transparent business logic: score × vulnerability × log(cases). |
| PCA vulnerability weights | Data-driven over manual assumption. 50/50 is the empirical result, not an arbitrary choice. |
| SMOTE on training folds only | Applying SMOTE to val/test/OOT would inflate metrics. Imbalance correction is a training-only step. |
| Single OOT window | Dataset ends Nov 2020. One OOT window is framed as a retrospective concept drift demonstration using the worst outbreak on record. |

---

## Docker Services

| Local | Purpose | AWS Equivalent |
|---|---|---|
| Postgres | Bronze/Silver/Gold/Operational tables | RDS (Postgres) |
| Redis | Online feature store (case_count, vulnerability_index) | ElastiCache |
| MLflow | Model registry + experiment tracking | MLflow on EC2 |
| Airflow | Batch DAG + retrain DAG orchestration | MWAA |
| FastAPI | Real-time inference endpoint | ECS (Fargate) |

---


