# CS611 MLE — Dengue Outbreak Risk Prediction
## Project Handoff Document

---

## Course context

- **Course:** SMU SCIS CS611 Machine Learning Engineering
- **Lecturer:** Ulysses Chong (Lead DS at Grab — propensity modelling, uplift modelling, recommender systems, GenAI, MLOps)
- **Rubric:** Design choices weighted over model complexity — every decision must be articulable
- **Deliverables:** Proposal deck 20% (done), Code + Report 30% (Week 10), Final Presentation 30% (Week 10), Peer Evaluation 20%

---

## Project brief

Predict which of Singapore's **330 residential subzones** will have an active dengue cluster in the **next 14 days**.

One trained model, two inference paths:
- **Batch** — weekly fogging deployment plan for NEA Ops
- **Real-time** — sub-500ms alert to CDC Officer at MOH when a new confirmed case is inserted

---

## Architecture

### Medallion data layers (Postgres)

```
Postgres (one instance)
│
├── Historical (Medallion)
│   ├── bronze.raw_clusters
│   ├── bronze.raw_weather
│   ├── bronze.raw_population
│   ├── bronze.raw_geodata
│   │
│   ├── silver.clusters_clean
│   ├── silver.weather_clean
│   ├── silver.population_clean
│   ├── silver.geodata_clean
│   │
│   └── gold.subzone_features       ← offline feature store
│
└── Operational
    ├── operational.confirmed_cases      ← RT trigger
    ├── operational.risk_tier            ← batch output
    └── operational.vulnerability_alerts ← RT output

Redis (separate server)              ← online feature store
    ├── current_week_case_count
    └── vulnerability_index
```

### Batch inference pipeline

```
Airflow DAG (Mon 06:00 SGT)
    → load Production model from MLflow
    → read gold.subzone_features (all 330 subzones)
    → score all 330 → apply Low/Medium/High tier logic
    → write operational.risk_tier
    → NEA dashboard reads via API
```

### Real-time inference pipeline

```
INSERT → operational.confirmed_cases
    → Postgres LISTEN/NOTIFY
    → FastAPI (Docker / AWS ECS)
    → update Redis: current_week_case_count
    → read Redis (online features) + Postgres Gold (static features)
    → MLflow Production model (loaded at startup, kept in memory)
    → alert rule: score × vulnerability_index × case_count > threshold
    → write operational.vulnerability_alerts
    → AWS SNS → CDC Officer mobile
    → target: < 500ms end-to-end
```

---

## Data sources

| Source | File | Content | Coverage |
|---|---|---|---|
| SGCharts | `BRONZE_sgcharts_dengue.csv` | Cluster snapshots, lat/lng, case counts | May 2013 – Nov 2020 |
| MSS Weather | `BRONZE_mss_weather_2013_2020.csv` | Daily rainfall, temp, wind per station | 2013 – 2020 |
| SingStat | `BRONZE_subzone_population.csv` | Population, age breakdown per subzone | 2019 census |
| SingStat | `BRONZE_planning_area_population.csv` | Planning area totals | 2019 census |
| URA GeoJSON | `BRONZE_MasterPlan2019SubzoneBoundaryNoSeaGEOJSON.geojson` | 332 subzone polygons | 2019 |

All five files are available and have been profiled.

---

## Key EDA findings (already done)

### SGCharts
- 74,469 rows, 256 snapshots, 393 unique clusters
- **17,511 rows have bad lat/lng** — `source_folder == 'incorrect_latitude_longitude'` — must be dropped
- 2020 peak: 393 active clusters on 2020-08-07 (DENV-3 serotype shift)
- Case counts are extremely right-skewed — log transform needed

### MSS Weather
- 63 stations, 168,697 rows
- **Rainfall coverage: 96.6%** — good for lag features
- **Temperature coverage: 28.7%** — too sparse to use reliably
- **Wind coverage: 34.2%** — too sparse to use reliably
- Use rainfall only for weather features

### Subzone population
- 283 subzones with population data
- GeoJSON has 332 features — **49 unmatched** (water bodies, ports, parks — exclude from scoring)
- 54 subzones have 0% elderly (industrial zones)
- 56 subzones missing age breakdown — impute from planning area average
- Outlier: Loyang West at 90.9% elderly (retirement home — keep but flag)
- 85 subzones with population < 1,000

### Class imbalance
- 330 subzones × 256 snapshots = 84,480 total slots
- Positive labels (label=1): 12,809 (15%)
- Negative labels (label=0): 71,671 (85%)
- **Ratio: 6:1** — SMOTE required on training set only

---

## Data split

| Split | Period | Purpose |
|---|---|---|
| Training | May 2013 – Dec 2018 | Model learns cluster patterns |
| Validation | Jan – Jun 2019 | Optuna HPT, threshold tuning |
| Test | Jul – Dec 2019 | Final evaluation — touched once |
| OOT | Jan – Nov 2020 | Concept drift case study (DENV-3 shift) |

- **5-fold time-series sliding-window CV within training window only**
- Each fold: earlier dates train, later dates validate — never reversed
- SMOTE applied to training fold only — never to val/test/OOT

---

## Preprocessing: Bronze → Silver (`preprocess.py`)

Per source:

### raw_clusters → clusters_clean
- Drop rows where `source_folder == 'incorrect_latitude_longitude'`
- Parse `date` → DATE
- Cast `case_count` to int
- Drop zero-count and duplicate clusters
- Normalise status strings → `'active'`
- Drop clusters outside Singapore bounding box (lat 1.15–1.48, lng 103.6–104.1)

### raw_weather → weather_clean
- Parse date from year/month/day columns → DATE
- Cast rainfall_mm to float
- Forward-fill gaps ≤ 3 days, interpolate longer gaps — never drop rows
- Cap rainfall > 300mm/day as outlier
- Drop temp and wind columns (too sparse)
- Aggregate to daily Singapore-wide mean across stations

### raw_population → population_clean
- Standardise subzone names → UPPERCASE, strip punctuation (canonical format)
- Impute missing age breakdowns from planning area average
- Derive `elderly_pct` = population aged 65+ / total population
- Drop subzones with population = 0

### raw_geodata → subzones_clean
- Parse GeoJSON → GeoPandas GeoDataFrame
- Validate all 330 residential subzones present (exclude 49 non-residential)
- Repair invalid geometries with `buffer(0)`
- Standardise subzone names to canonical format (match population)
- Compute `area_km2` in EPSG:3414, store as EPSG:4326
- Build date × subzone spine (cartesian product) — negative labels assigned here

---

## Feature engineering: Silver → Gold (`feature_engineering.py`)

Output: one wide table `gold.subzone_features` — one row per subzone per week

| Feature group | Features | Notes |
|---|---|---|
| Weather lags | rainfall_lag1w, rainfall_lag2w, rainfall_lag4w | Captures mosquito incubation cycle |
| Rolling clusters | cluster_count_rolling2w, cluster_count_rolling4w | Cluster momentum |
| Spatial | cluster_proximity, active_cluster_overlap | Spatial join clusters × subzones |
| Demographics | population_density, elderly_pct, area_km2 | From population_clean |
| Vulnerability | vulnerability_index | Composite: elderly_pct + population_density |
| Label | label | 1 if active cluster overlaps subzone in next 14 days |

**Label creation:** forward-looking 14-day window — features use data available BEFORE the label date (no leakage)

**Output written to:**
- `gold.subzone_features` table in Postgres
- `data/gold/subzone_features.parquet` on disk (for training)

---

## Model development

### train.py
1. Read `data/gold/subzone_features.parquet`
2. Split strictly by time (Train/Val/Test/OOT)
3. Apply SMOTE on training set only
4. 5-fold time-series CV within training window
5. Train XGBoost and LightGBM in parallel
6. Optuna HPT against validation set — target recall ≥ 0.70
7. Log all runs to MLflow
8. Register best model as `Candidate` in MLflow Registry

### evaluate.py
1. Load `Candidate` model from MLflow
2. **Test set evaluation:** Precision, Recall, F1, AUC-ROC, SHAP, spatial accuracy check
3. **OOT evaluation:** same metrics + PSI + CSI (drift detection)
4. **Promotion gate** — all four must pass:
   - Recall ≥ 0.70 on test set
   - OOT recall drop ≤ 10 percentage points vs test set
   - SHAP interpretable (top features make domain sense)
   - Spatial accuracy check passes
5. Pass → register as `Production` in MLflow
6. Fail → previous `Production` model remains active

---

## Model registry

MLflow stores models as folders:
```
dengue_cluster_model/
└── Production/
    ├── model.xgb
    ├── MLmodel
    ├── conda.yaml
    └── requirements.txt
```

Both Airflow and FastAPI load via:
```python
model = mlflow.xgboost.load_model("models:/dengue_cluster_model/Production")
```

---

## Redis online feature store

Two features only — updated on every new confirmed case:
- `current_week_case_count:{subzone_id}`
- `vulnerability_index:{subzone_id}` (pre-loaded weekly from Gold)

FastAPI updates Redis on LISTEN/NOTIFY event, then reads from it for scoring.

---

## Files to build

### Data pipeline
- `ingest_sgcharts.py`
- `ingest_mss_weather.py`
- `ingest_population.py`
- `ingest_geodata.py`
- `preprocess.py`
- `feature_engineering.py`

### Model
- `train.py`
- `evaluate.py`

### Inference
- `realtime_inference.py` (FastAPI app)
- `airflow/dags/batch_dag.py`
- `airflow/dags/retrain_dag.py`

### Monitoring
- `monitor.py` (PSI/CSI tracking, drift alarm, retrain trigger)

### Infrastructure
- `docker-compose.yml` (Postgres, Redis, Airflow, MLflow, FastAPI)
- `Dockerfile` (FastAPI)
- `requirements.txt`
- `.env` (connection strings)

---

## Design decisions (locked)

| Decision | Rationale |
|---|---|
| One model, two paths | Same cost function, different serving contracts |
| XGBoost + LightGBM | Tabular data, SHAP interpretability required |
| Time-series CV | Standard k-fold leaks future data |
| Recall ≥ 0.70 | False negative = missed outbreak (asymmetric cost) |
| 330 subzones not 55 planning areas | Operational precision for fogging deployment |
| MSS weather not NEA API | MSS extends back to 2013 (3 extra years of training data) |
| Alert is a rule not a model | No public outcome labels for individual cases |
| Local Docker | No cloud spend; architecture is cloud-portable |

---

## Production equivalent (AWS)

| Local | AWS |
|---|---|
| Postgres in Docker | AWS RDS (Postgres) |
| Redis in Docker | AWS ElastiCache (Redis) |
| Airflow in Docker | AWS MWAA |
| MLflow in Docker | MLflow on EC2 |
| FastAPI in Docker | AWS ECS (Fargate) |
| Parquet on disk | AWS S3 |
| Alert write to DB | AWS SNS → mobile push |
| NEA dashboard | AWS QuickSight or web app on EC2 |

---

## Key justification lines (preserve in report/presentation)

- *"The cost of a false negative in the batch model is a missed fogging run. The cost of a false negative in the real-time model is a preventable death."*
- *"One trained artifact — served two ways."*
- *"The alert rule is transparent business logic, not a second model. No public outcome labels exist for individual cases."*
- *"MSS station records extend back to 2013. The NEA API only starts at December 2016 — switching would cost us three years of training data."*
- *"We allow a maximum 10 percentage point degradation on OOT before flagging for retraining."*

---

## Environment

- Python 3.14 at `/Users/garethwang/Library/Python/3.14/`
- Required packages: `geopandas`, `pyspark`, `xgboost`, `lightgbm`, `mlflow`, `optuna`, `fastapi`, `redis`, `psycopg2`, `pandas`, `numpy`, `shap`, `imbalanced-learn`, `sqlalchemy`, `uvicorn`
- Dev: MacBook (primary), VSCode, GitHub
- Docker + docker-compose for local stack

---

*Generated from CS611 MLE project conversation — May 2026*
