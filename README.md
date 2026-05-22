# CS611 MLE — Dengue Outbreak Risk Prediction
**Group 4** | SMU SCIS CS611 Machine Learning Engineering

Predicts which of Singapore's 330 residential subzones will have an active dengue cluster in the next 14 days. One trained model, two inference paths: weekly batch for NEA Ops + real-time alert for CDA Officer at MOH.

---

## Quickstart

### 1. Prerequisites
- Python 3.11+
- Docker Desktop
- Git

### 2. Clone and setup
```bash
git clone https://github.com/notzdean/MLEgroup.git
cd MLEgroup
cp .env.example .env          # fill in passwords
pip install -r requirements.txt
```

### 3. Place data files
Copy the five source files into `data/bronze/`:
```
data/bronze/raw_sgcharts_dengue.csv
data/bronze/raw_mss_weather_2013_2020.csv
data/bronze/raw_mss_stations_list_mss.csv
data/bronze/raw_singstat_pop_17560.csv
data/bronze/raw_MasterPlan2019SubzoneBoundaryNoSeaGEOJSON.geojson
```

### 4. Run the pipeline
```bash
# Bronze ingestion
python pipeline/ingest_sgcharts.py
python pipeline/ingest_mss_weather.py
python pipeline/ingest_population.py
python pipeline/ingest_geodata.py

# Silver cleaning
python pipeline/preprocess.py

# Gold features
python pipeline/feature_engineering.py

# Train + evaluate
python model/train.py
python model/evaluate.py

# Monitor
python monitoring/monitor.py
```

### 5. Start Docker stack
```bash
docker-compose up -d
```

Services:
| Service | URL |
|---|---|
| MLflow | http://localhost:5000 |
| Airflow | http://localhost:8080 |
| FastAPI | http://localhost:8000/docs |

---

## Project structure
```
cs611-dengue/
├── data/
│   ├── bronze/          # raw ingested files
│   ├── silver/          # cleaned files
│   └── gold/            # feature store (parquet)
├── pipeline/            # ingest + preprocess + feature engineering
├── model/               # train + evaluate
├── inference/           # FastAPI + Airflow DAGs
│   └── airflow/
├── monitoring/          # PSI/CSI drift detection
├── notebooks/           # EDA only
├── tests/
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── .env.example
```

---

## Design decisions
| Decision | Rationale |
|---|---|
| One model, two paths | Same cost function, different serving contracts |
| XGBoost + LightGBM | Tabular data, SHAP interpretable, trains locally |
| Recall ≥ 0.70 target | False negative = missed outbreak (asymmetric cost) |
| Time-series CV | Standard k-fold leaks future cluster data |
| MSS weather not NEA API | MSS extends back to 2013 — 3 extra years of training data |
| Alert is a rule not a model | No public outcome labels for individual cases |
| PCA vulnerability weights | Data-driven weighting over manual 60/40 assumption |
