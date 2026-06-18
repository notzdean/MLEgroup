-- CS611 MLE Group Project — Postgres schema initialisation
-- Runs automatically on first docker-compose up

-- MLflow and Airflow each need their own database
SELECT 'CREATE DATABASE mlflow' WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'mlflow')\gexec
SELECT 'CREATE DATABASE airflow' WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'airflow')\gexec

-- Schemas
CREATE SCHEMA IF NOT EXISTS bronze;
CREATE SCHEMA IF NOT EXISTS silver;
CREATE SCHEMA IF NOT EXISTS gold;
CREATE SCHEMA IF NOT EXISTS operational;

-- Bronze tables
CREATE TABLE IF NOT EXISTS bronze.raw_clusters (
    id              SERIAL PRIMARY KEY,
    date            DATE,
    cluster_id      INTEGER,
    location        TEXT,
    latitude        FLOAT,
    longitude       FLOAT,
    cases           INTEGER,
    recent_cases    INTEGER,
    total_cases     INTEGER,
    source_file     TEXT,
    source_folder   TEXT,
    ingested_at     TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS bronze.raw_weather (
    id              SERIAL PRIMARY KEY,
    date            DATE,
    station_id      TEXT,
    station_name    TEXT,
    rainfall_mm     FLOAT,
    ingested_at     TIMESTAMP DEFAULT NOW()
);

-- Gold offline feature store
CREATE TABLE IF NOT EXISTS gold.subzone_features (
    id                          SERIAL PRIMARY KEY,
    date                        DATE NOT NULL,
    subzone_name                TEXT NOT NULL,
    rainfall_lag1w              FLOAT,
    rainfall_lag2w              FLOAT,
    rainfall_lag4w              FLOAT,
    cluster_count_rolling2w     FLOAT,
    cluster_count_rolling4w     FLOAT,
    population                  INTEGER,
    elderly_pct                 FLOAT,
    area_km2                    FLOAT,
    population_density          FLOAT,
    vulnerability_index         FLOAT,
    active_cluster_overlap      INTEGER,
    label                       INTEGER,
    created_at                  TIMESTAMP DEFAULT NOW(),
    UNIQUE (date, subzone_name)
);

-- Operational tables
CREATE TABLE IF NOT EXISTS operational.confirmed_cases (
    id              SERIAL PRIMARY KEY,
    subzone_name    TEXT NOT NULL,
    case_count      INTEGER DEFAULT 1,
    reported_at     TIMESTAMP DEFAULT NOW(),
    source          TEXT
);

CREATE TABLE IF NOT EXISTS operational.risk_tier (
    id              SERIAL PRIMARY KEY,
    scored_at       TIMESTAMP DEFAULT NOW(),
    subzone_name    TEXT NOT NULL,
    score           FLOAT,
    risk_tier       TEXT CHECK (risk_tier IN ('Low', 'Medium', 'High')),
    split           TEXT,
    week_start      DATE
);

CREATE TABLE IF NOT EXISTS operational.vulnerability_alerts (
    id              SERIAL PRIMARY KEY,
    alerted_at      TIMESTAMP DEFAULT NOW(),
    subzone_name    TEXT NOT NULL,
    score           FLOAT,
    vulnerability_index FLOAT,
    case_count      INTEGER,
    alert_score     FLOAT,
    action          TEXT
);

CREATE TABLE IF NOT EXISTS operational.drift_alarms (
    id              SERIAL PRIMARY KEY,
    alarm_time      TIMESTAMP DEFAULT NOW(),
    psi             FLOAT,
    psi_flag        TEXT,
    csi_json        JSONB,
    action          TEXT
);

-- Shadow deployment scoring table
-- Candidate model scores real traffic silently — no alerts fired
-- Used to compare candidate vs production before promotion
CREATE TABLE IF NOT EXISTS operational.shadow_scores (
    id                      SERIAL PRIMARY KEY,
    scored_at               TIMESTAMP DEFAULT NOW(),
    subzone_name            TEXT NOT NULL,
    production_score        FLOAT,
    shadow_score            FLOAT,
    shadow_model_version    TEXT,
    case_count              INTEGER,
    production_tier         TEXT,
    shadow_tier             TEXT,
    agreement               BOOLEAN   -- True if both models agree on High vs not-High
);

CREATE INDEX IF NOT EXISTS idx_shadow_scores_subzone ON operational.shadow_scores (subzone_name);
CREATE INDEX IF NOT EXISTS idx_shadow_scores_scored_at ON operational.shadow_scores (scored_at);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_risk_tier_week ON operational.risk_tier (week_start);
CREATE INDEX IF NOT EXISTS idx_confirmed_cases_subzone ON operational.confirmed_cases (subzone_name);
CREATE INDEX IF NOT EXISTS idx_subzone_features_date ON gold.subzone_features (date);

-- MLflow and Airflow use their own databases (created separately)
