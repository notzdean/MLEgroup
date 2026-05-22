"""
inference/realtime_inference.py
================================
CS611 MLE Group Project — Dengue Outbreak Risk Prediction
Real-time inference service — FastAPI.

Flow
----
1. Hospital submits confirmed case via POST /cases/confirmed
2. FastAPI writes to operational.confirmed_cases (Postgres)
3. Postgres LISTEN/NOTIFY fires on confirmed_cases INSERT trigger
4. FastAPI listener updates Redis: current_week_case_count for subzone
5. FastAPI reads online features from Redis + static features from Gold
6. Scores subzone with Production model (loaded at startup, kept in memory)
7. Computes alert score: model_score × vulnerability_index × log(case_count+1)
8. If alert_score > threshold: writes to operational.vulnerability_alerts
9. Returns score and alert decision to caller

Target latency: < 500ms end-to-end

Design decisions
----------------
- Model loaded at startup and kept in memory — avoids cold start on each request
- Redis holds only two online features per subzone (case_count, vulnerability_index)
- Alert is a rule not a model — no public outcome labels for individual cases
- Vulnerability index loaded weekly from Gold by batch DAG
"""

import asyncio
import json
import logging
import math
import os
from contextlib import asynccontextmanager
from datetime import datetime

import numpy as np
import pandas as pd
import redis
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from sqlalchemy import create_engine, text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
DATABASE_URL    = os.getenv("DATABASE_URL", "postgresql://dengue:dengue@localhost:5432/dengue")
REDIS_URL       = os.getenv("REDIS_URL", "redis://localhost:6379/0")
MLFLOW_URI      = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME      = os.getenv("MODEL_NAME", "dengue_cluster_model")
MODEL_STAGE     = os.getenv("MODEL_STAGE", "Production")
ALERT_THRESHOLD = float(os.getenv("ALERT_THRESHOLD", "0.6"))
GOLD_PARQUET    = os.getenv("GOLD_PARQUET", "/app/data/gold/subzone_features.parquet")
MODEL_DIR       = os.getenv("MODEL_DIR", "/app/model/candidate")

FEATURES = [
    "rainfall_lag1w", "rainfall_lag2w", "rainfall_lag4w",
    "cluster_count_rolling2w", "cluster_count_rolling4w",
    "population", "elderly_pct", "area_km2", "population_density",
    "vulnerability_index",
]

# ── Global state ──────────────────────────────────────────────────────────────
app_state = {
    "model":        None,
    "model_type":   None,
    "redis_client": None,
    "engine":       None,
    "gold_features": None,   # static features per subzone (latest snapshot)
}


# ── Model loader ──────────────────────────────────────────────────────────────

def load_model():
    """Load Production model — try MLflow first, fall back to local file."""
    try:
        import mlflow
        mlflow.set_tracking_uri(MLFLOW_URI)
        model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{MODEL_STAGE}")
        logger.info(f"Model loaded from MLflow: {MODEL_NAME}/{MODEL_STAGE}")
        return model, "mlflow"
    except Exception as e:
        logger.warning(f"MLflow unavailable ({e}) — loading local candidate model")

    try:
        meta_path = f"{MODEL_DIR}/candidate_meta.json"
        with open(meta_path) as f:
            meta = json.load(f)
        model_type = meta["model_type"]

        if model_type == "xgboost":
            import xgboost as xgb
            model = xgb.XGBClassifier()
            model.load_model(f"{MODEL_DIR}/model.xgb")
            logger.info("XGBoost model loaded from local file")
            return model, "xgboost"
        else:
            import lightgbm as lgb
            model = lgb.Booster(model_file=f"{MODEL_DIR}/model.lgb")
            logger.info("LightGBM model loaded from local file")
            return model, "lightgbm"
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


def load_gold_features():
    """Load latest static features per subzone from Gold parquet."""
    try:
        df = pd.read_parquet(GOLD_PARQUET)
        df["date"] = pd.to_datetime(df["date"])
        # Take the most recent snapshot per subzone
        latest = df.sort_values("date").groupby("subzone_name").last().reset_index()
        logger.info(f"Gold features loaded: {len(latest)} subzones")
        return latest.set_index("subzone_name")
    except Exception as e:
        logger.warning(f"Could not load Gold features ({e}) — will use zeros")
        return pd.DataFrame()


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and connections at startup, clean up at shutdown."""
    logger.info("Starting up — loading model and connections")

    # Model
    model, model_type = load_model()
    app_state["model"]      = model
    app_state["model_type"] = model_type

    # Gold static features
    app_state["gold_features"] = load_gold_features()

    # Redis
    try:
        r = redis.from_url(REDIS_URL, decode_responses=True)
        r.ping()
        app_state["redis_client"] = r
        logger.info("Redis connected")
    except Exception as e:
        logger.warning(f"Redis unavailable ({e}) — running without online store")

    # Postgres
    try:
        engine = create_engine(DATABASE_URL)
        app_state["engine"] = engine
        logger.info("Postgres connected")

        # Start LISTEN/NOTIFY listener in background
        asyncio.create_task(listen_for_cases())
    except Exception as e:
        logger.warning(f"Postgres unavailable ({e})")

    yield

    logger.info("Shutting down")
    if app_state["engine"]:
        app_state["engine"].dispose()


app = FastAPI(
    title="Dengue Outbreak Risk — Real-time Inference",
    description="Scores subzones on confirmed case events. CS611 Group 4.",
    version="1.0.0",
    lifespan=lifespan
)


# ── Schemas ───────────────────────────────────────────────────────────────────

class ConfirmedCase(BaseModel):
    subzone_name: str
    case_count:   int = 1
    source:       str = "hospital"


class ScoreResponse(BaseModel):
    subzone_name:       str
    model_score:        float
    vulnerability_index: float
    current_case_count: int
    alert_score:        float
    alert_triggered:    bool
    latency_ms:         float


# ── Scoring ───────────────────────────────────────────────────────────────────

def get_online_features(subzone_name: str) -> dict:
    """Read current_week_case_count and vulnerability_index from Redis."""
    r = app_state["redis_client"]
    if r is None:
        return {"case_count": 1, "vulnerability_index": 0.5}

    case_count = r.get(f"current_week_case_count:{subzone_name}")
    vuln_index = r.get(f"vulnerability_index:{subzone_name}")

    return {
        "case_count":        int(case_count)   if case_count else 1,
        "vulnerability_index": float(vuln_index) if vuln_index else 0.5,
    }


def get_static_features(subzone_name: str) -> np.ndarray:
    """Get latest static features for a subzone from Gold parquet."""
    gold = app_state["gold_features"]
    if gold.empty or subzone_name not in gold.index:
        return np.zeros(len(FEATURES))

    row = gold.loc[subzone_name]
    return np.array([row.get(f, 0) for f in FEATURES], dtype=float)


def score_subzone(subzone_name: str) -> tuple[float, dict]:
    """Score a subzone using the loaded model + online features."""
    model      = app_state["model"]
    model_type = app_state["model_type"]

    # Get features
    online  = get_online_features(subzone_name)
    X       = get_static_features(subzone_name).reshape(1, -1)

    # Override vulnerability_index with fresh Redis value
    vuln_idx = FEATURES.index("vulnerability_index") if "vulnerability_index" in FEATURES else -1
    if vuln_idx >= 0:
        X[0, vuln_idx] = online["vulnerability_index"]

    # Score
    if model_type == "lightgbm":
        import lightgbm as lgb
        score = float(model.predict(X)[0])
    elif model_type == "mlflow":
        score = float(model.predict(pd.DataFrame(X, columns=FEATURES))[0])
    else:
        score = float(model.predict_proba(X)[0, 1])

    return score, online


def compute_alert_score(model_score: float, vulnerability_index: float, case_count: int) -> float:
    """
    Alert rule: score × vulnerability_index × log(case_count + 1)

    This is transparent business logic, not a second model.
    No public outcome labels exist for individual cases.
    The log transform prevents a single high case count from dominating.
    """
    return round(model_score * vulnerability_index * math.log(case_count + 1), 4)


# ── Background tasks ──────────────────────────────────────────────────────────

async def listen_for_cases():
    """
    Listen for Postgres NOTIFY on confirmed_cases channel.
    Updates Redis case count on each INSERT.
    """
    import psycopg2
    try:
        conn = psycopg2.connect(DATABASE_URL)
        conn.set_isolation_level(0)  # autocommit for LISTEN
        cur = conn.cursor()
        cur.execute("LISTEN confirmed_cases;")
        logger.info("Listening on Postgres channel: confirmed_cases")

        while True:
            conn.poll()
            while conn.notifies:
                notify = conn.notifies.pop(0)
                payload = json.loads(notify.payload) if notify.payload else {}
                subzone = payload.get("subzone_name")
                if subzone and app_state["redis_client"]:
                    r = app_state["redis_client"]
                    r.incr(f"current_week_case_count:{subzone}")
                    logger.info(f"Case count incremented for {subzone}")
            await asyncio.sleep(0.1)

    except Exception as e:
        logger.warning(f"LISTEN/NOTIFY failed ({e}) — real-time updates unavailable")


def write_alert_to_postgres(subzone_name: str, score: float,
                             vulnerability_index: float, case_count: int,
                             alert_score: float):
    """Write alert to operational.vulnerability_alerts."""
    engine = app_state["engine"]
    if engine is None:
        return
    try:
        with engine.connect() as conn:
            conn.execute(text("""
                INSERT INTO operational.vulnerability_alerts
                    (subzone_name, score, vulnerability_index, case_count, alert_score, action)
                VALUES
                    (:subzone, :score, :vuln, :cases, :alert, 'alert_sent')
            """), {
                "subzone": subzone_name,
                "score":   score,
                "vuln":    vulnerability_index,
                "cases":   case_count,
                "alert":   alert_score
            })
            conn.commit()
    except Exception as e:
        logger.warning(f"Could not write alert to Postgres: {e}")


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status":     "ok",
        "model":      app_state["model_type"],
        "redis":      app_state["redis_client"] is not None,
        "postgres":   app_state["engine"] is not None,
    }


@app.post("/cases/confirmed", response_model=ScoreResponse)
def submit_confirmed_case(case: ConfirmedCase, background_tasks: BackgroundTasks):
    """
    Submit a confirmed dengue case.
    Scores the subzone and triggers alert if threshold exceeded.
    Target: < 500ms end-to-end.
    """
    start = datetime.now()

    # Write to Postgres (triggers LISTEN/NOTIFY)
    engine = app_state["engine"]
    if engine:
        try:
            with engine.connect() as conn:
                conn.execute(text("""
                    INSERT INTO operational.confirmed_cases
                        (subzone_name, case_count, source)
                    VALUES (:subzone, :count, :source)
                """), {
                    "subzone": case.subzone_name,
                    "count":   case.case_count,
                    "source":  case.source
                })
                conn.commit()
        except Exception as e:
            logger.warning(f"Could not write confirmed case: {e}")

    # Update Redis directly (belt + suspenders alongside LISTEN/NOTIFY)
    if app_state["redis_client"]:
        app_state["redis_client"].incr(f"current_week_case_count:{case.subzone_name}")

    # Score
    model_score, online = score_subzone(case.subzone_name)
    vulnerability_index = online["vulnerability_index"]
    case_count          = online["case_count"]

    # Alert rule
    alert_score     = compute_alert_score(model_score, vulnerability_index, case_count)
    alert_triggered = alert_score > ALERT_THRESHOLD

    # Write alert in background (keeps latency low)
    if alert_triggered:
        background_tasks.add_task(
            write_alert_to_postgres,
            case.subzone_name, model_score,
            vulnerability_index, case_count, alert_score
        )
        logger.info(f"ALERT triggered for {case.subzone_name} — score {alert_score:.4f}")

    latency_ms = (datetime.now() - start).total_seconds() * 1000

    return ScoreResponse(
        subzone_name        = case.subzone_name,
        model_score         = round(model_score, 4),
        vulnerability_index = round(vulnerability_index, 4),
        current_case_count  = case_count,
        alert_score         = alert_score,
        alert_triggered     = alert_triggered,
        latency_ms          = round(latency_ms, 2),
    )


@app.get("/score/{subzone_name}", response_model=ScoreResponse)
def get_subzone_score(subzone_name: str):
    """Get current risk score for a subzone on demand."""
    start = datetime.now()
    model_score, online = score_subzone(subzone_name)
    vulnerability_index = online["vulnerability_index"]
    case_count          = online["case_count"]
    alert_score         = compute_alert_score(model_score, vulnerability_index, case_count)
    latency_ms          = (datetime.now() - start).total_seconds() * 1000

    return ScoreResponse(
        subzone_name        = subzone_name,
        model_score         = round(model_score, 4),
        vulnerability_index = round(vulnerability_index, 4),
        current_case_count  = case_count,
        alert_score         = alert_score,
        alert_triggered     = alert_score > ALERT_THRESHOLD,
        latency_ms          = round(latency_ms, 2),
    )


@app.get("/subzones/high-risk")
def get_high_risk_subzones(threshold: float = 0.5):
    """Return all subzones currently above a risk threshold."""
    gold = app_state["gold_features"]
    if gold.empty:
        raise HTTPException(status_code=503, detail="Gold features not loaded")

    results = []
    for subzone_name in gold.index:
        model_score, online = score_subzone(subzone_name)
        if model_score >= threshold:
            results.append({
                "subzone_name": subzone_name,
                "model_score":  round(model_score, 4),
                "vulnerability_index": round(online["vulnerability_index"], 4),
            })

    results.sort(key=lambda x: x["model_score"], reverse=True)
    return {"high_risk_subzones": results, "count": len(results)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
