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
from typing import AsyncGenerator

import numpy as np
import pandas as pd
import redis
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
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
    "recent_cases_rolling2w", "recent_cases_rolling4w",
    "population", "elderly_pct", "area_km2", "population_density",
    "vulnerability_index",
]

# ── Global state ──────────────────────────────────────────────────────────────
app_state = {
    "model":               None,
    "model_type":          None,
    "redis_client":        None,
    "engine":              None,
    "gold_features":       None,
    # Shadow deployment — candidate model runs silently on live traffic
    "shadow_model":        None,
    "shadow_model_type":   None,
    "shadow_model_version": "candidate",
    "shadow_enabled":      False,
}


# ── SSE alert broadcast ───────────────────────────────────────────────────────
# Each connected /alerts/stream client gets its own asyncio.Queue.
# When an alert fires, broadcast_alert() pushes to every queue.
# The generator yields SSE-formatted events until the client disconnects.

_alert_subscribers: set[asyncio.Queue] = set()


def broadcast_alert(payload: dict):
    """Push alert payload to all connected SSE clients (CDC Officer devices)."""
    for q in _alert_subscribers:
        try:
            q.put_nowait(payload)
        except asyncio.QueueFull:
            pass


async def alert_event_generator() -> AsyncGenerator[str, None]:
    """SSE generator — keeps HTTP connection open, streams alerts as they arrive."""
    q: asyncio.Queue = asyncio.Queue(maxsize=20)
    _alert_subscribers.add(q)
    try:
        # Send a heartbeat immediately so the client knows the connection is live
        yield "event: connected\ndata: {\"status\": \"monitoring\"}\n\n"
        while True:
            try:
                payload = await asyncio.wait_for(q.get(), timeout=25)
                yield f"event: alert\ndata: {json.dumps(payload)}\n\n"
            except asyncio.TimeoutError:
                # Heartbeat every 25s to keep connection alive through proxies
                yield "event: heartbeat\ndata: {}\n\n"
    finally:
        _alert_subscribers.discard(q)


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


def load_shadow_model():
    """
    Load shadow (candidate) model for silent parallel scoring.
    Tries MLflow Staging stage first, falls back to local candidate file.
    Shadow model scores every live case but never fires alerts.
    """
    try:
        import mlflow
        mlflow.set_tracking_uri(MLFLOW_URI)
        model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/Staging")
        logger.info("Shadow model loaded from MLflow Staging")
        return model, "mlflow", "Staging"
    except Exception:
        pass

    try:
        import json
        meta_path = f"{MODEL_DIR}/candidate_meta.json"
        with open(meta_path) as f:
            meta = json.load(f)
        model_type = meta["model_type"]

        if model_type == "xgboost":
            import xgboost as xgb
            model = xgb.XGBClassifier()
            model.load_model(f"{MODEL_DIR}/model.xgb")
            logger.info("Shadow model loaded from local candidate (xgboost)")
            return model, "xgboost", "candidate-local"
        else:
            import lightgbm as lgb
            model = lgb.Booster(model_file=f"{MODEL_DIR}/model.lgb")
            logger.info("Shadow model loaded from local candidate (lightgbm)")
            return model, "lightgbm", "candidate-local"
    except Exception as e:
        logger.warning(f"Shadow model unavailable ({e}) — shadow scoring disabled")
        return None, None, None


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

    # Production model
    model, model_type = load_model()
    app_state["model"]      = model
    app_state["model_type"] = model_type

    # Shadow (candidate) model — silent parallel scoring
    shadow_model, shadow_type, shadow_version = load_shadow_model()
    app_state["shadow_model"]         = shadow_model
    app_state["shadow_model_type"]    = shadow_type
    app_state["shadow_model_version"] = shadow_version
    app_state["shadow_enabled"]       = shadow_model is not None
    if app_state["shadow_enabled"]:
        logger.info(f"Shadow deployment enabled — version: {shadow_version}")

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


# ── Shadow deployment ─────────────────────────────────────────────────────────

def score_with_shadow(subzone_name: str) -> float | None:
    """Score subzone with shadow (candidate) model. Returns None if disabled."""
    if not app_state["shadow_enabled"]:
        return None
    try:
        model      = app_state["shadow_model"]
        model_type = app_state["shadow_model_type"]
        X          = get_static_features(subzone_name).reshape(1, -1)

        online = get_online_features(subzone_name)
        vuln_idx = FEATURES.index("vulnerability_index") if "vulnerability_index" in FEATURES else -1
        if vuln_idx >= 0:
            X[0, vuln_idx] = online["vulnerability_index"]

        if model_type == "lightgbm":
            import lightgbm as lgb
            return float(model.predict(X)[0])
        elif model_type == "mlflow":
            return float(model.predict(pd.DataFrame(X, columns=FEATURES))[0])
        else:
            return float(model.predict_proba(X)[0, 1])
    except Exception as e:
        logger.warning(f"Shadow scoring failed for {subzone_name}: {e}")
        return None


def write_shadow_score(subzone_name: str, production_score: float,
                       shadow_score: float, case_count: int):
    """
    Write shadow vs production scores to operational.shadow_scores.
    Called as a background task — never blocks the alert path.
    Agreement = both models agree on High (>=0.6) vs not-High.
    """
    engine = app_state["engine"]
    if engine is None:
        return
    try:
        prod_tier   = "High" if production_score >= 0.6 else ("Medium" if production_score >= 0.3 else "Low")
        shadow_tier = "High" if shadow_score >= 0.6 else ("Medium" if shadow_score >= 0.3 else "Low")
        agreement   = (production_score >= 0.6) == (shadow_score >= 0.6)

        with engine.connect() as conn:
            conn.execute(text("""
                INSERT INTO operational.shadow_scores
                    (subzone_name, production_score, shadow_score,
                     shadow_model_version, case_count,
                     production_tier, shadow_tier, agreement)
                VALUES
                    (:subzone, :prod, :shadow, :version,
                     :cases, :prod_tier, :shadow_tier, :agreement)
            """), {
                "subzone":    subzone_name,
                "prod":       production_score,
                "shadow":     shadow_score,
                "version":    app_state["shadow_model_version"],
                "cases":      case_count,
                "prod_tier":  prod_tier,
                "shadow_tier": shadow_tier,
                "agreement":  agreement,
            })
            conn.commit()
    except Exception as e:
        logger.warning(f"Could not write shadow score: {e}")


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
        "status":          "ok",
        "model":           app_state["model_type"],
        "redis":           app_state["redis_client"] is not None,
        "postgres":        app_state["engine"] is not None,
        "shadow_enabled":  app_state["shadow_enabled"],
        "shadow_version":  app_state["shadow_model_version"],
        "sse_subscribers": len(_alert_subscribers),
    }


@app.get("/alerts/stream")
async def alerts_stream():
    """
    Server-Sent Events endpoint for CDC Officer mobile devices.
    Keeps HTTP connection open. Pushes alert events the instant a hospital
    submits a confirmed case that crosses the alert threshold.
    No polling needed — pure push from server to client.
    """
    return StreamingResponse(
        alert_event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # disable nginx buffering
        },
    )


@app.get("/shadow/comparison")
def shadow_comparison():
    """
    Compare shadow (candidate) vs production model on recent live traffic.
    Returns agreement rate, score differences, and tier-level breakdown.
    Use this to decide whether to promote the candidate to production.
    Promote when: agreement >= 90% sustained over 7+ days.
    """
    engine = app_state["engine"]
    if engine is None:
        return {"error": "Postgres not connected"}

    try:
        with engine.connect() as conn:
            rows = conn.execute(text("""
                SELECT
                    COUNT(*)                                        AS total_cases,
                    ROUND(AVG(CASE WHEN agreement THEN 1.0 ELSE 0.0 END) * 100, 1)
                                                                    AS agreement_pct,
                    ROUND(AVG(ABS(production_score - shadow_score))::numeric, 4)
                                                                    AS avg_score_diff,
                    ROUND(MAX(ABS(production_score - shadow_score))::numeric, 4)
                                                                    AS max_score_diff,
                    SUM(CASE WHEN production_tier = 'High' AND shadow_tier != 'High'
                             THEN 1 ELSE 0 END)                     AS prod_high_shadow_missed,
                    SUM(CASE WHEN shadow_tier = 'High' AND production_tier != 'High'
                             THEN 1 ELSE 0 END)                     AS shadow_high_prod_missed,
                    MIN(scored_at)                                  AS window_start,
                    MAX(scored_at)                                  AS window_end,
                    COUNT(DISTINCT shadow_model_version)            AS shadow_versions
                FROM operational.shadow_scores
                WHERE scored_at >= NOW() - INTERVAL '7 days'
            """)).fetchone()

        if not rows or rows[0] == 0:
            return {
                "message": "No shadow scores yet — submit cases via /cases/confirmed to populate",
                "shadow_enabled": app_state["shadow_enabled"],
            }

        agreement_pct = float(rows[1] or 0)
        recommendation = (
            "READY TO PROMOTE — agreement >= 90% sustained"
            if agreement_pct >= 90 else
            "NOT READY — monitor for more traffic before promoting"
        )

        return {
            "window":                 "last 7 days",
            "total_cases_scored":     rows[0],
            "agreement_pct":          agreement_pct,
            "avg_score_diff":         float(rows[2] or 0),
            "max_score_diff":         float(rows[3] or 0),
            "prod_high_shadow_missed": rows[4],
            "shadow_high_prod_missed": rows[5],
            "window_start":           str(rows[6]),
            "window_end":             str(rows[7]),
            "shadow_model_version":   app_state["shadow_model_version"],
            "recommendation":         recommendation,
        }
    except Exception as e:
        return {"error": str(e)}


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
        # Push to all connected CDC Officer SSE clients instantly
        broadcast_alert({
            "subzone_name":       case.subzone_name,
            "model_score":        round(model_score, 4),
            "alert_score":        alert_score,
            "vulnerability_index": round(vulnerability_index, 4),
            "case_count":         case_count,
            "source":             case.source,
            "timestamp":          datetime.now().isoformat(),
        })

    # Shadow deployment — score silently with candidate model, never fires alert
    if app_state["shadow_enabled"]:
        shadow_score = score_with_shadow(case.subzone_name)
        if shadow_score is not None:
            background_tasks.add_task(
                write_shadow_score,
                case.subzone_name, model_score, shadow_score, case_count
            )

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


# ── Dashboard & Mobile UI endpoints ──────────────────────────────────────────

import json as json_module
from fastapi.responses import HTMLResponse, JSONResponse


GEOJSON_PATH = os.getenv("GEOJSON_PATH", "/app/data/gold/../../../data/bronze/BRONZE_subzone_geodata.geojson")
# fallback paths to try
GEOJSON_CANDIDATES = [
    "/app/data/bronze/BRONZE_subzone_geodata.geojson",
    "/app/data/bronze/raw_MasterPlan2019SubzoneBoundaryNoSeaGEOJSON.geojson",
    "data/bronze/BRONZE_subzone_geodata.geojson",
    "data/bronze/raw_MasterPlan2019SubzoneBoundaryNoSeaGEOJSON.geojson",
]


def _load_geojson() -> dict:
    for path in GEOJSON_CANDIDATES:
        try:
            with open(path, encoding="utf-8") as f:
                return json_module.load(f)
        except FileNotFoundError:
            continue
    return {"type": "FeatureCollection", "features": []}


@app.get("/geojson")
def serve_geojson():
    """Serve Singapore subzone GeoJSON for the Leaflet map."""
    gj = _load_geojson()
    return JSONResponse(content=gj)


@app.get("/scores")
def serve_scores():
    """
    Return current risk scores for all subzones as JSON.
    Called by the dashboard map every 30s to update colours.
    """
    gold = app_state["gold_features"]
    if gold.empty:
        return JSONResponse(content={})

    scores = {}
    for subzone in gold.index:
        try:
            score, online = score_subzone(subzone)
            tier = "High" if score >= 0.6 else ("Medium" if score >= 0.3 else "Low")
            scores[subzone] = {
                "score": round(score, 4),
                "tier": tier,
                "vulnerability_index": round(online.get("vulnerability_index", 0), 4),
                "case_count": online.get("case_count", 0),
            }
        except Exception:
            continue
    return JSONResponse(content=scores)


@app.get("/dashboard", response_class=HTMLResponse)
def batch_dashboard():
    """
    NEA Operations batch inference dashboard.
    Tab 1: Leaflet choropleth map of Singapore subzones.
    Tab 2: Charts — score distribution, top 20, tier donut, scatter, case counts.
    Auto-refreshes every 30 seconds.
    """
    html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Dengue Risk Dashboard — NEA Operations</title>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap" rel="stylesheet">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.css"/>
<script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
  :root {
    --bg: #0a0f1e; --surface: #111827; --border: #1f2937;
    --text: #e5e7eb; --muted: #6b7280; --accent: #06b6d4;
    --high: #ef4444; --med: #f59e0b; --low: #10b981;
  }
  * { margin:0; padding:0; box-sizing:border-box; }
  body { background:var(--bg); color:var(--text); font-family:'IBM Plex Sans',sans-serif; height:100vh; display:flex; flex-direction:column; }

  .header {
    background:var(--surface); border-bottom:1px solid var(--border);
    padding:14px 24px; display:flex; align-items:center; justify-content:space-between;
    flex-shrink:0;
  }
  .header-left { display:flex; align-items:center; gap:12px; }
  .pulse { width:8px; height:8px; border-radius:50%; background:var(--low); animation:pulse 2s infinite; flex-shrink:0; }
  @keyframes pulse { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:.5;transform:scale(1.4)} }
  .title { font-family:'IBM Plex Mono',monospace; font-size:15px; font-weight:600; letter-spacing:.04em; }
  .subtitle { font-size:11px; color:var(--muted); margin-top:1px; }
  .header-right { display:flex; align-items:center; gap:16px; }
  .stat-pill {
    background:var(--bg); border:1px solid var(--border); border-radius:6px;
    padding:5px 12px; display:flex; align-items:center; gap:8px;
  }
  .stat-pill .dot { width:6px; height:6px; border-radius:50%; flex-shrink:0; }
  .stat-pill .label { font-size:10px; color:var(--muted); text-transform:uppercase; letter-spacing:.06em; }
  .stat-pill .val { font-family:'IBM Plex Mono',monospace; font-size:15px; font-weight:600; }
  .timestamp { font-family:'IBM Plex Mono',monospace; font-size:11px; color:var(--muted); }

  /* Tabs */
  .tab-bar {
    background:var(--surface); border-bottom:1px solid var(--border);
    display:flex; gap:0; flex-shrink:0;
  }
  .tab-btn {
    padding:10px 24px; font-size:13px; font-family:'IBM Plex Mono',monospace;
    color:var(--muted); background:none; border:none; cursor:pointer;
    border-bottom:2px solid transparent; transition:all .2s; letter-spacing:.04em;
  }
  .tab-btn:hover { color:var(--text); }
  .tab-btn.active { color:var(--accent); border-bottom-color:var(--accent); }

  .tab-content { display:none; flex:1; overflow:hidden; }
  .tab-content.active { display:flex; }

  /* Map tab */
  #map { flex:1; }
  .leaflet-tile { filter: brightness(0.6) invert(1) contrast(3) hue-rotate(200deg) saturate(0.3) brightness(0.7); }
  .sidebar {
    width:270px; background:var(--surface); border-left:1px solid var(--border);
    overflow-y:auto; flex-shrink:0; display:flex; flex-direction:column;
  }
  .sidebar-header {
    padding:12px 16px; border-bottom:1px solid var(--border);
    font-size:11px; color:var(--muted); text-transform:uppercase; letter-spacing:.08em;
    display:flex; justify-content:space-between; flex-shrink:0;
  }
  .subzone-item {
    padding:10px 16px; border-bottom:1px solid var(--border);
    cursor:pointer; transition:background .15s;
    display:flex; align-items:center; justify-content:space-between;
  }
  .subzone-item:hover { background:rgba(255,255,255,.04); }
  .subzone-name { font-family:'IBM Plex Mono',monospace; font-size:11px; }
  .tier-badge {
    padding:2px 7px; border-radius:3px; font-size:10px; font-weight:600;
    letter-spacing:.04em; font-family:'IBM Plex Mono',monospace; color:#fff;
  }
  .score-val { font-family:'IBM Plex Mono',monospace; font-size:11px; color:var(--muted); min-width:38px; text-align:right; }
  .legend {
    position:absolute; bottom:30px; left:10px; z-index:1000;
    background:rgba(17,24,39,.92); backdrop-filter:blur(8px);
    border:1px solid var(--border); border-radius:8px; padding:12px 16px;
  }
  .legend-title { font-size:10px; color:var(--muted); text-transform:uppercase; letter-spacing:.08em; margin-bottom:8px; }
  .legend-row { display:flex; align-items:center; gap:8px; margin-bottom:5px; font-size:12px; }
  .legend-swatch { width:13px; height:13px; border-radius:3px; flex-shrink:0; }
  .leaflet-popup-content-wrapper {
    background:var(--surface); border:1px solid var(--border); border-radius:10px;
    box-shadow:0 8px 32px rgba(0,0,0,.5);
  }
  .leaflet-popup-content { margin:14px; }
  .leaflet-popup-tip { background:var(--surface); }
  .popup-name { font-family:'IBM Plex Mono',monospace; font-size:13px; font-weight:600; color:var(--text); margin-bottom:8px; }
  .popup-row { display:flex; justify-content:space-between; font-size:12px; padding:4px 0; border-bottom:1px solid var(--border); }
  .popup-row:last-child { border:none; }
  .popup-key { color:var(--muted); }
  .popup-val { font-family:'IBM Plex Mono',monospace; font-weight:600; }
  .popup-tier { display:inline-block; padding:2px 8px; border-radius:3px; font-size:11px; font-weight:600; color:#fff; margin-bottom:6px; }
  .loading-overlay {
    position:absolute; inset:0; background:rgba(10,15,30,.8); z-index:2000;
    display:flex; align-items:center; justify-content:center; flex-direction:column; gap:12px;
  }
  .spinner { width:28px; height:28px; border:3px solid var(--border); border-top-color:var(--accent); border-radius:50%; animation:spin .8s linear infinite; }
  @keyframes spin { to{transform:rotate(360deg)} }
  .loading-text { font-family:'IBM Plex Mono',monospace; font-size:12px; color:var(--muted); }

  /* Charts tab */
  .charts-tab { flex:1; overflow-y:auto; padding:24px; display:grid; grid-template-columns:1fr 1fr; grid-template-rows:auto auto auto; gap:20px; }
  .chart-card {
    background:var(--surface); border:1px solid var(--border); border-radius:10px;
    padding:20px; display:flex; flex-direction:column; gap:12px;
  }
  .chart-card.wide { grid-column:1/-1; }
  .chart-title { font-family:'IBM Plex Mono',monospace; font-size:12px; color:var(--muted); text-transform:uppercase; letter-spacing:.06em; }
  .chart-wrap { position:relative; height:220px; }
  .chart-wrap.tall { height:280px; }
</style>
</head>
<body>

<div class="header">
  <div class="header-left">
    <div class="pulse"></div>
    <div>
      <div class="title">DENGUE RISK INTELLIGENCE</div>
      <div class="subtitle">NEA Operations — Weekly Fogging Deployment Plan · LightGBM · Recall 0.996</div>
    </div>
  </div>
  <div class="header-right">
    <div class="stat-pill">
      <div class="dot" style="background:var(--high)"></div>
      <div class="label">High</div>
      <div class="val" id="statHigh" style="color:var(--high)">—</div>
    </div>
    <div class="stat-pill">
      <div class="dot" style="background:var(--med)"></div>
      <div class="label">Medium</div>
      <div class="val" id="statMed" style="color:var(--med)">—</div>
    </div>
    <div class="stat-pill">
      <div class="dot" style="background:var(--low)"></div>
      <div class="label">Low</div>
      <div class="val" id="statLow" style="color:var(--low)">—</div>
    </div>
    <div class="timestamp" id="lastUpdate">Loading...</div>
  </div>
</div>

<div class="tab-bar">
  <button class="tab-btn active" onclick="switchTab('map')">🗺 Risk Map</button>
  <button class="tab-btn" onclick="switchTab('charts')">📊 Analytics</button>
</div>

<!-- MAP TAB -->
<div class="tab-content active" id="tab-map">
  <div id="map" style="position:relative">
    <div class="loading-overlay" id="loadingOverlay">
      <div class="spinner"></div>
      <div class="loading-text">Loading Singapore subzone data...</div>
    </div>
  </div>
  <div class="sidebar">
    <div class="sidebar-header">
      <span>Subzone Rankings</span>
      <span id="sidebarCount">—</span>
    </div>
    <div id="sidebarList"></div>
  </div>
  <div class="legend">
    <div class="legend-title">Risk Tier</div>
    <div class="legend-row"><div class="legend-swatch" style="background:#ef4444"></div><span>High (≥0.60) — Deploy fogging</span></div>
    <div class="legend-row"><div class="legend-swatch" style="background:#f59e0b"></div><span>Medium (0.30–0.59) — Monitor</span></div>
    <div class="legend-row"><div class="legend-swatch" style="background:#10b981"></div><span>Low (&lt;0.30) — Surveillance</span></div>
    <div class="legend-row"><div class="legend-swatch" style="background:#374151"></div><span>No data</span></div>
  </div>
</div>

<!-- CHARTS TAB -->
<div class="tab-content" id="tab-charts">
  <div class="charts-tab">

    <div class="chart-card">
      <div class="chart-title">Risk Tier Breakdown</div>
      <div class="chart-wrap"><canvas id="chartDonut"></canvas></div>
    </div>

    <div class="chart-card">
      <div class="chart-title">Score Distribution</div>
      <div class="chart-wrap"><canvas id="chartHist"></canvas></div>
    </div>

    <div class="chart-card wide">
      <div class="chart-title">Top 20 Highest Risk Subzones</div>
      <div class="chart-wrap tall"><canvas id="chartTop20"></canvas></div>
    </div>

    <div class="chart-card">
      <div class="chart-title">Vulnerability Index vs Model Score</div>
      <div class="chart-wrap"><canvas id="chartScatter"></canvas></div>
    </div>

    <div class="chart-card">
      <div class="chart-title">Active Case Count by Subzone (Top 20)</div>
      <div class="chart-wrap"><canvas id="chartCases"></canvas></div>
    </div>

  </div>
</div>

<script>
  const TIER_COLORS = { High:'#ef4444', Medium:'#f59e0b', Low:'#10b981' };
  const CHART_DEFAULTS = {
    color: '#e5e7eb',
    font: { family:"'IBM Plex Mono', monospace", size:11 },
    grid: { color:'#1f2937' },
  };

  let scoresData = {};
  let geoLayer = null;
  let mapInitialized = false;
  let chartsInitialized = false;
  const charts = {};

  // ── Tab switching ──────────────────────────────────────────────────────────
  function switchTab(tab) {
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
    document.querySelector(`[onclick="switchTab('${tab}')"]`).classList.add('active');
    document.getElementById(`tab-${tab}`).classList.add('active');

    if (tab === 'map' && !mapInitialized) initMap();
    if (tab === 'charts') {
      if (!chartsInitialized) { buildCharts(); chartsInitialized = true; }
      else updateCharts();
    }
  }

  // ── Scores ─────────────────────────────────────────────────────────────────
  async function loadScores() {
    try {
      const resp = await fetch('/scores');
      scoresData = await resp.json();

      const counts = {High:0, Medium:0, Low:0};
      Object.values(scoresData).forEach(s => { if(counts[s.tier]!==undefined) counts[s.tier]++; });
      document.getElementById('statHigh').textContent = counts.High;
      document.getElementById('statMed').textContent  = counts.Medium;
      document.getElementById('statLow').textContent  = counts.Low;
      document.getElementById('sidebarCount').textContent = Object.keys(scoresData).length + ' subzones';

      const now = new Date();
      document.getElementById('lastUpdate').textContent =
        'Updated ' + String(now.getHours()).padStart(2,'0') + ':' +
        String(now.getMinutes()).padStart(2,'0') + ':' +
        String(now.getSeconds()).padStart(2,'0');

      if (geoLayer) geoLayer.setStyle(styleFeature);
      updateSidebar();
      if (chartsInitialized) updateCharts();
    } catch(e) { console.error('Score load failed:', e); }
  }

  // ── Map ────────────────────────────────────────────────────────────────────
  let map;
  async function initMap() {
    mapInitialized = true;
    map = L.map('map', { center:[1.3521,103.8198], zoom:12 });
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      attribution:'© OpenStreetMap', maxZoom:18
    }).addTo(map);

    try {
      const resp = await fetch('/geojson');
      const gj   = await resp.json();
      geoLayer = L.geoJSON(gj, { style:styleFeature, onEachFeature }).addTo(map);
      map.fitBounds(geoLayer.getBounds(), {padding:[20,20]});
    } catch(e) { console.error('GeoJSON load failed:', e); }

    document.getElementById('loadingOverlay').style.display = 'none';
  }

  function getColor(name) {
    const s = scoresData[name];
    return s ? (TIER_COLORS[s.tier] || '#374151') : '#374151';
  }

  function styleFeature(feature) {
    const name = feature.properties.SUBZONE_N || feature.properties.subzone_name || '';
    return { fillColor:getColor(name), fillOpacity:0.65, color:'#1f2937', weight:0.8 };
  }

  function onEachFeature(feature, layer) {
    const name = feature.properties.SUBZONE_N || feature.properties.subzone_name || 'Unknown';
    layer.on({
      mouseover(e) { e.target.setStyle({fillOpacity:.85, weight:2, color:'#fff'}); },
      mouseout(e)  { geoLayer.resetStyle(e.target); },
      click(e) {
        const s = scoresData[name];
        const tier = s ? s.tier : 'No data';
        const color = TIER_COLORS[tier] || '#374151';
        layer.bindPopup(`
          <div class="popup-name">${name}</div>
          <div class="popup-tier" style="background:${color}">${tier}</div>
          <div class="popup-row"><span class="popup-key">Model Score</span><span class="popup-val">${s?s.score:'—'}</span></div>
          <div class="popup-row"><span class="popup-key">Vulnerability Index</span><span class="popup-val">${s?s.vulnerability_index:'—'}</span></div>
          <div class="popup-row"><span class="popup-key">Cases This Week</span><span class="popup-val">${s?s.case_count:'—'}</span></div>
          <div class="popup-row"><span class="popup-key">P(cluster 14d)</span><span class="popup-val">${s?s.score:'—'}</span></div>
        `, {maxWidth:240}).openPopup();
      }
    });
  }

  function updateSidebar() {
    const entries = Object.entries(scoresData).sort((a,b)=>b[1].score-a[1].score).slice(0,80);
    document.getElementById('sidebarList').innerHTML = entries.map(([name,s]) => {
      const color = TIER_COLORS[s.tier]||'#374151';
      return `<div class="subzone-item" onclick="flyTo('${name}')">
        <div class="subzone-name">${name}</div>
        <div style="display:flex;align-items:center;gap:6px">
          <span class="tier-badge" style="background:${color}">${s.tier}</span>
          <span class="score-val">${s.score}</span>
        </div>
      </div>`;
    }).join('');
  }

  function flyTo(name) {
    if (!geoLayer) { switchTab('map'); setTimeout(()=>flyTo(name),500); return; }
    geoLayer.eachLayer(layer => {
      const n = layer.feature.properties.SUBZONE_N || layer.feature.properties.subzone_name || '';
      if (n === name) { map.flyToBounds(layer.getBounds(),{duration:.8,padding:[40,40]}); layer.fire('click'); }
    });
  }

  // ── Charts ─────────────────────────────────────────────────────────────────
  function buildCharts() {
    const entries  = Object.entries(scoresData);
    const scores   = entries.map(([,s])=>s.score);
    const highList = entries.filter(([,s])=>s.tier==='High').sort((a,b)=>b[1].score-a[1].score).slice(0,20);
    const caseList = entries.filter(([,s])=>s.case_count>1).sort((a,b)=>b[1].case_count-a[1].case_count).slice(0,20);

    const counts = {High:0,Medium:0,Low:0};
    entries.forEach(([,s])=>{ if(counts[s.tier]!==undefined) counts[s.tier]++; });

    // Chart defaults helper
    const gridColor = '#1f2937';
    const textColor = '#9ca3af';
    const font = {family:"'IBM Plex Mono', monospace", size:10};

    // 1. Donut
    charts.donut = new Chart(document.getElementById('chartDonut'), {
      type:'doughnut',
      data:{
        labels:['High','Medium','Low'],
        datasets:[{
          data:[counts.High,counts.Medium,counts.Low],
          backgroundColor:['#ef4444','#f59e0b','#10b981'],
          borderColor:'#111827', borderWidth:3,
        }]
      },
      options:{
        responsive:true, maintainAspectRatio:false,
        plugins:{
          legend:{position:'bottom', labels:{color:textColor,font,padding:16,boxWidth:12}},
          tooltip:{callbacks:{label:ctx=>`${ctx.label}: ${ctx.raw} subzones`}}
        }
      }
    });

    // 2. Histogram
    const bins = Array.from({length:10},(_,i)=>i*0.1);
    const binCounts = bins.map(b=>scores.filter(s=>s>=b&&s<b+0.1).length);
    const binColors = bins.map(b=>b>=0.6?'#ef4444':b>=0.3?'#f59e0b':'#10b981');

    charts.hist = new Chart(document.getElementById('chartHist'), {
      type:'bar',
      data:{
        labels:bins.map(b=>`${b.toFixed(1)}-${(b+0.1).toFixed(1)}`),
        datasets:[{data:binCounts,backgroundColor:binColors,borderRadius:3,borderSkipped:false}]
      },
      options:{
        responsive:true, maintainAspectRatio:false,
        plugins:{legend:{display:false}},
        scales:{
          x:{ticks:{color:textColor,font},grid:{color:gridColor}},
          y:{ticks:{color:textColor,font},grid:{color:gridColor},title:{display:true,text:'Subzones',color:textColor,font}}
        }
      }
    });

    // 3. Top 20 horizontal bar
    charts.top20 = new Chart(document.getElementById('chartTop20'), {
      type:'bar',
      data:{
        labels:highList.map(([n])=>n),
        datasets:[{
          data:highList.map(([,s])=>s.score),
          backgroundColor:highList.map(([,s])=>TIER_COLORS[s.tier]||'#374151'),
          borderRadius:3, borderSkipped:false,
        }]
      },
      options:{
        indexAxis:'y', responsive:true, maintainAspectRatio:false,
        plugins:{legend:{display:false}},
        scales:{
          x:{min:0,max:1,ticks:{color:textColor,font},grid:{color:gridColor}},
          y:{ticks:{color:textColor,font:{family:"'IBM Plex Mono',monospace",size:10}},grid:{color:gridColor}}
        }
      }
    });

    // 4. Scatter: vulnerability vs score
    const scatterData = entries.map(([n,s])=>({x:s.vulnerability_index,y:s.score,label:n,tier:s.tier}));
    charts.scatter = new Chart(document.getElementById('chartScatter'), {
      type:'scatter',
      data:{
        datasets:[
          {label:'High', data:scatterData.filter(d=>d.tier==='High'), backgroundColor:'rgba(239,68,68,.6)', pointRadius:4},
          {label:'Medium',data:scatterData.filter(d=>d.tier==='Medium'),backgroundColor:'rgba(245,158,11,.6)',pointRadius:4},
          {label:'Low',  data:scatterData.filter(d=>d.tier==='Low'),  backgroundColor:'rgba(16,185,129,.6)', pointRadius:4},
        ]
      },
      options:{
        responsive:true, maintainAspectRatio:false,
        plugins:{legend:{labels:{color:textColor,font,boxWidth:10}}},
        scales:{
          x:{title:{display:true,text:'Vulnerability Index',color:textColor,font},ticks:{color:textColor,font},grid:{color:gridColor}},
          y:{title:{display:true,text:'Model Score',color:textColor,font},ticks:{color:textColor,font},grid:{color:gridColor},min:0,max:1}
        }
      }
    });

    // 5. Case counts bar
    charts.cases = new Chart(document.getElementById('chartCases'), {
      type:'bar',
      data:{
        labels:caseList.length?caseList.map(([n])=>n):['No active cases'],
        datasets:[{
          data:caseList.length?caseList.map(([,s])=>s.case_count):[0],
          backgroundColor:'#06b6d4', borderRadius:3, borderSkipped:false,
        }]
      },
      options:{
        responsive:true, maintainAspectRatio:false,
        plugins:{legend:{display:false}},
        scales:{
          x:{ticks:{color:textColor,font:{family:"'IBM Plex Mono',monospace",size:9}},grid:{color:gridColor}},
          y:{ticks:{color:textColor,font},grid:{color:gridColor},title:{display:true,text:'Cases',color:textColor,font}}
        }
      }
    });
  }

  function updateCharts() {
    if (!charts.donut) return;
    const entries = Object.entries(scoresData);
    const scores  = entries.map(([,s])=>s.score);
    const counts  = {High:0,Medium:0,Low:0};
    entries.forEach(([,s])=>{ if(counts[s.tier]!==undefined) counts[s.tier]++; });

    // Donut
    charts.donut.data.datasets[0].data = [counts.High,counts.Medium,counts.Low];
    charts.donut.update();

    // Hist
    const bins = Array.from({length:10},(_,i)=>i*0.1);
    charts.hist.data.datasets[0].data = bins.map(b=>scores.filter(s=>s>=b&&s<b+0.1).length);
    charts.hist.update();

    // Top 20
    const highList = entries.filter(([,s])=>s.tier==='High').sort((a,b)=>b[1].score-a[1].score).slice(0,20);
    charts.top20.data.labels = highList.map(([n])=>n);
    charts.top20.data.datasets[0].data = highList.map(([,s])=>s.score);
    charts.top20.update();

    // Scatter
    const scatterData = entries.map(([n,s])=>({x:s.vulnerability_index,y:s.score,tier:s.tier}));
    charts.scatter.data.datasets[0].data = scatterData.filter(d=>d.tier==='High');
    charts.scatter.data.datasets[1].data = scatterData.filter(d=>d.tier==='Medium');
    charts.scatter.data.datasets[2].data = scatterData.filter(d=>d.tier==='Low');
    charts.scatter.update();

    // Cases
    const caseList = entries.filter(([,s])=>s.case_count>1).sort((a,b)=>b[1].case_count-a[1].case_count).slice(0,20);
    charts.cases.data.labels = caseList.length?caseList.map(([n])=>n):['No active cases'];
    charts.cases.data.datasets[0].data = caseList.length?caseList.map(([,s])=>s.case_count):[0];
    charts.cases.update();
  }

  // ── Init ───────────────────────────────────────────────────────────────────
  async function init() {
    await loadScores();
    initMap();
    setInterval(loadScores, 30000);
  }

  init();
</script>
</body>
</html>"""
    return HTMLResponse(content=html)


@app.get("/hospital", response_class=HTMLResponse)
def hospital_submission():
    """
    Hospital / GP Clinic case submission portal.
    Desktop web UI for healthcare staff to submit confirmed dengue cases.
    In production this would be replaced by automated HIS API calls.
    Submits to POST /cases/confirmed and shows the scoring result.
    """
    html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Dengue Case Notification — Healthcare Portal</title>
<style>
  * { margin:0; padding:0; box-sizing:border-box; }
  body { font-family: 'Segoe UI', Arial, sans-serif; background: #f0f4f8; color: #1a1a2e; min-height: 100vh; }

  .topbar {
    background: #fff; border-bottom: 3px solid #c0392b;
    padding: 0 32px; display: flex; align-items: center; gap: 16px; height: 60px;
  }
  .logo { width: 36px; height: 36px; background: #c0392b; border-radius: 6px;
          display: flex; align-items: center; justify-content: center; color: #fff; font-weight: bold; font-size: 18px; }
  .topbar-title { font-size: 15px; font-weight: 600; color: #1a1a2e; }
  .topbar-sub   { font-size: 11px; color: #888; margin-top: 1px; }
  .topbar-right { margin-left: auto; font-size: 12px; color: #888; }

  .page { max-width: 900px; margin: 32px auto; padding: 0 16px; display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }

  .card {
    background: #fff; border-radius: 10px; padding: 28px;
    box-shadow: 0 1px 4px rgba(0,0,0,.08);
  }
  .card-title { font-size: 14px; font-weight: 600; color: #555; text-transform: uppercase;
                letter-spacing: .06em; margin-bottom: 20px; padding-bottom: 10px;
                border-bottom: 1px solid #eee; }

  .form-group { margin-bottom: 18px; }
  label { display: block; font-size: 12px; font-weight: 600; color: #444;
          margin-bottom: 6px; text-transform: uppercase; letter-spacing: .04em; }
  select, input[type=number], input[type=text] {
    width: 100%; padding: 10px 12px; border: 1px solid #ddd; border-radius: 6px;
    font-size: 14px; color: #1a1a2e; background: #fafafa;
    transition: border .2s;
  }
  select:focus, input:focus { outline: none; border-color: #c0392b; background: #fff; }

  .required { color: #c0392b; }

  .submit-btn {
    width: 100%; padding: 13px; background: #c0392b; color: #fff;
    border: none; border-radius: 6px; font-size: 14px; font-weight: 600;
    cursor: pointer; transition: background .2s; margin-top: 4px;
  }
  .submit-btn:hover { background: #a93226; }
  .submit-btn:disabled { background: #ccc; cursor: not-allowed; }

  .notice {
    background: #fff8e1; border: 1px solid #f9a825; border-radius: 6px;
    padding: 12px 14px; font-size: 12px; color: #7d6608; margin-top: 16px; line-height: 1.5;
  }

  /* Result card */
  .result { display: none; }
  .result.show { display: block; }
  .result-header {
    padding: 14px 16px; border-radius: 8px 8px 0 0; color: #fff; font-weight: 600; font-size: 14px;
  }
  .result-header.alert  { background: #c0392b; }
  .result-header.ok     { background: #1e8449; }
  .result-body { border: 1px solid #eee; border-top: none; border-radius: 0 0 8px 8px; padding: 16px; }
  .result-row { display: flex; justify-content: space-between; padding: 8px 0;
                border-bottom: 1px solid #f5f5f5; font-size: 13px; }
  .result-row:last-child { border: none; }
  .result-key { color: #777; }
  .result-val { font-weight: 600; font-family: monospace; }
  .result-val.danger { color: #c0392b; }
  .result-val.safe   { color: #1e8449; }
  .latency { font-size: 11px; color: #bbb; text-align: right; margin-top: 8px; font-family: monospace; }

  /* History table */
  .history-table { width: 100%; border-collapse: collapse; font-size: 12px; }
  .history-table th { text-align: left; padding: 7px 10px; background: #f5f5f5;
                      font-size: 11px; text-transform: uppercase; letter-spacing: .04em; color: #777; }
  .history-table td { padding: 7px 10px; border-bottom: 1px solid #f0f0f0; }
  .history-table tr:last-child td { border: none; }
  .badge { display: inline-block; padding: 1px 7px; border-radius: 3px;
           font-size: 10px; font-weight: bold; color: #fff; }
  .badge-alert { background: #c0392b; }
  .badge-ok    { background: #1e8449; }
  .badge-med   { background: #d68910; }
  .empty-history { color: #bbb; font-size: 12px; text-align: center; padding: 20px; }

  .error-box { background: #fdecea; border: 1px solid #e74c3c; border-radius: 6px;
               padding: 10px 14px; font-size: 12px; color: #c0392b; display: none; margin-top: 12px; }
  .error-box.show { display: block; }
</style>
</head>
<body>

<div class="topbar">
  <div class="logo">+</div>
  <div>
    <div class="topbar-title">MOH Dengue Case Notification Portal</div>
    <div class="topbar-sub">For use by registered healthcare facilities only</div>
  </div>
  <div class="topbar-right" id="clock"></div>
</div>

<div class="page">

  <!-- Left: submission form -->
  <div>
    <div class="card">
      <div class="card-title">Submit Confirmed Dengue Case</div>

      <div class="form-group">
        <label>Patient Residential Subzone <span class="required">*</span></label>
        <input type="text" id="subzoneInput" placeholder="e.g. TAMPINES EAST" list="subzoneList">
        <datalist id="subzoneList"></datalist>
      </div>

      <div class="form-group">
        <label>Number of Confirmed Cases <span class="required">*</span></label>
        <input type="number" id="caseCount" value="1" min="1" max="50">
      </div>

      <div class="form-group">
        <label>Reporting Facility <span class="required">*</span></label>
        <select id="source">
          <option value="hospital">Public Hospital (A&E / Inpatient)</option>
          <option value="clinic">GP / Polyclinic</option>
          <option value="nea">NEA Field Report</option>
        </select>
      </div>

      <div class="error-box" id="errorBox"></div>

      <button class="submit-btn" id="submitBtn" onclick="submitCase()">
        Submit Case Notification
      </button>

      <div class="notice">
        Submission triggers automated risk scoring. If the subzone risk score
        exceeds the alert threshold, a real-time notification is sent to the
        assigned CDA Officer at MOH.
      </div>
    </div>

    <!-- Result -->
    <div class="card result" id="resultCard" style="margin-top:20px">
      <div class="result-header" id="resultHeader">Submission Result</div>
      <div class="result-body">
        <div class="result-row"><span class="result-key">Subzone</span>       <span class="result-val" id="resSubzone">-</span></div>
        <div class="result-row"><span class="result-key">Model Score</span>   <span class="result-val" id="resScore">-</span></div>
        <div class="result-row"><span class="result-key">Alert Score</span>   <span class="result-val" id="resAlert">-</span></div>
        <div class="result-row"><span class="result-key">Vulnerability</span> <span class="result-val" id="resVuln">-</span></div>
        <div class="result-row"><span class="result-key">Cases This Week</span><span class="result-val" id="resCases">-</span></div>
        <div class="result-row"><span class="result-key">CDA Alert Sent</span><span class="result-val" id="resAlert2">-</span></div>
        <div class="latency" id="resLatency"></div>
      </div>
    </div>
  </div>

  <!-- Right: recent submissions -->
  <div>
    <div class="card">
      <div class="card-title">Recent Submissions — This Session</div>
      <table class="history-table">
        <thead>
          <tr><th>Time</th><th>Subzone</th><th>Score</th><th>Alert</th></tr>
        </thead>
        <tbody id="historyBody">
          <tr><td colspan="4" class="empty-history">No submissions yet</td></tr>
        </tbody>
      </table>
    </div>
  </div>

</div>

<script>
  // Clock
  function updateClock() {
    const now = new Date();
    document.getElementById('clock').textContent =
      now.toLocaleDateString('en-SG', {weekday:'short', day:'numeric', month:'short'}) +
      '  ' + String(now.getHours()).padStart(2,'0') + ':' +
      String(now.getMinutes()).padStart(2,'0') + ':' +
      String(now.getSeconds()).padStart(2,'0');
  }
  updateClock();
  setInterval(updateClock, 1000);

  // Populate subzone datalist from /scores
  fetch('/scores').then(r => r.json()).then(data => {
    const dl = document.getElementById('subzoneList');
    Object.keys(data).sort().forEach(name => {
      const opt = document.createElement('option');
      opt.value = name;
      dl.appendChild(opt);
    });
  }).catch(() => {});

  const history = [];

  async function submitCase() {
    const btn     = document.getElementById('submitBtn');
    const subzone = document.getElementById('subzoneInput').value.trim().toUpperCase();
    const count   = parseInt(document.getElementById('caseCount').value);
    const source  = document.getElementById('source').value;
    const errBox  = document.getElementById('errorBox');

    if (!subzone) {
      errBox.textContent = 'Please enter a subzone name.';
      errBox.classList.add('show');
      return;
    }

    btn.disabled = true;
    btn.textContent = 'Submitting...';
    errBox.classList.remove('show');

    try {
      const resp = await fetch('/cases/confirmed', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ subzone_name: subzone, case_count: count, source: source })
      });
      if (!resp.ok) throw new Error('Server returned ' + resp.status);
      const data = await resp.json();

      // Show result
      const card = document.getElementById('resultCard');
      card.classList.add('show');
      const header = document.getElementById('resultHeader');
      header.className = 'result-header ' + (data.alert_triggered ? 'alert' : 'ok');
      header.textContent = data.alert_triggered
        ? 'ALERT TRIGGERED — CDA Officer Notified'
        : 'Submitted — Risk Below Alert Threshold';

      document.getElementById('resSubzone').textContent = data.subzone_name;
      document.getElementById('resScore').textContent   = data.model_score;
      document.getElementById('resAlert').textContent   = data.alert_score;
      document.getElementById('resVuln').textContent    = data.vulnerability_index;
      document.getElementById('resCases').textContent   = data.current_case_count;
      document.getElementById('resLatency').textContent = 'Scored in ' + data.latency_ms + 'ms';

      const alertEl = document.getElementById('resAlert2');
      alertEl.textContent  = data.alert_triggered ? 'YES' : 'No';
      alertEl.className    = 'result-val ' + (data.alert_triggered ? 'danger' : 'safe');

      // Add to history
      const now = new Date();
      const timeStr = String(now.getHours()).padStart(2,'0') + ':' +
                      String(now.getMinutes()).padStart(2,'0') + ':' +
                      String(now.getSeconds()).padStart(2,'0');
      history.unshift({ time: timeStr, subzone: data.subzone_name,
                        score: data.model_score, alert: data.alert_triggered });
      if (history.length > 10) history.pop();

      const tbody = document.getElementById('historyBody');
      tbody.innerHTML = history.map(h => {
        const badgeClass = h.alert ? 'badge-alert' : (h.score >= 0.3 ? 'badge-med' : 'badge-ok');
        const badgeText  = h.alert ? 'ALERT' : (h.score >= 0.3 ? 'Monitor' : 'Low');
        return '<tr><td>' + h.time + '</td><td>' + h.subzone +
               '</td><td>' + h.score + '</td><td>' +
               '<span class="badge ' + badgeClass + '">' + badgeText + '</span></td></tr>';
      }).join('');

    } catch(e) {
      errBox.textContent = 'Submission failed: ' + e.message;
      errBox.classList.add('show');
    } finally {
      btn.disabled = false;
      btn.textContent = 'Submit Case Notification';
    }
  }
</script>
</body>
</html>"""
    return HTMLResponse(content=html)



@app.get("/mobile", response_class=HTMLResponse)
def mobile_emulator():
    """
    CDA Officer mobile alert receiver.
    Passively listens for alerts via Server-Sent Events (SSE).
    Alerts push instantly when a hospital submits a case that crosses the threshold.
    Officer does not submit anything — pure receiver.
    """
    html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CDA Officer - Live Alert Monitor</title>
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
* { margin:0; padding:0; box-sizing:border-box; }
body { background:linear-gradient(135deg,#1a1a2e,#16213e,#0f3460); min-height:100vh;
       display:flex; align-items:center; justify-content:center;
       font-family:"Space Grotesk",sans-serif; padding:20px; }
.scene { display:flex; gap:60px; align-items:flex-start; flex-wrap:wrap; justify-content:center; }
.phone { width:320px; background:#1c1c1e; border-radius:50px; padding:12px;
         box-shadow:0 0 0 2px #3a3a3c,0 40px 80px rgba(0,0,0,.6); }
.phone-screen { background:#000; border-radius:40px; overflow:hidden; min-height:620px; position:relative; }
.notch { width:120px; height:30px; background:#1c1c1e; border-radius:0 0 20px 20px; margin:0 auto; z-index:10; }
.lock-screen { background:linear-gradient(180deg,#1a1a2e,#0d1117); min-height:590px; padding:20px; display:flex; flex-direction:column; }
.status-bar { display:flex; justify-content:space-between; font-size:12px; color:#fff; padding:4px 8px 12px; }
.time-display { text-align:center; margin:20px 0 30px; }
.time-display .time { font-size:56px; font-weight:300; color:#fff; letter-spacing:-2px; line-height:1; }
.time-display .date { font-size:14px; color:rgba(255,255,255,.7); margin-top:4px; }
.notification { background:rgba(255,255,255,.12); backdrop-filter:blur(20px); border-radius:16px;
                padding:14px; margin-bottom:10px; border:1px solid rgba(255,255,255,.1); }
.notification.alert-notif { background:rgba(239,68,68,.2); border:1px solid rgba(239,68,68,.4);
                             animation:alertPulse 2s ease-in-out infinite; }
@keyframes alertPulse { 0%,100%{box-shadow:0 0 0 0 rgba(239,68,68,0)} 50%{box-shadow:0 0 0 8px rgba(239,68,68,.15)} }
.notif-header { display:flex; align-items:center; gap:8px; margin-bottom:8px; }
.notif-icon { width:28px; height:28px; border-radius:8px; display:flex; align-items:center; justify-content:center; font-size:14px; }
.notif-icon.health { background:#007aff; }
.notif-icon.danger { background:#ef4444; }
.notif-app  { font-size:11px; color:rgba(255,255,255,.6); text-transform:uppercase; letter-spacing:.06em; }
.notif-time { font-size:11px; color:rgba(255,255,255,.4); margin-left:auto; }
.notif-title { font-size:13px; font-weight:600; color:#fff; margin-bottom:2px; }
.notif-body  { font-size:12px; color:rgba(255,255,255,.75); line-height:1.4; }
.chips { display:flex; gap:8px; margin-top:10px; }
.chip { background:rgba(239,68,68,.25); border:1px solid rgba(239,68,68,.5); border-radius:8px;
        padding:6px 10px; flex:1; text-align:center; }
.chip-label { font-size:9px; color:rgba(255,255,255,.5); text-transform:uppercase; letter-spacing:.06em; }
.chip-value { font-size:15px; font-weight:700; color:#ef4444; font-family:monospace; }
.actions { display:flex; gap:8px; margin-top:10px; }
.act-btn { flex:1; padding:8px; border-radius:10px; border:none; font-size:12px; font-weight:600; cursor:pointer; }
.act-btn.primary   { background:#ef4444; color:#fff; }
.act-btn.secondary { background:rgba(255,255,255,.15); color:#fff; }
.info { width:320px; color:#fff; }
.info h2 { font-size:22px; font-weight:700; margin-bottom:4px;
           background:linear-gradient(135deg,#06b6d4,#3b82f6);
           -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
.desc { font-size:13px; color:rgba(255,255,255,.5); margin-bottom:24px; line-height:1.5; }
.status-box { background:rgba(255,255,255,.07); border:1px solid rgba(255,255,255,.12);
              border-radius:12px; padding:16px; }
.status-row { display:flex; justify-content:space-between; padding:8px 0;
              border-bottom:1px solid rgba(255,255,255,.06); font-size:13px; }
.status-row:last-child { border:none; }
.status-key { color:rgba(255,255,255,.5); }
.status-val { font-weight:600; font-family:monospace; }
.green { color:#10b981; } .red { color:#ef4444; }
.alert-log { margin-top:16px; background:rgba(255,255,255,.05); border:1px solid rgba(255,255,255,.1);
             border-radius:12px; padding:12px; max-height:180px; overflow-y:auto; }
.log-title { font-size:11px; color:rgba(255,255,255,.4); text-transform:uppercase; letter-spacing:.06em; margin-bottom:8px; }
.log-entry { font-size:11px; color:rgba(255,255,255,.7); padding:4px 0;
             border-bottom:1px solid rgba(255,255,255,.05); font-family:monospace; }
.log-entry:last-child { border:none; }
.log-alert { color:#ef4444; }
</style>
</head>
<body>
<div class="scene">
  <div class="phone">
    <div class="phone-screen">
      <div class="notch"></div>
      <div class="lock-screen">
        <div class="status-bar">
          <span id="clock">--:--</span>
          <span>CDA Officer</span>
          <span>MOH</span>
        </div>
        <div class="time-display">
          <div class="time" id="bigClock">--:--</div>
          <div class="date" id="bigDate"></div>
        </div>
        <div class="notification" id="idleNotif">
          <div class="notif-header">
            <div class="notif-icon health">+</div>
            <div class="notif-app">CDA Alert System</div>
            <div class="notif-time" id="idleTime">--:--</div>
          </div>
          <div class="notif-title">Monitoring Active</div>
          <div class="notif-body">Connected to live case stream. Awaiting hospital notifications.</div>
        </div>
        <div class="notification alert-notif" id="alertNotif" style="display:none">
          <div class="notif-header">
            <div class="notif-icon danger">!</div>
            <div class="notif-app">DENGUE ALERT</div>
            <div class="notif-time" id="alertTime">--:--</div>
          </div>
          <div class="notif-title" id="alertTitle">High Risk Alert</div>
          <div class="notif-body"  id="alertBody">Alert triggered</div>
          <div class="chips">
            <div class="chip"><div class="chip-label">Model Score</div><div class="chip-value" id="chipScore">-</div></div>
            <div class="chip"><div class="chip-label">Alert Score</div><div class="chip-value" id="chipAlert">-</div></div>
            <div class="chip"><div class="chip-label">Cases</div>     <div class="chip-value" id="chipCases">-</div></div>
          </div>
          <div class="actions">
            <button class="act-btn primary">Dispatch Team</button>
            <button class="act-btn secondary" onclick="dismissAlert()">Dismiss</button>
          </div>
        </div>
      </div>
    </div>
  </div>

  <div class="info">
    <h2>CDA Officer Live Monitor</h2>
    <p class="desc">
      Passively receives alerts via SSE push.<br>
      Alerts appear the instant a hospital submits a confirmed case.<br>
      No action required from the officer.
    </p>
    <div class="status-box">
      <div class="status-row">
        <span class="status-key">SSE Connection</span>
        <span class="status-val" id="sseStatus">Connecting...</span>
      </div>
      <div class="status-row">
        <span class="status-key">Alerts Received</span>
        <span class="status-val" id="alertCount">0</span>
      </div>
      <div class="status-row">
        <span class="status-key">Last Alert Subzone</span>
        <span class="status-val" id="lastAlert">None</span>
      </div>
    </div>
    <div class="alert-log">
      <div class="log-title">Alert Log</div>
      <div id="logEntries">
        <div style="color:rgba(255,255,255,.3);font-size:11px">Waiting for alerts...</div>
      </div>
    </div>
  </div>
</div>

<script>
  let alertCount = 0;

  function tick() {
    const now = new Date();
    const h = String(now.getHours()).padStart(2,'0');
    const m = String(now.getMinutes()).padStart(2,'0');
    const s = String(now.getSeconds()).padStart(2,'0');
    document.getElementById('clock').textContent    = h+':'+m;
    document.getElementById('bigClock').textContent = h+':'+m;
    document.getElementById('idleTime').textContent = h+':'+m+':'+s;
    document.getElementById('bigDate').textContent  =
      now.toLocaleDateString('en-SG',{weekday:'long',day:'numeric',month:'long',year:'numeric'});
  }
  tick();
  setInterval(tick, 1000);

  function dismissAlert() {
    document.getElementById('alertNotif').style.display = 'none';
    document.getElementById('idleNotif').style.display  = 'block';
  }

  function showAlert(data) {
    alertCount++;
    document.getElementById('alertCount').textContent = alertCount;
    document.getElementById('lastAlert').textContent  = data.subzone_name;

    const now = new Date();
    const t   = String(now.getHours()).padStart(2,'0')+':'+String(now.getMinutes()).padStart(2,'0');
    document.getElementById('alertTime').textContent  = t;
    document.getElementById('alertTitle').textContent = 'High Risk: '+data.subzone_name;
    document.getElementById('alertBody').textContent  =
      'Alert score '+data.alert_score+' exceeded threshold. Source: '+data.source+'. Triage required.';
    document.getElementById('chipScore').textContent  = data.model_score;
    document.getElementById('chipAlert').textContent  = data.alert_score;
    document.getElementById('chipCases').textContent  = data.case_count;

    document.getElementById('idleNotif').style.display  = 'none';
    document.getElementById('alertNotif').style.display = 'block';

    // Add to log
    const log   = document.getElementById('logEntries');
    const entry = document.createElement('div');
    entry.className   = 'log-entry log-alert';
    entry.textContent = t+' ALERT '+data.subzone_name+' score='+data.alert_score;
    log.insertBefore(entry, log.firstChild);
    if (log.children.length > 20) log.removeChild(log.lastChild);

    // Auto-dismiss after 15s
    setTimeout(dismissAlert, 15000);
  }

  function connectSSE() {
    const el = document.getElementById('sseStatus');
    el.textContent = 'Connecting...'; el.className = 'status-val';

    const es = new EventSource('/alerts/stream');

    es.addEventListener('connected', () => {
      el.textContent = 'LIVE'; el.className = 'status-val green';
      const log   = document.getElementById('logEntries');
      const entry = document.createElement('div');
      const now   = new Date();
      entry.className   = 'log-entry';
      entry.textContent = String(now.getHours()).padStart(2,'0')+':'+
                          String(now.getMinutes()).padStart(2,'0')+
                          ' Connected to alert stream';
      log.insertBefore(entry, log.firstChild);
    });

    es.addEventListener('alert', e => {
      showAlert(JSON.parse(e.data));
    });

    es.addEventListener('heartbeat', () => {
      el.textContent = 'LIVE'; el.className = 'status-val green';
    });

    es.onerror = () => {
      el.textContent = 'Reconnecting...'; el.className = 'status-val red';
      es.close();
      setTimeout(connectSSE, 3000);
    };
  }

  connectSSE();
</script>
</body>
</html>"""
    return HTMLResponse(content=html)


@app.get("/pest-control", response_class=HTMLResponse)
def pest_control_dashboard():
    """
    Pest Control Operators ranked list.
    Simple printable table of all subzones sorted by risk score.
    Covers the Pest Control Operator user story from the proposal:
    'Ranked list of high-risk subzones with confidence — prioritise contracts,
    allocate crews to predicted hotspots.'
    """
    html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Dengue Risk — Pest Control Deployment List</title>
<style>
  * { margin:0; padding:0; box-sizing:border-box; }
  body { font-family: Arial, sans-serif; font-size: 13px; background: #f5f5f5; color: #1a1a1a; }
  .header {
    background: #1a472a; color: #fff;
    padding: 16px 24px;
    display: flex; align-items: center; justify-content: space-between;
  }
  .header-title { font-size: 17px; font-weight: bold; }
  .header-sub { font-size: 11px; color: #a8d5b5; margin-top: 2px; }
  .controls {
    background: #fff; border-bottom: 1px solid #ddd;
    padding: 10px 24px; display: flex; gap: 12px; align-items: center;
  }
  .controls label { font-size: 12px; color: #555; }
  select, input { font-size: 12px; padding: 5px 8px; border: 1px solid #ccc; border-radius: 4px; }
  .btn { padding: 6px 14px; border: none; border-radius: 4px; font-size: 12px; font-weight: bold; cursor: pointer; }
  .btn-print { background: #1a472a; color: #fff; }
  .summary {
    padding: 10px 24px; background: #fff; border-bottom: 1px solid #eee;
    display: flex; gap: 24px;
  }
  .stat { display: flex; flex-direction: column; }
  .stat-val { font-size: 22px; font-weight: bold; }
  .stat-lbl { font-size: 11px; color: #777; text-transform: uppercase; }
  .stat-high .stat-val { color: #c0392b; }
  .stat-med  .stat-val { color: #d68910; }
  .stat-low  .stat-val { color: #1e8449; }
  .table-wrap { padding: 16px 24px; }
  table { width: 100%; border-collapse: collapse; background: #fff; box-shadow: 0 1px 3px rgba(0,0,0,.1); }
  thead { background: #1a472a; color: #fff; }
  th { padding: 9px 12px; text-align: left; font-size: 11px; text-transform: uppercase; letter-spacing: .06em; }
  td { padding: 8px 12px; border-bottom: 1px solid #eee; }
  tr:last-child td { border: none; }
  tr:hover td { background: #f9f9f9; }
  .tier-badge { display: inline-block; padding: 2px 8px; border-radius: 3px; font-size: 11px; font-weight: bold; color: #fff; }
  .tier-High   { background: #c0392b; }
  .tier-Medium { background: #d68910; }
  .tier-Low    { background: #1e8449; }
  .score-bar-wrap { width: 80px; background: #eee; border-radius: 3px; height: 8px; display:inline-block; vertical-align:middle; }
  .score-bar { height: 8px; border-radius: 3px; }
  .rank-num { font-weight: bold; color: #555; }
  .timestamp { font-size: 11px; color: #a8d5b5; }
  @media print {
    .controls, .btn { display: none !important; }
    .header { background: #000 !important; -webkit-print-color-adjust: exact; }
    .tier-badge, .score-bar { -webkit-print-color-adjust: exact; }
  }
</style>
</head>
<body>

<div class="header">
  <div>
    <div class="header-title">Dengue Risk — Pest Control Deployment List</div>
    <div class="header-sub">Weekly fogging allocation · Subzones ranked by P(active cluster in next 14 days)</div>
  </div>
  <div style="display:flex;gap:10px;align-items:center">
    <span class="timestamp" id="genTime"></span>
    <button class="btn btn-print" onclick="window.print()">Print / Export PDF</button>
  </div>
</div>

<div class="controls">
  <label>Show tier:</label>
  <select id="tierFilter" onchange="filterTable()">
    <option value="All">All subzones</option>
    <option value="High">High risk only</option>
    <option value="Medium">High + Medium</option>
  </select>
  <label style="margin-left:12px">Search:</label>
  <input type="text" id="searchBox" placeholder="Subzone name..." oninput="filterTable()" style="width:180px">
  <span style="margin-left:auto;font-size:11px;color:#777" id="rowCount"></span>
</div>

<div class="summary">
  <div class="stat stat-high"><span class="stat-val" id="cntHigh">-</span><span class="stat-lbl">High Risk</span></div>
  <div class="stat stat-med"> <span class="stat-val" id="cntMed">-</span> <span class="stat-lbl">Medium Risk</span></div>
  <div class="stat stat-low"> <span class="stat-val" id="cntLow">-</span> <span class="stat-lbl">Low Risk</span></div>
  <div class="stat">          <span class="stat-val" id="cntTotal">-</span><span class="stat-lbl">Total Subzones</span></div>
</div>

<div class="table-wrap">
  <table id="rankTable">
    <thead>
      <tr>
        <th>Rank</th>
        <th>Subzone</th>
        <th>Risk Tier</th>
        <th>Model Score</th>
        <th style="width:100px"></th>
        <th>Vulnerability</th>
        <th>Cases This Week</th>
        <th>Recommended Action</th>
      </tr>
    </thead>
    <tbody id="tableBody">
      <tr><td colspan="8" style="text-align:center;padding:24px;color:#aaa">Loading...</td></tr>
    </tbody>
  </table>
</div>

<script>
  let allRows = [];
  const TIER_COLOR = { High: '#c0392b', Medium: '#d68910', Low: '#1e8449' };
  const ACTION = {
    High:   'Deploy fogging crew this week',
    Medium: 'Schedule inspection - monitor closely',
    Low:    'Routine surveillance only',
  };

  async function loadData() {
    try {
      const resp = await fetch('/scores');
      const data = await resp.json();
      allRows = Object.entries(data)
        .map(([name, s]) => ({ name, ...s }))
        .sort((a, b) => b.score - a.score)
        .map((r, i) => ({ ...r, rank: i + 1 }));

      const counts = { High: 0, Medium: 0, Low: 0 };
      allRows.forEach(r => { if (counts[r.tier] !== undefined) counts[r.tier]++; });
      document.getElementById('cntHigh').textContent  = counts.High;
      document.getElementById('cntMed').textContent   = counts.Medium;
      document.getElementById('cntLow').textContent   = counts.Low;
      document.getElementById('cntTotal').textContent = allRows.length;

      const now = new Date();
      document.getElementById('genTime').textContent =
        'Generated ' + now.toLocaleDateString('en-SG') + ' ' +
        String(now.getHours()).padStart(2,'0') + ':' +
        String(now.getMinutes()).padStart(2,'0');

      filterTable();
    } catch(e) {
      document.getElementById('tableBody').innerHTML =
        '<tr><td colspan="8" style="text-align:center;color:#c0392b;padding:20px">Failed to load scores - is the server running?</td></tr>';
    }
  }

  function filterTable() {
    const tier   = document.getElementById('tierFilter').value;
    const search = document.getElementById('searchBox').value.toLowerCase();
    const filtered = allRows.filter(r => {
      const tierOk = tier === 'All' ||
                     (tier === 'High' && r.tier === 'High') ||
                     (tier === 'Medium' && (r.tier === 'High' || r.tier === 'Medium'));
      return tierOk && r.name.toLowerCase().includes(search);
    });
    document.getElementById('rowCount').textContent = filtered.length + ' subzones shown';
    document.getElementById('tableBody').innerHTML = filtered.map(r => {
      const color    = TIER_COLOR[r.tier] || '#555';
      const barWidth = Math.round(r.score * 100);
      return '<tr>' +
        '<td class="rank-num">' + r.rank + '</td>' +
        '<td><strong>' + r.name + '</strong></td>' +
        '<td><span class="tier-badge tier-' + r.tier + '">' + r.tier + '</span></td>' +
        '<td>' + r.score.toFixed(4) + '</td>' +
        '<td><div class="score-bar-wrap"><div class="score-bar" style="width:' + barWidth + '%;background:' + color + '"></div></div></td>' +
        '<td>' + r.vulnerability_index.toFixed(4) + '</td>' +
        '<td>' + r.case_count + '</td>' +
        '<td style="color:' + color + ';font-size:11px">' + (ACTION[r.tier] || '') + '</td>' +
        '</tr>';
    }).join('');
  }

  loadData();
</script>
</body>
</html>"""
    return HTMLResponse(content=html)
