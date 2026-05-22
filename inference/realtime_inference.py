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
    Leaflet.js choropleth map of Singapore subzones coloured by risk tier.
    Click any subzone for score details.
    Auto-refreshes scores every 30 seconds without reloading the page.
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
  .header-right { display:flex; align-items:center; gap:20px; }
  .stat-pill {
    background:var(--bg); border:1px solid var(--border); border-radius:6px;
    padding:6px 14px; display:flex; align-items:center; gap:8px;
  }
  .stat-pill .dot { width:6px; height:6px; border-radius:50%; flex-shrink:0; }
  .stat-pill .label { font-size:10px; color:var(--muted); text-transform:uppercase; letter-spacing:.06em; }
  .stat-pill .val { font-family:'IBM Plex Mono',monospace; font-size:16px; font-weight:600; }
  .timestamp { font-family:'IBM Plex Mono',monospace; font-size:11px; color:var(--muted); }

  .main { display:flex; flex:1; overflow:hidden; }

  #map { flex:1; }

  /* Override Leaflet tiles for dark mode */
  .leaflet-tile { filter: brightness(0.6) invert(1) contrast(3) hue-rotate(200deg) saturate(0.3) brightness(0.7); }

  .sidebar {
    width:280px; background:var(--surface); border-left:1px solid var(--border);
    overflow-y:auto; flex-shrink:0;
    display:flex; flex-direction:column;
  }
  .sidebar-header {
    padding:14px 16px; border-bottom:1px solid var(--border);
    font-size:11px; color:var(--muted); text-transform:uppercase; letter-spacing:.08em;
    display:flex; justify-content:space-between;
  }
  .subzone-item {
    padding:12px 16px; border-bottom:1px solid var(--border);
    cursor:pointer; transition:background .15s;
    display:flex; align-items:center; justify-content:space-between;
  }
  .subzone-item:hover { background:rgba(255,255,255,.04); }
  .subzone-name { font-family:'IBM Plex Mono',monospace; font-size:11px; color:var(--text); }
  .tier-badge {
    padding:2px 8px; border-radius:3px; font-size:10px; font-weight:600;
    letter-spacing:.04em; font-family:'IBM Plex Mono',monospace; color:#fff;
  }
  .score-val { font-family:'IBM Plex Mono',monospace; font-size:12px; color:var(--muted); min-width:36px; text-align:right; }

  .legend {
    position:absolute; bottom:30px; left:10px; z-index:1000;
    background:rgba(17,24,39,.9); backdrop-filter:blur(8px);
    border:1px solid var(--border); border-radius:8px; padding:12px 16px;
  }
  .legend-title { font-size:10px; color:var(--muted); text-transform:uppercase; letter-spacing:.08em; margin-bottom:8px; }
  .legend-row { display:flex; align-items:center; gap:8px; margin-bottom:5px; font-size:12px; }
  .legend-swatch { width:14px; height:14px; border-radius:3px; flex-shrink:0; }

  /* Popup */
  .leaflet-popup-content-wrapper {
    background:var(--surface); border:1px solid var(--border);
    border-radius:10px; box-shadow:0 8px 32px rgba(0,0,0,.5);
  }
  .leaflet-popup-content { margin:14px; }
  .leaflet-popup-tip { background:var(--surface); }
  .popup-name { font-family:'IBM Plex Mono',monospace; font-size:13px; font-weight:600; color:var(--text); margin-bottom:10px; }
  .popup-row { display:flex; justify-content:space-between; font-size:12px; padding:4px 0; border-bottom:1px solid var(--border); }
  .popup-row:last-child { border:none; }
  .popup-key { color:var(--muted); }
  .popup-val { font-family:'IBM Plex Mono',monospace; color:var(--text); font-weight:600; }
  .popup-tier { display:inline-block; padding:3px 10px; border-radius:4px; font-size:11px; font-weight:600; color:#fff; margin-bottom:8px; }

  .loading-overlay {
    position:absolute; inset:0; background:rgba(10,15,30,.8); z-index:2000;
    display:flex; align-items:center; justify-content:center; flex-direction:column; gap:12px;
  }
  .spinner { width:32px; height:32px; border:3px solid var(--border); border-top-color:var(--accent); border-radius:50%; animation:spin .8s linear infinite; }
  @keyframes spin { to{transform:rotate(360deg)} }
  .loading-text { font-family:'IBM Plex Mono',monospace; font-size:12px; color:var(--muted); }
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

<div class="main">
  <div id="map">
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
</div>

<div class="legend">
  <div class="legend-title">Risk Tier</div>
  <div class="legend-row"><div class="legend-swatch" style="background:#ef4444"></div><span>High (score ≥ 0.60) — Deploy fogging</span></div>
  <div class="legend-row"><div class="legend-swatch" style="background:#f59e0b"></div><span>Medium (0.30–0.59) — Monitor</span></div>
  <div class="legend-row"><div class="legend-swatch" style="background:#10b981"></div><span>Low (&lt;0.30) — Surveillance</span></div>
  <div class="legend-row"><div class="legend-swatch" style="background:#374151"></div><span>No data</span></div>
</div>

<script>
  const TIER_COLORS = { High:'#ef4444', Medium:'#f59e0b', Low:'#10b981' };

  // Init Leaflet
  const map = L.map('map', { center:[1.3521, 103.8198], zoom:12, zoomControl:true });
  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution:'© OpenStreetMap', maxZoom:18
  }).addTo(map);

  let geoLayer = null;
  let scoresData = {};

  function getColor(subzoneName) {
    const s = scoresData[subzoneName];
    if (!s) return '#374151';
    return TIER_COLORS[s.tier] || '#374151';
  }

  function styleFeature(feature) {
    const name = feature.properties.SUBZONE_N || feature.properties.subzone_name || '';
    return {
      fillColor: getColor(name),
      fillOpacity: 0.65,
      color: '#1f2937',
      weight: 0.8,
    };
  }

  function onEachFeature(feature, layer) {
    const name = feature.properties.SUBZONE_N || feature.properties.subzone_name || 'Unknown';
    layer.on({
      mouseover(e) {
        e.target.setStyle({ fillOpacity:0.85, weight:2, color:'#fff' });
      },
      mouseout(e) {
        geoLayer.resetStyle(e.target);
      },
      click(e) {
        const s = scoresData[name];
        const tier = s ? s.tier : 'No data';
        const score = s ? s.score : '—';
        const vuln  = s ? s.vulnerability_index : '—';
        const cases = s ? s.case_count : '—';
        const color = TIER_COLORS[tier] || '#374151';

        const popup = `
          <div class="popup-name">${name}</div>
          <div class="popup-tier" style="background:${color}">${tier}</div>
          <div class="popup-row"><span class="popup-key">Model Score</span><span class="popup-val">${score}</span></div>
          <div class="popup-row"><span class="popup-key">Vulnerability Index</span><span class="popup-val">${vuln}</span></div>
          <div class="popup-row"><span class="popup-key">Cases This Week</span><span class="popup-val">${cases}</span></div>
          <div class="popup-row"><span class="popup-key">P(cluster next 14d)</span><span class="popup-val">${score}</span></div>
        `;
        layer.bindPopup(popup, {maxWidth:240}).openPopup();
      }
    });
  }

  function updateSidebar() {
    const entries = Object.entries(scoresData)
      .sort((a,b) => b[1].score - a[1].score)
      .slice(0, 80);

    document.getElementById('sidebarCount').textContent = entries.length + ' subzones';

    const html = entries.map(([name, s]) => {
      const color = TIER_COLORS[s.tier] || '#374151';
      return `<div class="subzone-item" onclick="flyTo('${name}')">
        <div class="subzone-name">${name}</div>
        <div style="display:flex;align-items:center;gap:8px">
          <span class="tier-badge" style="background:${color}">${s.tier}</span>
          <span class="score-val">${s.score}</span>
        </div>
      </div>`;
    }).join('');

    document.getElementById('sidebarList').innerHTML = html;

    const counts = {High:0, Medium:0, Low:0};
    Object.values(scoresData).forEach(s => { if(counts[s.tier]!==undefined) counts[s.tier]++; });
    document.getElementById('statHigh').textContent = counts.High;
    document.getElementById('statMed').textContent  = counts.Medium;
    document.getElementById('statLow').textContent  = counts.Low;

    const now = new Date();
    document.getElementById('lastUpdate').textContent =
      'Updated ' + String(now.getHours()).padStart(2,'0') + ':' + String(now.getMinutes()).padStart(2,'0') + ':' + String(now.getSeconds()).padStart(2,'0');
  }

  function flyTo(name) {
    if (!geoLayer) return;
    geoLayer.eachLayer(layer => {
      const n = layer.feature.properties.SUBZONE_N || layer.feature.properties.subzone_name || '';
      if (n === name) {
        map.flyToBounds(layer.getBounds(), {duration:0.8, padding:[40,40]});
        layer.fire('click');
      }
    });
  }

  async function loadScores() {
    try {
      const resp = await fetch('/scores');
      scoresData = await resp.json();
      if (geoLayer) geoLayer.setStyle(styleFeature);
      updateSidebar();
    } catch(e) {
      console.error('Failed to load scores:', e);
    }
  }

  async function init() {
    // Load GeoJSON
    try {
      const resp = await fetch('/geojson');
      const gj   = await resp.json();

      geoLayer = L.geoJSON(gj, {
        style: styleFeature,
        onEachFeature: onEachFeature,
      }).addTo(map);

      map.fitBounds(geoLayer.getBounds(), {padding:[20,20]});
    } catch(e) {
      console.error('Failed to load GeoJSON:', e);
    }

    // Load scores
    await loadScores();

    // Hide loading overlay
    document.getElementById('loadingOverlay').style.display = 'none';

    // Auto-refresh scores every 30s
    setInterval(loadScores, 30000);
  }

  init();
</script>
</body>
</html>"""
    return HTMLResponse(content=html)


@app.get("/mobile", response_class=HTMLResponse)
def mobile_emulator():
    """
    CDA Officer mobile alert emulator.
    Simulates the real-time push notification a CDC Officer receives
    when a confirmed case triggers an alert.
    Connects live to the /cases/confirmed endpoint.
    """
    html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CDA Officer Mobile — RT Alert Emulator</title>
<link href="https://fonts.googleapis.com/css2?family=SF+Pro+Display:wght@300;400;600&family=Space+Grotesk:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
  * { margin:0; padding:0; box-sizing:border-box; }
  body {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    min-height:100vh; display:flex; align-items:center; justify-content:center;
    font-family:'Space Grotesk',sans-serif;
    padding: 20px;
  }

  .scene {
    display:flex; gap:60px; align-items:flex-start; flex-wrap:wrap; justify-content:center;
  }

  /* Phone frame */
  .phone {
    width: 320px;
    background: #1c1c1e;
    border-radius: 50px;
    padding: 12px;
    box-shadow:
      0 0 0 2px #3a3a3c,
      0 40px 80px rgba(0,0,0,.6),
      inset 0 0 0 1px rgba(255,255,255,.05);
    position:relative;
  }
  .phone-screen {
    background: #000;
    border-radius: 40px;
    overflow: hidden;
    min-height: 620px;
    position:relative;
  }
  .notch {
    width: 120px; height: 30px;
    background: #1c1c1e;
    border-radius: 0 0 20px 20px;
    margin: 0 auto;
    position:relative; z-index:10;
  }

  /* Lock screen */
  .lock-screen {
    background: linear-gradient(180deg, #1a1a2e 0%, #0d1117 100%);
    min-height: 590px;
    padding: 20px;
    display:flex; flex-direction:column;
  }
  .status-bar {
    display:flex; justify-content:space-between; align-items:center;
    font-size:12px; color:#fff; padding:4px 8px 12px;
  }
  .time-display {
    text-align:center; margin: 20px 0 30px;
  }
  .time-display .time {
    font-size:56px; font-weight:300; color:#fff; letter-spacing:-2px;
    line-height:1;
  }
  .time-display .date {
    font-size:14px; color:rgba(255,255,255,.7); margin-top:4px;
  }

  /* Notification card */
  .notification {
    background: rgba(255,255,255,.12);
    backdrop-filter: blur(20px);
    border-radius:16px;
    padding:14px;
    margin-bottom:10px;
    border:1px solid rgba(255,255,255,.1);
    transform: translateY(-20px);
    opacity:0;
    transition: all .5s cubic-bezier(.34,1.56,.64,1);
  }
  .notification.show {
    transform: translateY(0);
    opacity:1;
  }
  .notification.alert {
    background: rgba(239,68,68,.2);
    border:1px solid rgba(239,68,68,.4);
    animation: alertPulse 2s ease-in-out infinite;
  }
  @keyframes alertPulse {
    0%,100% { box-shadow: 0 0 0 0 rgba(239,68,68,0); }
    50% { box-shadow: 0 0 0 8px rgba(239,68,68,.15); }
  }
  .notif-header {
    display:flex; align-items:center; gap:8px; margin-bottom:8px;
  }
  .notif-icon {
    width:28px; height:28px; border-radius:8px;
    display:flex; align-items:center; justify-content:center;
    font-size:14px;
  }
  .notif-icon.health { background:#007aff; }
  .notif-icon.alert-icon { background:#ef4444; }
  .notif-app { font-size:11px; color:rgba(255,255,255,.6); text-transform:uppercase; letter-spacing:.06em; }
  .notif-time { font-size:11px; color:rgba(255,255,255,.4); margin-left:auto; }
  .notif-title { font-size:13px; font-weight:600; color:#fff; margin-bottom:2px; }
  .notif-body { font-size:12px; color:rgba(255,255,255,.75); line-height:1.4; }

  .score-display {
    display:flex; gap:8px; margin-top:10px;
  }
  .score-chip {
    background:rgba(239,68,68,.25); border:1px solid rgba(239,68,68,.5);
    border-radius:8px; padding:6px 10px; flex:1; text-align:center;
  }
  .score-chip .chip-label { font-size:9px; color:rgba(255,255,255,.5); text-transform:uppercase; letter-spacing:.06em; }
  .score-chip .chip-value { font-size:16px; font-weight:700; color:#ef4444; font-family:monospace; }

  .action-row {
    display:flex; gap:8px; margin-top:10px;
  }
  .action-btn {
    flex:1; padding:8px; border-radius:10px; border:none;
    font-size:12px; font-weight:600; cursor:pointer;
    font-family:'Space Grotesk',sans-serif;
  }
  .action-btn.primary { background:#ef4444; color:#fff; }
  .action-btn.secondary { background:rgba(255,255,255,.15); color:#fff; }

  /* Control panel */
  .controls {
    width:320px;
    color:#fff;
  }
  .controls h2 {
    font-size:22px; font-weight:700; margin-bottom:4px;
    background: linear-gradient(135deg, #06b6d4, #3b82f6);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
  }
  .controls .desc {
    font-size:13px; color:rgba(255,255,255,.5); margin-bottom:28px; line-height:1.5;
  }

  .form-group { margin-bottom:16px; }
  label { display:block; font-size:11px; color:rgba(255,255,255,.5); text-transform:uppercase; letter-spacing:.08em; margin-bottom:6px; }
  select, input {
    width:100%; padding:12px 14px;
    background:rgba(255,255,255,.07); border:1px solid rgba(255,255,255,.12);
    border-radius:10px; color:#fff; font-size:14px;
    font-family:'Space Grotesk',sans-serif;
    outline:none; transition:border .2s;
  }
  select:focus, input:focus { border-color:#06b6d4; }
  select option { background:#1c1c1e; }

  .submit-btn {
    width:100%; padding:14px;
    background:linear-gradient(135deg,#ef4444,#dc2626);
    border:none; border-radius:12px; color:#fff;
    font-size:15px; font-weight:600; cursor:pointer;
    font-family:'Space Grotesk',sans-serif;
    transition:transform .15s, box-shadow .15s;
    margin-bottom:16px;
  }
  .submit-btn:hover { transform:translateY(-1px); box-shadow:0 8px 24px rgba(239,68,68,.4); }
  .submit-btn:active { transform:translateY(0); }
  .submit-btn:disabled { opacity:.5; cursor:not-allowed; transform:none; }

  .result-box {
    background:rgba(255,255,255,.05); border:1px solid rgba(255,255,255,.1);
    border-radius:12px; padding:16px; display:none;
  }
  .result-box.show { display:block; animation:fadeIn .4s ease; }
  .result-box.triggered { border-color:rgba(239,68,68,.4); background:rgba(239,68,68,.08); }
  @keyframes fadeIn { from{opacity:0;transform:translateY(8px)} to{opacity:1;transform:none} }

  .result-row {
    display:flex; justify-content:space-between; align-items:center;
    padding:8px 0; border-bottom:1px solid rgba(255,255,255,.06); font-size:13px;
  }
  .result-row:last-child { border:none; }
  .result-key { color:rgba(255,255,255,.5); }
  .result-val { font-weight:600; font-family:monospace; }
  .result-val.red { color:#ef4444; }
  .result-val.green { color:#10b981; }
  .result-val.yellow { color:#f59e0b; }

  .latency { font-size:11px; color:rgba(255,255,255,.3); text-align:right; margin-top:8px; font-family:monospace; }

  .error-box {
    background:rgba(239,68,68,.1); border:1px solid rgba(239,68,68,.3);
    border-radius:10px; padding:12px; font-size:12px; color:#fca5a5;
    display:none; margin-bottom:12px;
  }
  .error-box.show { display:block; }
</style>
</head>
<body>

<div class="scene">

  <!-- Phone -->
  <div class="phone">
    <div class="phone-screen">
      <div class="notch"></div>
      <div class="lock-screen">
        <div class="status-bar">
          <span id="clock">--:--</span>
          <span>CDA · MOH</span>
          <span>⬛⬛⬛ 📶</span>
        </div>
        <div class="time-display">
          <div class="time" id="bigClock">--:--</div>
          <div class="date" id="bigDate">Thursday, 22 May 2026</div>
        </div>

        <!-- Idle notification -->
        <div class="notification show" id="idleNotif">
          <div class="notif-header">
            <div class="notif-icon health">🏥</div>
            <div class="notif-app">CDA Health System</div>
            <div class="notif-time">Now</div>
          </div>
          <div class="notif-title">System Active</div>
          <div class="notif-body">Monitoring confirmed case stream. Awaiting submissions.</div>
        </div>

        <!-- Alert notification (hidden until triggered) -->
        <div class="notification alert" id="alertNotif" style="display:none">
          <div class="notif-header">
            <div class="notif-icon alert-icon">🚨</div>
            <div class="notif-app">DENGUE ALERT</div>
            <div class="notif-time" id="alertTime">Now</div>
          </div>
          <div class="notif-title" id="alertTitle">High Risk Alert</div>
          <div class="notif-body" id="alertBody">Alert triggered for subzone</div>
          <div class="score-display">
            <div class="score-chip">
              <div class="chip-label">Model Score</div>
              <div class="chip-value" id="chipScore">—</div>
            </div>
            <div class="score-chip">
              <div class="chip-label">Alert Score</div>
              <div class="chip-value" id="chipAlert">—</div>
            </div>
            <div class="score-chip">
              <div class="chip-label">Cases</div>
              <div class="chip-value" id="chipCases">—</div>
            </div>
          </div>
          <div class="action-row">
            <button class="action-btn primary">Dispatch Team</button>
            <button class="action-btn secondary">View Details</button>
          </div>
        </div>

      </div>
    </div>
  </div>

  <!-- Controls -->
  <div class="controls">
    <h2>RT Alert Emulator</h2>
    <p class="desc">Simulates the CDA Officer mobile experience.<br>Submit a confirmed case to trigger the real-time inference pipeline.</p>

    <div class="error-box" id="errorBox"></div>

    <div class="form-group">
      <label>Subzone</label>
      <select id="subzoneSelect">
        <option value="TAMPINES EAST">TAMPINES EAST</option>
        <option value="BEDOK NORTH">BEDOK NORTH</option>
        <option value="JURONG WEST CENTRAL">JURONG WEST CENTRAL</option>
        <option value="WOODLANDS EAST">WOODLANDS EAST</option>
        <option value="YISHUN CENTRAL">YISHUN CENTRAL</option>
        <option value="SENGKANG WEST">SENGKANG WEST</option>
        <option value="PUNGGOL FIELD">PUNGGOL FIELD</option>
        <option value="ANG MO KIO TOWN CENTRE">ANG MO KIO TOWN CENTRE</option>
        <option value="BUKIT PANJANG">BUKIT PANJANG</option>
        <option value="CLEMENTI WEST">CLEMENTI WEST</option>
      </select>
    </div>

    <div class="form-group">
      <label>Confirmed Cases</label>
      <input type="number" id="caseCount" value="3" min="1" max="50">
    </div>

    <div class="form-group">
      <label>Source</label>
      <select id="source">
        <option value="hospital">Hospital</option>
        <option value="clinic">GP Clinic</option>
        <option value="nea">NEA Field Report</option>
      </select>
    </div>

    <button class="submit-btn" id="submitBtn" onclick="submitCase()">
      Submit Confirmed Case →
    </button>

    <div class="result-box" id="resultBox">
      <div class="result-row">
        <span class="result-key">Subzone</span>
        <span class="result-val" id="resSubzone">—</span>
      </div>
      <div class="result-row">
        <span class="result-key">Model Score</span>
        <span class="result-val yellow" id="resScore">—</span>
      </div>
      <div class="result-row">
        <span class="result-key">Vulnerability Index</span>
        <span class="result-val" id="resVuln">—</span>
      </div>
      <div class="result-row">
        <span class="result-key">Case Count (week)</span>
        <span class="result-val" id="resCases">—</span>
      </div>
      <div class="result-row">
        <span class="result-key">Alert Score</span>
        <span class="result-val" id="resAlert">—</span>
      </div>
      <div class="result-row">
        <span class="result-key">Alert Triggered</span>
        <span class="result-val" id="resTriggered">—</span>
      </div>
      <div class="latency" id="resLatency"></div>
    </div>
  </div>

</div>

<script>
  // Update clock
  function updateClock() {
    const now = new Date();
    const h = String(now.getHours()).padStart(2,'0');
    const m = String(now.getMinutes()).padStart(2,'0');
    document.getElementById('clock').textContent = h+':'+m;
    document.getElementById('bigClock').textContent = h+':'+m;
  }
  updateClock();
  setInterval(updateClock, 1000);

  async function submitCase() {
    const btn = document.getElementById('submitBtn');
    const subzone = document.getElementById('subzoneSelect').value;
    const count   = parseInt(document.getElementById('caseCount').value);
    const source  = document.getElementById('source').value;
    const errBox  = document.getElementById('errorBox');

    btn.disabled = true;
    btn.textContent = 'Submitting...';
    errBox.classList.remove('show');

    try {
      const resp = await fetch('/cases/confirmed', {
        method: 'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify({subzone_name: subzone, case_count: count, source: source})
      });

      if (!resp.ok) throw new Error('API returned ' + resp.status);
      const data = await resp.json();

      // Update result box
      const resultBox = document.getElementById('resultBox');
      resultBox.className = 'result-box show' + (data.alert_triggered ? ' triggered' : '');

      document.getElementById('resSubzone').textContent  = data.subzone_name;
      document.getElementById('resScore').textContent    = data.model_score;
      document.getElementById('resVuln').textContent     = data.vulnerability_index;
      document.getElementById('resCases').textContent    = data.current_case_count;
      document.getElementById('resAlert').textContent    = data.alert_score;
      document.getElementById('resLatency').textContent  = `Latency: ${data.latency_ms}ms`;

      const triggered = data.alert_triggered;
      const tElem = document.getElementById('resTriggered');
      tElem.textContent = triggered ? '🚨 YES' : '✓ No';
      tElem.className = 'result-val ' + (triggered ? 'red' : 'green');

      // Phone notification
      if (triggered) {
        document.getElementById('idleNotif').style.display = 'none';
        const alertNotif = document.getElementById('alertNotif');
        alertNotif.style.display = 'block';
        setTimeout(() => alertNotif.classList.add('show'), 50);

        const now = new Date();
        document.getElementById('alertTime').textContent =
          String(now.getHours()).padStart(2,'0') + ':' + String(now.getMinutes()).padStart(2,'0');
        document.getElementById('alertTitle').textContent =
          '🚨 High Risk Alert — ' + data.subzone_name;
        document.getElementById('alertBody').textContent =
          `Alert score ${data.alert_score} exceeded threshold. Immediate triage required.`;
        document.getElementById('chipScore').textContent  = data.model_score;
        document.getElementById('chipAlert').textContent  = data.alert_score;
        document.getElementById('chipCases').textContent  = data.current_case_count;
      } else {
        // Reset to idle
        document.getElementById('idleNotif').style.display = 'block';
        document.getElementById('alertNotif').style.display = 'none';
        document.getElementById('alertNotif').classList.remove('show');
      }

    } catch(e) {
      errBox.textContent = 'Error: ' + e.message + '. Is the FastAPI server running?';
      errBox.classList.add('show');
    } finally {
      btn.disabled = false;
      btn.textContent = 'Submit Confirmed Case →';
    }
  }
</script>

</body>
</html>"""
    return HTMLResponse(content=html)
