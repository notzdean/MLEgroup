"""
inference/airflow/batch_dag.py
================================
CS611 MLE Group Project — Dengue Outbreak Risk Prediction
Airflow DAG: Weekly batch inference — Monday 06:00 SGT.

What this DAG does
------------------
1. Load Production model from MLflow (or local fallback)
2. Read gold.subzone_features (all 320 subzones, latest snapshot)
3. Score all 320 subzones
4. Apply risk tier logic: Low / Medium / High
5. Write scores to operational.risk_tier
6. Pre-load vulnerability_index into Redis (for real-time inference)
7. Notify NEA dashboard (API call)

Risk tier thresholds
--------------------
High   : score >= 0.6
Medium : score >= 0.3
Low    : score < 0.3

Design note
-----------
Same Production model artifact used by both batch and real-time paths.
"One trained artifact — served two ways."
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

# ── DAG config ────────────────────────────────────────────────────────────────

default_args = {
    "owner":            "cs611_group4",
    "depends_on_past":  False,
    "email_on_failure": False,
    "email_on_retry":   False,
    "retries":          1,
    "retry_delay":      timedelta(minutes=5),
}

dag = DAG(
    dag_id="dengue_batch_inference",
    description="Weekly dengue risk scoring for all 320 subzones — NEA Ops",
    schedule="0 22 * * 0",   # Sunday 22:00 UTC = Monday 06:00 SGT
    start_date=datetime(2024, 1, 1),
    catchup=False,
    default_args=default_args,
    tags=["dengue", "batch", "inference"],
)

# ── Task functions ─────────────────────────────────────────────────────────────

def load_model(**context):
    """Load Production model from MLflow, push model_type to XCom."""
    import json
    import os

    model_dir = "/opt/airflow/model/candidate"
    try:
        import mlflow
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
        model = mlflow.pyfunc.load_model("models:/dengue_cluster_model/Production")
        context["ti"].xcom_push(key="model_type", value="mlflow")
        print("Model loaded from MLflow")
    except Exception as e:
        print(f"MLflow unavailable ({e}) — using local model")
        with open(f"{model_dir}/candidate_meta.json") as f:
            meta = json.load(f)
        context["ti"].xcom_push(key="model_type", value=meta["model_type"])
        print(f"Local model type: {meta['model_type']}")


def score_all_subzones(**context):
    """Score all subzones and push results to XCom."""
    import json
    import os

    import numpy as np
    import pandas as pd

    model_dir  = "/opt/airflow/model/candidate"
    gold_path  = "/opt/airflow/data/gold/subzone_features.parquet"
    model_type = context["ti"].xcom_pull(key="model_type")

    features = [
        "rainfall_lag1w", "rainfall_lag2w", "rainfall_lag4w",
        "cluster_count_rolling2w", "cluster_count_rolling4w",
        "recent_cases_rolling2w", "recent_cases_rolling4w",
        "population", "elderly_pct", "area_km2", "population_density",
        "vulnerability_index",
    ]

    # Load Gold features — latest snapshot per subzone
    df = pd.read_parquet(gold_path)
    df["date"] = pd.to_datetime(df["date"])
    latest = df.sort_values("date").groupby("subzone_name").last().reset_index()
    print(f"Scoring {len(latest)} subzones")

    X = latest[features].fillna(0).values

    # Load model
    if model_type == "mlflow":
        import mlflow
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
        model  = mlflow.pyfunc.load_model("models:/dengue_cluster_model/Production")
        scores = model.predict(pd.DataFrame(X, columns=features))
    elif model_type == "xgboost":
        import xgboost as xgb
        model = xgb.XGBClassifier()
        model.load_model(f"{model_dir}/model.xgb")
        scores = model.predict_proba(X)[:, 1]
    else:
        import lightgbm as lgb
        model  = lgb.Booster(model_file=f"{model_dir}/model.lgb")
        scores = model.predict(X)

    # Apply risk tier
    def risk_tier(score):
        if score >= 0.6:   return "High"
        elif score >= 0.3: return "Medium"
        else:              return "Low"

    results = []
    for i, row in latest.iterrows():
        results.append({
            "subzone_name":      row["subzone_name"],
            "score":             round(float(scores[i]), 4),
            "risk_tier":         risk_tier(float(scores[i])),
            "vulnerability_index": round(float(row.get("vulnerability_index", 0)), 4),
        })

    print(f"High: {sum(1 for r in results if r['risk_tier']=='High')} | "
          f"Medium: {sum(1 for r in results if r['risk_tier']=='Medium')} | "
          f"Low: {sum(1 for r in results if r['risk_tier']=='Low')}")

    context["ti"].xcom_push(key="scores", value=results)


def write_to_postgres(**context):
    """Write risk tier scores to operational.risk_tier."""
    import os
    from sqlalchemy import create_engine, text

    results   = context["ti"].xcom_pull(key="scores")
    db_url    = os.getenv("DATABASE_URL", "postgresql://dengue:dengue@postgres:5432/dengue")
    week_start = context["ds"]  # DAG run date

    engine = create_engine(db_url)
    with engine.begin() as conn:
        for r in results:
            conn.execute(text("""
                INSERT INTO operational.risk_tier
                    (subzone_name, score, risk_tier, week_start, split)
                VALUES
                    (:subzone, :score, :tier, :week, 'batch')
                ON CONFLICT DO NOTHING
            """), {
                "subzone": r["subzone_name"],
                "score":   r["score"],
                "tier":    r["risk_tier"],
                "week":    week_start,
            })
    print(f"Written {len(results)} scores to operational.risk_tier")


def preload_redis(**context):
    """
    Pre-load vulnerability_index into Redis for real-time inference.
    Resets current_week_case_count to 0 for the new week.
    """
    import os
    import redis

    results   = context["ti"].xcom_pull(key="scores")
    redis_url = os.getenv("REDIS_URL", "redis://redis:6379/0")

    r = redis.from_url(redis_url, decode_responses=True)

    for subzone in results:
        name = subzone["subzone_name"]
        r.set(f"vulnerability_index:{name}",        subzone["vulnerability_index"])
        r.set(f"current_week_case_count:{name}", 0)

    print(f"Redis preloaded: {len(results)} subzones")


def notify_nea_dashboard(**context):
    """Notify NEA dashboard that new scores are available."""
    import os
    import requests

    nea_webhook = os.getenv("NEA_WEBHOOK_URL", "")
    if not nea_webhook:
        print("NEA_WEBHOOK_URL not set — skipping notification")
        return

    results    = context["ti"].xcom_pull(key="scores")
    high_risk  = [r for r in results if r["risk_tier"] == "High"]
    week_start = context["ds"]

    try:
        resp = requests.post(nea_webhook, json={
            "event":      "weekly_risk_scores_ready",
            "week_start": week_start,
            "high_risk_count": len(high_risk),
            "high_risk_subzones": [r["subzone_name"] for r in high_risk],
        }, timeout=10)
        print(f"NEA notified — status {resp.status_code}")
    except Exception as e:
        print(f"NEA notification failed ({e}) — scores are still in Postgres")


# ── Task definitions ──────────────────────────────────────────────────────────

t1 = PythonOperator(task_id="load_model",          python_callable=load_model,          dag=dag)
t2 = PythonOperator(task_id="score_all_subzones",  python_callable=score_all_subzones,  dag=dag)
t3 = PythonOperator(task_id="write_to_postgres",   python_callable=write_to_postgres,   dag=dag)
t4 = PythonOperator(task_id="preload_redis",       python_callable=preload_redis,       dag=dag)
t5 = PythonOperator(task_id="notify_nea_dashboard",python_callable=notify_nea_dashboard,dag=dag)

t1 >> t2 >> t3 >> t4 >> t5
