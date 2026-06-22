"""
monitoring/monitor.py
======================
CS611 MLE Group Project — Dengue Outbreak Risk Prediction
Model monitoring: PSI / CSI drift detection + data quality checks + retrain trigger.

Scheduled via Airflow (Wed 06:00 SGT — see airflow/dags/retrain_dag.py).

What this script does
---------------------
1. DATA QUALITY CHECKS — pipeline health before model checks:
   - Freshness: how old is the latest Gold snapshot?
   - Completeness: null rates for key features
   - Subzone coverage: expected ~320 subzones in latest snapshot
   - Cluster staleness: cluster rolling counts suspiciously near zero?
   - Rainfall staleness: rainfall lags all zero (weather ingest broken)?
2. Loads recent model predictions from operational.risk_tier (Postgres)
   and compares score distribution to the training baseline.
3. Computes PSI (Population Stability Index) on prediction scores.
   PSI < 0.10  -> stable
   PSI 0.10-0.20 -> minor shift, log warning
   PSI > 0.20  -> significant drift, raise alarm
4. Computes CSI (Characteristic Stability Index) per feature.
5. Logs all metrics to MLflow.
6. If PSI > 0.20: writes a drift alarm to operational.drift_alarms
   and triggers the retrain DAG via Airflow REST API.

Drift context
-------------
Jan-Nov 2020 DENV-3 serotype shift is a known concept drift event.
PSI thresholds are calibrated to flag genuine distribution shifts
without false-alarming on seasonal variation.

Local fallback
--------------
If Postgres/MLflow/Airflow are not running (local dev without Docker),
the script reads from data/gold/subzone_features.parquet and
data/gold/training_score_baseline.parquet and writes results to
model/monitoring_report.json.
"""

import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

ROOT       = Path(__file__).resolve().parent.parent
GOLD       = ROOT / "data" / "gold"
MODEL_DIR  = ROOT / "model"
REPORT_DIR = ROOT / "model"

MLFLOW_TRACKING_URI = "http://172.18.0.4:5000"
AIRFLOW_API_URL     = "http://172.18.0.5:8080/api/v1"
AIRFLOW_DAG_ID      = "dengue_retrain_dag"
POSTGRES_DSN        = "postgresql://dengue:dengue@172.18.0.3:5432/dengue"

# PSI thresholds
PSI_WARN  = 0.10
PSI_ALARM = 0.20

# CSI threshold per feature
CSI_ALARM = 0.25

FEATURES = [
    "rainfall_lag1w", "rainfall_lag2w", "rainfall_lag4w",
    "cluster_count_rolling2w", "cluster_count_rolling4w",
    "recent_cases_rolling2w", "recent_cases_rolling4w",
    "population", "elderly_pct", "area_km2", "population_density",
    "vulnerability_index",
]


# ── Data quality checks ───────────────────────────────────────────────────────

# Expected values for quality gates
EXPECTED_SUBZONES       = 320   # residential subzones in Singapore
MAX_FRESHNESS_DAYS      = 21    # SGCharts is bi-weekly; >21 days = stale
MAX_NULL_RATE           = 0.05  # >5% nulls in a feature = problem
MIN_CLUSTER_COUNT       = 0.5   # avg cluster rolling count below this = SGCharts stale
MIN_RAINFALL_MM         = 0.01  # avg rainfall lag below this = weather ingest stale


def check_data_quality() -> dict:
    """
    Check pipeline data quality — runs before model drift checks.

    Checks:
    1. Freshness    — how old is the latest Gold snapshot date?
    2. Completeness — null rates for key features in latest snapshot
    3. Coverage     — how many subzones in latest snapshot vs expected 320
    4. Cluster data — cluster rolling counts suspiciously near zero (SGCharts stale?)
    5. Rainfall data — rainfall lags all near zero (weather ingest broken?)

    Returns a dict of check results with pass/fail flags.
    Falls back to local Gold parquet if Postgres unavailable.
    """
    print("\n[data quality] Checking pipeline health")
    checks = {}

    try:
        df = pd.read_parquet(GOLD / "subzone_features.parquet")
        df["date"] = pd.to_datetime(df["date"])
        latest_date = df["date"].max()
        latest = df[df["date"] == latest_date]
    except Exception as e:
        print(f"  [error] Cannot load Gold parquet: {e}")
        return {"error": str(e), "all_pass": False}

    # 1. Freshness — days since last snapshot
    days_old = (datetime.now() - latest_date.to_pydatetime()).days
    freshness_pass = days_old <= MAX_FRESHNESS_DAYS
    checks["freshness"] = {
        "latest_snapshot_date": str(latest_date.date()),
        "days_since_update":    days_old,
        "threshold_days":       MAX_FRESHNESS_DAYS,
        "pass":                 freshness_pass,
        "flag":                 "ok" if freshness_pass else "STALE — pipeline may be broken",
    }
    status = "ok" if freshness_pass else "STALE"
    print(f"  Freshness      : {status} — last snapshot {days_old} days ago ({latest_date.date()})")

    # 2. Completeness — null rates per feature in latest snapshot
    null_checks = {}
    any_null_fail = False
    for feat in FEATURES:
        if feat not in latest.columns:
            continue
        null_rate = round(latest[feat].isna().mean(), 4)
        passed    = null_rate <= MAX_NULL_RATE
        null_checks[feat] = {"null_rate": null_rate, "pass": passed}
        if not passed:
            any_null_fail = True
            print(f"  Null rate FAIL : {feat} = {null_rate:.1%} (threshold {MAX_NULL_RATE:.0%})")

    checks["completeness"] = {
        "feature_null_rates": null_checks,
        "any_fail":           any_null_fail,
        "pass":               not any_null_fail,
        "flag":               "ok" if not any_null_fail else "NULL RATE EXCEEDED",
    }
    if not any_null_fail:
        print(f"  Completeness   : ok — all features within null threshold")

    # 3. Coverage — subzone count in latest snapshot
    subzone_count = latest["subzone_name"].nunique() if "subzone_name" in latest.columns else len(latest)
    coverage_pass = subzone_count >= (EXPECTED_SUBZONES * 0.9)  # allow 10% missing
    checks["coverage"] = {
        "subzone_count":    subzone_count,
        "expected":         EXPECTED_SUBZONES,
        "pass":             coverage_pass,
        "flag":             "ok" if coverage_pass else f"LOW COVERAGE — only {subzone_count}/{EXPECTED_SUBZONES} subzones",
    }
    status = "ok" if coverage_pass else "LOW"
    print(f"  Coverage       : {status} — {subzone_count}/{EXPECTED_SUBZONES} subzones in latest snapshot")

    # 4. Cluster staleness — if rolling counts near zero, SGCharts may have stopped
    cluster_mean = 0.0
    cluster_pass = True
    if "cluster_count_rolling2w" in latest.columns:
        cluster_mean = float(latest["cluster_count_rolling2w"].mean())
        cluster_pass = cluster_mean >= MIN_CLUSTER_COUNT
    checks["cluster_activity"] = {
        "avg_cluster_count_rolling2w": round(cluster_mean, 3),
        "threshold":                   MIN_CLUSTER_COUNT,
        "pass":                        cluster_pass,
        "flag":                        "ok" if cluster_pass else "NEAR ZERO — SGCharts ingest may be stale",
    }
    status = "ok" if cluster_pass else "STALE"
    print(f"  Cluster data   : {status} — avg rolling2w count = {cluster_mean:.2f}")

    # 5. Rainfall staleness — if lags all near zero, weather ingest may be broken
    rainfall_mean = 0.0
    rainfall_pass = True
    if "rainfall_lag1w" in latest.columns:
        rainfall_mean = float(latest["rainfall_lag1w"].mean())
        rainfall_pass = rainfall_mean >= MIN_RAINFALL_MM
    checks["rainfall_data"] = {
        "avg_rainfall_lag1w_mm": round(rainfall_mean, 4),
        "threshold_mm":          MIN_RAINFALL_MM,
        "pass":                  rainfall_pass,
        "flag":                  "ok" if rainfall_pass else "NEAR ZERO — MSS weather ingest may be broken",
    }
    status = "ok" if rainfall_pass else "STALE"
    print(f"  Rainfall data  : {status} — avg lag1w = {rainfall_mean:.4f} mm")

    # Overall
    all_pass = (
        freshness_pass and
        not any_null_fail and
        coverage_pass and
        cluster_pass and
        rainfall_pass
    )
    checks["all_pass"] = all_pass
    overall = "ALL CHECKS PASSED" if all_pass else "DATA QUALITY ISSUES DETECTED"
    print(f"  Overall        : {overall}")

    return checks


# ── PSI ───────────────────────────────────────────────────────────────────────

def compute_psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """
    Population Stability Index.
    Measures shift in score/feature distribution between baseline and current.
    PSI = Σ (Actual% - Expected%) × ln(Actual% / Expected%)
    """
    breakpoints = np.linspace(
        min(expected.min(), actual.min()),
        max(expected.max(), actual.max()),
        bins + 1
    )
    e_counts = np.histogram(expected, bins=breakpoints)[0]
    a_counts = np.histogram(actual,   bins=breakpoints)[0]

    # Replace zeros to avoid log(0)
    e_pct = np.where(e_counts == 0, 1e-4, e_counts / len(expected))
    a_pct = np.where(a_counts == 0, 1e-4, a_counts / len(actual))

    psi = float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))
    return round(psi, 4)


def psi_flag(psi: float) -> str:
    if psi < PSI_WARN:
        return "stable"
    elif psi < PSI_ALARM:
        return "minor_shift"
    else:
        return "significant_drift"


# ── CSI ───────────────────────────────────────────────────────────────────────

def compute_csi(baseline_df: pd.DataFrame, current_df: pd.DataFrame) -> dict:
    """
    Characteristic Stability Index — PSI applied per feature.
    Measures if individual feature distributions have shifted.
    """
    csi_results = {}
    for feat in FEATURES:
        if feat not in baseline_df.columns or feat not in current_df.columns:
            continue
        b = baseline_df[feat].dropna().values
        c = current_df[feat].dropna().values
        if len(b) < 10 or len(c) < 10:
            continue
        csi_results[feat] = {
            "csi":  compute_psi(b, c),
            "flag": psi_flag(compute_psi(b, c))
        }
    return csi_results


# ── data loading ──────────────────────────────────────────────────────────────

def load_baseline() -> tuple[np.ndarray, pd.DataFrame]:
    """
    Load training score baseline and feature distributions.
    Tries Postgres first, falls back to local parquet.
    """
    try:
        from sqlalchemy import create_engine, text
        engine = create_engine(POSTGRES_DSN)
        with engine.connect() as conn:
            scores = pd.read_sql(
                text("SELECT score FROM operational.risk_tier WHERE split = 'train'"),
                conn
            )
            features = pd.read_sql(
                text("SELECT * FROM gold.subzone_features WHERE date <= '2018-12-31'"),
                conn
            )
        if len(scores) == 0:
            raise ValueError("empty baseline table")
        print(f"[load] Baseline from Postgres — {len(scores):,} scores")
        return scores["score"].values, features

    except Exception:
        print("[load] Postgres unavailable — loading baseline from parquet")
        df = pd.read_parquet(GOLD / "subzone_features.parquet")
        df["date"] = pd.to_datetime(df["date"])
        train = df[df["date"] <= "2018-12-31"]
        # Generate pseudo-scores using the saved model
        scores = _score_with_model(train)
        return scores, train


def load_recent_predictions(lookback_days: int = 14) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Load recent predictions (last N days) for monitoring.
    Tries Postgres first, falls back to local parquet (OOT window).
    """
    cutoff = datetime.now() - timedelta(days=lookback_days)
    try:
        from sqlalchemy import create_engine, text
        engine = create_engine(POSTGRES_DSN)
        with engine.connect() as conn:
            scores = pd.read_sql(
                text(f"SELECT score FROM operational.risk_tier WHERE scored_at >= '{cutoff}'"),
                conn
            )
            features = pd.read_sql(
                text(f"SELECT * FROM gold.subzone_features WHERE date >= '{cutoff.date()}'"),
                conn
            )
        if len(scores) == 0:
            raise ValueError("empty predictions table")
        print(f"[load] Recent predictions from Postgres — {len(scores):,} scores")
        return scores["score"].values, features

    except Exception:
        print("[load] Postgres unavailable — using OOT window as recent data")
        df = pd.read_parquet(GOLD / "subzone_features.parquet")
        df["date"] = pd.to_datetime(df["date"])
        oot = df[df["date"] >= "2020-01-01"]
        scores = _score_with_model(oot)
        return scores, oot


def _score_with_model(df: pd.DataFrame) -> np.ndarray:
    """Score a dataframe using the saved candidate model."""
    try:
        meta_path = MODEL_DIR / "candidate" / "candidate_meta.json"
        with open(meta_path) as f:
            meta = json.load(f)

        model_type = meta["model_type"]
        features   = meta["features"]
        X = df[features].fillna(0).values

        if model_type == "xgboost":
            import xgboost as xgb
            model = xgb.XGBClassifier()
            model.load_model(MODEL_DIR / "candidate" / "model.xgb")
            return model.predict_proba(X)[:, 1]
        else:
            import lightgbm as lgb
            model = lgb.Booster(model_file=str(MODEL_DIR / "candidate" / "model.lgb"))
            return model.predict(X)

    except Exception as e:
        print(f"  [warn] Could not score with model ({e}) - using random scores as placeholder")
        return np.random.uniform(0, 1, len(df))


# ── Feature attribution drift ─────────────────────────────────────────────────

# Thresholds for SHAP drift
SHAP_RANK_SHIFT_ALARM  = 3    # flag if a feature moves more than 3 positions in ranking
SHAP_MAG_CHANGE_ALARM  = 0.50 # flag if mean |SHAP| changes by more than 50% relative


def check_feature_attribution_drift(recent_features: pd.DataFrame) -> dict:
    """
    Compare current SHAP feature importance against the baseline from training.

    Baseline: shap_importance from model/evaluation_report.json
    Current:  SHAP computed on a sample of recent_features

    Checks:
    - Ranking shift: did any feature move > 3 positions in importance rank?
    - Magnitude change: did mean |SHAP| change by > 50% relative to baseline?

    Large ranking shifts or magnitude changes indicate the model is relying
    on different features in production vs training — a sign of concept drift
    even if PSI alone doesn't catch it.
    """
    print("\n[shap drift] Checking feature attribution drift")

    # Load baseline SHAP from evaluation report
    eval_report_path = MODEL_DIR / "evaluation_report.json"
    try:
        with open(eval_report_path) as f:
            eval_report = json.load(f)
        baseline_shap = eval_report.get("shap_importance", {})
        if not baseline_shap:
            print("  [warn] No baseline SHAP in evaluation_report.json - skipping")
            return {"skipped": True, "reason": "no baseline SHAP found"}
    except Exception as e:
        print(f"  [warn] Could not load evaluation_report.json ({e})")
        return {"skipped": True, "reason": str(e)}

    # Compute current SHAP on a sample of recent data
    try:
        import shap
        meta_path = MODEL_DIR / "candidate" / "candidate_meta.json"
        with open(meta_path) as f:
            meta = json.load(f)

        model_type = meta["model_type"]
        features   = meta["features"]
        X = recent_features[features].fillna(0).values

        # Sample max 300 rows for speed
        idx = np.random.choice(len(X), min(300, len(X)), replace=False)
        X_sample = X[idx]

        if model_type == "xgboost":
            import xgboost as xgb
            model = xgb.XGBClassifier()
            model.load_model(MODEL_DIR / "candidate" / "model.xgb")
            explainer = shap.TreeExplainer(model)
        else:
            import lightgbm as lgb
            model = lgb.Booster(model_file=str(MODEL_DIR / "candidate" / "model.lgb"))
            explainer = shap.TreeExplainer(model)

        shap_values = explainer.shap_values(X_sample)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        mean_shap = np.abs(shap_values).mean(axis=0)
        current_shap = {f: round(float(v), 4) for f, v in zip(features, mean_shap)}

    except Exception as e:
        print(f"  [warn] SHAP computation failed ({e})")
        return {"skipped": True, "reason": str(e)}

    # Compare rankings
    baseline_ranked = sorted(baseline_shap.keys(), key=lambda f: baseline_shap[f], reverse=True)
    current_ranked  = sorted(current_shap.keys(),  key=lambda f: current_shap[f],  reverse=True)

    rank_shifts = {}
    alarmed_features = []
    for feat in features:
        if feat not in baseline_shap or feat not in current_shap:
            continue
        baseline_rank = baseline_ranked.index(feat) + 1 if feat in baseline_ranked else None
        current_rank  = current_ranked.index(feat)  + 1 if feat in current_ranked  else None
        if baseline_rank is None or current_rank is None:
            continue

        rank_shift = abs(current_rank - baseline_rank)
        baseline_mag = baseline_shap[feat]
        current_mag  = current_shap[feat]
        mag_change_pct = round(
            abs(current_mag - baseline_mag) / (baseline_mag + 1e-9) * 100, 1
        )

        rank_alarm = rank_shift > SHAP_RANK_SHIFT_ALARM
        mag_alarm  = (abs(current_mag - baseline_mag) / (baseline_mag + 1e-9)) > SHAP_MAG_CHANGE_ALARM

        rank_shifts[feat] = {
            "baseline_rank":   baseline_rank,
            "current_rank":    current_rank,
            "rank_shift":      rank_shift,
            "baseline_shap":   baseline_mag,
            "current_shap":    current_mag,
            "magnitude_change_pct": mag_change_pct,
            "rank_alarm":      rank_alarm,
            "magnitude_alarm": mag_alarm,
        }

        if rank_alarm or mag_alarm:
            alarmed_features.append(feat)
            flags = []
            if rank_alarm: flags.append(f"rank shifted {rank_shift} positions")
            if mag_alarm:  flags.append(f"magnitude changed {mag_change_pct}%")
            print(f"  [ALARM] {feat:<35} {', '.join(flags)}")
        else:
            print(f"  [ok]    {feat:<35} rank {baseline_rank}->{current_rank}  "
                  f"SHAP {baseline_mag:.4f}->{current_mag:.4f}")

    top_feature_changed = (
        len(baseline_ranked) > 0 and len(current_ranked) > 0 and
        baseline_ranked[0] != current_ranked[0]
    )
    if top_feature_changed:
        print(f"  [ALARM] Top feature changed: {baseline_ranked[0]} -> {current_ranked[0]}")

    all_ok = len(alarmed_features) == 0 and not top_feature_changed
    print(f"  Overall: {'ok - attribution stable' if all_ok else 'DRIFT DETECTED in feature attribution'}")

    return {
        "baseline_shap":       baseline_shap,
        "current_shap":        current_shap,
        "feature_details":     rank_shifts,
        "alarmed_features":    alarmed_features,
        "top_feature_baseline": baseline_ranked[0] if baseline_ranked else None,
        "top_feature_current":  current_ranked[0]  if current_ranked  else None,
        "top_feature_changed":  top_feature_changed,
        "all_ok":              all_ok,
    }


# ── alarm & retrain trigger ───────────────────────────────────────────────────

def write_alarm_to_postgres(psi: float, csi_results: dict):
    """Write drift alarm to operational.drift_alarms table."""
    try:
        from sqlalchemy import create_engine, text
        engine = create_engine(POSTGRES_DSN)
        with engine.connect() as conn:
            conn.execute(text("""
                INSERT INTO operational.drift_alarms
                    (alarm_time, psi, psi_flag, csi_json, action)
                VALUES
                    (NOW(), :psi, :flag, :csi, 'retrain_triggered')
            """), {
                "psi":  psi,
                "flag": psi_flag(psi),
                "csi":  json.dumps(csi_results)
            })
            conn.commit()
        print(f"  [alarm] Written to operational.drift_alarms")
    except Exception as e:
        print(f"  [warn] Could not write alarm to Postgres ({e})")


def trigger_retrain_dag():
    """Trigger Airflow retrain DAG via REST API."""
    try:
        import requests
        resp = requests.post(
            f"{AIRFLOW_API_URL}/dags/{AIRFLOW_DAG_ID}/dagRuns",
            json={"conf": {"triggered_by": "monitor.py", "reason": "psi_threshold_exceeded"}},
            auth=("airflow", "airflow"),
            timeout=10
        )
        if resp.status_code in (200, 201):
            print(f"  [trigger] Retrain DAG triggered — run_id: {resp.json().get('dag_run_id')}")
        else:
            print(f"  [warn] Airflow trigger failed: {resp.status_code} {resp.text}")
    except Exception as e:
        print(f"  [warn] Could not trigger Airflow DAG ({e})")


def log_to_mlflow(psi: float, csi_results: dict, score_psi_flag: str):
    """Log monitoring metrics to MLflow."""
    try:
        import mlflow
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment("dengue_monitoring")

        with mlflow.start_run(run_name=f"monitor_{datetime.now().strftime('%Y%m%d_%H%M')}"):
            mlflow.log_metric("score_psi", psi)
            mlflow.log_param("score_psi_flag", score_psi_flag)
            for feat, result in csi_results.items():
                mlflow.log_metric(f"csi_{feat}", result["csi"])
        print("  [mlflow] Metrics logged")
    except Exception as e:
        print(f"  [warn] MLflow logging failed ({e})")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Model Monitoring — Data Quality + PSI / CSI Drift Detection")
    print(f"Run time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # 1. Data quality checks — run first before model checks
    dq_results = check_data_quality()

    # Load data for model drift checks
    baseline_scores, baseline_features = load_baseline()
    recent_scores,   recent_features   = load_recent_predictions()

    print(f"\n[data] Baseline: {len(baseline_scores):,} scores | "
          f"Recent: {len(recent_scores):,} scores")

    # 2. PSI on prediction scores
    score_psi = compute_psi(baseline_scores, recent_scores)
    flag      = psi_flag(score_psi)
    print(f"\n[psi]  Score PSI: {score_psi:.4f} - {flag}")
    if flag == "minor_shift":
        print("       [warn] Minor shift detected - monitoring closely")
    elif flag == "significant_drift":
        print("       [ALARM] SIGNIFICANT DRIFT - retrain required")

    # 3. CSI per feature
    print("\n[csi]  Feature stability:")
    csi_results = compute_csi(baseline_features, recent_features)
    alarmed_features = []
    for feat, result in csi_results.items():
        status = "ok" if result["flag"] == "stable" else ("warn" if result["flag"] == "minor_shift" else "ALARM")
        print(f"       [{status}] {feat:<35} CSI={result['csi']:.4f}  {result['flag']}")
        if result["csi"] > CSI_ALARM:
            alarmed_features.append(feat)

    if alarmed_features:
        print(f"\n  Features with significant shift: {alarmed_features}")

    # 4. Feature attribution drift — SHAP comparison vs baseline
    shap_drift = check_feature_attribution_drift(recent_features)

    # Log to MLflow
    log_to_mlflow(score_psi, csi_results, flag)

    # Build report — data quality + model drift + feature attribution drift
    report = {
        "run_time":              datetime.now().isoformat(),
        "data_quality":          dq_results,
        "score_psi":             score_psi,
        "score_psi_flag":        flag,
        "csi_results":           csi_results,
        "alarmed_features":      alarmed_features,
        "shap_drift":            shap_drift,
        "retrain_triggered":     False,
    }

    # Trigger retrain if PSI exceeds alarm threshold
    if score_psi > PSI_ALARM:
        print(f"\n[alarm] PSI {score_psi:.4f} > {PSI_ALARM} threshold - triggering retrain")
        write_alarm_to_postgres(score_psi, csi_results)
        trigger_retrain_dag()
        report["retrain_triggered"] = True
    else:
        print(f"\n[ok]   PSI {score_psi:.4f} within acceptable range - no action required")

    # Warn if data quality failed but don't trigger retrain
    # (data issues need human investigation, not automatic retraining)
    if not dq_results.get("all_pass", True):
        print("\n[warn] Data quality checks failed - review pipeline before next retrain")
        report["data_quality_warning"] = True

    # Save report — convert numpy types to native Python for JSON serialization
    def _json_safe(obj):
        if hasattr(obj, "item"):   # numpy scalar
            return obj.item()
        raise TypeError(f"Not serializable: {type(obj)}")

    report_path = REPORT_DIR / "monitoring_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=_json_safe)
    print(f"\n[save] Report -> {report_path.name}")
    print("[done] Monitoring complete.")

    return report


if __name__ == "__main__":
    main()
