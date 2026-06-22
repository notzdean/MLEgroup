"""
inference/airflow/monitor_retrain_dag.py
=========================================
CS611 MLE Group Project — Dengue Outbreak Risk Prediction
Airflow DAG: Weekly monitoring + conditional retrain.

Schedule: Wednesday 06:00 SGT (Tue 22:00 UTC)

Flow
----
run_monitoring
    └── check_drift (branch)
            ├── [drift]    run_feature_engineering → run_training → run_evaluation
            │                   └── check_promotion (branch)
            │                           ├── model_promoted     → end
            │                           └── model_not_promoted → end
            └── [no drift] no_retrain_needed → end
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator

default_args = {
    "owner":            "cs611_group4",
    "depends_on_past":  False,
    "email_on_failure": False,
    "retries":          0,
}

dag = DAG(
    dag_id="dengue_monitor_retrain_dag",
    description="Weekly PSI/CSI monitoring — auto-retrains if drift alarm fires",
    schedule="0 22 * * 1",   # Tuesday 22:00 UTC = Wednesday 06:00 SGT
    start_date=datetime(2024, 1, 1),
    catchup=False,
    default_args=default_args,
    tags=["dengue", "monitoring", "retrain", "mlops"],
)


def _subprocess_env():
    import os
    return {**os.environ, "HOME": "/home/airflow", "SKIP_DAG_TRIGGER": "true"}


# ── Monitoring ────────────────────────────────────────────────────────────────

def run_monitoring(**context):
    """Run PSI/CSI drift detection. Pushes drift_alarm bool to XCom."""
    import json
    import subprocess

    result = subprocess.run(
        ["python", "/opt/airflow/monitoring/monitor.py"],
        capture_output=True, text=True, env=_subprocess_env()
    )
    print(result.stdout)
    if result.returncode != 0:
        raise RuntimeError(f"monitor.py failed:\n{result.stderr}")

    try:
        with open("/opt/airflow/model/monitoring_report.json") as f:
            report = json.load(f)
        psi  = report.get("score_psi", 0)
        flag = report.get("score_psi_flag", "stable")
        context["ti"].xcom_push(key="score_psi",   value=psi)
        context["ti"].xcom_push(key="drift_alarm", value=(flag == "significant_drift"))
        print(f"PSI: {psi:.4f} | Flag: {flag}")
    except Exception as e:
        print(f"Could not read monitoring_report.json ({e}) — assuming no drift")
        context["ti"].xcom_push(key="drift_alarm", value=False)


def check_drift(**context):
    """Branch: significant drift → retrain. Otherwise → skip."""
    drift_alarm = context["ti"].xcom_pull(key="drift_alarm", task_ids="run_monitoring")
    if drift_alarm:
        print("Drift alarm — proceeding to retrain")
        return "run_feature_engineering"
    print("No drift — skipping retrain")
    return "no_retrain_needed"


# ── Retrain ───────────────────────────────────────────────────────────────────

def run_feature_engineering(**context):
    """Rebuild Gold feature table with latest data."""
    import subprocess
    result = subprocess.run(
        ["python", "/opt/airflow/pipeline/feature_engineering.py"],
        capture_output=True, text=True, env=_subprocess_env()
    )
    print(result.stdout)
    if result.returncode != 0:
        raise RuntimeError(f"feature_engineering.py failed:\n{result.stderr}")
    print("Feature engineering complete")


def run_training(**context):
    """Retrain XGBoost + LightGBM with Optuna HPT."""
    import subprocess
    result = subprocess.run(
        ["python", "/opt/airflow/model/train.py"],
        capture_output=True, text=True, env=_subprocess_env()
    )
    print(result.stdout)
    if result.returncode != 0:
        raise RuntimeError(f"train.py failed:\n{result.stderr}")
    print("Training complete")


def run_evaluation(**context):
    """Evaluate candidate — test set + OOT + promotion gate."""
    import json
    import subprocess

    result = subprocess.run(
        ["python", "/opt/airflow/model/evaluate.py"],
        capture_output=True, text=True, env=_subprocess_env()
    )
    print(result.stdout)
    if result.returncode != 0:
        raise RuntimeError(f"evaluate.py failed:\n{result.stderr}")

    try:
        with open("/opt/airflow/model/evaluation_report.json") as f:
            report = json.load(f)
        promoted = report.get("promoted", False)
        context["ti"].xcom_push(key="promoted", value=promoted)
        print(f"Promotion gate: {'PASS' if promoted else 'FAIL'}")
    except Exception as e:
        print(f"Could not read evaluation_report.json ({e})")
        context["ti"].xcom_push(key="promoted", value=False)


def check_promotion(**context):
    """Branch based on promotion gate result."""
    promoted = context["ti"].xcom_pull(key="promoted", task_ids="run_evaluation")
    return "model_promoted" if promoted else "model_not_promoted"


def notify_promoted(**context):
    """Notify team — new model is in Production."""
    import json, os, requests
    webhook = os.getenv("TEAM_WEBHOOK_URL", "")
    if not webhook:
        print("TEAM_WEBHOOK_URL not set — skipping notification")
        return
    try:
        with open("/opt/airflow/model/evaluation_report.json") as f:
            report = json.load(f)
        requests.post(webhook, json={
            "event":       "model_promoted",
            "model_type":  report.get("model_type"),
            "test_recall": report.get("test_metrics", {}).get("recall"),
            "oot_recall":  report.get("oot_metrics",  {}).get("recall"),
        }, timeout=10)
        print("Team notified — model promoted")
    except Exception as e:
        print(f"Notification failed ({e})")


def notify_not_promoted(**context):
    """Notify team — gate failed, Production model unchanged."""
    import os, requests
    webhook = os.getenv("TEAM_WEBHOOK_URL", "")
    if not webhook:
        print("TEAM_WEBHOOK_URL not set — skipping notification")
        return
    try:
        requests.post(webhook, json={
            "event":   "model_retrain_gate_failed",
            "message": "Candidate did not pass promotion gate. Production unchanged.",
        }, timeout=10)
        print("Team notified — gate failed")
    except Exception as e:
        print(f"Notification failed ({e})")


# ── Task definitions ──────────────────────────────────────────────────────────

t_monitor   = PythonOperator(task_id="run_monitoring",          python_callable=run_monitoring,          dag=dag)
t_branch    = BranchPythonOperator(task_id="check_drift",       python_callable=check_drift,             dag=dag)
t_skip      = EmptyOperator(task_id="no_retrain_needed",                                                 dag=dag)
t_fe        = PythonOperator(task_id="run_feature_engineering", python_callable=run_feature_engineering, dag=dag)
t_train     = PythonOperator(task_id="run_training",            python_callable=run_training,            dag=dag)
t_eval      = PythonOperator(task_id="run_evaluation",          python_callable=run_evaluation,          dag=dag)
t_gate      = BranchPythonOperator(task_id="check_promotion",   python_callable=check_promotion,         dag=dag)
t_promoted  = PythonOperator(task_id="model_promoted",          python_callable=notify_promoted,         dag=dag)
t_not_prom  = PythonOperator(task_id="model_not_promoted",      python_callable=notify_not_promoted,     dag=dag)
t_end       = EmptyOperator(task_id="end", trigger_rule="none_failed_min_one_success",                   dag=dag)

t_monitor >> t_branch >> [t_fe, t_skip]
t_fe >> t_train >> t_eval >> t_gate >> [t_promoted, t_not_prom]
[t_promoted, t_not_prom, t_skip] >> t_end
