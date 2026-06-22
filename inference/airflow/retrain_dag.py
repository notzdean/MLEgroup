"""
inference/airflow/retrain_dag.py
==================================
CS611 MLE Group Project — Dengue Outbreak Risk Prediction
Airflow DAG: Model retraining — triggered by monitor.py on drift alarm.

What this DAG does
------------------
1. Run feature_engineering.py to rebuild Gold table with fresh data
2. Run train.py to retrain XGBoost + LightGBM with Optuna HPT
3. Run evaluate.py — test set + OOT evaluation + promotion gate
4. If gate passes: new model promoted to Production in MLflow
5. If gate fails: previous Production model remains active
6. Notify team via webhook

Trigger
-------
Triggered by monitor.py when PSI > 0.20 (significant drift).
Also scheduled weekly (Wednesday 06:00 SGT) as a safety net.

Design note
-----------
The promotion gate in evaluate.py acts as the quality gate.
A failed retrain never touches the Production model.
"We allow a maximum 10 percentage point degradation on OOT
before flagging for retraining."
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator

default_args = {
    "owner":            "cs611_group4",
    "depends_on_past":  False,
    "email_on_failure": False,
    "retries":          0,
}

dag = DAG(
    dag_id="dengue_retrain_dag",
    description="Model retraining pipeline — triggered on drift or weekly",
    schedule="0 22 * * 1",   # Monday 22:00 UTC = Tuesday 06:00 SGT
    start_date=datetime(2024, 1, 1),
    catchup=False,
    default_args=default_args,
    tags=["dengue", "retrain", "mlops"],
)


# ── Task functions ─────────────────────────────────────────────────────────────

def _subprocess_env():
    """Return env with HOME set so user-installed packages in ~/.local are found."""
    import os
    return {**os.environ, "HOME": "/home/airflow"}


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
    """Retrain model with Optuna HPT."""
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
    """Evaluate candidate model — test set + OOT + promotion gate."""
    import json
    import subprocess

    result = subprocess.run(
        ["python", "/opt/airflow/model/evaluate.py"],
        capture_output=True, text=True, env=_subprocess_env()
    )
    print(result.stdout)
    if result.returncode != 0:
        raise RuntimeError(f"evaluate.py failed:\n{result.stderr}")

    # Read promotion decision from report
    try:
        with open("/opt/airflow/model/evaluation_report.json") as f:
            report = json.load(f)
        promoted = report.get("promoted", False)
        context["ti"].xcom_push(key="promoted", value=promoted)
        print(f"Promotion gate: {'PASS' if promoted else 'FAIL'}")
    except Exception as e:
        print(f"Could not read evaluation report ({e})")
        context["ti"].xcom_push(key="promoted", value=False)


def check_promotion(**context):
    """Branch based on promotion gate result."""
    promoted = context["ti"].xcom_pull(key="promoted", task_ids="run_evaluation")
    if promoted:
        return "model_promoted"
    else:
        return "model_not_promoted"


def notify_team_promoted(**context):
    """Notify team that new model is in Production."""
    import json
    import os
    import requests

    webhook = os.getenv("TEAM_WEBHOOK_URL", "")
    if not webhook:
        print("TEAM_WEBHOOK_URL not set — skipping notification")
        return

    try:
        with open("/opt/airflow/model/evaluation_report.json") as f:
            report = json.load(f)
        requests.post(webhook, json={
            "event":        "model_promoted",
            "model_type":   report.get("model_type"),
            "test_recall":  report.get("test_metrics", {}).get("recall"),
            "oot_recall":   report.get("oot_metrics", {}).get("recall"),
            "triggered_by": context["dag_run"].conf.get("triggered_by", "schedule"),
        }, timeout=10)
        print("Team notified — model promoted")
    except Exception as e:
        print(f"Notification failed ({e})")


def notify_team_not_promoted(**context):
    """Notify team that retrain failed the gate — Production unchanged."""
    import os
    import requests

    webhook = os.getenv("TEAM_WEBHOOK_URL", "")
    if not webhook:
        print("TEAM_WEBHOOK_URL not set — skipping notification")
        return

    try:
        requests.post(webhook, json={
            "event":        "model_retrain_gate_failed",
            "message":      "Candidate model did not pass promotion gate. Production model unchanged.",
            "triggered_by": context["dag_run"].conf.get("triggered_by", "schedule"),
        }, timeout=10)
        print("Team notified — gate failed, Production unchanged")
    except Exception as e:
        print(f"Notification failed ({e})")


# ── Task definitions ──────────────────────────────────────────────────────────

t1 = PythonOperator(task_id="run_feature_engineering", python_callable=run_feature_engineering, dag=dag)
t2 = PythonOperator(task_id="run_training",            python_callable=run_training,            dag=dag)
t3 = PythonOperator(task_id="run_evaluation",          python_callable=run_evaluation,          dag=dag)
t4 = BranchPythonOperator(task_id="check_promotion",   python_callable=check_promotion,         dag=dag)

t5 = PythonOperator(task_id="model_promoted",          python_callable=notify_team_promoted,     dag=dag)
t6 = PythonOperator(task_id="model_not_promoted",      python_callable=notify_team_not_promoted, dag=dag)

t_end = EmptyOperator(task_id="end", trigger_rule="none_failed_min_one_success", dag=dag)

t1 >> t2 >> t3 >> t4 >> [t5, t6] >> t_end
