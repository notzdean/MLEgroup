"""
model/evaluate.py
==================
CS611 MLE Group Project — Dengue Outbreak Risk Prediction
Model evaluation: Test set + OOT + promotion gate.

Reads : model/candidate/model.xgb (or model.lgb) + candidate_meta.json
        data/gold/subzone_features.parquet
Writes: model/evaluation_report.json
        MLflow: promotes Candidate → Production if gate passes

Promotion gate — ALL four must pass
------------------------------------
1. Recall ≥ 0.70 on test set
2. OOT recall drop ≤ 10 percentage points vs test set
3. SHAP top features make domain sense (logged, manual check)
4. AUC-ROC ≥ 0.75 on test set

If gate passes  → model registered as Production in MLflow
If gate fails   → previous Production model remains; report written with failure reason

OOT interpretation
------------------
Jan–Nov 2020 is the DENV-3 serotype shift outbreak (35,000+ cases).
Model performance degradation on OOT is expected and documented as
a concept drift case study, not a model failure.
The 10pp degradation allowance accounts for this.
"""

import json
import warnings
import pandas as pd
import numpy as np
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT       = Path(__file__).resolve().parent.parent
GOLD       = ROOT / "data" / "gold"
MODEL_DIR  = ROOT / "model" / "candidate"
REPORT_DIR = ROOT / "model"

TRAIN_END  = pd.Timestamp("2018-12-31")
VAL_END    = pd.Timestamp("2019-06-30")
TEST_START = pd.Timestamp("2019-07-01")
TEST_END   = pd.Timestamp("2019-12-31")
OOT_START  = pd.Timestamp("2020-01-01")

RECALL_GATE     = 0.70
OOT_DROP_LIMIT  = 0.10   # max allowed pp drop from test to OOT
AUC_GATE        = 0.75

MLFLOW_TRACKING_URI = "http://localhost:5000"
MODEL_NAME          = "dengue_cluster_model"


# ── load ──────────────────────────────────────────────────────────────────────

def load_candidate():
    meta_path = MODEL_DIR / "candidate_meta.json"
    with open(meta_path) as f:
        meta = json.load(f)

    model_type = meta["model_type"]
    features   = meta["features"]

    if model_type == "xgboost":
        import xgboost as xgb
        model = xgb.XGBClassifier()
        model.load_model(MODEL_DIR / "model.xgb")
    else:
        import lightgbm as lgb
        model = lgb.Booster(model_file=str(MODEL_DIR / "model.lgb"))

    print(f"[load] Candidate model: {model_type}")
    return model, model_type, features, meta


def load_splits(features):
    df = pd.read_parquet(GOLD / "subzone_features.parquet")
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=features + ["label"])

    test = df[(df["date"] >= TEST_START) & (df["date"] <= TEST_END)]
    oot  = df[df["date"] >= OOT_START]
    return test, oot


# ── metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(model, model_type, X, y, split_name):
    from sklearn.metrics import (
        recall_score, precision_score, f1_score,
        roc_auc_score, confusion_matrix
    )

    if model_type == "lightgbm":
        import lightgbm as lgb
        y_prob = model.predict(X)
        y_pred = (y_prob >= 0.5).astype(int)
    else:
        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)[:, 1]

    cm = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

    metrics = {
        "split":     split_name,
        "n_rows":    len(y),
        "n_positive": int(y.sum()),
        "recall":    round(recall_score(y, y_pred, zero_division=0), 4),
        "precision": round(precision_score(y, y_pred, zero_division=0), 4),
        "f1":        round(f1_score(y, y_pred, zero_division=0), 4),
        "auc_roc":   round(roc_auc_score(y, y_prob), 4),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }

    print(f"\n[{split_name}] Metrics")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  F1        : {metrics['f1']:.4f}")
    print(f"  AUC-ROC   : {metrics['auc_roc']:.4f}")
    print(f"  TP={tp} FP={fp} TN={tn} FN={fn}")
    return metrics


# ── SHAP ──────────────────────────────────────────────────────────────────────

def compute_shap(model, model_type, X, features):
    try:
        import shap
        print("\n[shap] Computing SHAP values")

        if model_type == "xgboost":
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.TreeExplainer(model)

        # Sample 500 rows for speed
        idx = np.random.choice(len(X), min(500, len(X)), replace=False)
        shap_values = explainer.shap_values(X[idx])

        # Mean absolute SHAP per feature
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        mean_shap = np.abs(shap_values).mean(axis=0)
        feature_importance = sorted(
            zip(features, mean_shap),
            key=lambda x: x[1], reverse=True
        )

        print("  Top features by mean |SHAP|:")
        for feat, val in feature_importance[:8]:
            print(f"    {feat:<35} {val:.4f}")

        return {f: round(float(v), 4) for f, v in feature_importance}

    except Exception as e:
        print(f"  [warn] SHAP failed ({e})")
        return {}


# ── PSI ───────────────────────────────────────────────────────────────────────

def compute_psi(expected, actual, bins=10):
    """
    Population Stability Index — measures score distribution shift.
    PSI < 0.1: stable | 0.1-0.2: minor shift | > 0.2: significant drift
    """
    def psi_bucket(e, a, bins):
        breakpoints = np.linspace(0, 1, bins + 1)
        e_pct = np.histogram(e, bins=breakpoints)[0] / len(e)
        a_pct = np.histogram(a, bins=breakpoints)[0] / len(a)
        e_pct = np.where(e_pct == 0, 1e-4, e_pct)
        a_pct = np.where(a_pct == 0, 1e-4, a_pct)
        return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))

    return round(psi_bucket(expected, actual, bins), 4)


# ── promotion gate ────────────────────────────────────────────────────────────

def evaluate_gate(test_metrics, oot_metrics, shap_dict):
    recall_test = test_metrics["recall"]
    recall_oot  = oot_metrics["recall"]
    auc_test    = test_metrics["auc_roc"]
    oot_drop    = recall_test - recall_oot

    gates = {
        "recall_test_pass":  recall_test >= RECALL_GATE,
        "oot_drop_pass":     oot_drop <= OOT_DROP_LIMIT,
        "auc_test_pass":     auc_test >= AUC_GATE,
        "shap_logged":       len(shap_dict) > 0,
    }

    all_pass = all(gates.values())

    print("\n[gate] Promotion gate")
    print(f"  Recall >= {RECALL_GATE} on test    : {'PASS' if gates['recall_test_pass'] else 'FAIL'} ({recall_test:.4f})")
    print(f"  OOT drop <= {OOT_DROP_LIMIT*100:.0f}pp             : {'PASS' if gates['oot_drop_pass'] else 'FAIL'} ({oot_drop*100:.1f}pp)")
    print(f"  AUC-ROC >= {AUC_GATE} on test     : {'PASS' if gates['auc_test_pass'] else 'FAIL'} ({auc_test:.4f})")
    print(f"  SHAP values logged             : {'PASS' if gates['shap_logged'] else 'FAIL'}")
    print(f"  Overall: {'PASS -- promote to Production' if all_pass else 'FAIL -- keep existing Production model'}")

    return gates, all_pass


# ── MLflow promotion ──────────────────────────────────────────────────────────

def promote_to_production(model, model_type):
    try:
        import mlflow
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = mlflow.tracking.MlflowClient()

        # Get latest Candidate version
        versions = client.get_latest_versions(MODEL_NAME, stages=["None", "Staging"])
        if not versions:
            print("  [warn] No candidate version found in MLflow registry")
            return

        latest = sorted(versions, key=lambda v: int(v.version))[-1]
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=latest.version,
            stage="Production"
        )
        print(f"  MLflow: version {latest.version} -> Production")

    except Exception as e:
        print(f"  [warn] MLflow promotion failed ({e})")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Model Evaluation — Test + OOT + Promotion Gate")
    print("=" * 60)

    model, model_type, features, meta = load_candidate()
    test, oot = load_splits(features)

    X_test = test[features].values
    y_test = test["label"].values
    X_oot  = oot[features].values
    y_oot  = oot["label"].values

    # Metrics
    test_metrics = compute_metrics(model, model_type, X_test, y_test, "Test")
    oot_metrics  = compute_metrics(model, model_type, X_oot,  y_oot,  "OOT")

    # PSI — score distribution shift from test to OOT
    if model_type == "lightgbm":
        import lightgbm as lgb
        test_scores = model.predict(X_test)
        oot_scores  = model.predict(X_oot)
    else:
        test_scores = model.predict_proba(X_test)[:, 1]
        oot_scores  = model.predict_proba(X_oot)[:, 1]

    psi = compute_psi(test_scores, oot_scores)
    psi_flag = "stable" if psi < 0.1 else ("minor shift" if psi < 0.2 else "significant drift")
    print(f"\n[psi] Score PSI (test -> OOT): {psi:.4f} - {psi_flag}")

    # SHAP
    shap_dict = compute_shap(model, model_type, X_test, features)

    # Promotion gate
    gates, all_pass = evaluate_gate(test_metrics, oot_metrics, shap_dict)

    # Build report
    report = {
        "model_type":   model_type,
        "features":     features,
        "test_metrics": test_metrics,
        "oot_metrics":  oot_metrics,
        "psi":          psi,
        "psi_flag":     psi_flag,
        "shap_importance": shap_dict,
        "gate_results": gates,
        "promoted":     all_pass,
        "val_metrics":  meta.get("val_metrics", {}),
        "note_oot": (
            "OOT period (Jan-Nov 2020) covers the DENV-3 serotype shift outbreak "
            "(35,000+ cases). Performance degradation is expected and documented "
            "as a concept drift case study, not a model failure."
        )
    }

    report_path = REPORT_DIR / "evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[save] Report -> {report_path.name}")

    if all_pass:
        promote_to_production(model, model_type)
        print("\n[done] Model promoted to Production.")
    else:
        print("\n[done] Gate failed — existing Production model unchanged.")


if __name__ == "__main__":
    main()
