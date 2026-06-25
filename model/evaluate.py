"""
model/evaluate.py
==================
CS611 MLE Group Project — Dengue Outbreak Risk Prediction
Model evaluation: Val + Test set + OOT + promotion gate.

Reads : model/candidate/model.xgb                           — raw XGBoost booster
        model/candidate/model.lgb                           — raw LightGBM booster
        model/candidate/model_xgboost_calibrated.joblib     — XGBoost calibrated wrapper
        model/candidate/model_lightgbm_calibrated.joblib    — LightGBM calibrated wrapper
        model/candidate/best_model_calibrated.joblib        — fallback alias
        model/candidate/candidate_meta.json                 — all_models block read
        data/gold/subzone_features.parquet
Writes: model/evaluation_report.json
        MLflow: promotes Candidate → Production if gate passes

Promotion gate — ALL four must pass
------------------------------------
1. Recall ≥ 0.70 on test set          (at the calibrated threshold)
2. OOT recall drop ≤ 10 percentage points vs test set
3. AUC-ROC ≥ 0.75 on test set         (threshold-independent)
4. SHAP top features logged            (manual domain sanity check)

Key changes from earlier version
---------------------------------
- Calibrated model (.joblib) used for all predictions; raw model used for SHAP only.
- Calibrated threshold from candidate_meta.json used instead of default 0.5.
- compute_metrics accepts a threshold parameter and also reports PR-AUC.
- Threshold sensitivity check on Test included in report.
- Feature ~ label correlation cross-check included in report.
- PSI computed from calibrated probabilities.
- Evaluation covers Val, Test, and OOT (not just Test and OOT).

OOT interpretation
------------------
Jan–Nov 2020 is the DENV-3 serotype shift outbreak (35,000+ cases).
Performance degradation on OOT is expected and documented as a concept
drift case study, not a model failure. The 10pp gate accounts for this.
"""

import json
import warnings
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT       = Path(__file__).resolve().parent.parent
GOLD       = ROOT / "data" / "gold"
MODEL_DIR  = ROOT / "model" / "candidate"
REPORT_DIR = ROOT / "model"

TRAIN_END  = pd.Timestamp("2018-12-31")
VAL_START  = pd.Timestamp("2019-01-01")
VAL_END    = pd.Timestamp("2019-06-30")
TEST_START = pd.Timestamp("2019-07-01")
TEST_END   = pd.Timestamp("2019-12-31")
OOT_START  = pd.Timestamp("2020-01-01")

RECALL_GATE    = 0.70
OOT_DROP_LIMIT = 0.10   # max allowed pp drop from test to OOT recall
AUC_GATE       = 0.75

MLFLOW_TRACKING_URI = "http://172.18.0.4:5000"
MODEL_NAME          = "dengue_cluster_model"


# ── load ──────────────────────────────────────────────────────────────────────

def load_candidate():
    """
    Returns:
        models      — dict {model_type: (raw_model, cal_model, threshold)}
        best_type   — model_type string of the winning model
        features    — list of feature names
        meta        — full metadata dict

    Loads every model listed in meta["all_models"] so evaluate.py can
    compare XGBoost and LightGBM side by side.  Falls back to the legacy
    single-model layout (no "all_models" key) for backwards compatibility.
    """
    meta_path = MODEL_DIR / "candidate_meta.json"
    with open(meta_path) as f:
        meta = json.load(f)

    features  = meta["features"]
    best_type = meta["model_type"]

    def _load_raw(mt):
        if mt == "xgboost":
            import xgboost as xgb
            m = xgb.XGBClassifier()
            m.load_model(MODEL_DIR / "model.xgb")
        else:
            import lightgbm as lgb
            m = lgb.Booster(model_file=str(MODEL_DIR / "model.lgb"))
        return m

    def _load_cal(mt, meta_entry):
        # Try per-model joblib first (new format), fall back to best alias
        specific = MODEL_DIR / meta_entry.get("calibrated_joblib",
                                               f"model_{mt}_calibrated.joblib")
        alias    = MODEL_DIR / "best_model_calibrated.joblib"
        path = specific if specific.exists() else alias
        if path.exists():
            cal = joblib.load(path)
            print(f"[load] {mt} calibrated model: {path.name}")
            return cal
        print(f"[load] ⚠ No calibrated joblib for {mt}; falling back to raw model")
        return _load_raw(mt)

    # ── new format: all_models block ─────────────────────────────────────────
    if "all_models" in meta:
        models = {}
        for mt, entry in meta["all_models"].items():
            raw = _load_raw(mt)
            cal = _load_cal(mt, entry)
            thr = entry["calibrated_threshold"]
            models[mt] = (raw, cal, thr)
            print(f"[load] {mt:10s}  threshold={thr:.4f}")

    # ── legacy format: single best model only ────────────────────────────────
    else:
        print("[load] No all_models block found — loading best model only")
        mt  = best_type
        raw = _load_raw(mt)
        cal = _load_cal(mt, {"calibrated_joblib": "best_model_calibrated.joblib"})
        thr = meta.get("calibrated_threshold", 0.5)
        models = {mt: (raw, cal, thr)}
        print(f"[load] {mt:10s}  threshold={thr:.4f}")

    print(f"[load] Best model: {best_type}")
    return models, best_type, features, meta


def load_splits(features: list):
    """Load val, test, and OOT splits from the gold parquet."""
    df = pd.read_parquet(GOLD / "subzone_features.parquet")
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=features + ["label"])

    val  = df[(df["date"] >= VAL_START)  & (df["date"] <= VAL_END)]
    test = df[(df["date"] >= TEST_START) & (df["date"] <= TEST_END)]
    oot  = df[df["date"] >= OOT_START]

    for name, s in [("Val", val), ("Test", test), ("OOT", oot)]:
        print(f"  {name:5s}: {len(s):6,} rows | "
              f"positive: {s['label'].sum():,} ({s['label'].mean():.1%})")
    return val, test, oot


# ── metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(cal_model, X: np.ndarray, y: np.ndarray,
                    split_name: str, threshold: float = 0.5) -> dict:
    """
    Computes recall, precision, F1, AUC-ROC, and PR-AUC.
    Uses predict_proba from the calibrated wrapper and applies the
    calibrated threshold (not the raw model's default 0.5).
    PR-AUC (average_precision_score) is threshold-independent and is
    the more informative metric for imbalanced data.
    """
    from sklearn.metrics import (
        recall_score, precision_score, f1_score,
        roc_auc_score, average_precision_score, confusion_matrix,
    )

    y_prob = cal_model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    cm     = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

    metrics = {
        "split":     split_name,
        "threshold": round(threshold, 4),
        "n_rows":    len(y),
        "n_positive": int(y.sum()),
        "recall":    round(recall_score(y, y_pred, zero_division=0), 4),
        "precision": round(precision_score(y, y_pred, zero_division=0), 4),
        "f1":        round(f1_score(y, y_pred, zero_division=0), 4),
        "auc_roc":   round(roc_auc_score(y, y_prob), 4),
        "pr_auc":    round(average_precision_score(y, y_prob), 4),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }

    print(f"\n[{split_name}] Metrics (thr={threshold:.3f})")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  F1        : {metrics['f1']:.4f}")
    print(f"  AUC-ROC   : {metrics['auc_roc']:.4f}")
    print(f"  PR-AUC    : {metrics['pr_auc']:.4f}")
    print(f"  TP={tp} FP={fp} TN={tn} FN={fn}")
    return metrics


# ── threshold sensitivity ─────────────────────────────────────────────────────

def threshold_sensitivity(cal_model, X: np.ndarray, y: np.ndarray,
                           threshold: float,
                           deltas=(-0.05, -0.02, 0.0, 0.02, 0.05)) -> list:
    """
    Recall/precision at the calibrated threshold ± small deltas, evaluated
    on Test (NOT on val_thresh where the threshold was chosen).
    A flat profile = threshold on a stable plateau; a sharp swing = cliff risk.
    """
    from sklearn.metrics import precision_score, recall_score
    y_prob = cal_model.predict_proba(X)[:, 1]
    rows = []
    for d in deltas:
        t = float(np.clip(threshold + d, 0.0, 1.0))
        y_pred = (y_prob >= t).astype(int)
        rows.append(dict(
            delta=round(d, 4), threshold=round(t, 4),
            recall=round(recall_score(y, y_pred, zero_division=0), 4),
            precision=round(precision_score(y, y_pred, zero_division=0), 4),
        ))
    print(f"  Threshold sensitivity (Δ around {threshold:.3f}):")
    for row in rows:
        marker = "  <-- chosen" if row["delta"] == 0.0 else ""
        print(f"    Δ={row['delta']:+.2f}  thr={row['threshold']:.3f}  "
              f"recall={row['recall']:.4f}  precision={row['precision']:.4f}{marker}")
    return rows


# ── feature ~ label correlation ───────────────────────────────────────────────

def feature_label_correlation(features: list) -> dict:
    """
    Point-biserial correlation between each raw feature and the binary label,
    computed on the full training set. Model-free cross-check against SHAP:
    a feature with low SHAP importance but real raw correlation is worth
    investigating.
    """
    df = pd.read_parquet(GOLD / "subzone_features.parquet")
    df["date"] = pd.to_datetime(df["date"])
    train = df[df["date"] <= TRAIN_END].dropna(subset=features + ["label"])

    corrs = {}
    y = train["label"].values.astype(float)
    for f in features:
        x = train[f].values.astype(float)
        corrs[f] = 0.0 if np.std(x) == 0 else float(np.corrcoef(x, y)[0, 1])
    ranked = dict(sorted(corrs.items(), key=lambda kv: abs(kv[1]), reverse=True))

    print("\n[corr] Feature ~ label correlation (train set, |r| descending):")
    for f, r in ranked.items():
        print(f"    {f:<35} {r:+.4f}")
    return ranked


# ── SHAP ──────────────────────────────────────────────────────────────────────

def compute_shap(raw_model, model_type: str,
                 X: np.ndarray, features: list) -> dict:
    """
    SHAP must use the RAW tree model (not the CalibratedClassifierCV wrapper),
    because TreeExplainer expects native XGB/LGB estimators.
    """
    try:
        import shap
        print("\n[shap] Computing SHAP values (raw model)")

        explainer = shap.TreeExplainer(raw_model)

        idx = np.random.choice(len(X), min(500, len(X)), replace=False)
        shap_values = explainer.shap_values(X[idx])

        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        mean_shap = np.abs(shap_values).mean(axis=0)
        feature_importance = sorted(
            zip(features, mean_shap),
            key=lambda x: x[1], reverse=True,
        )

        print("  Top features by mean |SHAP|:")
        for feat, val in feature_importance[:8]:
            print(f"    {feat:<35} {val:.4f}")

        return {f: round(float(v), 4) for f, v in feature_importance}

    except Exception as e:
        print(f"  [warn] SHAP failed ({e})")
        return {}


# ── PSI ───────────────────────────────────────────────────────────────────────

def compute_psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """
    Population Stability Index on calibrated probability scores.
    PSI < 0.1: stable | 0.1–0.2: minor shift | > 0.2: significant drift
    """
    bp    = np.linspace(0, 1, bins + 1)
    e_pct = np.histogram(expected, bins=bp)[0] / len(expected)
    a_pct = np.histogram(actual,   bins=bp)[0] / len(actual)
    e_pct = np.where(e_pct == 0, 1e-4, e_pct)
    a_pct = np.where(a_pct == 0, 1e-4, a_pct)
    return round(float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct))), 4)


# ── promotion gate ────────────────────────────────────────────────────────────

def evaluate_gate(test_metrics: dict, oot_metrics: dict, shap_dict: dict):
    recall_test = test_metrics["recall"]
    recall_oot  = oot_metrics["recall"]
    auc_test    = test_metrics["auc_roc"]
    oot_drop    = recall_test - recall_oot

    gates = {
        f"recall_test >= {RECALL_GATE}":          recall_test >= RECALL_GATE,
        f"oot_drop   <= {OOT_DROP_LIMIT*100:.0f}pp": oot_drop <= OOT_DROP_LIMIT,
        f"auc_roc    >= {AUC_GATE} (test)":       auc_test >= AUC_GATE,
        "shap_values_logged":                      len(shap_dict) > 0,
    }
    all_pass = all(gates.values())

    print("\n[gate] Promotion gate")
    print(f"  Recall >= {RECALL_GATE} on test    : "
          f"{'PASS' if gates[f'recall_test >= {RECALL_GATE}'] else 'FAIL'} ({recall_test:.4f})")
    print(f"  OOT drop <= {OOT_DROP_LIMIT*100:.0f}pp             : "
          f"{'PASS' if gates[f'oot_drop   <= {OOT_DROP_LIMIT*100:.0f}pp'] else 'FAIL'} ({oot_drop*100:.1f}pp)")
    print(f"  AUC-ROC >= {AUC_GATE} on test     : "
          f"{'PASS' if gates[f'auc_roc    >= {AUC_GATE} (test)'] else 'FAIL'} ({auc_test:.4f})")
    print(f"  SHAP values logged             : "
          f"{'PASS' if gates['shap_values_logged'] else 'FAIL'}")
    print(f"  Overall: {'PASS -- promote to Production' if all_pass else 'FAIL -- keep existing Production model'}")
    return gates, all_pass


# ── MLflow promotion ──────────────────────────────────────────────────────────

def promote_to_production(model_type: str):
    try:
        import mlflow
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = mlflow.tracking.MlflowClient()

        versions = client.get_latest_versions(MODEL_NAME, stages=["None", "Staging"])
        if not versions:
            print("  [warn] No candidate version found in MLflow registry")
            return

        latest = sorted(versions, key=lambda v: int(v.version))[-1]
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=latest.version,
            stage="Production",
        )
        print(f"  MLflow: version {latest.version} → Production")

    except Exception as e:
        print(f"  [warn] MLflow promotion failed ({e})")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Model Evaluation — Val + Test + OOT + Promotion Gate")
    print("=" * 60)

    # ── 1. load all candidate models ──────────────────────────────────────────
    models, best_type, features, meta = load_candidate()
    val, test, oot = load_splits(features)

    X_val  = val[features].values;   y_val  = val["label"].values
    X_test = test[features].values;  y_test = test["label"].values
    X_oot  = oot[features].values;   y_oot  = oot["label"].values

    # ── 2. evaluate every model on Val / Test / OOT ───────────────────────────
    all_metrics: dict = {}   # model_type -> {val, test, oot}

    for mt, (raw_model, cal_model, threshold) in models.items():
        print(f"\n{'='*40}")
        print(f"Evaluating: {mt.upper()}  (threshold={threshold:.4f})")
        print(f"{'='*40}")
        all_metrics[mt] = {
            "val":  compute_metrics(cal_model, X_val,  y_val,  "Val",  threshold),
            "test": compute_metrics(cal_model, X_test, y_test, "Test", threshold),
            "oot":  compute_metrics(cal_model, X_oot,  y_oot,  "OOT",  threshold),
        }

    # ── 3. PSI, SHAP, sensitivity, correlation — best model only ─────────────
    best_raw, best_cal, best_thr = models[best_type]

    test_scores = best_cal.predict_proba(X_test)[:, 1]
    oot_scores  = best_cal.predict_proba(X_oot)[:, 1]
    psi_val     = compute_psi(test_scores, oot_scores)
    psi_flag    = ("stable"           if psi_val < 0.1
                   else "minor shift" if psi_val < 0.2
                   else "significant drift")
    print(f"\n[psi] {best_type} — test→OOT: {psi_val:.4f} — {psi_flag}")

    shap_dict = compute_shap(best_raw, best_type, X_test, features)

    print(f"\n[sensitivity] Threshold stability on Test ({best_type})")
    sens = threshold_sensitivity(best_cal, X_test, y_test, best_thr)

    corr_dict = feature_label_correlation(features)

    # ── 4. promotion gate — based on best model's Test / OOT metrics ──────────
    gates, all_pass = evaluate_gate(
        all_metrics[best_type]["test"],
        all_metrics[best_type]["oot"],
        shap_dict,
    )

    # ── 5. build report ───────────────────────────────────────────────────────
    # Flat per-split metrics for the best model (kept for backwards compat)
    report = {
        "model_type":                  best_type,
        "calibrated_threshold":        best_thr,
        "calibration_method":          meta.get("calibration_method", "sigmoid"),
        "features":                    features,
        # best model metrics (backwards-compatible keys)
        "val_metrics":                 all_metrics[best_type]["val"],
        "test_metrics":                all_metrics[best_type]["test"],
        "oot_metrics":                 all_metrics[best_type]["oot"],
        # all-model comparison block (new)
        "all_model_metrics": {
            mt: {
                "calibrated_threshold": models[mt][2],
                "val":  splits_m["val"],
                "test": splits_m["test"],
                "oot":  splits_m["oot"],
            }
            for mt, splits_m in all_metrics.items()
        },
        "psi":                         psi_val,
        "psi_flag":                    psi_flag,
        "threshold_sensitivity":       sens,
        "shap_importance":             shap_dict,
        "feature_label_correlation":   corr_dict,
        "gate_results":                gates,
        "promoted":                    all_pass,
        "training_val_metrics":        meta.get("val_metrics", {}),
        "note_oot": (
            "OOT period (Jan–Nov 2020) covers the DENV-3 serotype shift outbreak "
            "(35,000+ cases). Performance degradation is expected and documented "
            "as a concept drift case study, not a model failure."
        ),
    }

    report_path = REPORT_DIR / "evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[save] Report → {report_path.name}")

    # ── 6. summary table ──────────────────────────────────────────────────────
    print("\n[summary] All models — Val / Test / OOT")
    header = f"  {'Model':10s}  {'Split':5s}  {'Thr':6s}  {'Recall':7s}  "
    header += f"{'Prec':7s}  {'F1':6s}  {'AUC':6s}  {'PR-AUC':7s}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for mt, splits_m in all_metrics.items():
        thr = models[mt][2]
        for split_name, m in splits_m.items():
            marker = " ← best" if mt == best_type else ""
            print(f"  {mt:10s}  {m['split']:5s}  {thr:.3f}  "
                  f"{m['recall']:.4f}   {m['precision']:.4f}   "
                  f"{m['f1']:.4f}  {m['auc_roc']:.4f}  "
                  f"{m['pr_auc']:.4f}{marker if split_name == 'test' else ''}")

    # ── 7. MLflow promotion ───────────────────────────────────────────────────
    if all_pass:
        promote_to_production(best_type)
        print("\n[done] Model promoted to Production.")
    else:
        print("\n[done] Gate failed — existing Production model unchanged.")


if __name__ == "__main__":
    main()
