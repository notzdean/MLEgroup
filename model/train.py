"""
model/train.py
===============
CS611 MLE Group Project — Dengue Outbreak Risk Prediction
Model training: XGBoost + LightGBM with Optuna HPT.

Reads : data/gold/subzone_features.parquet
Writes: MLflow experiment runs + Candidate model in MLflow Registry
        model/candidate/model.xgb (or model.lgb)           — raw booster
        model/candidate/best_model_calibrated.joblib        — calibrated wrapper
        model/candidate/candidate_meta.json

Pipeline
--------
1.  Load Gold feature table
2.  Strict temporal split — Train / Val[calib+thresh] / Test / OOT
3.  5-fold expanding-window CV within training window (NO resampling)
4.  Optuna HPT — optimise F2-score (beta=2) on CV folds
    Class imbalance handled via tuned scale_pos_weight (1–10×), not SMOTE
5.  Train final XGBoost + LightGBM on full raw training set with best params
6.  Probability calibration (Platt/sigmoid) on Val-calib slice
7.  Threshold calibration on Val-thresh — max precision s.t. recall ≥ 0.70
8.  Threshold sensitivity check on Test
9.  Log all runs to MLflow
10. Register best model (selected by precision at recall floor) as Candidate

Data splits
-----------
Train      : up to 2018-12-31
Val-calib  : 2019-01-01 – 2019-03-31   (fit Platt calibrator)
Val-thresh : 2019-04-01 – 2019-06-30   (pick decision threshold)
Test       : 2019-07-01 – 2019-12-31   (held out — touched once in evaluate.py)
OOT        : 2020-01-01 – 2020-11-06   (concept drift — DENV-3 serotype shift)

Design decisions
----------------
- F2-score CV objective: pure recall rewards degenerate "flag everything positive"
  solutions when scale_pos_weight is free up to 10×. F2 (beta=2) still weights
  recall above precision but bounds the degenerate solution.
- No SMOTE: scale_pos_weight in the loss function achieves the same effect without
  synthesising points in the feature space.
- Disjoint calib/thresh splits: fitting the Platt scaler and picking the threshold
  on the same rows lets calibration silently overfit the threshold choice.
- Best model selected by precision at the recall floor, not by recall alone.
- XGBoost + LightGBM: tabular data, SHAP interpretable, trains locally.
- Optuna: efficient HPT with pruning, better than grid search.
"""

import json
import warnings
import pandas as pd
import numpy as np
from pathlib import Path

warnings.filterwarnings("ignore")

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parent.parent
GOLD      = ROOT / "data" / "gold"
INPUT     = GOLD / "subzone_features.parquet"
MODEL_DIR = ROOT / "model" / "candidate"

# ── split boundaries ──────────────────────────────────────────────────────────
TRAIN_END  = pd.Timestamp("2018-12-31")
VAL_START  = pd.Timestamp("2019-01-01")
VAL_END    = pd.Timestamp("2019-06-30")
VAL_MID    = pd.Timestamp("2019-03-31")   # splits Val → calib (Jan–Mar) / thresh (Apr–Jun)
TEST_START = pd.Timestamp("2019-07-01")
TEST_END   = pd.Timestamp("2019-12-31")
OOT_START  = pd.Timestamp("2020-01-01")

FEATURES = [
    "rainfall_lag1w", "rainfall_lag2w", "rainfall_lag4w",
    "cluster_count_rolling2w", "cluster_count_rolling4w",
    "recent_cases_rolling2w",  "recent_cases_rolling4w",
    "population", "elderly_pct", "area_km2",
    "population_density", "vulnerability_index",
]
LABEL = "label"

# ── hyperparameters ───────────────────────────────────────────────────────────
N_OPTUNA_TRIALS = 150
N_CV_FOLDS      = 5
RECALL_TARGET   = 0.70
F2_BETA         = 2.0   # CV objective: F-beta with beta=2 (recall weighted 2× precision)

MLFLOW_TRACKING_URI = "http://localhost:5000"
EXPERIMENT_NAME     = "dengue_cluster_model"
MODEL_NAME          = "dengue_cluster_model"


# ── data loading & splitting ──────────────────────────────────────────────────

def load_and_split(path: Path) -> dict:
    """
    Returns a dict with keys:
        train / val / val_calib / val_thresh / test / oot
    Val is split at VAL_MID so that probability calibration (fit on val_calib)
    and threshold selection (fit on val_thresh) use disjoint rows.
    """
    print(f"[load] {path.name}")
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=FEATURES + [LABEL])

    splits = {
        "train":      df[df["date"] <= TRAIN_END],
        "val":        df[(df["date"] >= VAL_START)  & (df["date"] <= VAL_END)],
        "val_calib":  df[(df["date"] >= VAL_START)  & (df["date"] <= VAL_MID)],
        "val_thresh": df[(df["date"] >  VAL_MID)    & (df["date"] <= VAL_END)],
        "test":       df[(df["date"] >= TEST_START) & (df["date"] <= TEST_END)],
        "oot":        df[df["date"] >= OOT_START],
    }
    for name, s in splits.items():
        print(f"  {name:10s}: {len(s):6,} rows | "
              f"positive: {s[LABEL].sum():,} ({s[LABEL].mean():.1%})")
    return splits


# ── time-series CV folds ──────────────────────────────────────────────────────

def ts_cv_folds(train_df: pd.DataFrame, n_folds: int = N_CV_FOLDS):
    """
    Expanding-window CV within the training set.
    Each fold: earlier dates = train, next block = val. Future never leaks into past.
    No SMOTE inside folds — class imbalance handled by scale_pos_weight in the model.
    """
    dates     = sorted(train_df["date"].unique())
    fold_size = len(dates) // (n_folds + 1)
    folds = []
    for i in range(1, n_folds + 1):
        t_end = i * fold_size
        v_end = t_end + fold_size
        t_idx = dates[:t_end]
        v_idx = dates[t_end:v_end]
        if not v_idx:
            continue
        folds.append((
            train_df[train_df["date"].isin(t_idx)],
            train_df[train_df["date"].isin(v_idx)],
        ))
    return folds


# ── Optuna objective ──────────────────────────────────────────────────────────

def make_objective(train_df: pd.DataFrame, model_type: str):
    """
    CV objective: mean F2-score across folds (beta=2 weights recall 2× precision).
    Using raw recall as the objective rewards the degenerate "flag everything"
    solution when scale_pos_weight is free up to 10×; F2 prevents that.
    """
    from sklearn.metrics import fbeta_score

    def objective(trial):
        if model_type == "xgboost":
            import xgboost as xgb
            params = dict(
                n_estimators     = trial.suggest_int("n_estimators", 100, 500),
                max_depth        = trial.suggest_int("max_depth", 3, 8),
                learning_rate    = trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                subsample        = trial.suggest_float("subsample", 0.6, 1.0),
                colsample_bytree = trial.suggest_float("colsample_bytree", 0.6, 1.0),
                min_child_weight = trial.suggest_int("min_child_weight", 1, 10),
                scale_pos_weight = trial.suggest_float("scale_pos_weight", 1.0, 10.0),
                use_label_encoder=False, eval_metric="logloss", random_state=42,
            )
            clf = xgb.XGBClassifier(**params)
        else:
            import lightgbm as lgb
            params = dict(
                n_estimators      = trial.suggest_int("n_estimators", 100, 500),
                max_depth         = trial.suggest_int("max_depth", 3, 8),
                learning_rate     = trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                subsample         = trial.suggest_float("subsample", 0.6, 1.0),
                colsample_bytree  = trial.suggest_float("colsample_bytree", 0.6, 1.0),
                min_child_samples = trial.suggest_int("min_child_samples", 5, 50),
                scale_pos_weight  = trial.suggest_float("scale_pos_weight", 1.0, 10.0),
                random_state=42, verbose=-1,
            )
            clf = lgb.LGBMClassifier(**params)

        cv_scores = []
        for fold_tr, fold_vl in ts_cv_folds(train_df):
            Xf = fold_tr[FEATURES].values;  yf = fold_tr[LABEL].values
            Xv = fold_vl[FEATURES].values;  yv = fold_vl[LABEL].values
            clf.fit(Xf, yf)
            yp = clf.predict(Xv)
            cv_scores.append(fbeta_score(yv, yp, beta=F2_BETA, zero_division=0))

        return float(np.mean(cv_scores))

    return objective


# ── model training ────────────────────────────────────────────────────────────

def train_model(model_type: str, best_params: dict,
                X_train: np.ndarray, y_train: np.ndarray):
    """Train final model on the full raw training set (no SMOTE)."""
    if model_type == "xgboost":
        import xgboost as xgb
        model = xgb.XGBClassifier(
            **best_params,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
        )
    else:
        import lightgbm as lgb
        model = lgb.LGBMClassifier(**best_params, random_state=42, verbose=-1)
    model.fit(X_train, y_train)
    return model


# ── probability calibration ───────────────────────────────────────────────────

def calibrate_model(clf, X_calib: np.ndarray, y_calib: np.ndarray,
                    method: str = "sigmoid"):
    """
    Wrap an already-fitted classifier with Platt/sigmoid probability calibration,
    fit on a held-out slice (val_calib) the model was NOT trained on.

    Raw GBM probabilities under a tuned scale_pos_weight are not well-calibrated;
    this correction improves threshold portability from Val to Test/OOT.

    Returns a CalibratedClassifierCV wrapper — NOT the raw tree model.
    Keep the original clf reference for SHAP and native-format saving.
    """
    from sklearn.calibration import CalibratedClassifierCV
    try:
        # sklearn >= 1.6: cv="prefit" deprecated in favour of FrozenEstimator
        from sklearn.frozen import FrozenEstimator
        calibrated = CalibratedClassifierCV(FrozenEstimator(clf), method=method)
    except ImportError:
        calibrated = CalibratedClassifierCV(clf, method=method, cv="prefit")
    calibrated.fit(X_calib, y_calib)
    return calibrated


# ── threshold selection ───────────────────────────────────────────────────────

def select_threshold(clf, X: np.ndarray, y: np.ndarray,
                     recall_target: float = RECALL_TARGET):
    """
    Pick the operating threshold from the PR curve on val_thresh:
    maximise precision subject to recall >= recall_target.
    Falls back to closest-to-target recall if target is unreachable.
    """
    from sklearn.metrics import precision_recall_curve

    y_prob = clf.predict_proba(X)[:, 1]
    prec, rec, thr = precision_recall_curve(y, y_prob)
    if len(thr) == 0:
        return 0.5, float(prec[-1]), float(rec[-1])

    # precision_recall_curve appends a sentinel (prec=1, rec=0) with no threshold
    prec, rec = prec[:-1], rec[:-1]

    meets_target = rec >= recall_target
    if meets_target.any():
        cand = np.where(meets_target)[0]
        idx  = cand[np.argmax(prec[cand])]
    else:
        idx = int(np.argmin(np.abs(rec - recall_target)))

    return float(thr[idx]), float(prec[idx]), float(rec[idx])


# ── threshold sensitivity ─────────────────────────────────────────────────────

def threshold_sensitivity(clf, X: np.ndarray, y: np.ndarray, threshold: float,
                           deltas=(-0.05, -0.02, 0.0, 0.02, 0.05)) -> list:
    """
    Recall/precision at threshold ± small deltas on the Test set
    (NOT on val_thresh where the threshold was chosen).
    A flat profile means the threshold sits on a stable plateau;
    a sharp swing means it sits on a cliff that may not transfer to OOT.
    """
    from sklearn.metrics import precision_score, recall_score
    y_prob = clf.predict_proba(X)[:, 1]
    rows = []
    for d in deltas:
        t = float(np.clip(threshold + d, 0.0, 1.0))
        y_pred = (y_prob >= t).astype(int)
        rows.append(dict(
            delta=round(d, 4), threshold=round(t, 4),
            recall=round(recall_score(y, y_pred, zero_division=0), 4),
            precision=round(precision_score(y, y_pred, zero_division=0), 4),
        ))
    return rows


# ── feature ~ label correlation ───────────────────────────────────────────────

def feature_label_correlation(df: pd.DataFrame, features: list, label: str) -> dict:
    """
    Point-biserial correlation between each raw feature and the binary label.
    Model-free cross-check against SHAP: a feature that ranks low on SHAP but
    has a meaningful raw correlation warrants investigation.
    """
    corrs = {}
    y = df[label].values.astype(float)
    for f in features:
        x = df[f].values.astype(float)
        corrs[f] = 0.0 if np.std(x) == 0 else float(np.corrcoef(x, y)[0, 1])
    ranked = dict(sorted(corrs.items(), key=lambda kv: abs(kv[1]), reverse=True))
    print("  Feature ~ label correlation (train set, |r| descending):")
    for f, r in ranked.items():
        print(f"    {f:<35} {r:+.4f}")
    return ranked


# ── MLflow logging ────────────────────────────────────────────────────────────

def log_to_mlflow(model, model_type: str, best_params: dict, val_metrics: dict):
    try:
        import mlflow
        import mlflow.xgboost
        import mlflow.lightgbm

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(EXPERIMENT_NAME)

        with mlflow.start_run(run_name=f"{model_type}_optuna") as run:
            mlflow.log_params(best_params)
            mlflow.log_param("model_type", model_type)
            mlflow.log_param("f2_beta", F2_BETA)
            mlflow.log_param("recall_target", RECALL_TARGET)
            mlflow.log_param("n_optuna_trials", N_OPTUNA_TRIALS)
            mlflow.log_param("smote", False)
            mlflow.log_param("calibration_method", "sigmoid")
            mlflow.log_param("features", json.dumps(FEATURES))
            mlflow.log_metrics(val_metrics)

            if model_type == "xgboost":
                mlflow.xgboost.log_model(model, "model",
                                         registered_model_name=MODEL_NAME)
            else:
                mlflow.lightgbm.log_model(model, "model",
                                          registered_model_name=MODEL_NAME)

            print(f"  MLflow run: {run.info.run_id}")
            return run.info.run_id

    except Exception as e:
        print(f"  ⚠ MLflow logging failed ({e}) — continuing without MLflow")
        return None


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    import optuna
    from sklearn.metrics import (
        recall_score, precision_score, f1_score, roc_auc_score,
        average_precision_score,
    )
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    print("=" * 60)
    print("Model Training — Dengue Cluster Prediction")
    print("=" * 60)

    # ── 1. data ───────────────────────────────────────────────────────────────
    splits = load_and_split(INPUT)
    X_train = splits["train"][FEATURES].values
    y_train = splits["train"][LABEL].values
    X_val_calib  = splits["val_calib"][FEATURES].values
    y_val_calib  = splits["val_calib"][LABEL].values
    X_val_thresh = splits["val_thresh"][FEATURES].values
    y_val_thresh = splits["val_thresh"][LABEL].values
    X_val = splits["val"][FEATURES].values
    y_val = splits["val"][LABEL].values
    X_test = splits["test"][FEATURES].values
    y_test = splits["test"][LABEL].values

    # ── 2. track best across both model types ─────────────────────────────────
    # Best selected by precision at the recall floor (not raw recall), because
    # a model that hits ≥ RECALL_TARGET with higher precision is more operationally
    # useful: fewer false alarms for the same outbreak catch rate.
    best_run = {
        "precision":  -1,
        "model":      None,
        "model_type": None,
        "params":     None,
        "threshold":  None,
        "raw_model":  None,
    }

    for model_type in ["xgboost", "lightgbm"]:
        print(f"\n[optuna] Tuning {model_type} — {N_OPTUNA_TRIALS} trials "
              f"(objective: F{F2_BETA:.0f}-score, no SMOTE)")
        study = optuna.create_study(direction="maximize")
        study.optimize(
            make_objective(splits["train"], model_type),
            n_trials=N_OPTUNA_TRIALS,
            show_progress_bar=True,
        )

        best_params    = study.best_params
        best_cv_score  = study.best_value
        print(f"  Best CV F{F2_BETA:.0f}-score : {best_cv_score:.4f}")
        print(f"  Best params     : {best_params}")

        # ── train final model on full raw training set ────────────────────────
        model = train_model(model_type, best_params, X_train, y_train)

        # ── probability calibration on val_calib ─────────────────────────────
        cal_model = calibrate_model(model, X_val_calib, y_val_calib, method="sigmoid")

        # ── threshold calibration on val_thresh ──────────────────────────────
        thr, val_prec, val_rec = select_threshold(
            cal_model, X_val_thresh, y_val_thresh, RECALL_TARGET
        )
        recall_flag = "PASS" if val_rec >= RECALL_TARGET else "FAIL"
        print(f"  Calibrated threshold : {thr:.3f}  "
              f"(val-thresh recall={val_rec:.4f}, precision={val_prec:.4f})  "
              f"{recall_flag}")

        # ── threshold sensitivity on Test ─────────────────────────────────────
        sens = threshold_sensitivity(cal_model, X_test, y_test, thr)
        print(f"  Threshold sensitivity ({model_type}, Test, Δ around {thr:.3f}):")
        for row in sens:
            marker = "  <-- chosen" if row["delta"] == 0.0 else ""
            print(f"    Δ={row['delta']:+.2f}  thr={row['threshold']:.3f}  "
                  f"recall={row['recall']:.4f}  precision={row['precision']:.4f}{marker}")

        # ── val metrics for MLflow (evaluated at calibrated threshold) ────────
        y_prob_val = cal_model.predict_proba(X_val)[:, 1]
        y_pred_val = (y_prob_val >= thr).astype(int)
        val_metrics = {
            "val_recall":    round(recall_score(y_val, y_pred_val, zero_division=0), 4),
            "val_precision": round(precision_score(y_val, y_pred_val, zero_division=0), 4),
            "val_f1":        round(f1_score(y_val, y_pred_val, zero_division=0), 4),
            "val_auc_roc":   round(roc_auc_score(y_val, y_prob_val), 4),
            "val_pr_auc":    round(average_precision_score(y_val, y_prob_val), 4),
        }
        print(f"  Val metrics (thr={thr:.3f}): {val_metrics}")

        # ── log to MLflow ─────────────────────────────────────────────────────
        log_to_mlflow(model, model_type, best_params, val_metrics)

        # ── update best run ───────────────────────────────────────────────────
        if val_prec > best_run["precision"]:
            best_run = {
                "precision":  val_prec,
                "model":      cal_model,
                "model_type": model_type,
                "params":     best_params,
                "threshold":  thr,
                "raw_model":  model,
                "metrics":    val_metrics,
                "sensitivity": sens,
            }

    print(f"\n[best] {best_run['model_type']} — "
          f"val precision={best_run['precision']:.4f} "
          f"@ threshold={best_run['threshold']:.3f}")

    # ── feature ~ label correlation (diagnostic) ──────────────────────────────
    print("\n[corr] Feature ~ label correlation (diagnostic cross-check vs SHAP)")
    corr_dict = feature_label_correlation(splits["train"], FEATURES, LABEL)

    # ── save best model ───────────────────────────────────────────────────────
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Native format — raw booster (no calibration layer, SHAP-compatible)
    if best_run["model_type"] == "xgboost":
        best_run["raw_model"].save_model(MODEL_DIR / "model.xgb")
    else:
        best_run["raw_model"].booster_.save_model(str(MODEL_DIR / "model.lgb"))

    # Calibrated wrapper — the actual scoring artifact (threshold applies to this)
    import joblib
    joblib.dump(best_run["model"], MODEL_DIR / "best_model_calibrated.joblib")

    # Metadata
    meta = {
        "model_type":               best_run["model_type"],
        "calibrated_threshold":     round(best_run["threshold"], 4),
        "calibration_method":       "sigmoid",
        "val_precision_at_thr":     round(best_run["precision"], 4),
        "val_recall_at_thr":        round(
            recall_score(
                y_val_thresh,
                (best_run["model"].predict_proba(X_val_thresh)[:, 1]
                 >= best_run["threshold"]).astype(int),
                zero_division=0,
            ), 4,
        ),
        "val_metrics":              best_run["metrics"],
        "params":                   best_run["params"],
        "features":                 FEATURES,
        "f2_beta":                  F2_BETA,
        "recall_target":            RECALL_TARGET,
        "n_optuna_trials":          N_OPTUNA_TRIALS,
        "n_cv_folds":               N_CV_FOLDS,
        "smote":                    False,
        "threshold_sensitivity":    best_run["sensitivity"],
        "feature_label_correlation": {f: round(r, 4) for f, r in corr_dict.items()},
        "train_end":                str(TRAIN_END.date()),
        "val_start":                str(VAL_START.date()),
        "val_mid":                  str(VAL_MID.date()),
        "val_end":                  str(VAL_END.date()),
    }
    with open(MODEL_DIR / "candidate_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[save] Candidate model saved → {MODEL_DIR}")
    print(f"[done] Training complete. Run evaluate.py to assess on test set.")

    recall_at_thr = meta["val_recall_at_thr"]
    if recall_at_thr < RECALL_TARGET:
        print(f"\n⚠ WARNING: Best val recall {recall_at_thr:.4f} < target {RECALL_TARGET}")
        print("  Consider: more Optuna trials, wider scale_pos_weight range, or "
              "additional features.")
    else:
        print(f"\n✓ Recall target met ({recall_at_thr:.4f} ≥ {RECALL_TARGET}).")


if __name__ == "__main__":
    main()
