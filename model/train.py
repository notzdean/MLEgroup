"""
model/train.py
===============
CS611 MLE Group Project — Dengue Outbreak Risk Prediction
Model training: XGBoost + LightGBM ensemble with Optuna HPT.

Reads : data/gold/subzone_features.parquet
Writes: MLflow experiment runs + Candidate model in MLflow Registry

Pipeline
--------
1. Load Gold feature table
2. Strict temporal split — Train / Val / Test / OOT
3. SMOTE on training set only (never val/test/OOT)
4. 5-fold time-series sliding-window CV within training window
5. Optuna HPT — optimise recall ≥ 0.70 against validation set
6. Train final XGBoost + LightGBM on full training set with best params
7. Log all runs to MLflow
8. Register best model as Candidate in MLflow Registry

Data splits
-----------
Train     : up to 2018-12-31
Validation: 2019-01-01 – 2019-06-30   (Optuna HPT target)
Test      : 2019-07-01 – 2019-12-31   (held out — touched once in evaluate.py)
OOT       : 2020-01-01 – 2020-11-06   (concept drift — DENV-3 serotype shift)

Design decisions
----------------
- Recall ≥ 0.70 target: false negative = missed outbreak (asymmetric cost)
- SMOTE on training fold only: prevents data leakage into val/test
- Time-series CV: standard k-fold leaks future cluster data
- XGBoost + LightGBM: tabular data, SHAP interpretable, trains locally
- Optuna: efficient HPT with pruning, better than grid search
"""

import json
import warnings
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import date

warnings.filterwarnings("ignore")

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT  = Path(__file__).resolve().parent.parent
GOLD  = ROOT / "data" / "gold"
INPUT = GOLD / "subzone_features.parquet"

# ── split boundaries ──────────────────────────────────────────────────────────
TRAIN_END = pd.Timestamp("2018-12-31")
VAL_START = pd.Timestamp("2019-01-01")
VAL_END   = pd.Timestamp("2019-06-30")
TEST_START = pd.Timestamp("2019-07-01")
TEST_END   = pd.Timestamp("2019-12-31")
OOT_START  = pd.Timestamp("2020-01-01")

FEATURES = [
    "rainfall_lag1w", "rainfall_lag2w", "rainfall_lag4w",
    "cluster_count_rolling2w", "cluster_count_rolling4w",
    "population", "elderly_pct", "area_km2", "population_density",
    "vulnerability_index",
]
LABEL = "label"

MLFLOW_TRACKING_URI = "http://localhost:5000"
EXPERIMENT_NAME     = "dengue_cluster_model"
MODEL_NAME          = "dengue_cluster_model"
N_OPTUNA_TRIALS     = 30
RECALL_TARGET       = 0.70
N_CV_FOLDS          = 5


# ── data loading & splitting ──────────────────────────────────────────────────

def load_and_split(path: Path):
    print(f"[load] {path.name}")
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=FEATURES + [LABEL])
    print(f"  Total rows: {len(df):,} | Positive labels: {df[LABEL].sum():,} ({df[LABEL].mean():.1%})")

    train = df[df["date"] <= TRAIN_END]
    val   = df[(df["date"] >= VAL_START) & (df["date"] <= VAL_END)]
    test  = df[(df["date"] >= TEST_START) & (df["date"] <= TEST_END)]
    oot   = df[df["date"] >= OOT_START]

    for name, split in [("Train", train), ("Val", val), ("Test", test), ("OOT", oot)]:
        print(f"  {name:5s}: {len(split):6,} rows | "
              f"positive: {split[LABEL].sum():,} ({split[LABEL].mean():.1%})")

    return train, val, test, oot


# ── SMOTE ─────────────────────────────────────────────────────────────────────

def apply_smote(X_train, y_train):
    from imblearn.over_sampling import SMOTE
    print(f"[smote] Before: {y_train.sum():,} positive / {len(y_train):,} total")
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    print(f"[smote] After : {y_res.sum():,} positive / {len(y_res):,} total")
    return X_res, y_res


# ── time-series CV ────────────────────────────────────────────────────────────

def time_series_cv_folds(train: pd.DataFrame, n_folds: int = N_CV_FOLDS):
    """
    Sliding-window CV within training set.
    Each fold: earlier dates = train, later dates = val. Never reversed.
    """
    dates = sorted(train["date"].unique())
    fold_size = len(dates) // (n_folds + 1)
    folds = []
    for i in range(1, n_folds + 1):
        train_end_idx = i * fold_size
        val_end_idx   = train_end_idx + fold_size
        train_dates = dates[:train_end_idx]
        val_dates   = dates[train_end_idx:val_end_idx]
        if not val_dates:
            continue
        fold_train = train[train["date"].isin(train_dates)]
        fold_val   = train[train["date"].isin(val_dates)]
        folds.append((fold_train, fold_val))
    return folds


# ── Optuna objective ──────────────────────────────────────────────────────────

def make_objective(train, val, model_type="xgboost"):
    from sklearn.metrics import recall_score

    X_val = val[FEATURES].values
    y_val = val[LABEL].values

    def objective(trial):
        if model_type == "xgboost":
            import xgboost as xgb
            params = {
                "n_estimators":     trial.suggest_int("n_estimators", 100, 500),
                "max_depth":        trial.suggest_int("max_depth", 3, 8),
                "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 10.0),
                "use_label_encoder": False,
                "eval_metric": "logloss",
                "random_state": 42,
            }
            model = xgb.XGBClassifier(**params)
        else:
            import lightgbm as lgb
            params = {
                "n_estimators":   trial.suggest_int("n_estimators", 100, 500),
                "max_depth":      trial.suggest_int("max_depth", 3, 8),
                "learning_rate":  trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample":      trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
                "scale_pos_weight":  trial.suggest_float("scale_pos_weight", 1.0, 10.0),
                "random_state": 42,
                "verbose": -1,
            }
            import lightgbm as lgb
            model = lgb.LGBMClassifier(**params)

        # CV recall
        folds = time_series_cv_folds(train)
        cv_recalls = []
        for fold_train, fold_val_inner in folds:
            X_ft = fold_train[FEATURES].values
            y_ft = fold_train[LABEL].values
            X_fv = fold_val_inner[FEATURES].values
            y_fv = fold_val_inner[LABEL].values
            X_ft_s, y_ft_s = apply_smote(X_ft, y_ft) if y_ft.sum() > 5 else (X_ft, y_ft)
            model.fit(X_ft_s, y_ft_s)
            preds = model.predict(X_fv)
            cv_recalls.append(recall_score(y_fv, preds, zero_division=0))

        return float(np.mean(cv_recalls))

    return objective


# ── training ──────────────────────────────────────────────────────────────────

def train_model(model_type, best_params, X_train, y_train):
    if model_type == "xgboost":
        import xgboost as xgb
        model = xgb.XGBClassifier(
            **best_params,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42
        )
    else:
        import lightgbm as lgb
        model = lgb.LGBMClassifier(**best_params, random_state=42, verbose=-1)
    model.fit(X_train, y_train)
    return model


# ── MLflow logging ────────────────────────────────────────────────────────────

def log_to_mlflow(model, model_type, best_params, val_metrics, X_train):
    try:
        import mlflow
        import mlflow.xgboost
        import mlflow.lightgbm
        from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(EXPERIMENT_NAME)

        with mlflow.start_run(run_name=f"{model_type}_optuna") as run:
            mlflow.log_params(best_params)
            mlflow.log_param("model_type", model_type)
            mlflow.log_metrics(val_metrics)
            mlflow.log_param("features", json.dumps(FEATURES))

            if model_type == "xgboost":
                mlflow.xgboost.log_model(
                    model, "model",
                    registered_model_name=MODEL_NAME
                )
            else:
                mlflow.lightgbm.log_model(
                    model, "model",
                    registered_model_name=MODEL_NAME
                )

            print(f"  MLflow run: {run.info.run_id}")
            return run.info.run_id

    except Exception as e:
        print(f"  ⚠ MLflow logging failed ({e}) — continuing without MLflow")
        return None


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    import optuna
    from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    print("=" * 60)
    print("Model Training — Dengue Cluster Prediction")
    print("=" * 60)

    train, val, test, oot = load_and_split(INPUT)

    X_train_raw = train[FEATURES].values
    y_train_raw = train[LABEL].values
    X_val = val[FEATURES].values
    y_val = val[LABEL].values

    # Apply SMOTE to full training set for final model training
    X_train, y_train = apply_smote(X_train_raw, y_train_raw)

    best_run = {"recall": 0, "model": None, "model_type": None, "params": None}

    for model_type in ["xgboost", "lightgbm"]:
        print(f"\n[optuna] Tuning {model_type} — {N_OPTUNA_TRIALS} trials")
        study = optuna.create_study(direction="maximize")
        study.optimize(
            make_objective(train, val, model_type),
            n_trials=N_OPTUNA_TRIALS,
            show_progress_bar=True
        )

        best_params = study.best_params
        best_cv_recall = study.best_value
        print(f"  Best CV recall: {best_cv_recall:.4f}")
        print(f"  Best params: {best_params}")

        # Train final model on full training set
        model = train_model(model_type, best_params, X_train, y_train)

        # Evaluate on validation set
        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val)[:, 1]

        val_metrics = {
            "val_recall":    round(recall_score(y_val, y_pred, zero_division=0), 4),
            "val_precision": round(precision_score(y_val, y_pred, zero_division=0), 4),
            "val_f1":        round(f1_score(y_val, y_pred, zero_division=0), 4),
            "val_auc_roc":   round(roc_auc_score(y_val, y_prob), 4),
        }

        print(f"  Val metrics: {val_metrics}")

        recall_flag = "PASS" if val_metrics["val_recall"] >= RECALL_TARGET else "FAIL"
        print(f"  Recall >= {RECALL_TARGET}: {recall_flag} ({val_metrics['val_recall']:.4f})")

        # Log to MLflow
        log_to_mlflow(model, model_type, best_params, val_metrics, X_train)

        # Track best model
        if val_metrics["val_recall"] > best_run["recall"]:
            best_run = {
                "recall":     val_metrics["val_recall"],
                "model":      model,
                "model_type": model_type,
                "params":     best_params,
                "metrics":    val_metrics,
            }

    print(f"\n[best] {best_run['model_type']} — val recall {best_run['recall']:.4f}")

    # Save best model locally as fallback (if MLflow not running)
    model_dir = ROOT / "model" / "candidate"
    model_dir.mkdir(parents=True, exist_ok=True)

    if best_run["model_type"] == "xgboost":
        best_run["model"].save_model(model_dir / "model.xgb")
    else:
        import lightgbm as lgb
        best_run["model"].booster_.save_model(str(model_dir / "model.lgb"))

    # Save metadata
    meta = {
        "model_type":  best_run["model_type"],
        "val_metrics": best_run["metrics"],
        "params":      best_run["params"],
        "features":    FEATURES,
        "train_end":   str(TRAIN_END.date()),
        "val_start":   str(VAL_START.date()),
        "val_end":     str(VAL_END.date()),
        "recall_target": RECALL_TARGET,
    }
    with open(model_dir / "candidate_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[save] Candidate model saved → {model_dir}")
    print(f"[done] Training complete.")

    if best_run["recall"] < RECALL_TARGET:
        print(f"\n⚠ WARNING: Best val recall {best_run['recall']:.4f} < target {RECALL_TARGET}")
        print("  Consider: more Optuna trials, class weight tuning, or additional features.")
    else:
        print(f"\n✓ Recall target met. Run evaluate.py to assess on test set.")


if __name__ == "__main__":
    main()
