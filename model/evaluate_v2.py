"""
evaluate.py — Dengue Cluster Model Evaluation
Loads thresholds from train.py outputs and produces full metric report + plots.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os, json, warnings
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score,
    precision_recall_curve, average_precision_score
)
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────
# 0. CONFIG  — must match train.py
# ─────────────────────────────────────────
RANDOM_STATE = 42
OUTPUT_DIR   = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────
# 1. LOAD DATA  (same split as train.py)
# ─────────────────────────────────────────
# Replace with:
#   df = pd.read_csv("dengue_features.csv")
#   X = df.drop(columns=["label"])
#   y = df["label"]

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X, y = make_classification(
    n_samples=5000, n_features=20, n_informative=10,
    weights=[0.85, 0.15], random_state=RANDOM_STATE
)
X = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(X.shape[1])])
y = pd.Series(y, name="label")

X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.15, stratify=y, random_state=RANDOM_STATE
)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.15, stratify=y_trainval, random_state=RANDOM_STATE
)

# ─────────────────────────────────────────
# 2. LOAD THRESHOLDS
# ─────────────────────────────────────────
thr_path = os.path.join(OUTPUT_DIR, "thresholds.json")
if not os.path.exists(thr_path):
    raise FileNotFoundError(
        "thresholds.json not found. Run train.py first."
    )
with open(thr_path) as f:
    thr = json.load(f)

XGB_THR       = thr["xgb_threshold"]
LGB_THR       = thr["lgb_threshold"]
RECALL_FLOOR  = thr["recall_floor"]
print(f"Loaded thresholds — XGB: {XGB_THR:.4f}  LGB: {LGB_THR:.4f}  (recall floor: {RECALL_FLOOR})")

# ─────────────────────────────────────────
# 3. RETRAIN MODELS (same config as train.py)
#    In production: load persisted models via joblib/pickle instead
# ─────────────────────────────────────────
scale_pos = (y_train == 0).sum() / (y_train == 1).sum()

xgb_model = xgb.XGBClassifier(
    n_estimators=400, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, scale_pos_weight=scale_pos,
    eval_metric="aucpr", early_stopping_rounds=30,
    use_label_encoder=False, random_state=RANDOM_STATE, verbosity=0,
)
lgb_model = lgb.LGBMClassifier(
    n_estimators=400, num_leaves=31, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, scale_pos_weight=scale_pos,
    metric="average_precision", n_jobs=-1,
    random_state=RANDOM_STATE, verbose=-1,
)

xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
lgb_model.fit(
    X_train, y_train, eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(period=-1)],
)

# ─────────────────────────────────────────
# 4. PREDICT WITH RECALL ≥ 0.80 THRESHOLDS
# ─────────────────────────────────────────
def predict_at_threshold(model, X, threshold):
    scores = model.predict_proba(X)[:, 1]
    preds  = (scores >= threshold).astype(int)
    return scores, preds

def compute_metrics(y_true, y_pred, y_score, label):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "Model":      label,
        "Recall":     round(recall_score(y_true, y_pred), 3),
        "Precision":  round(precision_score(y_true, y_pred, zero_division=0), 3),
        "F1":         round(f1_score(y_true, y_pred), 3),
        "PR-AUC":     round(average_precision_score(y_true, y_score), 3),
        "ROC-AUC":    round(roc_auc_score(y_true, y_score), 3),
        "TP":         int(tp), "FP": int(fp),
        "TN":         int(tn), "FN": int(fn),
        "FP per TP":  round(fp / max(tp, 1), 2),
    }

# Test set
xgb_scores_test, xgb_preds_test = predict_at_threshold(xgb_model, X_test, XGB_THR)
lgb_scores_test, lgb_preds_test = predict_at_threshold(lgb_model, X_test, LGB_THR)

# Val set (for cross-checking)
xgb_scores_val, xgb_preds_val = predict_at_threshold(xgb_model, X_val, XGB_THR)
lgb_scores_val, lgb_preds_val = predict_at_threshold(lgb_model, X_val, LGB_THR)

test_results = pd.DataFrame([
    compute_metrics(y_test, xgb_preds_test, xgb_scores_test, "XGBoost (test)"),
    compute_metrics(y_test, lgb_preds_test, lgb_scores_test, "LightGBM (test)"),
])
val_results = pd.DataFrame([
    compute_metrics(y_val, xgb_preds_val, xgb_scores_val, "XGBoost (val)"),
    compute_metrics(y_val, lgb_preds_val, lgb_scores_val, "LightGBM (val)"),
])

results_all = pd.concat([val_results, test_results]).reset_index(drop=True)

print("\n" + "="*60)
print(f"EVALUATION RESULTS  (threshold recall floor: ≥{RECALL_FLOOR})")
print("="*60)
print(results_all[["Model","Recall","Precision","F1","PR-AUC","ROC-AUC","TP","FP","FN"]].to_string(index=False))
print("="*60)

results_all.to_csv(os.path.join(OUTPUT_DIR, "evaluation_results.csv"), index=False)
print(f"\nResults saved → {OUTPUT_DIR}/evaluation_results.csv")

# ─────────────────────────────────────────
# 5. SCORE DISTRIBUTION PLOT
# ─────────────────────────────────────────
def plot_score_dist(ax, scores, y_true, threshold, model_name, color_pos, color_neg):
    pos = scores[y_true == 1]
    neg = scores[y_true == 0]
    bins = np.linspace(0, 1, 40)

    ax.hist(neg, bins=bins, density=True, alpha=0.55, color=color_neg,
            label=f"Negative (n={len(neg):,})", edgecolor="none")
    ax.hist(pos, bins=bins, density=True, alpha=0.70, color=color_pos,
            label=f"Positive (n={len(pos):,})", edgecolor="none")
    ax.axvline(threshold, color="#e3b341", lw=2.2, ls="--",
               label=f"Threshold = {threshold:.3f}")

    # Overlap zone shading
    ax.axvspan(0, threshold * 0.95, alpha=0.05, color=color_neg)

    # Separation annotation
    median_sep = abs(np.median(pos) - np.median(neg))
    ax.set_title(
        f"{model_name}  |  Score Distribution (Test Set)\n"
        f"Median gap = {median_sep:.3f}  |  Threshold = {threshold:.3f}  (recall ≥ {RECALL_FLOOR})",
        fontsize=11, fontweight="bold", color="#e6edf3", pad=8
    )
    ax.set_xlabel("Predicted Probability Score", fontsize=10, color="#b0b8c1")
    ax.set_ylabel("Density", fontsize=10, color="#b0b8c1")
    ax.tick_params(colors="#b0b8c1", labelsize=9)
    for spine in ax.spines.values(): spine.set_edgecolor("#30363d")
    ax.set_facecolor("#0d1117")
    ax.legend(fontsize=9, facecolor="#161b22", edgecolor="#30363d",
              labelcolor="#e6edf3", framealpha=0.9)

    # Annotation: median of positive scores
    med_pos = np.median(pos)
    ylim = ax.get_ylim()
    ax.annotate(
        f"Median positive\n= {med_pos:.3f}",
        xy=(med_pos, ylim[1] * 0.5),
        xytext=(min(med_pos + 0.15, 0.85), ylim[1] * 0.72),
        arrowprops=dict(arrowstyle="->", color=color_pos, lw=1.5),
        fontsize=9, color=color_pos,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#0d1117",
                  edgecolor=color_pos, alpha=0.9)
    )


fig, axes = plt.subplots(2, 1, figsize=(12, 10))
fig.patch.set_facecolor("#0d1117")
fig.suptitle(
    f"Score Distribution by True Label — Test Set\n"
    f"Threshold strategy: max precision s.t. recall ≥ {RECALL_FLOOR}  |  "
    f"XGB thr={XGB_THR:.3f}  LGB thr={LGB_THR:.3f}",
    fontsize=13, fontweight="bold", color="#e6edf3", y=1.01
)

plot_score_dist(axes[0], xgb_scores_test, y_test, XGB_THR,
                "XGBoost", "#58a6ff", "#f78166")
plot_score_dist(axes[1], lgb_scores_test, y_test, LGB_THR,
                "LightGBM", "#3fb950", "#f78166")

plt.tight_layout()
dist_path = os.path.join(OUTPUT_DIR, "score_distributions_eval.png")
plt.savefig(dist_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
plt.close()
print(f"Score distribution plot saved → {dist_path}")

# ─────────────────────────────────────────
# 6. THRESHOLD SENSITIVITY PLOT
#    shows recall vs threshold across ±0.10 window
# ─────────────────────────────────────────
def threshold_sensitivity(model, X, y, center_thr, window=0.10, steps=40):
    lo = max(0.01, center_thr - window)
    hi = min(0.99, center_thr + window)
    thresholds = np.linspace(lo, hi, steps)
    scores = model.predict_proba(X)[:, 1]
    recalls, precisions, f1s = [], [], []
    for t in thresholds:
        preds = (scores >= t).astype(int)
        recalls.append(recall_score(y, preds, zero_division=0))
        precisions.append(precision_score(y, preds, zero_division=0))
        f1s.append(f1_score(y, preds, zero_division=0))
    return thresholds, recalls, precisions, f1s

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
fig.patch.set_facecolor("#0d1117")
fig.suptitle("Threshold Sensitivity — Test Set  (±0.10 window around chosen threshold)",
             fontsize=13, fontweight="bold", color="#e6edf3")

for ax, model, thr_val, name, color in [
    (axes[0], xgb_model, XGB_THR, "XGBoost", "#58a6ff"),
    (axes[1], lgb_model, LGB_THR, "LightGBM", "#3fb950"),
]:
    thrs, recs, precs, f1s = threshold_sensitivity(model, X_test, y_test, thr_val)
    ax.plot(thrs, recs,  color=color,    lw=2.5, label="Recall")
    ax.plot(thrs, precs, color="#e3b341", lw=2,   label="Precision", ls="--")
    ax.plot(thrs, f1s,   color="#bc8cff", lw=2,   label="F1",        ls=":")
    ax.axvline(thr_val, color="white", lw=1.5, ls="--", alpha=0.7,
               label=f"Chosen thr = {thr_val:.3f}")
    ax.axhline(RECALL_FLOOR, color="#f78166", lw=1.2, ls=":", alpha=0.8,
               label=f"Recall floor = {RECALL_FLOOR}")

    ax.set_title(f"{name}", fontsize=12, fontweight="bold", color="#e6edf3")
    ax.set_xlabel("Threshold", fontsize=10, color="#b0b8c1")
    ax.set_ylabel("Score", fontsize=10, color="#b0b8c1")
    ax.tick_params(colors="#b0b8c1")
    for spine in ax.spines.values(): spine.set_edgecolor("#30363d")
    ax.set_facecolor("#0d1117")
    ax.legend(fontsize=9, facecolor="#161b22", edgecolor="#30363d",
              labelcolor="#e6edf3", framealpha=0.9)

plt.tight_layout()
sens_path = os.path.join(OUTPUT_DIR, "threshold_sensitivity.png")
plt.savefig(sens_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
plt.close()
print(f"Threshold sensitivity plot saved → {sens_path}")

print("\n✅ Evaluation complete.")
print(f"   Outputs in: {OUTPUT_DIR}/")
print("   - evaluation_results.csv")
print("   - score_distributions_eval.png")
print("   - threshold_sensitivity.png")
