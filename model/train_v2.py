"""
train.py — Dengue Cluster Model Training
Threshold selected via: max precision subject to recall >= 0.80 (val PR curve)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os, json, warnings

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV

import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────
# 0. CONFIG
# ─────────────────────────────────────────
RECALL_FLOOR      = 0.80       # minimum recall constraint
RANDOM_STATE      = 42
OUTPUT_DIR        = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────
# Replace with your actual data loading:
#   df = pd.read_csv("dengue_features.csv")
#   X = df.drop(columns=["label"])
#   y = df["label"]
#
# Placeholder for reproducibility (remove when using real data):
from sklearn.datasets import make_classification
X, y = make_classification(
    n_samples=5000, n_features=20, n_informative=10,
    weights=[0.85, 0.15],          # ~15% positive — typical cluster imbalance
    random_state=RANDOM_STATE
)
X = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(X.shape[1])])
y = pd.Series(y, name="label")

# ─────────────────────────────────────────
# 2. SPLIT: TRAIN / VAL / TEST / OOT
# ─────────────────────────────────────────
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.15, stratify=y, random_state=RANDOM_STATE
)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.15, stratify=y_trainval, random_state=RANDOM_STATE
)

print(f"Train: {len(y_train):,}  |  Val: {len(y_val):,}  |  Test: {len(y_test):,}")
print(f"Positive rate → Train: {y_train.mean():.2%}  Val: {y_val.mean():.2%}  Test: {y_test.mean():.2%}")

# ─────────────────────────────────────────
# 3. MODELS
# ─────────────────────────────────────────
scale_pos = (y_train == 0).sum() / (y_train == 1).sum()

xgb_model = xgb.XGBClassifier(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos,
    eval_metric="aucpr",
    early_stopping_rounds=30,
    use_label_encoder=False,
    random_state=RANDOM_STATE,
    verbosity=0,
)

lgb_model = lgb.LGBMClassifier(
    n_estimators=400,
    num_leaves=31,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos,
    metric="average_precision",
    n_jobs=-1,
    random_state=RANDOM_STATE,
    verbose=-1,
)

# ─────────────────────────────────────────
# 4. TRAIN
# ─────────────────────────────────────────
print("\nTraining XGBoost...")
xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False,
)

print("Training LightGBM...")
lgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(period=-1)],
)

# ─────────────────────────────────────────
# 5. THRESHOLD SELECTION (Val PR curve)
#    max precision subject to recall >= RECALL_FLOOR
# ─────────────────────────────────────────
def select_threshold(model, X_val, y_val, recall_floor=0.80):
    """
    Find threshold from val PR curve:
    argmax precision s.t. recall >= recall_floor
    """
    scores = model.predict_proba(X_val)[:, 1]
    prec, rec, thresholds = precision_recall_curve(y_val, scores)
    # prec/rec have len(thresholds)+1 elements; align them
    prec_t  = prec[:-1]
    rec_t   = rec[:-1]
    mask = rec_t >= recall_floor
    if not mask.any():
        raise ValueError(
            f"No threshold satisfies recall >= {recall_floor}. "
            "Lower the recall floor or retune the model."
        )
    best_idx  = np.argmax(prec_t[mask])
    best_thr  = thresholds[mask][best_idx]
    best_prec = prec_t[mask][best_idx]
    best_rec  = rec_t[mask][best_idx]
    return best_thr, best_prec, best_rec, prec, rec, thresholds

xgb_thr, xgb_val_prec, xgb_val_rec, xgb_prec_all, xgb_rec_all, xgb_thr_all =     select_threshold(xgb_model, X_val, y_val, RECALL_FLOOR)

lgb_thr, lgb_val_prec, lgb_val_rec, lgb_prec_all, lgb_rec_all, lgb_thr_all =     select_threshold(lgb_model, X_val, y_val, RECALL_FLOOR)

print(f"\nXGBoost selected threshold : {xgb_thr:.4f}  (val recall={xgb_val_rec:.3f}, val prec={xgb_val_prec:.3f})")
print(f"LightGBM selected threshold: {lgb_thr:.4f}  (val recall={lgb_val_rec:.3f}, val prec={lgb_val_prec:.3f})")

# Save thresholds for evaluate.py
thresholds_dict = {
    "xgb_threshold": float(xgb_thr),
    "lgb_threshold": float(lgb_thr),
    "recall_floor":  RECALL_FLOOR,
}
with open(os.path.join(OUTPUT_DIR, "thresholds.json"), "w") as f:
    json.dump(thresholds_dict, f, indent=2)
print(f"\nThresholds saved → {OUTPUT_DIR}/thresholds.json")

# ─────────────────────────────────────────
# 6. SCORE DISTRIBUTION PLOT
# ─────────────────────────────────────────
def plot_score_distribution(model, X_set, y_set, threshold, model_name, ax):
    scores = model.predict_proba(X_set)[:, 1]
    pos_scores = scores[y_set == 1]
    neg_scores = scores[y_set == 0]

    bins = np.linspace(0, 1, 40)
    ax.hist(neg_scores, bins=bins, alpha=0.55, color="#f78166", label="True Negative",
            edgecolor="none", density=True)
    ax.hist(pos_scores, bins=bins, alpha=0.65, color="#3fb950", label="True Positive",
            edgecolor="none", density=True)
    ax.axvline(threshold, color="#e3b341", linewidth=2.2, linestyle="--",
               label=f"Threshold = {threshold:.3f}")

    # Shade saturation zone for positives
    ax.axvspan(0, threshold, alpha=0.06, color="#f78166", label="_nolegend_")

    ax.set_title(f"{model_name} — Score Distribution (Test Set)",
                 fontsize=13, fontweight="bold", color="#e6edf3", pad=10)
    ax.set_xlabel("Predicted Probability", fontsize=11, color="#b0b8c1")
    ax.set_ylabel("Density", fontsize=11, color="#b0b8c1")
    ax.tick_params(colors="#b0b8c1")
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")
    ax.set_facecolor("#0d1117")
    ax.legend(fontsize=10, facecolor="#161b22", edgecolor="#30363d",
              labelcolor="#e6edf3", framealpha=0.9)

    # Annotation: cluster of pos scores
    med_pos = np.median(pos_scores)
    ax.annotate(
        f"Median positive score\n= {med_pos:.3f}",
        xy=(med_pos, ax.get_ylim()[1] * 0.6 if ax.get_ylim()[1] > 0 else 1),
        xytext=(med_pos + 0.12, ax.get_ylim()[1] * 0.75 if ax.get_ylim()[1] > 0 else 1.5),
        arrowprops=dict(arrowstyle="->", color="#3fb950", lw=1.5),
        fontsize=9, color="#3fb950",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#0d1117",
                  edgecolor="#3fb950", alpha=0.9)
    )
    return scores, pos_scores, neg_scores


fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor("#0d1117")
fig.suptitle(
    "Predicted Score Distributions — Test Set\n"
    f"(Threshold floor: recall ≥ {RECALL_FLOOR})",
    fontsize=14, fontweight="bold", color="#e6edf3", y=1.02
)

xgb_scores, xgb_pos, xgb_neg = plot_score_distribution(
    xgb_model, X_test, y_test, xgb_thr, "XGBoost", axes[0]
)
# Re-run annotation after ylim is set
axes[0].relim(); axes[0].autoscale_view()

lgb_scores, lgb_pos, lgb_neg = plot_score_distribution(
    lgb_model, X_test, y_test, lgb_thr, "LightGBM", axes[1]
)

plt.tight_layout()
dist_path = os.path.join(OUTPUT_DIR, "score_distributions.png")
plt.savefig(dist_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
plt.close()
print(f"Score distribution plot saved → {dist_path}")

# ─────────────────────────────────────────
# 7. VAL PR CURVE PLOT
# ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
fig.patch.set_facecolor("#0d1117")
ax.set_facecolor("#0d1117")

ax.plot(xgb_rec_all, xgb_prec_all, color="#58a6ff", linewidth=2.5, label="XGBoost PR curve")
ax.plot(lgb_rec_all, lgb_prec_all, color="#3fb950", linewidth=2.5, label="LightGBM PR curve")
ax.axvline(RECALL_FLOOR, color="#e3b341", linewidth=1.8, linestyle="--",
           label=f"Recall floor = {RECALL_FLOOR}")
ax.scatter([xgb_val_rec], [xgb_val_prec], color="#58a6ff", s=100, zorder=5,
           label=f"XGB operating point (rec={xgb_val_rec:.2f}, prec={xgb_val_prec:.2f})")
ax.scatter([lgb_val_rec], [lgb_val_prec], color="#3fb950", s=100, zorder=5,
           label=f"LGB operating point (rec={lgb_val_rec:.2f}, prec={lgb_val_prec:.2f})")

ax.set_title("Validation PR Curve — Threshold Selection\n"
             f"Max precision s.t. recall ≥ {RECALL_FLOOR}",
             fontsize=13, fontweight="bold", color="#e6edf3")
ax.set_xlabel("Recall", fontsize=11, color="#b0b8c1")
ax.set_ylabel("Precision", fontsize=11, color="#b0b8c1")
ax.tick_params(colors="#b0b8c1")
for spine in ax.spines.values(): spine.set_edgecolor("#30363d")
ax.legend(fontsize=9, facecolor="#161b22", edgecolor="#30363d",
          labelcolor="#e6edf3", framealpha=0.9, loc="upper right")

plt.tight_layout()
pr_path = os.path.join(OUTPUT_DIR, "val_pr_curve.png")
plt.savefig(pr_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
plt.close()
print(f"Val PR curve plot saved → {pr_path}")

print("\n✅ Training complete. Run evaluate.py to get full metrics.")
