#!/usr/bin/env python3
"""
03_model_evaluation.py — Comprehensive model evaluation.

1. Results comparison table (all models x all metrics)
2. Confusion matrices for all models
3. ROC curves (one-vs-rest) for top 3 models
4. Calibration plots for top 3 models
5. Per-class precision/recall/F1 (focus on draw class)
6. Test set evaluation (holdout: 2022 WC onward)
7. 2022 World Cup specific analysis
8. Time-series cross-validation
"""

import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import (confusion_matrix, classification_report,
                              roc_curve, auc, f1_score, accuracy_score,
                              log_loss, roc_auc_score)
from sklearn.calibration import calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier

from src.models import evaluate_model

# ── Setup ──────────────────────────────────────────────────────────────
SPLITS = BASE / "data" / "processed" / "splits"
ARTIFACTS = BASE / "outputs" / "model_artifacts"
FIG_DIR = BASE / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 11,
    "figure.facecolor": "white",
})
sns.set_style("whitegrid")

CLASS_NAMES = ["Home Win", "Draw", "Away Win"]
CLASS_COLORS = ["#2a9d8f", "#e9c46a", "#e63946"]

# ── Load data ──────────────────────────────────────────────────────────
print("Loading data and models...")
X_val = pd.read_csv(SPLITS / "X_val.csv")
y_val = pd.read_csv(SPLITS / "y_val.csv")["result"].values
X_test = pd.read_csv(SPLITS / "X_test.csv")
y_test = pd.read_csv(SPLITS / "y_test.csv")["result"].values
X_val_med = pd.read_csv(SPLITS / "X_val_median.csv")
X_test_med = pd.read_csv(SPLITS / "X_test_median.csv")

results_df = pd.read_csv(ARTIFACTS / "results_summary.csv")

# Load key models
models = {}
model_files = list(ARTIFACTS.glob("*.joblib"))
for f in model_files:
    if f.stem in ["optuna_results"]:
        continue
    try:
        models[f.stem] = joblib.load(f)
    except Exception as e:
        print(f"  Warning: couldn't load {f.stem}: {e}")

print(f"Loaded {len(models)} models: {list(models.keys())}")

optuna_data = joblib.load(ARTIFACTS / "optuna_results.joblib")


# ══════════════════════════════════════════════════════════════════════
# 1. Results comparison table
# ══════════════════════════════════════════════════════════════════════
print("\n[1/8] Results comparison table...")
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)
print(results_df.sort_values("macro_f1", ascending=False).to_string(index=False, float_format="{:.4f}".format))


# ══════════════════════════════════════════════════════════════════════
# 2. Confusion matrices for key models
# ══════════════════════════════════════════════════════════════════════
print("\n[2/8] Confusion matrices...")

key_models = ["XGBoost_tuned", "XGBoost_tuned_balanced", "HistGBM_tuned",
              "HistGBM_tuned_balanced", "StackingEnsemble", "LogisticRegression"]
key_models = [m for m in key_models if m in models]

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

for i, name in enumerate(key_models[:6]):
    model = models[name]
    # Choose correct X based on model type
    if name in ["StackingEnsemble", "RandomForest"]:
        X_v = X_val_med
    else:
        X_v = X_val
    y_pred = model.predict(X_v)
    cm = confusion_matrix(y_val, y_pred)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    sns.heatmap(cm_pct, annot=True, fmt=".1f", cmap="Blues", ax=axes[i],
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                cbar_kws={"label": "%"})
    # Add count annotations
    for r in range(3):
        for c in range(3):
            axes[i].text(c + 0.5, r + 0.75, f"(n={cm[r,c]})",
                        ha="center", va="center", fontsize=7, color="gray")
    axes[i].set_ylabel("Actual")
    axes[i].set_xlabel("Predicted")
    axes[i].set_title(name.replace("_", " "), fontsize=10)

for j in range(len(key_models), 6):
    axes[j].set_visible(False)

plt.suptitle("Confusion Matrices (Validation Set)", fontsize=14)
plt.tight_layout()
fig.savefig(FIG_DIR / "10_confusion_matrices.png")
plt.close()
print("  Saved: 10_confusion_matrices.png")


# ══════════════════════════════════════════════════════════════════════
# 3. ROC curves (one-vs-rest) for top 3 models
# ══════════════════════════════════════════════════════════════════════
print("\n[3/8] ROC curves...")

top3 = ["XGBoost_tuned_balanced", "HistGBM_tuned_balanced", "HistGBM_tuned"]
top3 = [m for m in top3 if m in models]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for cls_idx in range(3):
    ax = axes[cls_idx]
    y_binary = (y_val == cls_idx).astype(int)

    for m_name in top3:
        model = models[m_name]
        X_v = X_val_med if m_name in ["StackingEnsemble", "RandomForest"] else X_val
        y_proba = model.predict_proba(X_v)[:, cls_idx]
        fpr, tpr, _ = roc_curve(y_binary, y_proba)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, linewidth=2, label=f"{m_name.replace('_', ' ')} (AUC={roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC — {CLASS_NAMES[cls_idx]}")
    ax.legend(fontsize=7, loc="lower right")

plt.suptitle("One-vs-Rest ROC Curves (Validation Set)", fontsize=14)
plt.tight_layout()
fig.savefig(FIG_DIR / "11_roc_curves.png")
plt.close()
print("  Saved: 11_roc_curves.png")


# ══════════════════════════════════════════════════════════════════════
# 4. Calibration plots for top 3 models
# ══════════════════════════════════════════════════════════════════════
print("\n[4/8] Calibration plots...")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for cls_idx in range(3):
    ax = axes[cls_idx]
    y_binary = (y_val == cls_idx).astype(int)

    for m_name in top3:
        model = models[m_name]
        X_v = X_val_med if m_name in ["StackingEnsemble", "RandomForest"] else X_val
        y_proba = model.predict_proba(X_v)[:, cls_idx]
        prob_true, prob_pred = calibration_curve(y_binary, y_proba, n_bins=10, strategy="uniform")
        ax.plot(prob_pred, prob_true, "o-", linewidth=2,
                label=m_name.replace("_", " "))

    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Perfectly calibrated")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title(f"Calibration — {CLASS_NAMES[cls_idx]}")
    ax.legend(fontsize=7)

plt.suptitle("Calibration Plots (Validation Set)", fontsize=14)
plt.tight_layout()
fig.savefig(FIG_DIR / "12_calibration_plots.png")
plt.close()
print("  Saved: 12_calibration_plots.png")


# ══════════════════════════════════════════════════════════════════════
# 5. Per-class precision/recall/F1 (draw focus)
# ══════════════════════════════════════════════════════════════════════
print("\n[5/8] Per-class P/R/F1 table...")

print("\nPer-class metrics for top models on validation set:")
for m_name in key_models:
    model = models[m_name]
    X_v = X_val_med if m_name in ["StackingEnsemble", "RandomForest"] else X_val
    y_pred = model.predict(X_v)
    print(f"\n--- {m_name} ---")
    print(classification_report(y_val, y_pred, target_names=CLASS_NAMES, digits=3))


# ══════════════════════════════════════════════════════════════════════
# 6. Test set evaluation (holdout)
# ══════════════════════════════════════════════════════════════════════
print("\n[6/8] Test set evaluation (2022 WC holdout)...")

# Evaluate best models on test set
best_models = ["XGBoost_tuned_balanced", "HistGBM_tuned_balanced", "XGBoost_tuned", "StackingEnsemble"]
best_models = [m for m in best_models if m in models]

test_results = []
for m_name in best_models:
    model = models[m_name]
    X_t = X_test_med if m_name in ["StackingEnsemble", "RandomForest"] else X_test
    metrics = evaluate_model(model, X_t, y_test, model_name=m_name)
    test_results.append(metrics)
    print(f"\n  {m_name}:")
    print(f"    Accuracy: {metrics['accuracy']:.4f}  |  Macro F1: {metrics['macro_f1']:.4f}  |  "
          f"Log Loss: {metrics['log_loss']:.4f}  |  AUC-ROC: {metrics['auc_roc_ovr']:.4f}")
    print(f"    Per-class acc: HW={metrics['acc_class_0']:.3f}  D={metrics['acc_class_1']:.3f}  AW={metrics['acc_class_2']:.3f}")

test_df = pd.DataFrame(test_results)
test_df.to_csv(ARTIFACTS / "test_results.csv", index=False)

# Confusion matrix for best model on test set
fig, axes = plt.subplots(1, len(best_models), figsize=(5 * len(best_models), 5))
if len(best_models) == 1:
    axes = [axes]

for i, m_name in enumerate(best_models):
    model = models[m_name]
    X_t = X_test_med if m_name in ["StackingEnsemble", "RandomForest"] else X_test
    y_pred = model.predict(X_t)
    cm = confusion_matrix(y_test, y_pred)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    sns.heatmap(cm_pct, annot=True, fmt=".1f", cmap="Oranges", ax=axes[i],
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    axes[i].set_ylabel("Actual")
    axes[i].set_xlabel("Predicted")
    axes[i].set_title(f"{m_name.replace('_', ' ')}\n(Test Set)")

plt.suptitle("Test Set Confusion Matrices", fontsize=14)
plt.tight_layout()
fig.savefig(FIG_DIR / "13_test_confusion_matrices.png")
plt.close()
print("  Saved: 13_test_confusion_matrices.png")


# ══════════════════════════════════════════════════════════════════════
# 7. 2022 World Cup specific analysis
# ══════════════════════════════════════════════════════════════════════
print("\n[7/8] 2022 World Cup analysis...")

# Load full feature matrix to get 2022 WC matches
fm = pd.read_csv(BASE / "data" / "processed" / "feature_matrix.csv", parse_dates=["date"])
wc22 = fm[(fm["tournament"].str.contains("FIFA World Cup", na=False)) &
          (fm["date"] >= "2022-11-20") & (fm["date"] <= "2022-12-18")].copy()

if len(wc22) > 0:
    print(f"  Found {len(wc22)} 2022 World Cup matches")

    DROP_COLS = ["date", "home_team", "away_team", "home_score", "away_score", "tournament", "result"]
    X_wc22 = wc22.drop(columns=DROP_COLS, errors="ignore")
    y_wc22 = wc22["result"].values

    # Use best model
    best_name = "XGBoost_tuned_balanced"
    if best_name in models:
        model = models[best_name]
        y_pred = model.predict(X_wc22)
        y_proba = model.predict_proba(X_wc22)

        print(f"\n  2022 WC predictions ({best_name}):")
        print(f"    Accuracy: {accuracy_score(y_wc22, y_pred):.4f}")
        print(f"    Macro F1: {f1_score(y_wc22, y_pred, average='macro'):.4f}")

        # Per-match predictions
        print("\n  Match-by-match predictions:")
        correct = 0
        for idx, (_, row) in enumerate(wc22.iterrows()):
            pred = y_pred[idx]
            actual = int(row["result"])
            pred_label = CLASS_NAMES[pred]
            actual_label = CLASS_NAMES[actual]
            match_str = f"  {row['home_team']} vs {row['away_team']}"
            probs_str = f"[HW:{y_proba[idx,0]:.2f} D:{y_proba[idx,1]:.2f} AW:{y_proba[idx,2]:.2f}]"
            check = "OK" if pred == actual else "XX"
            if pred == actual:
                correct += 1
            print(f"    {check} {match_str:<40s} Pred: {pred_label:<10s} Actual: {actual_label:<10s} {probs_str}")

        print(f"\n    Correct: {correct}/{len(wc22)} ({100*correct/len(wc22):.1f}%)")

        # Confusion matrix for WC 2022
        cm = confusion_matrix(y_wc22, y_pred, labels=[0, 1, 2])
        fig, ax = plt.subplots(figsize=(6, 5))
        cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
        sns.heatmap(cm_pct, annot=True, fmt=".1f", cmap="Greens", ax=ax,
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
        for r in range(3):
            for c in range(3):
                ax.text(c + 0.5, r + 0.75, f"(n={cm[r,c]})",
                       ha="center", va="center", fontsize=8, color="gray")
        ax.set_title(f"2022 World Cup Predictions — {best_name.replace('_', ' ')}")
        ax.set_ylabel("Actual")
        ax.set_xlabel("Predicted")
        plt.tight_layout()
        fig.savefig(FIG_DIR / "14_wc2022_confusion_matrix.png")
        plt.close()
        print("  Saved: 14_wc2022_confusion_matrix.png")
else:
    print("  No 2022 WC matches found in feature matrix")


# ══════════════════════════════════════════════════════════════════════
# 8. Time-series cross-validation
# ══════════════════════════════════════════════════════════════════════
print("\n[8/8] Time-series cross-validation...")

# Expanding window: train to each WC, test on next cycle
fm_full = pd.read_csv(BASE / "data" / "processed" / "feature_matrix.csv", parse_dates=["date"])
DROP_COLS = ["date", "home_team", "away_team", "home_score", "away_score", "tournament", "result"]

windows = [
    ("Train to 2010", "2004-01-01", "2010-07-11", "2010-07-12", "2014-07-12"),
    ("Train to 2014", "2004-01-01", "2014-07-13", "2014-07-14", "2018-07-15"),
    ("Train to 2018", "2004-01-01", "2018-07-15", "2018-07-16", "2022-12-18"),
    ("Train to 2022", "2004-01-01", "2022-12-18", "2022-12-19", "2026-12-31"),
]

cv_results = []
for name, train_start, train_end, test_start, test_end in windows:
    train = fm_full[(fm_full["date"] >= train_start) & (fm_full["date"] <= train_end)]
    test = fm_full[(fm_full["date"] >= test_start) & (fm_full["date"] <= test_end)]

    if len(test) == 0:
        continue

    X_tr = train.drop(columns=DROP_COLS, errors="ignore")
    y_tr = train["result"].values
    X_te = test.drop(columns=DROP_COLS, errors="ignore")
    y_te = test["result"].values

    # Use XGBoost with best params
    xgb_params = optuna_data["xgboost_best_params"]
    model_cv = XGBClassifier(
        objective="multi:softprob", num_class=3,
        random_state=42, n_jobs=-1, tree_method="hist",
        **xgb_params
    )
    # Balanced weights
    class_counts = np.bincount(y_tr)
    cw = len(y_tr) / (3 * class_counts)
    sw = np.array([cw[y] for y in y_tr])
    model_cv.fit(X_tr, y_tr, sample_weight=sw)

    y_pred = model_cv.predict(X_te)
    y_proba = model_cv.predict_proba(X_te)

    cv_res = {
        "window": name,
        "train_size": len(train),
        "test_size": len(test),
        "accuracy": accuracy_score(y_te, y_pred),
        "macro_f1": f1_score(y_te, y_pred, average="macro"),
        "log_loss": log_loss(y_te, y_proba),
    }
    try:
        cv_res["auc_roc"] = roc_auc_score(y_te, y_proba, multi_class="ovr", average="macro")
    except ValueError:
        cv_res["auc_roc"] = np.nan
    cv_results.append(cv_res)
    print(f"  {name}: acc={cv_res['accuracy']:.4f}, F1={cv_res['macro_f1']:.4f}, "
          f"logloss={cv_res['log_loss']:.4f}, train={cv_res['train_size']}, test={cv_res['test_size']}")

cv_df = pd.DataFrame(cv_results)
cv_df.to_csv(ARTIFACTS / "timeseries_cv_results.csv", index=False)

# Plot CV metrics over time
if len(cv_results) > 1:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metrics_to_plot = [("accuracy", "Accuracy"), ("macro_f1", "Macro F1"), ("log_loss", "Log Loss")]
    for ax, (col, label) in zip(axes, metrics_to_plot):
        ax.plot(range(len(cv_df)), cv_df[col], "o-", color="#264653", linewidth=2, markersize=8)
        ax.set_xticks(range(len(cv_df)))
        ax.set_xticklabels(cv_df["window"], rotation=30, ha="right")
        ax.set_ylabel(label)
        ax.set_title(f"Time-Series CV: {label}")
        ax.grid(True, alpha=0.3)

    plt.suptitle("Expanding Window Cross-Validation (XGBoost Balanced)", fontsize=14)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "15_timeseries_cv.png")
    plt.close()
    print("  Saved: 15_timeseries_cv.png")

print("\n" + "=" * 60)
print("EVALUATION COMPLETE")
print("=" * 60)
