#!/usr/bin/env python3
"""
02_model_training.py — Train all 7 models with Optuna tuning + class imbalance experiments.

Outputs:
  outputs/model_artifacts/*.joblib — saved models
  outputs/model_artifacts/results_summary.csv — model x metric table
"""

import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

import numpy as np
import pandas as pd
import joblib
import optuna
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import (RandomForestClassifier, HistGradientBoostingClassifier,
                               StackingClassifier)
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, f1_score, log_loss, roc_auc_score)
from xgboost import XGBClassifier

from src.models import evaluate_model

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Paths ──────────────────────────────────────────────────────────────
SPLITS = BASE / "data" / "processed" / "splits"
ARTIFACTS = BASE / "outputs" / "model_artifacts"
ARTIFACTS.mkdir(parents=True, exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────
print("=" * 60)
print("02_model_training.py — Training 7 models")
print("=" * 60)

X_train = pd.read_csv(SPLITS / "X_train.csv")
y_train = pd.read_csv(SPLITS / "y_train.csv")["result"].values
X_val = pd.read_csv(SPLITS / "X_val.csv")
y_val = pd.read_csv(SPLITS / "y_val.csv")["result"].values
X_train_smote = pd.read_csv(SPLITS / "X_train_smote.csv")
y_train_smote = pd.read_csv(SPLITS / "y_train_smote.csv")["result"].values
X_train_med = pd.read_csv(SPLITS / "X_train_median.csv")
X_val_med = pd.read_csv(SPLITS / "X_val_median.csv")

print(f"Train: {X_train.shape}, Val: {X_val.shape}")
print(f"Train SMOTE: {X_train_smote.shape}")

# RF can't handle NaN — use median-imputed for RF
X_train_rf = X_train_med.copy()
X_val_rf = X_val_med.copy()

all_results = []


# ── Helper ─────────────────────────────────────────────────────────────
def train_and_log(model, name, X_tr, y_tr, X_v, y_v, save=True):
    """Train model, log on val, save artifact."""
    print(f"\n  Training {name}...")
    model.fit(X_tr, y_tr)
    metrics = evaluate_model(model, X_v, y_v, model_name=name)
    print(f"    Accuracy: {metrics['accuracy']:.4f}  |  Macro F1: {metrics['macro_f1']:.4f}  |  "
          f"Log Loss: {metrics['log_loss']:.4f}  |  AUC-ROC: {metrics['auc_roc_ovr']:.4f}")
    print(f"    Per-class acc: HW={metrics['acc_class_0']:.3f}  D={metrics['acc_class_1']:.3f}  AW={metrics['acc_class_2']:.3f}")
    if save:
        joblib.dump(model, ARTIFACTS / f"{name}.joblib")
    all_results.append(metrics)
    return model


# ══════════════════════════════════════════════════════════════════════
# PART 1: Base models (no class imbalance handling)
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PART 1: Base models (no class imbalance handling)")
print("=" * 60)

# 1. Logistic Regression
lr = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000, random_state=42))
])
train_and_log(lr, "LogisticRegression", X_train, y_train, X_val, y_val)

# 2. KNN (try k=5,10,20,50)
best_knn_f1 = -1
best_knn_k = 5
for k in [5, 10, 20, 50]:
    knn = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier(n_neighbors=k))
    ])
    m = train_and_log(knn, f"KNN_k{k}", X_train, y_train, X_val, y_val, save=True)
    f1_val = all_results[-1]["macro_f1"]
    if f1_val > best_knn_f1:
        best_knn_f1 = f1_val
        best_knn_k = k

print(f"\n  >> Best KNN: k={best_knn_k} (F1={best_knn_f1:.4f})")

# 3. Random Forest (needs imputed data)
rf = RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1)
train_and_log(rf, "RandomForest", X_train_rf, y_train, X_val_rf, y_val)

# 4. XGBoost (untuned)
xgb = XGBClassifier(
    objective="multi:softprob", num_class=3,
    random_state=42, n_jobs=-1, tree_method="hist"
)
train_and_log(xgb, "XGBoost_untuned", X_train, y_train, X_val, y_val)

# 5. HistGBM (untuned)
hgbm = HistGradientBoostingClassifier(random_state=42, max_iter=500)
train_and_log(hgbm, "HistGBM_untuned", X_train, y_train, X_val, y_val)

# 6. SVM RBF
svm = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("clf", SVC(kernel="rbf", probability=True, random_state=42))
])
train_and_log(svm, "SVM_RBF", X_train, y_train, X_val, y_val)


# ══════════════════════════════════════════════════════════════════════
# PART 2: Optuna hyperparameter tuning (50 trials each)
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PART 2: Optuna hyperparameter tuning")
print("=" * 60)

# XGBoost tuning
print("\n  Tuning XGBoost (50 trials)...")

def xgb_objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }
    model = XGBClassifier(
        objective="multi:softprob", num_class=3,
        random_state=42, n_jobs=-1, tree_method="hist", **params
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    return f1_score(y_val, y_pred, average="macro")

xgb_study = optuna.create_study(direction="maximize", study_name="xgboost")
xgb_study.optimize(xgb_objective, n_trials=50, show_progress_bar=True)
print(f"  Best XGBoost F1: {xgb_study.best_value:.4f}")
print(f"  Best params: {xgb_study.best_params}")

xgb_tuned = XGBClassifier(
    objective="multi:softprob", num_class=3,
    random_state=42, n_jobs=-1, tree_method="hist",
    **xgb_study.best_params
)
train_and_log(xgb_tuned, "XGBoost_tuned", X_train, y_train, X_val, y_val)

# HistGBM tuning
print("\n  Tuning HistGBM (50 trials)...")

def hgbm_objective(trial):
    params = {
        "max_iter": trial.suggest_int("max_iter", 100, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 127),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 50),
        "l2_regularization": trial.suggest_float("l2_regularization", 1e-8, 10.0, log=True),
        "max_bins": trial.suggest_int("max_bins", 64, 255),
    }
    model = HistGradientBoostingClassifier(random_state=42, **params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    return f1_score(y_val, y_pred, average="macro")

hgbm_study = optuna.create_study(direction="maximize", study_name="histgbm")
hgbm_study.optimize(hgbm_objective, n_trials=50, show_progress_bar=True)
print(f"  Best HistGBM F1: {hgbm_study.best_value:.4f}")
print(f"  Best params: {hgbm_study.best_params}")

hgbm_tuned = HistGradientBoostingClassifier(
    random_state=42, **hgbm_study.best_params
)
train_and_log(hgbm_tuned, "HistGBM_tuned", X_train, y_train, X_val, y_val)


# ══════════════════════════════════════════════════════════════════════
# PART 3: Class imbalance experiments
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PART 3: Class imbalance experiments")
print("=" * 60)

# (b) class_weight='balanced' for best XGB via sample_weight
print("\n  [b] XGBoost with balanced sample weights...")
class_counts = np.bincount(y_train)
class_weights = len(y_train) / (3 * class_counts)
sample_weights = np.array([class_weights[y] for y in y_train])

xgb_balanced = XGBClassifier(
    objective="multi:softprob", num_class=3,
    random_state=42, n_jobs=-1, tree_method="hist",
    **xgb_study.best_params
)
print(f"    Class weights: {dict(zip(range(3), class_weights.round(3)))}")
xgb_balanced.fit(X_train, y_train, sample_weight=sample_weights)
metrics_bal = evaluate_model(xgb_balanced, X_val, y_val, model_name="XGBoost_tuned_balanced")
print(f"    Accuracy: {metrics_bal['accuracy']:.4f}  |  Macro F1: {metrics_bal['macro_f1']:.4f}  |  "
      f"Log Loss: {metrics_bal['log_loss']:.4f}")
print(f"    Per-class acc: HW={metrics_bal['acc_class_0']:.3f}  D={metrics_bal['acc_class_1']:.3f}  AW={metrics_bal['acc_class_2']:.3f}")
joblib.dump(xgb_balanced, ARTIFACTS / "XGBoost_tuned_balanced.joblib")
all_results.append(metrics_bal)

# (c) SMOTE training set
print("\n  [c] XGBoost tuned on SMOTE training set...")
xgb_smote = XGBClassifier(
    objective="multi:softprob", num_class=3,
    random_state=42, n_jobs=-1, tree_method="hist",
    **xgb_study.best_params
)
train_and_log(xgb_smote, "XGBoost_tuned_SMOTE", X_train_smote, y_train_smote, X_val, y_val)

# HistGBM balanced
print("\n  [b] HistGBM with class_weight='balanced'...")
hgbm_balanced = HistGradientBoostingClassifier(
    random_state=42, class_weight="balanced", **hgbm_study.best_params
)
train_and_log(hgbm_balanced, "HistGBM_tuned_balanced", X_train, y_train, X_val, y_val)

# HistGBM SMOTE
print("\n  [c] HistGBM tuned on SMOTE training set...")
hgbm_smote = HistGradientBoostingClassifier(
    random_state=42, **hgbm_study.best_params
)
train_and_log(hgbm_smote, "HistGBM_tuned_SMOTE", X_train_smote, y_train_smote, X_val, y_val)


# ══════════════════════════════════════════════════════════════════════
# PART 4: Two-stage approach (win/not-win then draw/loss)
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PART 4: Two-stage approach")
print("=" * 60)
print("  Stage 1: Win (class 0) vs Not-Win (class 1+2)")
print("  Stage 2: Draw (class 1) vs Away Win (class 2)")

# Stage 1: binary win/not-win
y_train_s1 = (y_train != 0).astype(int)  # 0=Win, 1=Not-Win
y_val_s1 = (y_val != 0).astype(int)

stage1 = XGBClassifier(
    objective="binary:logistic",
    random_state=42, n_jobs=-1, tree_method="hist",
    n_estimators=xgb_study.best_params.get("n_estimators", 300),
    max_depth=xgb_study.best_params.get("max_depth", 6),
    learning_rate=xgb_study.best_params.get("learning_rate", 0.1),
)
stage1.fit(X_train, y_train_s1)
s1_proba = stage1.predict_proba(X_val)  # [P(Win), P(Not-Win)]
print(f"  Stage 1 acc: {(stage1.predict(X_val) == y_val_s1).mean():.4f}")

# Stage 2: draw/loss on not-win subset
not_win_mask_train = y_train != 0
not_win_mask_val = y_val != 0
y_train_s2 = y_train[not_win_mask_train] - 1  # 0=Draw, 1=Away Win
y_val_s2 = y_val[not_win_mask_val] - 1

stage2 = XGBClassifier(
    objective="binary:logistic",
    random_state=42, n_jobs=-1, tree_method="hist",
    n_estimators=xgb_study.best_params.get("n_estimators", 300),
    max_depth=xgb_study.best_params.get("max_depth", 6),
    learning_rate=xgb_study.best_params.get("learning_rate", 0.1),
)
stage2.fit(X_train[not_win_mask_train], y_train_s2)
print(f"  Stage 2 acc (not-win subset): {(stage2.predict(X_val[not_win_mask_val]) == y_val_s2).mean():.4f}")

# Combine: P(HW) = P(Win), P(D) = P(NotWin)*P(D|NotWin), P(AW) = P(NotWin)*P(AW|NotWin)
p_win = s1_proba[:, 0]
p_not_win = s1_proba[:, 1]
s2_all = stage2.predict_proba(X_val)
p_draw = p_not_win * s2_all[:, 0]
p_away = p_not_win * s2_all[:, 1]

two_stage_proba = np.column_stack([p_win, p_draw, p_away])
two_stage_pred = np.argmax(two_stage_proba, axis=1)

ts_metrics = {
    "model": "TwoStage_XGBoost",
    "accuracy": accuracy_score(y_val, two_stage_pred),
    "macro_f1": f1_score(y_val, two_stage_pred, average="macro"),
    "log_loss": log_loss(y_val, two_stage_proba),
    "auc_roc_ovr": roc_auc_score(y_val, two_stage_proba, multi_class="ovr", average="macro"),
}
for cls in [0, 1, 2]:
    mask = y_val == cls
    ts_metrics[f"acc_class_{cls}"] = accuracy_score(y_val[mask], two_stage_pred[mask])

print(f"\n  Two-Stage Combined:")
print(f"    Accuracy: {ts_metrics['accuracy']:.4f}  |  Macro F1: {ts_metrics['macro_f1']:.4f}  |  "
      f"Log Loss: {ts_metrics['log_loss']:.4f}")
print(f"    Per-class acc: HW={ts_metrics['acc_class_0']:.3f}  D={ts_metrics['acc_class_1']:.3f}  AW={ts_metrics['acc_class_2']:.3f}")
all_results.append(ts_metrics)

joblib.dump({"stage1": stage1, "stage2": stage2}, ARTIFACTS / "TwoStage_XGBoost.joblib")


# ══════════════════════════════════════════════════════════════════════
# PART 5: Stacking Ensemble
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PART 5: Stacking Ensemble (top base models)")
print("=" * 60)

print("  Building stacking with: XGBoost_tuned, HistGBM_tuned, RandomForest, LogisticRegression")

stack = StackingClassifier(
    estimators=[
        ("xgb", XGBClassifier(
            objective="multi:softprob", num_class=3,
            random_state=42, n_jobs=-1, tree_method="hist",
            **xgb_study.best_params
        )),
        ("hgbm", HistGradientBoostingClassifier(
            random_state=42, **hgbm_study.best_params
        )),
        ("rf", RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1)),
        ("lr", Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42))
        ])),
    ],
    final_estimator=LogisticRegression(max_iter=1000, random_state=42),
    cv=5,
    stack_method="predict_proba",
    n_jobs=-1,
    passthrough=False
)
train_and_log(stack, "StackingEnsemble", X_train_med, y_train, X_val_med, y_val)


# ══════════════════════════════════════════════════════════════════════
# RESULTS SUMMARY
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)

results_df = pd.DataFrame(all_results)
results_df = results_df.sort_values("macro_f1", ascending=False).reset_index(drop=True)

pd.set_option("display.max_columns", 20)
pd.set_option("display.width", 200)
print("\n", results_df.to_string(index=False, float_format="{:.4f}".format))

results_df.to_csv(ARTIFACTS / "results_summary.csv", index=False)
print(f"\nResults saved to {ARTIFACTS / 'results_summary.csv'}")

optuna_results = {
    "xgboost_best_params": xgb_study.best_params,
    "xgboost_best_f1": xgb_study.best_value,
    "histgbm_best_params": hgbm_study.best_params,
    "histgbm_best_f1": hgbm_study.best_value,
}
joblib.dump(optuna_results, ARTIFACTS / "optuna_results.joblib")
print("Optuna results saved.")

print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)
