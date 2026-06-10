#!/usr/bin/env python3
"""
06_enhanced_models.py — CatBoost + LightGBM + Calibrated Ensemble.

Adds two new gradient boosting models (CatBoost, LightGBM), tunes them
with Optuna, applies isotonic probability calibration to the top models,
and builds a calibrated weighted ensemble — the best possible model for
Monte Carlo simulation where log loss matters more than accuracy.
"""

import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

import numpy as np
import pandas as pd
import joblib
import optuna
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (accuracy_score, f1_score, log_loss,
                              roc_auc_score, brier_score_loss)
from sklearn.model_selection import StratifiedKFold

from src.models import evaluate_model

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Paths ──────────────────────────────────────────────────────────────
SPLITS = BASE / "data" / "processed" / "splits"
ARTIFACTS = BASE / "outputs" / "model_artifacts"
ARTIFACTS.mkdir(parents=True, exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────
print("=" * 60)
print("06_enhanced_models.py — CatBoost + LightGBM + Calibrated Ensemble")
print("=" * 60)

X_train = pd.read_csv(SPLITS / "X_train.csv")
y_train = pd.read_csv(SPLITS / "y_train.csv")["result"].values
X_val = pd.read_csv(SPLITS / "X_val.csv")
y_val = pd.read_csv(SPLITS / "y_val.csv")["result"].values
X_test = pd.read_csv(SPLITS / "X_test.csv")
y_test = pd.read_csv(SPLITS / "y_test.csv")["result"].values

feature_names = X_train.columns.tolist()
print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
print(f"Features: {len(feature_names)}")

# Class weights for balanced training
class_counts = np.bincount(y_train)
class_weights_arr = len(y_train) / (3 * class_counts)
sample_weights = np.array([class_weights_arr[y] for y in y_train])
class_weight_dict = {0: class_weights_arr[0], 1: class_weights_arr[1], 2: class_weights_arr[2]}
print(f"Class weights: {dict(zip(['HW','D','AW'], class_weights_arr.round(3)))}")

all_results = []


def evaluate_and_log(model, name, X_v, y_v, save=True):
    """Evaluate a fitted model, print metrics, optionally save."""
    metrics = evaluate_model(model, X_v, y_v, model_name=name)
    print(f"  {name}:")
    print(f"    Acc={metrics['accuracy']:.4f}  F1={metrics['macro_f1']:.4f}  "
          f"LL={metrics['log_loss']:.4f}  AUC={metrics['auc_roc_ovr']:.4f}")
    print(f"    Per-class: HW={metrics['acc_class_0']:.3f}  "
          f"D={metrics['acc_class_1']:.3f}  AW={metrics['acc_class_2']:.3f}")
    if save:
        joblib.dump(model, ARTIFACTS / f"{name}.joblib")
    all_results.append(metrics)
    return metrics


# ══════════════════════════════════════════════════════════════════════
# PART 1: CatBoost with Optuna tuning
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PART 1: CatBoost (50 Optuna trials)")
print("=" * 60)


def catboost_objective(trial):
    params = {
        "iterations": trial.suggest_int("iterations", 200, 1500),
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
        "border_count": trial.suggest_int("border_count", 32, 255),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
        "random_strength": trial.suggest_float("random_strength", 1e-9, 10.0, log=True),
    }
    model = CatBoostClassifier(
        loss_function="MultiClass",
        random_seed=42,
        verbose=0,
        auto_class_weights="Balanced",
        **params
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val).astype(int).flatten()
    return f1_score(y_val, y_pred, average="macro")


print("  Tuning CatBoost...")
cb_study = optuna.create_study(direction="maximize", study_name="catboost")
cb_study.optimize(catboost_objective, n_trials=50, show_progress_bar=True)
print(f"  Best CatBoost F1: {cb_study.best_value:.4f}")

# Train final CatBoost with best params
cb_balanced = CatBoostClassifier(
    loss_function="MultiClass",
    random_seed=42,
    verbose=0,
    auto_class_weights="Balanced",
    **cb_study.best_params
)
cb_balanced.fit(X_train, y_train)
evaluate_and_log(cb_balanced, "CatBoost_tuned_balanced", X_val, y_val)

# Also train unbalanced version (for log loss comparison)
cb_unbalanced = CatBoostClassifier(
    loss_function="MultiClass",
    random_seed=42,
    verbose=0,
    **cb_study.best_params
)
cb_unbalanced.fit(X_train, y_train)
evaluate_and_log(cb_unbalanced, "CatBoost_tuned", X_val, y_val)


# ══════════════════════════════════════════════════════════════════════
# PART 2: LightGBM with Optuna tuning
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PART 2: LightGBM (50 Optuna trials)")
print("=" * 60)


def lgbm_objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1500),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }
    model = LGBMClassifier(
        objective="multiclass",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        verbose=-1,
        **params
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    return f1_score(y_val, y_pred, average="macro")


print("  Tuning LightGBM...")
lgbm_study = optuna.create_study(direction="maximize", study_name="lightgbm")
lgbm_study.optimize(lgbm_objective, n_trials=50, show_progress_bar=True)
print(f"  Best LightGBM F1: {lgbm_study.best_value:.4f}")

lgbm_balanced = LGBMClassifier(
    objective="multiclass",
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
    verbose=-1,
    **lgbm_study.best_params
)
lgbm_balanced.fit(X_train, y_train)
evaluate_and_log(lgbm_balanced, "LightGBM_tuned_balanced", X_val, y_val)

lgbm_unbalanced = LGBMClassifier(
    objective="multiclass",
    random_state=42,
    n_jobs=-1,
    verbose=-1,
    **lgbm_study.best_params
)
lgbm_unbalanced.fit(X_train, y_train)
evaluate_and_log(lgbm_unbalanced, "LightGBM_tuned", X_val, y_val)


# ══════════════════════════════════════════════════════════════════════
# PART 3: Probability calibration (isotonic regression)
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PART 3: Isotonic Probability Calibration")
print("=" * 60)

# Load existing best models
print("  Loading existing models for calibration...")
xgb_balanced = joblib.load(ARTIFACTS / "XGBoost_tuned_balanced.joblib")
hgbm_tuned = joblib.load(ARTIFACTS / "HistGBM_tuned.joblib")

# Models to calibrate: top performers by log loss + the new ones
models_to_calibrate = {
    "XGBoost_cal": xgb_balanced,
    "HistGBM_cal": hgbm_tuned,
    "CatBoost_cal": cb_unbalanced,
    "LightGBM_cal": lgbm_unbalanced,
}

calibrated_models = {}
for name, base_model in models_to_calibrate.items():
    print(f"\n  Calibrating {name}...")
    cal_model = CalibratedClassifierCV(
        base_model,
        method="isotonic",
        cv=5,
    )
    cal_model.fit(X_train, y_train)
    calibrated_models[name] = cal_model
    evaluate_and_log(cal_model, name, X_val, y_val)


# ══════════════════════════════════════════════════════════════════════
# PART 4: Calibrated Weighted Ensemble
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PART 4: Calibrated Weighted Ensemble")
print("=" * 60)

# Use the calibrated models + the balanced versions for diversity
ensemble_models = {
    "XGBoost_cal": calibrated_models["XGBoost_cal"],
    "HistGBM_cal": calibrated_models["HistGBM_cal"],
    "CatBoost_cal": calibrated_models["CatBoost_cal"],
    "LightGBM_cal": calibrated_models["LightGBM_cal"],
    "CatBoost_balanced": cb_balanced,
}

# Optimize weights on validation set
print("  Finding optimal ensemble weights (minimize log loss)...")
from scipy.optimize import minimize


def ensemble_log_loss(weights):
    """Log loss of weighted probability average."""
    w = np.array(weights)
    w = w / w.sum()  # normalize
    proba = np.zeros((len(X_val), 3))
    for (name, model), wi in zip(ensemble_models.items(), w):
        proba += wi * model.predict_proba(X_val)
    proba = np.clip(proba, 1e-15, 1.0)
    proba = proba / proba.sum(axis=1, keepdims=True)
    return log_loss(y_val, proba)


n_models = len(ensemble_models)
init_weights = np.ones(n_models) / n_models
bounds = [(0.01, 1.0)] * n_models

result = minimize(ensemble_log_loss, init_weights, method="L-BFGS-B", bounds=bounds)
optimal_weights = result.x / result.x.sum()

print("  Optimal weights:")
for (name, _), w in zip(ensemble_models.items(), optimal_weights):
    print(f"    {name:25s}: {w:.4f}")


from src.models import CalibratedEnsemble

ensemble = CalibratedEnsemble(ensemble_models, optimal_weights)
evaluate_and_log(ensemble, "CalibratedEnsemble", X_val, y_val)


# ══════════════════════════════════════════════════════════════════════
# PART 5: Final comparison on TEST SET
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PART 5: Test Set Evaluation (held-out 2022 WC + beyond)")
print("=" * 60)

test_models = {
    "XGBoost_tuned_balanced": xgb_balanced,
    "CatBoost_tuned_balanced": cb_balanced,
    "LightGBM_tuned_balanced": lgbm_balanced,
    "HistGBM_tuned": hgbm_tuned,
    "XGBoost_cal": calibrated_models["XGBoost_cal"],
    "CatBoost_cal": calibrated_models["CatBoost_cal"],
    "LightGBM_cal": calibrated_models["LightGBM_cal"],
    "HistGBM_cal": calibrated_models["HistGBM_cal"],
    "CalibratedEnsemble": ensemble,
}

test_results = []
print()
for name, model in test_models.items():
    metrics = evaluate_model(model, X_test, y_test, model_name=name)
    test_results.append(metrics)
    print(f"  {name:30s}  Acc={metrics['accuracy']:.4f}  F1={metrics['macro_f1']:.4f}  "
          f"LL={metrics['log_loss']:.4f}  AUC={metrics['auc_roc_ovr']:.4f}")

test_df = pd.DataFrame(test_results).sort_values("log_loss")
test_df.to_csv(ARTIFACTS / "enhanced_test_results.csv", index=False)


# ══════════════════════════════════════════════════════════════════════
# SAVE BEST MODEL FOR SIMULATION
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SAVING BEST MODEL")
print("=" * 60)

# The ensemble is best for log loss (what Monte Carlo needs)
joblib.dump(ensemble, ARTIFACTS / "CalibratedEnsemble.joblib")
print(f"  Saved: CalibratedEnsemble.joblib")

# Also save individual calibrated models
for name, model in calibrated_models.items():
    joblib.dump(model, ARTIFACTS / f"{name}.joblib")
    print(f"  Saved: {name}.joblib")

# Save Optuna results
enhanced_optuna = {
    "catboost_best_params": cb_study.best_params,
    "catboost_best_f1": cb_study.best_value,
    "lightgbm_best_params": lgbm_study.best_params,
    "lightgbm_best_f1": lgbm_study.best_value,
    "ensemble_weights": dict(zip(ensemble_models.keys(), optimal_weights.tolist())),
}
joblib.dump(enhanced_optuna, ARTIFACTS / "enhanced_optuna_results.joblib")

# Full results summary
all_df = pd.DataFrame(all_results).sort_values("macro_f1", ascending=False)
all_df.to_csv(ARTIFACTS / "enhanced_results_summary.csv", index=False)
print(f"\n  Results: {ARTIFACTS / 'enhanced_results_summary.csv'}")

print("\n" + "=" * 60)
print("ENHANCED MODEL TRAINING COMPLETE")
print("=" * 60)
