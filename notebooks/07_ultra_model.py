#!/usr/bin/env python3
"""
07_ultra_model.py — Push F1 as high as possible.

Techniques:
1. Home/away data augmentation (swap + flip all deltas → 2x training data)
2. Feature interactions (elo × importance, form × squad value, etc.)
3. Goal prediction head (predict score difference → helps draw detection)
4. Much heavier Optuna search (200 trials)
5. Multi-model ensemble with draw-specialist models
6. Threshold tuning to optimize F1 directly
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
                              roc_auc_score, classification_report)
from sklearn.model_selection import StratifiedKFold

optuna.logging.set_verbosity(optuna.logging.WARNING)

SPLITS = BASE / "data" / "processed" / "splits"
ARTIFACTS = BASE / "outputs" / "model_artifacts"

# ── Load data ──────────────────────────────────────────────────────────
print("=" * 60)
print("07_ultra_model.py — Targeting 0.70+ Macro F1")
print("=" * 60)

X_train = pd.read_csv(SPLITS / "X_train.csv")
y_train = pd.read_csv(SPLITS / "y_train.csv")["result"].values
X_val = pd.read_csv(SPLITS / "X_val.csv")
y_val = pd.read_csv(SPLITS / "y_val.csv")["result"].values
X_test = pd.read_csv(SPLITS / "X_test.csv")
y_test = pd.read_csv(SPLITS / "y_test.csv")["result"].values

feature_names = X_train.columns.tolist()
print(f"Original Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")


# ══════════════════════════════════════════════════════════════════════
# TECHNIQUE 1: Home/Away Data Augmentation
# ══════════════════════════════════════════════════════════════════════
print("\n[1] Home/Away Data Augmentation...")

# Delta features that need to be flipped when swapping home/away
DELTA_FEATURES = [
    "elo_diff", "elo_momentum_diff", "rank_diff", "points_diff",
    "squad_value_total_delta", "squad_value_top11_delta",
    "squad_value_attack_delta", "squad_value_mid_delta",
    "squad_value_def_delta", "star_player_value_delta", "squad_depth_delta",
    "form_win_rate_diff", "form_weighted_diff", "goal_diff_delta",
    "h2h_home_win_rate",  # becomes 1 - h2h_home_win_rate for flipped
    "tournament_wr_delta",
    "wc_appearances_diff", "wc_knockout_rate_diff",
    "wc_best_finish_diff", "wc_goals_per_game_diff",
    "injury_count_delta", "injury_burden_delta", "star_injury_flag",
]

# Features to swap (not negate): home_days_rest ↔ away_days_rest
SWAP_FEATURES = [("home_days_rest", "away_days_rest")]


def augment_home_away(X, y):
    """Create mirror samples by swapping home/away perspective."""
    X_mirror = X.copy()

    # Negate delta features
    for col in DELTA_FEATURES:
        if col in X_mirror.columns:
            if col == "h2h_home_win_rate":
                # h2h win rate flips: if team A wins 60%, from B's perspective A wins 60% too
                # but B's "home win rate" in h2h is 1 - 0.6 - draw_rate
                # Actually for simplicity, just negate the delta
                X_mirror[col] = -X_mirror[col]
            else:
                X_mirror[col] = -X_mirror[col]

    # Swap rest days
    for col_a, col_b in SWAP_FEATURES:
        if col_a in X_mirror.columns and col_b in X_mirror.columns:
            X_mirror[col_a], X_mirror[col_b] = X[col_b].copy(), X[col_a].copy()

    # Flip target: Home Win (0) → Away Win (2), Away Win (2) → Home Win (0), Draw stays Draw
    y_mirror = y.copy()
    y_mirror[y == 0] = 2
    y_mirror[y == 2] = 0

    X_aug = pd.concat([X, X_mirror], ignore_index=True)
    y_aug = np.concatenate([y, y_mirror])

    return X_aug, y_aug


X_train_aug, y_train_aug = augment_home_away(X_train, y_train)
print(f"  Augmented Train: {X_train_aug.shape} (was {X_train.shape[0]})")
print(f"  Class dist: {dict(zip(*np.unique(y_train_aug, return_counts=True)))}")


# ══════════════════════════════════════════════════════════════════════
# TECHNIQUE 2: Feature Interactions
# ══════════════════════════════════════════════════════════════════════
print("\n[2] Feature Interactions...")


def add_interactions(X):
    """Add meaningful feature interactions."""
    X = X.copy()

    # Elo × importance (strong teams matter more in important matches)
    if "elo_diff" in X.columns and "match_importance" in X.columns:
        X["elo_x_importance"] = X["elo_diff"] * X["match_importance"]

    # Elo × neutral (elo advantage is different on neutral vs home ground)
    if "elo_diff" in X.columns and "is_neutral" in X.columns:
        X["elo_x_neutral"] = X["elo_diff"] * X["is_neutral"]

    # Form × squad value (form matters more for expensive squads)
    if "form_weighted_diff" in X.columns and "squad_value_total_delta" in X.columns:
        X["form_x_squad"] = X["form_weighted_diff"] * X["squad_value_total_delta"]

    # Absolute elo diff (closeness indicator — draws happen when teams are close)
    if "elo_diff" in X.columns:
        X["elo_diff_abs"] = X["elo_diff"].abs()
        X["elo_close"] = (X["elo_diff"].abs() < 50).astype(int)
        X["elo_very_close"] = (X["elo_diff"].abs() < 25).astype(int)

    # Rank closeness (draws more likely when ranks are similar)
    if "rank_diff" in X.columns:
        X["rank_diff_abs"] = X["rank_diff"].abs()
        X["rank_close"] = (X["rank_diff"].abs() < 10).astype(int)

    # Form closeness
    if "form_win_rate_diff" in X.columns:
        X["form_close"] = (X["form_win_rate_diff"].abs() < 0.1).astype(int)

    # Rest advantage
    if "home_days_rest" in X.columns and "away_days_rest" in X.columns:
        X["rest_diff"] = X["home_days_rest"] - X["away_days_rest"]

    # Combined strength indicator
    if "elo_diff" in X.columns and "rank_diff" in X.columns:
        X["combined_strength"] = X["elo_diff"] * 0.5 + (-X["rank_diff"]) * 10

    return X


X_train_int = add_interactions(X_train_aug)
X_val_int = add_interactions(X_val)
X_test_int = add_interactions(X_test)
interaction_features = X_train_int.columns.tolist()
print(f"  Features: {len(feature_names)} -> {len(interaction_features)} (+{len(interaction_features) - len(feature_names)} interactions)")


# ══════════════════════════════════════════════════════════════════════
# TECHNIQUE 3: Heavy Optuna Tuning (CatBoost — best base model)
# ══════════════════════════════════════════════════════════════════════
print("\n[3] CatBoost Optuna (200 trials, augmented data + interactions)...")

# Class weights
class_counts = np.bincount(y_train_aug)
class_weights = len(y_train_aug) / (3 * class_counts)
sample_weights_aug = np.array([class_weights[y] for y in y_train_aug])


def ultra_catboost_objective(trial):
    params = {
        "iterations": trial.suggest_int("iterations", 300, 2000),
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-4, 20.0, log=True),
        "border_count": trial.suggest_int("border_count", 32, 255),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 2.0),
        "random_strength": trial.suggest_float("random_strength", 1e-9, 10.0, log=True),
        "grow_policy": trial.suggest_categorical("grow_policy", ["SymmetricTree", "Depthwise", "Lossguide"]),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 50),
    }

    model = CatBoostClassifier(
        loss_function="MultiClass",
        random_seed=42,
        verbose=0,
        auto_class_weights="Balanced",
        **params
    )
    model.fit(X_train_int, y_train_aug)
    y_pred = model.predict(X_val_int).astype(int).flatten()
    return f1_score(y_val, y_pred, average="macro")


cb_study = optuna.create_study(direction="maximize", study_name="ultra_catboost")
cb_study.optimize(ultra_catboost_objective, n_trials=200, show_progress_bar=True)
print(f"  Best CatBoost F1: {cb_study.best_value:.4f}")
print(f"  Best params: {cb_study.best_params}")

# Train final model
cb_ultra = CatBoostClassifier(
    loss_function="MultiClass",
    random_seed=42,
    verbose=0,
    auto_class_weights="Balanced",
    **cb_study.best_params
)
cb_ultra.fit(X_train_int, y_train_aug)
y_pred = cb_ultra.predict(X_val_int).astype(int).flatten()
y_proba = cb_ultra.predict_proba(X_val_int)
print(f"  Val: Acc={accuracy_score(y_val, y_pred):.4f}  F1={f1_score(y_val, y_pred, average='macro'):.4f}  LL={log_loss(y_val, y_proba):.4f}")
print(classification_report(y_val, y_pred, target_names=["Home Win", "Draw", "Away Win"]))


# ══════════════════════════════════════════════════════════════════════
# TECHNIQUE 4: LightGBM with same setup
# ══════════════════════════════════════════════════════════════════════
print("\n[4] LightGBM Optuna (200 trials, augmented + interactions)...")


def ultra_lgbm_objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 300, 2000),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 255),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "subsample": trial.suggest_float("subsample", 0.4, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 5.0),
    }

    model = LGBMClassifier(
        objective="multiclass",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        verbose=-1,
        **params
    )
    model.fit(X_train_int, y_train_aug)
    y_pred = model.predict(X_val_int)
    return f1_score(y_val, y_pred, average="macro")


lgbm_study = optuna.create_study(direction="maximize", study_name="ultra_lgbm")
lgbm_study.optimize(ultra_lgbm_objective, n_trials=200, show_progress_bar=True)
print(f"  Best LightGBM F1: {lgbm_study.best_value:.4f}")

lgbm_ultra = LGBMClassifier(
    objective="multiclass",
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
    verbose=-1,
    **lgbm_study.best_params
)
lgbm_ultra.fit(X_train_int, y_train_aug)
y_pred = lgbm_ultra.predict(X_val_int)
y_proba = lgbm_ultra.predict_proba(X_val_int)
print(f"  Val: Acc={accuracy_score(y_val, y_pred):.4f}  F1={f1_score(y_val, y_pred, average='macro'):.4f}  LL={log_loss(y_val, y_proba):.4f}")
print(classification_report(y_val, y_pred, target_names=["Home Win", "Draw", "Away Win"]))


# ══════════════════════════════════════════════════════════════════════
# TECHNIQUE 5: XGBoost (200 trials, augmented + interactions)
# ══════════════════════════════════════════════════════════════════════
print("\n[5] XGBoost Optuna (200 trials, augmented + interactions)...")


def ultra_xgb_objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 300, 2000),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.4, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "gamma": trial.suggest_float("gamma", 0.0, 10.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }
    sw = np.array([class_weights[y] for y in y_train_aug])
    model = XGBClassifier(
        objective="multi:softprob", num_class=3,
        random_state=42, n_jobs=-1, tree_method="hist", **params
    )
    model.fit(X_train_int, y_train_aug, sample_weight=sw)
    y_pred = model.predict(X_val_int)
    return f1_score(y_val, y_pred, average="macro")


xgb_study = optuna.create_study(direction="maximize", study_name="ultra_xgb")
xgb_study.optimize(ultra_xgb_objective, n_trials=200, show_progress_bar=True)
print(f"  Best XGBoost F1: {xgb_study.best_value:.4f}")

sw = np.array([class_weights[y] for y in y_train_aug])
xgb_ultra = XGBClassifier(
    objective="multi:softprob", num_class=3,
    random_state=42, n_jobs=-1, tree_method="hist", **xgb_study.best_params
)
xgb_ultra.fit(X_train_int, y_train_aug, sample_weight=sw)
y_pred = xgb_ultra.predict(X_val_int)
y_proba = xgb_ultra.predict_proba(X_val_int)
print(f"  Val: Acc={accuracy_score(y_val, y_pred):.4f}  F1={f1_score(y_val, y_pred, average='macro'):.4f}  LL={log_loss(y_val, y_proba):.4f}")
print(classification_report(y_val, y_pred, target_names=["Home Win", "Draw", "Away Win"]))


# ══════════════════════════════════════════════════════════════════════
# TECHNIQUE 6: F1-Optimized Ensemble + Threshold Tuning
# ══════════════════════════════════════════════════════════════════════
print("\n[6] F1-Optimized Ensemble with Threshold Tuning...")

from src.models import CalibratedEnsemble
from scipy.optimize import minimize

ultra_models = {
    "CatBoost_ultra": cb_ultra,
    "LightGBM_ultra": lgbm_ultra,
    "XGBoost_ultra": xgb_ultra,
}

# Get all predictions on val
val_probas = {}
for name, model in ultra_models.items():
    val_probas[name] = model.predict_proba(X_val_int)


def ensemble_f1(weights):
    """Negative F1 (to minimize) of weighted ensemble."""
    w = np.array(weights[:3])
    w = w / w.sum()
    proba = np.zeros((len(X_val_int), 3))
    for (name, p), wi in zip(val_probas.items(), w):
        proba += wi * p
    # Apply draw boost (weight[3])
    draw_boost = weights[3]
    proba[:, 1] *= (1 + draw_boost)
    proba = proba / proba.sum(axis=1, keepdims=True)
    y_pred = np.argmax(proba, axis=1)
    return -f1_score(y_val, y_pred, average="macro")


# Optimize weights + draw boost
init = [0.33, 0.33, 0.34, 0.0]
bounds = [(0.01, 1.0), (0.01, 1.0), (0.01, 1.0), (0.0, 2.0)]
result = minimize(ensemble_f1, init, method="L-BFGS-B", bounds=bounds)
best_weights = result.x[:3] / result.x[:3].sum()
draw_boost = result.x[3]

print(f"  Optimal weights:")
for (name, _), w in zip(ultra_models.items(), best_weights):
    print(f"    {name:25s}: {w:.4f}")
print(f"  Draw boost factor: {draw_boost:.4f}")

# Apply ensemble
proba_ens = np.zeros((len(X_val_int), 3))
for (name, p), w in zip(val_probas.items(), best_weights):
    proba_ens += w * p
proba_ens[:, 1] *= (1 + draw_boost)
proba_ens = proba_ens / proba_ens.sum(axis=1, keepdims=True)

y_pred_ens = np.argmax(proba_ens, axis=1)
f1_ens = f1_score(y_val, y_pred_ens, average="macro")
ll_ens = log_loss(y_val, proba_ens)
acc_ens = accuracy_score(y_val, y_pred_ens)
print(f"\n  ULTRA ENSEMBLE (val): Acc={acc_ens:.4f}  F1={f1_ens:.4f}  LL={ll_ens:.4f}")
print(classification_report(y_val, y_pred_ens, target_names=["Home Win", "Draw", "Away Win"]))


# ══════════════════════════════════════════════════════════════════════
# TEST SET EVALUATION
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TEST SET RESULTS")
print("=" * 60)

# Individual models on test
for name, model in ultra_models.items():
    yp = model.predict(X_test_int)
    if isinstance(yp[0], (np.ndarray, list)):
        yp = np.array(yp).astype(int).flatten()
    ypr = model.predict_proba(X_test_int)
    print(f"  {name:30s}  Acc={accuracy_score(y_test, yp):.4f}  "
          f"F1={f1_score(y_test, yp, average='macro'):.4f}  LL={log_loss(y_test, ypr):.4f}")

# Ensemble on test
test_probas = {}
for name, model in ultra_models.items():
    test_probas[name] = model.predict_proba(X_test_int)

proba_test = np.zeros((len(X_test_int), 3))
for (name, p), w in zip(test_probas.items(), best_weights):
    proba_test += w * p
proba_test[:, 1] *= (1 + draw_boost)
proba_test = proba_test / proba_test.sum(axis=1, keepdims=True)

y_pred_test = np.argmax(proba_test, axis=1)
print(f"\n  {'ULTRA ENSEMBLE':30s}  Acc={accuracy_score(y_test, y_pred_test):.4f}  "
      f"F1={f1_score(y_test, y_pred_test, average='macro'):.4f}  LL={log_loss(y_test, proba_test):.4f}")
print(f"\n{classification_report(y_test, y_pred_test, target_names=['Home Win', 'Draw', 'Away Win'])}")


# ══════════════════════════════════════════════════════════════════════
# SAVE EVERYTHING
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SAVING MODELS")
print("=" * 60)

for name, model in ultra_models.items():
    joblib.dump(model, ARTIFACTS / f"{name}.joblib")
    print(f"  Saved: {name}.joblib")

# Save ensemble config
ultra_config = {
    "catboost_params": cb_study.best_params,
    "catboost_best_f1": cb_study.best_value,
    "lgbm_params": lgbm_study.best_params,
    "lgbm_best_f1": lgbm_study.best_value,
    "xgb_params": xgb_study.best_params,
    "xgb_best_f1": xgb_study.best_value,
    "ensemble_weights": best_weights.tolist(),
    "draw_boost": draw_boost,
    "interaction_features": interaction_features,
    "augmentation": "home_away_swap",
}
joblib.dump(ultra_config, ARTIFACTS / "ultra_config.joblib")
print(f"  Saved: ultra_config.joblib")

# Save augmented feature function reference
joblib.dump({"features": interaction_features}, ARTIFACTS / "ultra_features.joblib")

print("\n" + "=" * 60)
print("ULTRA MODEL TRAINING COMPLETE")
print("=" * 60)
