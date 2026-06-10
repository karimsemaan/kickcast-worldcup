#!/usr/bin/env python3
"""
08_kickcast_net.py -- Train the custom KickCastNet neural network.

Searches over architectures (hidden dim, depth, dropout, focal gamma, LR)
using Optuna, trains with snapshot ensembling, compares to all baselines.
"""

import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

import numpy as np
import pandas as pd
import joblib
import optuna
import torch
from sklearn.metrics import f1_score, log_loss, accuracy_score, classification_report

from src.kickcast_net import KickCastTrainer

optuna.logging.set_verbosity(optuna.logging.WARNING)

SPLITS = BASE / "data" / "processed" / "splits"
ARTIFACTS = BASE / "outputs" / "model_artifacts"

print("=" * 60)
print("08_kickcast_net.py -- Custom KickCastNet")
print("=" * 60)

X_train = pd.read_csv(SPLITS / "X_train.csv")
y_train = pd.read_csv(SPLITS / "y_train.csv")["result"].values
X_val = pd.read_csv(SPLITS / "X_val.csv")
y_val = pd.read_csv(SPLITS / "y_val.csv")["result"].values
X_test = pd.read_csv(SPLITS / "X_test.csv")
y_test = pd.read_csv(SPLITS / "y_test.csv")["result"].values

feature_names = X_train.columns.tolist()
n_features = len(feature_names)

class_counts = np.bincount(y_train)
class_weights = len(y_train) / (3 * class_counts)
print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
print(f"Class weights: {dict(zip(['HW','D','AW'], class_weights.round(3)))}")


# Part 1: Optuna hyperparameter search
print("\n" + "=" * 60)
print("PART 1: Optuna Architecture Search (100 trials)")
print("=" * 60)


def objective(trial):
    hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256, 384])
    n_blocks = trial.suggest_int("n_blocks", 2, 6)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    feature_dropout = trial.suggest_float("feature_dropout", 0.0, 0.2)
    lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
    focal_gamma = trial.suggest_float("focal_gamma", 0.5, 3.0)
    epochs_per_cycle = trial.suggest_int("epochs_per_cycle", 20, 60)

    trainer = KickCastTrainer(
        n_features=n_features,
        feature_names=feature_names,
        hidden_dim=hidden_dim,
        n_blocks=n_blocks,
        dropout=dropout,
        feature_dropout=feature_dropout,
        lr=lr,
        weight_decay=weight_decay,
        focal_gamma=focal_gamma,
        class_weights=class_weights.tolist(),
        n_cycles=3,
        epochs_per_cycle=epochs_per_cycle,
        patience=5,
    )

    # Suppress cycle prints during search
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        trainer.fit(X_train, y_train, X_val, y_val)

    y_pred = trainer.predict(X_val)
    val_f1 = f1_score(y_val, y_pred, average="macro")

    # Also check train F1 to detect overfitting
    y_train_pred = trainer.predict(X_train)
    train_f1 = f1_score(y_train, y_train_pred, average="macro")
    gap = train_f1 - val_f1

    # Penalize overfitting: if gap > 0.08, reduce score
    if gap > 0.08:
        val_f1 -= (gap - 0.08) * 0.5

    return val_f1


study = optuna.create_study(direction="maximize", study_name="kickcast_net")
study.optimize(objective, n_trials=100, show_progress_bar=True)

print(f"\nBest trial F1: {study.best_value:.4f}")
print(f"Best params: {study.best_params}")


# Part 2: Train final model with best params + full snapshot ensemble
print("\n" + "=" * 60)
print("PART 2: Train Final KickCastNet (8 snapshots)")
print("=" * 60)

best = study.best_params
final_trainer = KickCastTrainer(
    n_features=n_features,
    feature_names=feature_names,
    hidden_dim=best["hidden_dim"],
    n_blocks=best["n_blocks"],
    dropout=best["dropout"],
    feature_dropout=best["feature_dropout"],
    lr=best["lr"],
    weight_decay=best["weight_decay"],
    focal_gamma=best["focal_gamma"],
    class_weights=class_weights.tolist(),
    n_cycles=8,
    epochs_per_cycle=best["epochs_per_cycle"],
    patience=6,
)

final_trainer.fit(X_train, y_train, X_val, y_val)


# Part 3: Evaluate
print("\n" + "=" * 60)
print("PART 3: Evaluation")
print("=" * 60)

for label, X, y in [("Train", X_train, y_train), ("Val", X_val, y_val), ("Test", X_test, y_test)]:
    y_pred = final_trainer.predict(X)
    y_proba = final_trainer.predict_proba(X)
    f1 = f1_score(y, y_pred, average="macro")
    ll = log_loss(y, y_proba)
    acc = accuracy_score(y, y_pred)
    print(f"\n  {label}: Acc={acc:.4f}  F1={f1:.4f}  LL={ll:.4f}")
    if label == "Test":
        print(classification_report(y, y_pred, target_names=["Home Win", "Draw", "Away Win"]))

# Overfit check
train_f1 = f1_score(y_train, final_trainer.predict(X_train), average="macro")
test_f1 = f1_score(y_test, final_trainer.predict(X_test), average="macro")
train_ll = log_loss(y_train, final_trainer.predict_proba(X_train))
test_ll = log_loss(y_test, final_trainer.predict_proba(X_test))
print(f"\n  OVERFIT CHECK:")
print(f"    F1  gap: {train_f1 - test_f1:+.4f} (train={train_f1:.4f}, test={test_f1:.4f})")
print(f"    LL  gap: {test_ll - train_ll:+.4f} (train={train_ll:.4f}, test={test_ll:.4f})")


# Part 4: Compare to all baselines
print("\n" + "=" * 60)
print("PART 4: Head-to-Head Comparison")
print("=" * 60)

baselines = {
    "CatBoost_tuned_balanced": "CatBoost_tuned_balanced",
    "XGBoost_tuned_balanced": "XGBoost_tuned_balanced",
    "LightGBM_tuned_balanced": "LightGBM_tuned_balanced",
    "CalibratedEnsemble": "CalibratedEnsemble",
}

print(f"\n  {'Model':35s} {'Test F1':>8s} {'Test LL':>8s} {'Test Acc':>8s} {'Overfit':>8s}")
print("  " + "-" * 75)

# KickCastNet
gap = train_f1 - test_f1
print(f"  {'>>> KickCastNet (ours)':35s} {test_f1:8.4f} {test_ll:8.4f} "
      f"{accuracy_score(y_test, final_trainer.predict(X_test)):8.4f} {gap:+8.4f}")

for label, name in baselines.items():
    try:
        model = joblib.load(ARTIFACTS / f"{name}.joblib")
        yp = model.predict(X_test)
        ypr = model.predict_proba(X_test)
        f1 = f1_score(y_test, yp, average="macro")
        ll = log_loss(y_test, ypr)
        acc = accuracy_score(y_test, yp)
        train_pred = model.predict(X_train)
        train_f1_b = f1_score(y_train, train_pred, average="macro")
        gap_b = train_f1_b - f1
        print(f"  {label:35s} {f1:8.4f} {ll:8.4f} {acc:8.4f} {gap_b:+8.4f}")
    except Exception as e:
        print(f"  {label:35s} ERROR: {e}")


# Part 5: Save
print("\n" + "=" * 60)
print("SAVING")
print("=" * 60)

joblib.dump(final_trainer, ARTIFACTS / "KickCastNet.joblib")
print(f"  Saved: KickCastNet.joblib")

net_config = {
    "best_params": study.best_params,
    "best_optuna_f1": study.best_value,
    "n_snapshots": len(final_trainer.snapshots),
    "architecture": "ResidualMLP + FocalLoss + SnapshotEnsemble",
}
joblib.dump(net_config, ARTIFACTS / "kickcast_net_config.joblib")

print("\n" + "=" * 60)
print("KICKCASTNET TRAINING COMPLETE")
print("=" * 60)
