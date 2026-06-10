#!/usr/bin/env python3
"""10_kickcast_v3.py -- Train KickCastNet v3 (dual-path + threshold tuning)."""

import sys
from pathlib import Path
BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

import numpy as np
import pandas as pd
import joblib
import optuna
import torch
from sklearn.metrics import f1_score, log_loss, accuracy_score, roc_auc_score, classification_report
optuna.logging.set_verbosity(optuna.logging.WARNING)

from src.kickcast_net_v3 import KickCastTrainerV3

SPLITS = BASE / "data" / "processed" / "splits"
ARTIFACTS = BASE / "outputs" / "model_artifacts"

print("=" * 60)
print("10_kickcast_v3.py -- KickCastNet v3 (Dual-Path + Thresholds)")
print("=" * 60)

X_train = pd.read_csv(SPLITS / "X_train.csv")
y_train = pd.read_csv(SPLITS / "y_train.csv")["result"].values
X_val = pd.read_csv(SPLITS / "X_val.csv")
y_val = pd.read_csv(SPLITS / "y_val.csv")["result"].values
X_test = pd.read_csv(SPLITS / "X_test.csv")
y_test = pd.read_csv(SPLITS / "y_test.csv")["result"].values
feature_names = X_train.columns.tolist()
n_features = len(feature_names)
print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# ── PART 1: Optuna Search (30 trials) ───────────────────────────────
print("\n" + "=" * 60)
print("PART 1: Optuna Search (30 trials)")
print("=" * 60, flush=True)

def objective(trial):
    d_token = trial.suggest_categorical("d_token", [48, 64])
    n_heads = trial.suggest_categorical("n_heads", [2, 4])
    hidden_dim = trial.suggest_categorical("hidden_dim", [96, 128, 192])
    n_blocks = trial.suggest_int("n_blocks", 2, 4)
    dropout = trial.suggest_float("dropout", 0.2, 0.4)
    noise_std = trial.suggest_float("noise_std", 0.03, 0.08)
    attention_dropout = trial.suggest_float("attention_dropout", 0.3, 0.5)
    lr = trial.suggest_float("lr", 1e-3, 5e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 5e-4, 5e-3, log=True)
    focal_gamma_draw = trial.suggest_float("focal_gamma_draw", 1.5, 3.0)
    focal_alpha_draw = trial.suggest_float("focal_alpha_draw", 1.5, 3.0)
    mixup_alpha = trial.suggest_float("mixup_alpha", 0.3, 0.6)
    epochs_per_cycle = trial.suggest_int("epochs_per_cycle", 20, 35)
    if d_token % n_heads != 0: return 0.0

    import io, contextlib
    trainer = KickCastTrainerV3(
        n_features=n_features, feature_names=feature_names,
        d_token=d_token, n_heads=n_heads, hidden_dim=hidden_dim,
        n_blocks=n_blocks, dropout=dropout, noise_std=noise_std,
        attention_dropout=attention_dropout, lr=lr, weight_decay=weight_decay,
        focal_gamma_draw=focal_gamma_draw, focal_alpha_draw=focal_alpha_draw,
        mixup_alpha=mixup_alpha, n_cycles=3, epochs_per_cycle=epochs_per_cycle, patience=5,
    )
    with contextlib.redirect_stdout(io.StringIO()):
        trainer.fit(X_train, y_train, X_val, y_val, verbose=False)

    # Use threshold-tuned F1 as objective (since v3 has built-in thresholds)
    val_pred = trainer.predict(X_val)
    val_f1 = f1_score(y_val, val_pred, average="macro")

    # Overfit penalty
    train_pred = trainer.predict(X_train)
    train_f1 = f1_score(y_train, train_pred, average="macro")
    gap = train_f1 - val_f1
    if gap > 0.08: val_f1 -= (gap - 0.08) * 0.5
    return val_f1

study = optuna.create_study(direction="maximize", study_name="kickcast_v3")
study.optimize(objective, n_trials=30, show_progress_bar=True)
print(f"\nBest trial F1: {study.best_value:.4f}", flush=True)
print(f"Best params: {study.best_params}", flush=True)

# ── PART 2: Final Training (5 cycles) ───────────────────────────────
print("\n" + "=" * 60)
print("PART 2: Train Final KickCastNet v3 (5 snapshots)")
print("=" * 60, flush=True)

best = study.best_params
final = KickCastTrainerV3(
    n_features=n_features, feature_names=feature_names,
    d_token=best["d_token"], n_heads=best["n_heads"],
    hidden_dim=best["hidden_dim"], n_blocks=best["n_blocks"],
    dropout=best["dropout"], noise_std=best["noise_std"],
    attention_dropout=best["attention_dropout"],
    lr=best["lr"], weight_decay=best["weight_decay"],
    focal_gamma_draw=best["focal_gamma_draw"],
    focal_alpha_draw=best["focal_alpha_draw"],
    mixup_alpha=best["mixup_alpha"],
    n_cycles=5, epochs_per_cycle=best["epochs_per_cycle"], patience=6,
)
final.fit(X_train, y_train, X_val, y_val)

# ── PART 3: Evaluation ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("PART 3: Evaluation")
print("=" * 60, flush=True)

for label, X, y in [("Val", X_val, y_val), ("Test", X_test, y_test)]:
    pred = final.predict(X)
    proba = final.predict_proba(X)
    f1 = f1_score(y, pred, average="macro")
    ll = log_loss(y, proba)
    acc = accuracy_score(y, pred)
    auc = roc_auc_score(y, proba, multi_class="ovr")
    per_class = [accuracy_score(y[y==c], pred[y==c]) for c in range(3)]
    print(f"\n  {label}: Acc={acc:.4f}  F1={f1:.4f}  LL={ll:.4f}  AUC={auc:.4f}")
    print(f"    HW={per_class[0]:.4f}  Draw={per_class[1]:.4f}  AW={per_class[2]:.4f}")
    if label == "Test":
        print(classification_report(y, pred, target_names=["Home Win", "Draw", "Away Win"]), flush=True)

# Overfit check
tr_pred = final.predict(X_train)
tr_f1 = f1_score(y_train, tr_pred, average="macro")
te_f1 = f1_score(y_test, final.predict(X_test), average="macro")
print(f"\n  OVERFIT: train_F1={tr_f1:.4f} test_F1={te_f1:.4f} gap={tr_f1-te_f1:+.4f}", flush=True)

# ── PART 4: Compare ─────────────────────────────────────────────────
print("\n" + "=" * 60)
print("PART 4: Head-to-Head")
print("=" * 60, flush=True)

v3_val_proba = final.predict_proba(X_val)
v3_test_proba = final.predict_proba(X_test)
v3_val_f1 = f1_score(y_val, final.predict(X_val), average="macro")
v3_test_f1 = f1_score(y_test, final.predict(X_test), average="macro")

print(f"\n  {'Model':30s} {'Val F1':>8s} {'Val LL':>8s} {'Test F1':>8s} {'Test LL':>8s}")
print("  " + "-" * 60)
print(f"  {'>>> KickCastNet v3':30s} {v3_val_f1:8.4f} {log_loss(y_val, v3_val_proba):8.4f} "
      f"{v3_test_f1:8.4f} {log_loss(y_test, v3_test_proba):8.4f}")

for name in ["KickCastNet_v2", "LightGBM_tuned_balanced", "CatBoost_tuned_balanced", "CalibratedEnsemble"]:
    try:
        m = joblib.load(ARTIFACTS / f"{name}.joblib")
        vp, tp = m.predict_proba(X_val), m.predict_proba(X_test)
        vf = f1_score(y_val, m.predict(X_val), average="macro")
        tf = f1_score(y_test, m.predict(X_test), average="macro")
        print(f"  {name:30s} {vf:8.4f} {log_loss(y_val, vp):8.4f} {tf:8.4f} {log_loss(y_test, tp):8.4f}")
    except Exception as e:
        print(f"  {name:30s} ERROR: {e}")

# ── PART 5: v3 Ensemble ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("PART 5: v3 Ensemble")
print("=" * 60, flush=True)

from scipy.optimize import minimize

ens_models = {"KickCastNet v3": v3_val_proba}
ens_test = {"KickCastNet v3": v3_test_proba}
for name in ["LightGBM_tuned_balanced", "CatBoost_tuned_balanced"]:
    try:
        m = joblib.load(ARTIFACTS / f"{name}.joblib")
        ens_models[name] = m.predict_proba(X_val)
        ens_test[name] = m.predict_proba(X_test)
    except: pass

keys = list(ens_models.keys())
val_list = [ens_models[k] for k in keys]
test_list = [ens_test[k] for k in keys]

def ens_ll(w, preds, y):
    w = np.abs(w); w = w / w.sum()
    avg = np.clip(sum(wi*p for wi, p in zip(w, preds)), 1e-6, 1.0)
    return log_loss(y, avg / avg.sum(axis=1, keepdims=True))

res = minimize(ens_ll, np.ones(len(keys))/len(keys), args=(val_list, y_val), method="Nelder-Mead")
w = np.abs(res.x); w = w / w.sum()
print(f"  Weights: {dict(zip(keys, w.round(3)))}")

ens_v = np.clip(sum(wi*p for wi, p in zip(w, val_list)), 1e-6, 1.0)
ens_v = ens_v / ens_v.sum(axis=1, keepdims=True)
ens_t = np.clip(sum(wi*p for wi, p in zip(w, test_list)), 1e-6, 1.0)
ens_t = ens_t / ens_t.sum(axis=1, keepdims=True)

# Apply threshold tuning to ensemble too
from src.kickcast_net_v3 import find_thresholds
ens_thresh = find_thresholds(ens_v, y_val)
ens_val_pred = np.argmax(ens_v / ens_thresh, axis=1)
ens_test_pred = np.argmax(ens_t / ens_thresh, axis=1)

ens_val_f1 = f1_score(y_val, ens_val_pred, average="macro")
ens_test_f1 = f1_score(y_test, ens_test_pred, average="macro")
print(f"  Ensemble Val F1:  {ens_val_f1:.4f}  LL: {log_loss(y_val, ens_v):.4f}")
print(f"  Ensemble Test F1: {ens_test_f1:.4f}  LL: {log_loss(y_test, ens_t):.4f}")
print(f"  Thresholds: {ens_thresh.round(3)}", flush=True)

# ── SAVE ─────────────────────────────────────────────────────────────
joblib.dump(final, ARTIFACTS / "KickCastNet_v3.joblib")
joblib.dump({
    "best_params": best, "best_optuna_f1": study.best_value,
    "temperature": final.temperature, "thresholds": final.thresholds.tolist(),
    "architecture": "DualPath(Attention+Bypass) + AttentionPool + FeatureGate + Thresholds",
    "ensemble_weights": dict(zip(keys, w.tolist())),
    "ensemble_thresholds": ens_thresh.tolist(),
}, ARTIFACTS / "kickcast_net_v3_config.joblib")
print(f"\n  Saved KickCastNet_v3.joblib", flush=True)

print("\n" + "=" * 60)
print("DONE")
print("=" * 60, flush=True)
