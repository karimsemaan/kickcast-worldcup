#!/usr/bin/env python3
"""
09_kickcast_v2.py -- Train KickCastNet v2, evaluate against all models,
build ensemble, and generate presentation-quality figures.

Run from repo root: python notebooks/09_kickcast_v2.py
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.metrics import (
    f1_score, log_loss, accuracy_score, classification_report,
    confusion_matrix, roc_auc_score,
)
from sklearn.calibration import calibration_curve
from scipy.optimize import minimize

from src.kickcast_net_v2 import KickCastTrainerV2

optuna.logging.set_verbosity(optuna.logging.WARNING)

SPLITS = BASE / "data" / "processed" / "splits"
ARTIFACTS = BASE / "outputs" / "model_artifacts"
FIGURES = BASE / "outputs" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

# ── Style ────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0e0e11",
    "axes.facecolor": "#141418",
    "axes.edgecolor": "#27272a",
    "axes.labelcolor": "#a1a1aa",
    "text.color": "#e4e4e7",
    "xtick.color": "#71717a",
    "ytick.color": "#71717a",
    "grid.color": "#1e1e22",
    "grid.alpha": 0.6,
    "font.family": "sans-serif",
    "font.size": 10,
    "figure.dpi": 150,
})
EMERALD = "#22c55e"
YELLOW = "#eab308"
INDIGO = "#6366f1"
RED = "#ef4444"
BLUE = "#3b82f6"
CYAN = "#06b6d4"
MUTED = "#71717a"

print("=" * 60)
print("09_kickcast_v2.py -- KickCastNet v2")
print("=" * 60)

# ── Load data ────────────────────────────────────────────────────────
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
print(f"Missing rate: {X_train.isnull().mean().mean():.1%}")


# ══════════════════════════════════════════════════════════════════════
# PART 1: Optuna Architecture Search (100 trials)
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PART 1: Optuna Architecture Search (100 trials)")
print("=" * 60)


def objective(trial):
    d_token = trial.suggest_categorical("d_token", [32, 48, 64])
    n_heads = trial.suggest_categorical("n_heads", [2, 4])
    hidden_dim = trial.suggest_categorical("hidden_dim", [96, 128, 192, 256])
    n_blocks = trial.suggest_int("n_blocks", 2, 4)
    dropout = trial.suggest_float("dropout", 0.15, 0.45)
    feature_dropout = trial.suggest_float("feature_dropout", 0.0, 0.15)
    noise_std = trial.suggest_float("noise_std", 0.01, 0.08)
    attention_dropout = trial.suggest_float("attention_dropout", 0.15, 0.45)
    lr = trial.suggest_float("lr", 5e-4, 5e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 5e-3, log=True)
    focal_gamma_draw = trial.suggest_float("focal_gamma_draw", 1.5, 3.5)
    focal_alpha_draw = trial.suggest_float("focal_alpha_draw", 1.2, 3.0)
    mixup_alpha = trial.suggest_float("mixup_alpha", 0.05, 0.5)
    label_smoothing = trial.suggest_float("label_smoothing", 0.0, 0.1)
    epochs_per_cycle = trial.suggest_int("epochs_per_cycle", 20, 40)

    # d_token must be divisible by n_heads
    if d_token % n_heads != 0:
        return 0.0

    trainer = KickCastTrainerV2(
        n_features=n_features,
        feature_names=feature_names,
        d_token=d_token,
        n_heads=n_heads,
        hidden_dim=hidden_dim,
        n_blocks=n_blocks,
        dropout=dropout,
        feature_dropout=feature_dropout,
        noise_std=noise_std,
        attention_dropout=attention_dropout,
        lr=lr,
        weight_decay=weight_decay,
        focal_gamma_draw=focal_gamma_draw,
        focal_alpha_draw=focal_alpha_draw,
        label_smoothing=label_smoothing,
        mixup_alpha=mixup_alpha,
        n_cycles=3,  # Fewer cycles during search
        epochs_per_cycle=epochs_per_cycle,
        patience=5,
    )

    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        trainer.fit(X_train, y_train, X_val, y_val, verbose=False)

    y_pred = trainer.predict(X_val)
    val_f1 = f1_score(y_val, y_pred, average="macro")

    # Overfit check
    y_train_pred = trainer.predict(X_train)
    train_f1 = f1_score(y_train, y_train_pred, average="macro")
    gap = train_f1 - val_f1
    if gap > 0.08:
        val_f1 -= (gap - 0.08) * 0.5

    return val_f1


study = optuna.create_study(direction="maximize", study_name="kickcast_net_v2")
study.optimize(objective, n_trials=30, show_progress_bar=True)

print(f"\nBest trial F1: {study.best_value:.4f}")
print(f"Best params: {study.best_params}")


# ══════════════════════════════════════════════════════════════════════
# PART 2: Train Final Model (8 snapshot cycles)
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PART 2: Train Final KickCastNet v2 (8 snapshots)")
print("=" * 60)

best = study.best_params
final_trainer = KickCastTrainerV2(
    n_features=n_features,
    feature_names=feature_names,
    d_token=best["d_token"],
    n_heads=best["n_heads"],
    hidden_dim=best["hidden_dim"],
    n_blocks=best["n_blocks"],
    dropout=best["dropout"],
    feature_dropout=best["feature_dropout"],
    noise_std=best["noise_std"],
    attention_dropout=best["attention_dropout"],
    lr=best["lr"],
    weight_decay=best["weight_decay"],
    focal_gamma_draw=best["focal_gamma_draw"],
    focal_alpha_draw=best["focal_alpha_draw"],
    label_smoothing=best["label_smoothing"],
    mixup_alpha=best["mixup_alpha"],
    n_cycles=8,
    epochs_per_cycle=best["epochs_per_cycle"],
    patience=6,
)

final_trainer.fit(X_train, y_train, X_val, y_val)


# ══════════════════════════════════════════════════════════════════════
# PART 3: Evaluation
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PART 3: Evaluation")
print("=" * 60)

results_v2 = {}
for label, X, y in [("Train", X_train, y_train), ("Val", X_val, y_val), ("Test", X_test, y_test)]:
    y_pred = final_trainer.predict(X)
    y_proba = final_trainer.predict_proba(X)
    f1 = f1_score(y, y_pred, average="macro")
    ll = log_loss(y, y_proba)
    acc = accuracy_score(y, y_pred)
    auc = roc_auc_score(y, y_proba, multi_class="ovr")
    per_class = [accuracy_score(y[y == c], y_pred[y == c]) for c in range(3)]

    results_v2[label] = {
        "accuracy": acc, "macro_f1": f1, "log_loss": ll, "auc_roc": auc,
        "acc_hw": per_class[0], "acc_draw": per_class[1], "acc_aw": per_class[2],
    }
    print(f"\n  {label}: Acc={acc:.4f}  F1={f1:.4f}  LL={ll:.4f}  AUC={auc:.4f}")
    print(f"    HW={per_class[0]:.4f}  Draw={per_class[1]:.4f}  AW={per_class[2]:.4f}")
    if label == "Test":
        print(classification_report(y, y_pred, target_names=["Home Win", "Draw", "Away Win"]))

# Overfit check
train_f1 = results_v2["Train"]["macro_f1"]
test_f1 = results_v2["Test"]["macro_f1"]
print(f"\n  OVERFIT CHECK:")
print(f"    F1  gap: {train_f1 - test_f1:+.4f} (train={train_f1:.4f}, test={test_f1:.4f})")
print(f"    LL  gap: {results_v2['Test']['log_loss'] - results_v2['Train']['log_loss']:+.4f}")


# ══════════════════════════════════════════════════════════════════════
# PART 4: Compare All Models
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PART 4: Head-to-Head Comparison")
print("=" * 60)

baselines = {
    "LightGBM_tuned_balanced": "LightGBM_tuned_balanced",
    "CatBoost_tuned_balanced": "CatBoost_tuned_balanced",
    "XGBoost_tuned_balanced": "XGBoost_tuned_balanced",
    "CalibratedEnsemble": "CalibratedEnsemble",
    "KickCastNet v1": "KickCastNet",
}

all_model_results = {}
all_val_probas = {}
all_test_probas = {}

# KickCastNet v2
v2_val_proba = final_trainer.predict_proba(X_val)
v2_test_proba = final_trainer.predict_proba(X_test)
v2_val_pred = final_trainer.predict(X_val)
v2_test_pred = final_trainer.predict(X_test)

all_model_results["KickCastNet v2"] = {
    "val_f1": results_v2["Val"]["macro_f1"],
    "val_ll": results_v2["Val"]["log_loss"],
    "val_acc": results_v2["Val"]["accuracy"],
    "val_auc": results_v2["Val"]["auc_roc"],
    "test_f1": results_v2["Test"]["macro_f1"],
    "test_ll": results_v2["Test"]["log_loss"],
    "test_acc": results_v2["Test"]["accuracy"],
    "test_auc": results_v2["Test"]["auc_roc"],
    "val_draw_acc": results_v2["Val"]["acc_draw"],
    "test_draw_acc": results_v2["Test"]["acc_draw"],
}
all_val_probas["KickCastNet v2"] = v2_val_proba
all_test_probas["KickCastNet v2"] = v2_test_proba

print(f"\n  {'Model':30s} {'Val F1':>8s} {'Val LL':>8s} {'Test F1':>8s} {'Test LL':>8s} {'Draw%':>7s}")
print("  " + "-" * 75)
print(f"  {'>>> KickCastNet v2':30s} "
      f"{results_v2['Val']['macro_f1']:8.4f} {results_v2['Val']['log_loss']:8.4f} "
      f"{results_v2['Test']['macro_f1']:8.4f} {results_v2['Test']['log_loss']:8.4f} "
      f"{results_v2['Val']['acc_draw']:6.1%}")

for label, name in baselines.items():
    try:
        model = joblib.load(ARTIFACTS / f"{name}.joblib")
        val_pred = model.predict(X_val)
        val_proba = model.predict_proba(X_val)
        test_pred = model.predict(X_test)
        test_proba = model.predict_proba(X_test)

        val_f1 = f1_score(y_val, val_pred, average="macro")
        val_ll = log_loss(y_val, val_proba)
        test_f1 = f1_score(y_test, test_pred, average="macro")
        test_ll = log_loss(y_test, test_proba)
        val_draw = accuracy_score(y_val[y_val == 1], val_pred[y_val == 1])

        all_model_results[label] = {
            "val_f1": val_f1, "val_ll": val_ll,
            "test_f1": test_f1, "test_ll": test_ll,
            "val_draw_acc": val_draw,
        }
        all_val_probas[label] = val_proba
        all_test_probas[label] = test_proba

        print(f"  {label:30s} {val_f1:8.4f} {val_ll:8.4f} "
              f"{test_f1:8.4f} {test_ll:8.4f} {val_draw:6.1%}")
    except Exception as e:
        print(f"  {label:30s} ERROR: {e}")


# ══════════════════════════════════════════════════════════════════════
# PART 5: Build Ensemble
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PART 5: KickCast Ensemble (Weighted Averaging)")
print("=" * 60)

# Pick models for ensemble
ensemble_models = ["KickCastNet v2"]
for name in ["LightGBM_tuned_balanced", "CatBoost_tuned_balanced"]:
    if name in all_val_probas:
        ensemble_models.append(name)

print(f"  Ensemble members: {ensemble_models}")

ensemble_val = [all_val_probas[m] for m in ensemble_models]
ensemble_test = [all_test_probas[m] for m in ensemble_models]


def ensemble_log_loss(weights, pred_list, y_true):
    w = np.array(weights)
    w = np.abs(w)
    w = w / w.sum()
    avg = sum(wi * p for wi, p in zip(w, pred_list))
    avg = np.clip(avg, 1e-6, 1.0)
    avg = avg / avg.sum(axis=1, keepdims=True)
    return log_loss(y_true, avg)


n_models = len(ensemble_models)
x0 = np.ones(n_models) / n_models

result = minimize(
    ensemble_log_loss, x0, args=(ensemble_val, y_val),
    method="Nelder-Mead",
    options={"maxiter": 5000, "xatol": 1e-6},
)

opt_weights = np.abs(result.x)
opt_weights = opt_weights / opt_weights.sum()

print(f"  Optimal weights:")
for m, w in zip(ensemble_models, opt_weights):
    print(f"    {m}: {w:.3f}")

# Ensemble predictions
ens_val_proba = sum(w * p for w, p in zip(opt_weights, ensemble_val))
ens_val_proba = np.clip(ens_val_proba, 1e-6, 1.0)
ens_val_proba = ens_val_proba / ens_val_proba.sum(axis=1, keepdims=True)
ens_val_pred = np.argmax(ens_val_proba, axis=1)

ens_test_proba = sum(w * p for w, p in zip(opt_weights, ensemble_test))
ens_test_proba = np.clip(ens_test_proba, 1e-6, 1.0)
ens_test_proba = ens_test_proba / ens_test_proba.sum(axis=1, keepdims=True)
ens_test_pred = np.argmax(ens_test_proba, axis=1)

ens_val_f1 = f1_score(y_val, ens_val_pred, average="macro")
ens_val_ll = log_loss(y_val, ens_val_proba)
ens_test_f1 = f1_score(y_test, ens_test_pred, average="macro")
ens_test_ll = log_loss(y_test, ens_test_proba)
ens_val_draw = accuracy_score(y_val[y_val == 1], ens_val_pred[y_val == 1])

print(f"\n  KickCast Ensemble Results:")
print(f"    Val:  F1={ens_val_f1:.4f}  LL={ens_val_ll:.4f}  DrawAcc={ens_val_draw:.1%}")
print(f"    Test: F1={ens_test_f1:.4f}  LL={ens_test_ll:.4f}")

all_model_results["KickCast Ensemble"] = {
    "val_f1": ens_val_f1, "val_ll": ens_val_ll,
    "test_f1": ens_test_f1, "test_ll": ens_test_ll,
    "val_draw_acc": ens_val_draw,
}


# ══════════════════════════════════════════════════════════════════════
# PART 6: Save Artifacts
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SAVING ARTIFACTS")
print("=" * 60)

joblib.dump(final_trainer, ARTIFACTS / "KickCastNet_v2.joblib")
joblib.dump({
    "best_params": best,
    "best_optuna_f1": study.best_value,
    "architecture": "FeatureTokenizer + SelfAttention + ResidualMLP + AdaptiveFocalLoss + Mixup",
    "temperature": final_trainer.temperature,
    "n_snapshots": len(final_trainer.snapshots),
    "ensemble_weights": dict(zip(ensemble_models, opt_weights.tolist())),
}, ARTIFACTS / "kickcast_net_v2_config.joblib")

print(f"  Saved KickCastNet_v2.joblib")
print(f"  Saved kickcast_net_v2_config.joblib")


# ══════════════════════════════════════════════════════════════════════
# PART 7: Presentation Figures
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PART 7: Generating Presentation Figures")
print("=" * 60)


# ── Figure 1: Model Comparison Bar Chart ─────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

models_to_plot = [k for k in all_model_results.keys() if k != "KickCast Ensemble"]
models_to_plot.append("KickCast Ensemble")

val_f1s = [all_model_results[m]["val_f1"] for m in models_to_plot]
val_lls = [all_model_results[m]["val_ll"] for m in models_to_plot]

colors = []
for m in models_to_plot:
    if "v2" in m:
        colors.append(EMERALD)
    elif "Ensemble" in m and "Calibrated" not in m:
        colors.append(CYAN)
    elif "v1" in m:
        colors.append(YELLOW)
    else:
        colors.append(MUTED)

# F1 chart
bars = axes[0].barh(range(len(models_to_plot)), val_f1s, color=colors, height=0.6)
axes[0].set_yticks(range(len(models_to_plot)))
axes[0].set_yticklabels(models_to_plot, fontsize=8)
axes[0].set_xlabel("Validation Macro F1")
axes[0].set_title("Macro F1 (higher is better)", fontsize=11, fontweight="bold")
axes[0].invert_yaxis()
for bar, val in zip(bars, val_f1s):
    axes[0].text(val + 0.002, bar.get_y() + bar.get_height()/2,
                 f"{val:.4f}", va="center", fontsize=7, color="#a1a1aa")

# Log Loss chart
bars = axes[1].barh(range(len(models_to_plot)), val_lls, color=colors, height=0.6)
axes[1].set_yticks(range(len(models_to_plot)))
axes[1].set_yticklabels(models_to_plot, fontsize=8)
axes[1].set_xlabel("Validation Log Loss")
axes[1].set_title("Log Loss (lower is better)", fontsize=11, fontweight="bold")
axes[1].invert_yaxis()
for bar, val in zip(bars, val_lls):
    axes[1].text(val + 0.002, bar.get_y() + bar.get_height()/2,
                 f"{val:.4f}", va="center", fontsize=7, color="#a1a1aa")

plt.tight_layout()
plt.savefig(FIGURES / "pres_06_model_comparison_v2.png", bbox_inches="tight")
plt.close()
print("  Saved pres_06_model_comparison_v2.png")


# ── Figure 2: Per-Class Accuracy ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))

class_names = ["Home Win", "Draw", "Away Win"]
models_for_class = ["KickCastNet v2", "KickCastNet v1", "LightGBM_tuned_balanced", "CalibratedEnsemble"]
models_for_class = [m for m in models_for_class if m in all_val_probas or m == "KickCastNet v2"]

x = np.arange(len(class_names))
width = 0.18
class_colors = [EMERALD, CYAN, YELLOW, MUTED]

for i, model_name in enumerate(models_for_class):
    if model_name == "KickCastNet v2":
        preds = v2_val_pred
    elif model_name in all_val_probas:
        preds = np.argmax(all_val_probas[model_name], axis=1)
    else:
        continue

    accs = [accuracy_score(y_val[y_val == c], preds[y_val == c]) for c in range(3)]
    offset = (i - len(models_for_class) / 2 + 0.5) * width
    bars = ax.bar(x + offset, accs, width, label=model_name, color=class_colors[i], alpha=0.85)
    for bar, val in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.1%}", ha="center", fontsize=7, color="#a1a1aa")

ax.set_xticks(x)
ax.set_xticklabels(class_names)
ax.set_ylabel("Accuracy")
ax.set_title("Per-Class Accuracy (Validation Set)", fontsize=11, fontweight="bold")
ax.legend(fontsize=8, loc="upper right")
ax.set_ylim(0, 1.0)
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(FIGURES / "pres_07_per_class_accuracy.png", bbox_inches="tight")
plt.close()
print("  Saved pres_07_per_class_accuracy.png")


# ── Figure 3: Confusion Matrices ────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

cm_models = {
    "KickCastNet v1": all_val_probas.get("KickCastNet v1"),
    "KickCastNet v2": v2_val_proba,
    "KickCast Ensemble": ens_val_proba,
}

for idx, (name, proba) in enumerate(cm_models.items()):
    if proba is None:
        continue
    preds = np.argmax(proba, axis=1)
    cm = confusion_matrix(y_val, preds, normalize="true")

    im = axes[idx].imshow(cm, cmap="YlGn", vmin=0, vmax=0.8)
    axes[idx].set_xticks(range(3))
    axes[idx].set_yticks(range(3))
    axes[idx].set_xticklabels(["HW", "D", "AW"], fontsize=9)
    axes[idx].set_yticklabels(["HW", "D", "AW"], fontsize=9)
    axes[idx].set_title(name, fontsize=10, fontweight="bold")
    axes[idx].set_xlabel("Predicted")
    if idx == 0:
        axes[idx].set_ylabel("Actual")

    for i in range(3):
        for j in range(3):
            axes[idx].text(j, i, f"{cm[i, j]:.2f}",
                          ha="center", va="center", fontsize=11,
                          color="white" if cm[i, j] > 0.4 else "#333")

plt.tight_layout()
plt.savefig(FIGURES / "pres_08_confusion_matrices.png", bbox_inches="tight")
plt.close()
print("  Saved pres_08_confusion_matrices.png")


# ── Figure 4: Training Curves ───────────────────────────────────────
if final_trainer.training_history:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    history = final_trainer.training_history
    global_epochs = list(range(len(history)))
    losses = [h["loss"] for h in history]
    f1s = [h["val_f1"] for h in history]

    # Mark cycle boundaries
    cycle_boundaries = []
    for i in range(1, len(history)):
        if history[i]["cycle"] != history[i-1]["cycle"]:
            cycle_boundaries.append(i)

    ax1.plot(global_epochs, losses, color=RED, linewidth=1, alpha=0.8)
    ax1.set_ylabel("Training Loss")
    ax1.set_title("Training Curves — KickCastNet v2", fontsize=11, fontweight="bold")
    ax1.grid(alpha=0.3)
    for cb in cycle_boundaries:
        ax1.axvline(cb, color=MUTED, alpha=0.3, linestyle="--", linewidth=0.8)

    ax2.plot(global_epochs, f1s, color=EMERALD, linewidth=1, alpha=0.8)
    ax2.set_ylabel("Validation Macro F1")
    ax2.set_xlabel("Epoch (across all cycles)")
    ax2.grid(alpha=0.3)
    for cb in cycle_boundaries:
        ax2.axvline(cb, color=MUTED, alpha=0.3, linestyle="--", linewidth=0.8)

    # Mark snapshot captures (cycle boundaries)
    for cb in cycle_boundaries:
        ax2.plot(cb - 1, f1s[cb - 1], "o", color=YELLOW, markersize=6, zorder=5)
    if len(f1s) > 0:
        ax2.plot(len(f1s) - 1, f1s[-1], "o", color=YELLOW, markersize=6, zorder=5)

    plt.tight_layout()
    plt.savefig(FIGURES / "pres_09_training_curves.png", bbox_inches="tight")
    plt.close()
    print("  Saved pres_09_training_curves.png")


# ── Figure 5: Calibration Plot ──────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

cal_models = {
    "KickCastNet v2": v2_val_proba,
    "LightGBM_tuned_balanced": all_val_probas.get("LightGBM_tuned_balanced"),
    "KickCast Ensemble": ens_val_proba,
}
cal_colors = [EMERALD, YELLOW, CYAN]

for c, class_name in enumerate(class_names):
    for (name, proba), color in zip(cal_models.items(), cal_colors):
        if proba is None:
            continue
        y_binary = (y_val == c).astype(int)
        prob_true, prob_pred = calibration_curve(y_binary, proba[:, c], n_bins=10)
        axes[c].plot(prob_pred, prob_true, "o-", color=color, label=name,
                    markersize=4, linewidth=1.5)

    axes[c].plot([0, 1], [0, 1], "--", color=MUTED, alpha=0.5, linewidth=1)
    axes[c].set_title(class_name, fontsize=10, fontweight="bold")
    axes[c].set_xlabel("Predicted Probability")
    if c == 0:
        axes[c].set_ylabel("True Frequency")
    axes[c].legend(fontsize=7)
    axes[c].grid(alpha=0.3)
    axes[c].set_xlim(0, 1)
    axes[c].set_ylim(0, 1)

plt.suptitle("Calibration Plots (Validation Set)", fontsize=12, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(FIGURES / "pres_10_calibration.png", bbox_inches="tight")
plt.close()
print("  Saved pres_10_calibration.png")


# ── Figure 6: Attention Weights Heatmap ─────────────────────────────
try:
    attn_weights = final_trainer.get_attention_weights(X_val.head(100))
    if attn_weights is not None:
        # Average across samples and heads: (n_features, n_features)
        avg_attn = attn_weights.mean(axis=(0, 1))  # (31, 31)

        fig, ax = plt.subplots(figsize=(12, 10))
        im = ax.imshow(avg_attn, cmap="YlGn", aspect="auto")

        short_names = [f.replace("_delta", "").replace("_diff", "")
                       .replace("squad_value_", "sv_").replace("_", "\n")[:12]
                       for f in feature_names]
        ax.set_xticks(range(len(feature_names)))
        ax.set_yticks(range(len(feature_names)))
        ax.set_xticklabels(short_names, fontsize=5, rotation=90)
        ax.set_yticklabels(short_names, fontsize=5)
        ax.set_title("Feature Self-Attention Weights (averaged)", fontsize=11, fontweight="bold")
        plt.colorbar(im, ax=ax, shrink=0.8)
        plt.tight_layout()
        plt.savefig(FIGURES / "pres_11_attention_weights.png", bbox_inches="tight")
        plt.close()
        print("  Saved pres_11_attention_weights.png")
except Exception as e:
    print(f"  Attention weights visualization skipped: {e}")


# ── Figure 7: v1 vs v2 Improvement ─────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))

improvement_metrics = ["Macro F1", "Log Loss", "Draw Recall", "AUC-ROC"]
v1_vals = []
v2_vals = []

# Get v1 metrics
if "KickCastNet v1" in all_model_results:
    v1_r = all_model_results["KickCastNet v1"]
    v1_vals = [v1_r["val_f1"], v1_r["val_ll"], v1_r.get("val_draw_acc", 0.316), 0.7359]
else:
    v1_vals = [0.5230, 0.9393, 0.316, 0.7359]

v2_r = results_v2["Val"]
v2_vals = [v2_r["macro_f1"], v2_r["log_loss"], v2_r["acc_draw"], v2_r["auc_roc"]]

x = np.arange(len(improvement_metrics))
width = 0.35

bars1 = ax.bar(x - width/2, v1_vals, width, label="v1 (ResidualMLP)", color=YELLOW, alpha=0.8)
bars2 = ax.bar(x + width/2, v2_vals, width, label="v2 (Attention + Missing)", color=EMERALD, alpha=0.8)

for bar, val in zip(bars1, v1_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f"{val:.3f}", ha="center", fontsize=8, color=YELLOW)
for bar, val in zip(bars2, v2_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f"{val:.3f}", ha="center", fontsize=8, color=EMERALD)

ax.set_xticks(x)
ax.set_xticklabels(improvement_metrics)
ax.set_title("KickCastNet v1 vs v2 — Key Metrics", fontsize=11, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(FIGURES / "pres_12_v1_vs_v2.png", bbox_inches="tight")
plt.close()
print("  Saved pres_12_v1_vs_v2.png")


# ══════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)
print(f"\n  KickCastNet v1:         Val F1={0.5230:.4f}  LL={0.9393:.4f}")
print(f"  KickCastNet v2:         Val F1={results_v2['Val']['macro_f1']:.4f}  LL={results_v2['Val']['log_loss']:.4f}")
print(f"  LightGBM_tuned_bal:     Val F1={all_model_results.get('LightGBM_tuned_balanced', {}).get('val_f1', 0):.4f}")
print(f"  KickCast Ensemble:      Val F1={ens_val_f1:.4f}  LL={ens_val_ll:.4f}")
print(f"\n  Temperature: {final_trainer.temperature:.3f}")
print(f"  Snapshots: {len(final_trainer.snapshots)}")
print(f"  Figures saved to: {FIGURES}")
print("=" * 60)
