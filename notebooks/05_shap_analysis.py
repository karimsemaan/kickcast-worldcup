#!/usr/bin/env python3
"""
05_shap_analysis.py — SHAP analysis for the best XGBoost model.
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
import shap
from sklearn.metrics import f1_score, accuracy_score, log_loss
from sklearn.inspection import permutation_importance
from xgboost import XGBClassifier
from scipy.stats import spearmanr

SPLITS = BASE / "data" / "processed" / "splits"
ARTIFACTS = BASE / "outputs" / "model_artifacts"
FIG_DIR = BASE / "outputs" / "figures"

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": 11, "figure.facecolor": "white",
})

CLASS_NAMES = ["Home Win", "Draw", "Away Win"]

print("Loading data and models...")
X_test = pd.read_csv(SPLITS / "X_test.csv")
y_test = pd.read_csv(SPLITS / "y_test.csv")["result"].values
X_train = pd.read_csv(SPLITS / "X_train.csv")
y_train = pd.read_csv(SPLITS / "y_train.csv")["result"].values

xgb_model = joblib.load(ARTIFACTS / "XGBoost_tuned.joblib")
optuna_data = joblib.load(ARTIFACTS / "optuna_results.joblib")

feature_names = X_test.columns.tolist()
print(f"Features: {len(feature_names)}, Test samples: {len(X_test)}")

# ══════════════════════════════════════════════════════════════════════
# 1. Compute SHAP values
# ══════════════════════════════════════════════════════════════════════
print("\n[1/7] Computing SHAP values (TreeExplainer)...")
explainer = shap.TreeExplainer(xgb_model)
shap_explanation = explainer(X_test)
# shap_explanation.values shape: (n_samples, n_features, n_classes)
print(f"  SHAP explanation shape: {shap_explanation.values.shape}")

# Extract per-class arrays for compatibility with older API calls
shap_vals_per_class = [shap_explanation.values[:, :, c] for c in range(3)]
base_values_per_class = [shap_explanation.base_values[:, c] for c in range(3)]


# ══════════════════════════════════════════════════════════════════════
# 2. Beeswarm (summary) plot — top 20 features
# ══════════════════════════════════════════════════════════════════════
print("\n[2/7] Beeswarm plot...")

fig, axes = plt.subplots(1, 3, figsize=(20, 8))
for cls in range(3):
    plt.sca(axes[cls])
    shap.summary_plot(shap_vals_per_class[cls], X_test, feature_names=feature_names,
                      max_display=20, show=False, plot_size=None)
    axes[cls].set_title(f"SHAP — {CLASS_NAMES[cls]}", fontsize=11)

plt.suptitle("SHAP Beeswarm Plots (Top 20 Features per Class)", fontsize=14, y=1.02)
plt.tight_layout()
fig.savefig(FIG_DIR / "16_shap_beeswarm.png")
plt.close()
print("  Saved: 16_shap_beeswarm.png")


# ══════════════════════════════════════════════════════════════════════
# 3. Bar plot of mean |SHAP| values
# ══════════════════════════════════════════════════════════════════════
print("\n[3/7] SHAP bar plot...")

mean_abs_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_vals_per_class], axis=0)
shap_importance = pd.Series(mean_abs_shap, index=feature_names).sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(10, 8))
shap_importance.plot(kind="barh", color="#264653", ax=ax)
ax.set_xlabel("Mean |SHAP value| (averaged across classes)")
ax.set_title("Feature Importance (SHAP)")
plt.tight_layout()
fig.savefig(FIG_DIR / "17_shap_bar.png")
plt.close()
print("  Saved: 17_shap_bar.png")


# ══════════════════════════════════════════════════════════════════════
# 4. Dependence plots for top 5 features
# ══════════════════════════════════════════════════════════════════════
print("\n[4/7] Dependence plots...")

top5_features = shap_importance.tail(5).index.tolist()[::-1]

fig, axes = plt.subplots(1, 5, figsize=(25, 5))
for i, feat in enumerate(top5_features):
    feat_idx = feature_names.index(feat)
    plt.sca(axes[i])
    shap.dependence_plot(feat_idx, shap_vals_per_class[0], X_test,
                         feature_names=feature_names, show=False, ax=axes[i])
    axes[i].set_title(f"{feat}", fontsize=9)

plt.suptitle("SHAP Dependence Plots (Home Win Class, Top 5 Features)", fontsize=13, y=1.02)
plt.tight_layout()
fig.savefig(FIG_DIR / "18_shap_dependence.png")
plt.close()
print("  Saved: 18_shap_dependence.png")


# ══════════════════════════════════════════════════════════════════════
# 5. Waterfall plots for specific 2022 WC matches
# ══════════════════════════════════════════════════════════════════════
print("\n[5/7] Waterfall plots for 2022 WC matches...")

fm = pd.read_csv(BASE / "data" / "processed" / "feature_matrix.csv", parse_dates=["date"])
wc22 = fm[(fm["tournament"].str.contains("FIFA World Cup", na=False)) &
          (fm["date"] >= "2022-11-20") & (fm["date"] <= "2022-12-18")].copy()

DROP_COLS = ["date", "home_team", "away_team", "home_score", "away_score", "tournament", "result"]

target_matches = [
    ("Argentina", "France"),
    ("Argentina", "Saudi Arabia"),
    ("Morocco", "Spain"),
    ("Germany", "Japan"),
]

for home, away in target_matches:
    match = wc22[(wc22["home_team"] == home) & (wc22["away_team"] == away)]
    if len(match) == 0:
        match = wc22[(wc22["home_team"] == away) & (wc22["away_team"] == home)]
        if len(match) == 0:
            print(f"  Match not found: {home} vs {away}")
            continue

    row = match.iloc[0]
    X_match = match.drop(columns=DROP_COLS, errors="ignore")

    match_expl = explainer(X_match)
    actual = int(row["result"])
    pred = xgb_model.predict(X_match)[0]

    # Build single-sample Explanation for waterfall
    shap_exp = shap.Explanation(
        values=match_expl.values[0, :, actual],
        base_values=match_expl.base_values[0, actual],
        data=X_match.values[0],
        feature_names=feature_names
    )

    plt.figure(figsize=(14, 4))
    shap.waterfall_plot(shap_exp, max_display=15, show=False)
    actual_label = CLASS_NAMES[actual]
    pred_label = CLASS_NAMES[pred]
    score_str = f"{int(row['home_score'])}-{int(row['away_score'])}"
    plt.title(f"{row['home_team']} {score_str} {row['away_team']} | Actual: {actual_label}, Pred: {pred_label}")
    safe_name = f"{row['home_team']}_{row['away_team']}".replace(" ", "_")
    plt.savefig(FIG_DIR / f"19_shap_waterfall_{safe_name}.png", bbox_inches="tight")
    plt.close()
    print(f"  Saved: 19_shap_waterfall_{safe_name}.png")


# ══════════════════════════════════════════════════════════════════════
# 6. Feature importance comparison
# ══════════════════════════════════════════════════════════════════════
print("\n[6/7] Feature importance comparison...")

xgb_importance = pd.Series(xgb_model.feature_importances_, index=feature_names).sort_values(ascending=False)
shap_imp = shap_importance.sort_values(ascending=False)

print("  Computing permutation importance...")
perm_result = permutation_importance(
    xgb_model, X_test, y_test, n_repeats=10, random_state=42,
    scoring="f1_macro", n_jobs=-1
)
perm_importance = pd.Series(perm_result.importances_mean, index=feature_names).sort_values(ascending=False)

fig, axes = plt.subplots(1, 3, figsize=(18, 8))
methods = [
    ("SHAP (mean |value|)", shap_imp, "#264653"),
    ("XGBoost Built-in (gain)", xgb_importance, "#2a9d8f"),
    ("Permutation (F1 drop)", perm_importance, "#e76f51"),
]

for ax, (title, imp, color) in zip(axes, methods):
    imp_sorted = imp.sort_values(ascending=True)
    imp_sorted.plot(kind="barh", color=color, ax=ax)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Importance")

plt.suptitle("Feature Importance: SHAP vs XGBoost vs Permutation", fontsize=14)
plt.tight_layout()
fig.savefig(FIG_DIR / "20_feature_importance_comparison.png")
plt.close()
print("  Saved: 20_feature_importance_comparison.png")

shap_ranks = shap_imp.rank(ascending=False)
xgb_ranks = xgb_importance.rank(ascending=False)
perm_ranks = perm_importance.rank(ascending=False)

r_sx, _ = spearmanr(shap_ranks[feature_names], xgb_ranks[feature_names])
r_sp, _ = spearmanr(shap_ranks[feature_names], perm_ranks[feature_names])
r_xp, _ = spearmanr(xgb_ranks[feature_names], perm_ranks[feature_names])
print(f"  Spearman rank correlations:")
print(f"    SHAP vs XGBoost: {r_sx:.3f}")
print(f"    SHAP vs Permutation: {r_sp:.3f}")
print(f"    XGBoost vs Permutation: {r_xp:.3f}")


# ══════════════════════════════════════════════════════════════════════
# 7. Ablation: remove bottom 30% features
# ══════════════════════════════════════════════════════════════════════
print("\n[7/7] Feature ablation (remove bottom 30%)...")

n_remove = int(0.3 * len(feature_names))
bottom_features = shap_imp.tail(n_remove).index.tolist()
keep_features = [f for f in feature_names if f not in bottom_features]

print(f"  Removing {n_remove} features: {bottom_features}")
print(f"  Keeping {len(keep_features)} features")

X_train_reduced = X_train[keep_features]
X_test_reduced = X_test[keep_features]

xgb_params = optuna_data["xgboost_best_params"]

xgb_full = XGBClassifier(
    objective="multi:softprob", num_class=3,
    random_state=42, n_jobs=-1, tree_method="hist", **xgb_params
)
xgb_full.fit(X_train, y_train)
y_pred_full = xgb_full.predict(X_test)
y_proba_full = xgb_full.predict_proba(X_test)

xgb_reduced = XGBClassifier(
    objective="multi:softprob", num_class=3,
    random_state=42, n_jobs=-1, tree_method="hist", **xgb_params
)
xgb_reduced.fit(X_train_reduced, y_train)
y_pred_red = xgb_reduced.predict(X_test_reduced)
y_proba_red = xgb_reduced.predict_proba(X_test_reduced)

print(f"\n  Full model ({len(feature_names)} features):")
print(f"    Accuracy: {accuracy_score(y_test, y_pred_full):.4f}")
print(f"    Macro F1: {f1_score(y_test, y_pred_full, average='macro'):.4f}")
print(f"    Log Loss: {log_loss(y_test, y_proba_full):.4f}")

print(f"\n  Reduced model ({len(keep_features)} features):")
print(f"    Accuracy: {accuracy_score(y_test, y_pred_red):.4f}")
print(f"    Macro F1: {f1_score(y_test, y_pred_red, average='macro'):.4f}")
print(f"    Log Loss: {log_loss(y_test, y_proba_red):.4f}")

joblib.dump(xgb_reduced, ARTIFACTS / "XGBoost_tuned_reduced.joblib")

print("\n" + "=" * 60)
print("SHAP ANALYSIS COMPLETE")
print("=" * 60)
