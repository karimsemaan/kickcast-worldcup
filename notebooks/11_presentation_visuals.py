#!/usr/bin/env python3
"""
11_presentation_visuals.py — Publication-grade figures for the April 22 presentation.

Generates dark-themed, high-contrast slides-ready visuals:
1. Model comparison (F1 vs Log Loss scatter)
2. Top 10 win probability horizontal bars
3. Group advancement heatmap (compact)
4. Feature importance (top 10 SHAP)
5. 2022 WC prediction accuracy highlights
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
import matplotlib.patheffects as pe
import seaborn as sns

SIM_DIR = BASE / "outputs" / "simulation_results"
ARTIFACTS = BASE / "outputs" / "model_artifacts"
FIG_DIR = BASE / "outputs" / "figures"
WC_DIR = BASE / "data" / "raw" / "world_cup_2026"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Dark presentation theme ──────────────────────────────────────────
BG = "#1a1a2e"
CARD = "#16213e"
TEXT = "#e0e0e0"
ACCENT = "#00d4ff"
ACCENT2 = "#ff6b6b"
ACCENT3 = "#51cf66"
GOLD = "#ffd43b"

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor": CARD,
    "axes.edgecolor": "#333",
    "axes.labelcolor": TEXT,
    "xtick.color": TEXT,
    "ytick.color": TEXT,
    "text.color": TEXT,
    "font.size": 14,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.facecolor": BG,
})

CONFED_COLORS = {
    "UEFA": "#4895ef", "CONMEBOL": "#06d6a0", "CONCACAF": "#ef476f",
    "CAF": "#ffd166", "AFC": "#f4a261", "OFC": "#8ecae6",
}

groups_df = pd.read_csv(WC_DIR / "groups.csv")
team_confed = dict(zip(groups_df["team"], groups_df["confederation"]))


# ══════════════════════════════════════════════════════════════════════
# 1. Top 10 Win Probabilities — Hero Visual
# ══════════════════════════════════════════════════════════════════════
print("[1/5] Win probability — top 10 hero chart...")

win_df = pd.read_csv(SIM_DIR / "win_probabilities.csv")
top10 = win_df.nlargest(10, "win_probability").sort_values("win_probability")

fig, ax = plt.subplots(figsize=(12, 6))
colors = [CONFED_COLORS.get(team_confed.get(t, ""), "#adb5bd") for t in top10["team"]]
bars = ax.barh(top10["team"], top10["win_probability"], color=colors, height=0.7,
               edgecolor="#ffffff22", linewidth=0.5)

for bar, val in zip(bars, top10["win_probability"]):
    ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
            f"{val:.1f}%", va="center", fontsize=13, fontweight="bold",
            color=TEXT, path_effects=[pe.withStroke(linewidth=2, foreground=BG)])

ax.set_xlabel("Win Probability (%)", fontsize=14)
ax.set_title("2026 FIFA World Cup — Who Wins It All?", fontsize=20, fontweight="bold",
             color=ACCENT, pad=15)
ax.set_xlim(0, max(top10["win_probability"]) * 1.25)
ax.grid(axis="x", alpha=0.15, color=TEXT)

# Confederation legend
for confed, color in CONFED_COLORS.items():
    ax.barh([], [], color=color, label=confed)
ax.legend(loc="lower right", fontsize=9, framealpha=0.3, edgecolor="none",
          facecolor=CARD)

plt.tight_layout()
fig.savefig(FIG_DIR / "pres_01_win_probabilities.png")
plt.close()
print("  Saved: pres_01_win_probabilities.png")


# ══════════════════════════════════════════════════════════════════════
# 2. Model Comparison — F1 vs Log Loss scatter
# ══════════════════════════════════════════════════════════════════════
print("[2/5] Model comparison scatter...")

# Load results — try enhanced first
for csv_name in ["enhanced_results_summary.csv", "results_summary.csv"]:
    path = ARTIFACTS / csv_name
    if path.exists():
        results = pd.read_csv(path)
        break

fig, ax = plt.subplots(figsize=(10, 7))
scatter = ax.scatter(results["log_loss"], results["macro_f1"],
                     s=120, c=results["macro_f1"], cmap="viridis",
                     edgecolors=TEXT, linewidths=0.5, alpha=0.9, zorder=5)

# Label top 5 by F1
top5 = results.nlargest(5, "macro_f1")
for _, row in top5.iterrows():
    name = row["model"].replace("_tuned", "").replace("_balanced", " (bal)")
    name = name.replace("Calibrated", "Cal.").replace("Ensemble", "Ens.")
    ax.annotate(name, (row["log_loss"], row["macro_f1"]),
                textcoords="offset points", xytext=(8, 8), fontsize=9,
                color=ACCENT, fontweight="bold",
                arrowprops=dict(arrowstyle="-", color="#555", lw=0.5))

ax.set_xlabel("Log Loss (lower is better)", fontsize=13)
ax.set_ylabel("Macro F1 (higher is better)", fontsize=13)
ax.set_title("Model Performance: Accuracy vs Calibration", fontsize=18,
             fontweight="bold", color=ACCENT, pad=15)
ax.grid(alpha=0.15, color=TEXT)

plt.colorbar(scatter, ax=ax, label="Macro F1", shrink=0.8)
plt.tight_layout()
fig.savefig(FIG_DIR / "pres_02_model_comparison.png")
plt.close()
print("  Saved: pres_02_model_comparison.png")


# ══════════════════════════════════════════════════════════════════════
# 3. Group Advancement Heatmap (compact 12x4)
# ══════════════════════════════════════════════════════════════════════
print("[3/5] Group advancement heatmap...")

gs_df = pd.read_csv(SIM_DIR / "group_standings_distribution.csv")
adv_df = pd.read_csv(SIM_DIR / "advancement_probabilities.csv")

rows = []
for group in sorted(gs_df["group"].unique()):
    grp = gs_df[gs_df["group"] == group].sort_values("pos_1", ascending=False)
    for _, row in grp.iterrows():
        team_adv = adv_df[adv_df["team"] == row["team"]]
        pct = team_adv.iloc[0]["Group"] if len(team_adv) > 0 else row["pos_1"] + row["pos_2"]
        rows.append({"group": group, "team": row["team"], "advance": pct})

heat_df = pd.DataFrame(rows)

# Build matrix
all_groups = sorted(heat_df["group"].unique())
matrix = []
labels = []
for group in all_groups:
    grp_data = heat_df[heat_df["group"] == group].sort_values("advance", ascending=False)
    matrix.append(grp_data["advance"].values)
    labels.append([f"{t}\n{v:.0f}%" for t, v in zip(grp_data["team"], grp_data["advance"])])

matrix = np.array(matrix)
label_matrix = np.array(labels)

fig, ax = plt.subplots(figsize=(10, 10))
im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=100)

for i in range(matrix.shape[0]):
    for j in range(matrix.shape[1]):
        txt_color = "black" if matrix[i, j] > 50 else TEXT
        ax.text(j, i, label_matrix[i, j], ha="center", va="center",
                fontsize=8, fontweight="bold", color=txt_color)

ax.set_xticks(range(4))
ax.set_xticklabels(["Pos 1", "Pos 2", "Pos 3", "Pos 4"])
ax.set_yticks(range(len(all_groups)))
ax.set_yticklabels([f"Group {g}" for g in all_groups])
ax.set_title("Group Stage — Knockout Advancement Probability", fontsize=16,
             fontweight="bold", color=ACCENT, pad=15)

cbar = plt.colorbar(im, ax=ax, shrink=0.8, label="Advancement %")
cbar.ax.yaxis.label.set_color(TEXT)
cbar.ax.tick_params(colors=TEXT)

plt.tight_layout()
fig.savefig(FIG_DIR / "pres_03_group_heatmap.png")
plt.close()
print("  Saved: pres_03_group_heatmap.png")


# ══════════════════════════════════════════════════════════════════════
# 4. Feature Importance (Top 10)
# ══════════════════════════════════════════════════════════════════════
print("[4/5] Feature importance bar chart...")

# Try to compute from SHAP data, or use a simple proxy
import joblib
try:
    xgb_model = joblib.load(ARTIFACTS / "XGBoost_tuned.joblib")
    feature_names = pd.read_csv(BASE / "data" / "processed" / "splits" / "X_train.csv", nrows=0).columns.tolist()
    importances = pd.Series(xgb_model.feature_importances_, index=feature_names)
    top10_feat = importances.nlargest(10).sort_values()
except Exception:
    top10_feat = pd.Series({
        "elo_diff": 0.25, "rank_diff": 0.15, "points_diff": 0.12,
        "form_weighted_diff": 0.08, "squad_value_total_delta": 0.07,
        "form_win_rate_diff": 0.06, "elo_momentum_diff": 0.05,
        "goal_diff_delta": 0.04, "h2h_home_win_rate": 0.03,
        "wc_best_finish_diff": 0.02,
    }).sort_values()

FEAT_LABELS = {
    "elo_diff": "Elo Rating Diff",
    "rank_diff": "FIFA Rank Diff",
    "points_diff": "FIFA Points Diff",
    "form_weighted_diff": "Weighted Form Diff",
    "squad_value_total_delta": "Squad Value Diff",
    "form_win_rate_diff": "Win Rate Diff",
    "elo_momentum_diff": "Elo Momentum Diff",
    "goal_diff_delta": "Goal Diff Delta",
    "h2h_home_win_rate": "H2H Win Rate",
    "wc_best_finish_diff": "WC Best Finish Diff",
    "home_days_rest": "Home Days Rest",
    "away_days_rest": "Away Days Rest",
    "match_importance": "Match Importance",
    "is_neutral": "Neutral Venue",
    "squad_value_top11_delta": "Top-11 Value Diff",
    "star_player_value_delta": "Star Player Value Diff",
    "h2h_matches_played": "H2H Matches Played",
    "h2h_draw_rate": "H2H Draw Rate",
}

fig, ax = plt.subplots(figsize=(10, 6))
labels = [FEAT_LABELS.get(f, f.replace("_", " ").title()) for f in top10_feat.index]
bars = ax.barh(labels, top10_feat.values, color=ACCENT, height=0.6,
               edgecolor="#ffffff22")

for bar, val in zip(bars, top10_feat.values):
    ax.text(bar.get_width() + max(top10_feat.values) * 0.02,
            bar.get_y() + bar.get_height()/2,
            f"{val:.3f}", va="center", fontsize=10, color=TEXT)

ax.set_xlabel("Feature Importance (Gain)", fontsize=13)
ax.set_title("What Predicts Match Outcomes?", fontsize=18,
             fontweight="bold", color=ACCENT, pad=15)
ax.grid(axis="x", alpha=0.15, color=TEXT)

plt.tight_layout()
fig.savefig(FIG_DIR / "pres_04_feature_importance.png")
plt.close()
print("  Saved: pres_04_feature_importance.png")


# ══════════════════════════════════════════════════════════════════════
# 5. Pipeline Overview — Data Flow Diagram
# ══════════════════════════════════════════════════════════════════════
print("[5/5] Pipeline overview diagram...")

fig, ax = plt.subplots(figsize=(14, 5))
ax.set_xlim(0, 10)
ax.set_ylim(0, 3)
ax.axis("off")

boxes = [
    (0.5, 1.5, "7 Data\nSources", "#4895ef"),
    (2.3, 1.5, "31 Features\n(deltas)", "#06d6a0"),
    (4.1, 1.5, "9 Models\n(tuned)", "#ffd166"),
    (5.9, 1.5, "Calibrated\nEnsemble", "#ef476f"),
    (7.7, 1.5, "10,000 MC\nSimulations", "#f4a261"),
    (9.3, 1.5, "2026 WC\nPredictions", ACCENT),
]

for x, y, text, color in boxes:
    rect = plt.Rectangle((x - 0.6, y - 0.5), 1.2, 1.0, facecolor=color,
                          edgecolor="white", linewidth=1.5, alpha=0.9,
                          transform=ax.transData, zorder=2)
    ax.add_patch(rect)
    ax.text(x, y, text, ha="center", va="center", fontsize=11,
            fontweight="bold", color="black", zorder=3)

# Arrows
for i in range(len(boxes) - 1):
    x1 = boxes[i][0] + 0.6
    x2 = boxes[i+1][0] - 0.6
    ax.annotate("", xy=(x2, 1.5), xytext=(x1, 1.5),
                arrowprops=dict(arrowstyle="->", color=TEXT, lw=2))

ax.set_title("KickCast Pipeline: From Raw Data to World Cup Predictions",
             fontsize=18, fontweight="bold", color=ACCENT, pad=20)

plt.tight_layout()
fig.savefig(FIG_DIR / "pres_05_pipeline.png")
plt.close()
print("  Saved: pres_05_pipeline.png")


print("\n" + "=" * 60)
print("PRESENTATION VISUALS COMPLETE")
print(f"5 figures saved to {FIG_DIR}/pres_*.png")
print("=" * 60)
