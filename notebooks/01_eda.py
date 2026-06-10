#!/usr/bin/env python3
"""
01_eda.py — Exploratory Data Analysis for FIFA World Cup Prediction Project.

Generates 9 publication-quality figures saved to outputs/figures/.
Converted to 01_eda.ipynb after execution.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ── Setup ──────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data" / "processed"
FIG_DIR = BASE / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Publication style
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.facecolor": "white",
})
sns.set_style("whitegrid")

# Load data
df = pd.read_csv(DATA / "feature_matrix.csv", parse_dates=["date"])
print(f"Loaded feature matrix: {df.shape}")

# ── 1. Match distribution over time by tournament type ─────────────────
print("\n[1/9] Match distribution over time by tournament type...")

df["year"] = df["date"].dt.year

# Group tournaments into categories
def categorize_tournament(t):
    t_lower = t.lower()
    if "world cup" in t_lower and "qualification" not in t_lower:
        return "World Cup"
    elif "qualification" in t_lower or "qualifier" in t_lower:
        return "Qualifiers"
    elif "friendly" in t_lower:
        return "Friendlies"
    elif any(x in t_lower for x in ["euro", "copa am", "african cup", "asian cup",
                                     "gold cup", "nations league"]):
        return "Continental"
    else:
        return "Other"

df["tournament_cat"] = df["tournament"].apply(categorize_tournament)

fig, ax = plt.subplots(figsize=(14, 5))
cat_order = ["World Cup", "Continental", "Qualifiers", "Friendlies", "Other"]
colors = ["#e63946", "#457b9d", "#2a9d8f", "#e9c46a", "#adb5bd"]
yearly = df.groupby(["year", "tournament_cat"]).size().unstack(fill_value=0)
yearly = yearly.reindex(columns=cat_order, fill_value=0)
yearly.plot(kind="bar", stacked=True, ax=ax, color=colors, width=0.85, edgecolor="none")
ax.set_xlabel("Year")
ax.set_ylabel("Number of Matches")
ax.set_title("International Match Distribution by Tournament Type (2004–2026)")
ax.legend(title="Tournament", bbox_to_anchor=(1.02, 1), loc="upper left")
# Show every other year label
labels = [item.get_text() for item in ax.get_xticklabels()]
ax.set_xticklabels([l if i % 2 == 0 else "" for i, l in enumerate(labels)], rotation=45)
plt.tight_layout()
fig.savefig(FIG_DIR / "01_match_distribution_by_year.png")
plt.close()
print("  Saved: 01_match_distribution_by_year.png")

# ── 2. Target class distribution overall and by tournament type ────────
print("[2/9] Target class distribution...")

labels_map = {0: "Home Win", 1: "Draw", 2: "Away Win"}
class_colors = ["#2a9d8f", "#e9c46a", "#e63946"]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Overall
counts = df["result"].value_counts().sort_index()
bars = axes[0].bar([labels_map[i] for i in counts.index], counts.values, color=class_colors, edgecolor="white")
for bar, val in zip(bars, counts.values):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                 f"{val}\n({100*val/len(df):.1f}%)", ha="center", fontsize=10)
axes[0].set_title("Overall Class Distribution")
axes[0].set_ylabel("Number of Matches")

# By tournament category
cat_results = df.groupby(["tournament_cat", "result"]).size().unstack(fill_value=0)
cat_results = cat_results.reindex(columns=[0, 1, 2])
cat_pct = cat_results.div(cat_results.sum(axis=1), axis=0) * 100
cat_pct = cat_pct.reindex(cat_order)
cat_pct.plot(kind="barh", stacked=True, ax=axes[1], color=class_colors, edgecolor="white")
axes[1].set_xlabel("Percentage (%)")
axes[1].set_title("Class Distribution by Tournament Type")
axes[1].legend([labels_map[i] for i in [0, 1, 2]], loc="lower right")
axes[1].set_xlim(0, 100)

plt.tight_layout()
fig.savefig(FIG_DIR / "02_class_distribution.png")
plt.close()
print("  Saved: 02_class_distribution.png")

# ── 3. Home advantage: win rates at home vs away vs neutral ────────────
print("[3/9] Home advantage analysis...")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Home vs neutral
for i, (label, subset) in enumerate([("Home (non-neutral)", df[df["is_neutral"] == 0]),
                                       ("Neutral venue", df[df["is_neutral"] == 1])]):
    dist = subset["result"].value_counts(normalize=True).sort_index() * 100
    bars = axes[0].bar([x + i*0.35 for x in range(3)], [dist.get(j, 0) for j in range(3)],
                       width=0.33, label=label, color=["#264653", "#e76f51"][i], alpha=0.85)

axes[0].set_xticks([x + 0.175 for x in range(3)])
axes[0].set_xticklabels(["Home Win", "Draw", "Away Win"])
axes[0].set_ylabel("Win Rate (%)")
axes[0].set_title("Home Advantage: Home vs Neutral Venue")
axes[0].legend()

# Home advantage over time
yearly_home = df.groupby("year").apply(
    lambda x: (x["result"] == 0).mean() * 100, include_groups=False
).reset_index(name="home_win_pct")
axes[1].plot(yearly_home["year"], yearly_home["home_win_pct"], "o-", color="#264653", linewidth=2, markersize=4)
axes[1].axhline(y=df["result"].eq(0).mean()*100, color="#e63946", linestyle="--", alpha=0.7, label="Overall avg")
axes[1].set_xlabel("Year")
axes[1].set_ylabel("Home Win Rate (%)")
axes[1].set_title("Home Win Rate Over Time")
axes[1].legend()

plt.tight_layout()
fig.savefig(FIG_DIR / "03_home_advantage.png")
plt.close()
print("  Saved: 03_home_advantage.png")

# ── 4. Feature correlation heatmap (top 20 most correlated with target) ──
print("[4/9] Feature correlation heatmap...")

feature_cols = [c for c in df.columns if c not in
                ["date", "home_team", "away_team", "home_score", "away_score",
                 "tournament", "result", "year", "tournament_cat"]]

corr_with_target = df[feature_cols + ["result"]].corr()["result"].drop("result").abs().sort_values(ascending=False)
top20 = corr_with_target.head(20).index.tolist()

fig, ax = plt.subplots(figsize=(12, 10))
corr_matrix = df[top20].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, square=True, linewidths=0.5, ax=ax,
            cbar_kws={"shrink": 0.8, "label": "Correlation"})
ax.set_title("Feature Correlation Heatmap (Top 20 Features by |corr| with Target)")
plt.tight_layout()
fig.savefig(FIG_DIR / "04_correlation_heatmap.png")
plt.close()
print("  Saved: 04_correlation_heatmap.png")

# ── 5. Elo diff vs outcome: box plot ──────────────────────────────────
print("[5/9] Elo diff vs outcome...")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Box plot
plot_df = df[["elo_diff", "result"]].copy()
plot_df["Outcome"] = plot_df["result"].map(labels_map)
order = ["Home Win", "Draw", "Away Win"]
sns.boxplot(data=plot_df, x="Outcome", y="elo_diff", order=order,
            palette=class_colors, ax=axes[0], fliersize=1)
axes[0].axhline(y=0, color="black", linestyle="--", alpha=0.3)
axes[0].set_ylabel("Elo Difference (Home - Away)")
axes[0].set_title("Elo Difference Distribution by Match Outcome")

# Violin plot for more detail
sns.violinplot(data=plot_df, x="Outcome", y="elo_diff", order=order,
               palette=class_colors, ax=axes[1], inner="quartile", cut=0)
axes[1].axhline(y=0, color="black", linestyle="--", alpha=0.3)
axes[1].set_ylabel("Elo Difference (Home - Away)")
axes[1].set_title("Elo Difference Density by Match Outcome")

plt.tight_layout()
fig.savefig(FIG_DIR / "05_elo_diff_vs_outcome.png")
plt.close()
print("  Saved: 05_elo_diff_vs_outcome.png")

# ── 6. Squad value delta distribution ──────────────────────────────────
print("[6/9] Squad value delta distribution...")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

sv_col = "squad_value_total_delta"
sv_data = df[[sv_col, "result"]].dropna()
sv_data["Outcome"] = sv_data["result"].map(labels_map)

sns.boxplot(data=sv_data, x="Outcome", y=sv_col, order=order,
            palette=class_colors, ax=axes[0], fliersize=1)
axes[0].axhline(y=0, color="black", linestyle="--", alpha=0.3)
axes[0].set_ylabel("Squad Value Delta (log-transformed)")
axes[0].set_title("Transfermarkt Squad Value Delta by Outcome")

# KDE plot
for i, outcome in enumerate(order):
    subset = sv_data[sv_data["Outcome"] == outcome][sv_col]
    subset.plot.kde(ax=axes[1], label=outcome, color=class_colors[i], linewidth=2)
axes[1].set_xlabel("Squad Value Delta (log-transformed)")
axes[1].set_title("Squad Value Delta Density by Outcome")
axes[1].legend()
axes[1].set_xlim(sv_data[sv_col].quantile(0.01), sv_data[sv_col].quantile(0.99))

plt.tight_layout()
fig.savefig(FIG_DIR / "06_squad_value_distribution.png")
plt.close()
print("  Saved: 06_squad_value_distribution.png")

# ── 7. Feature distributions: histograms of top 10 features ───────────
print("[7/9] Feature distributions (top 10)...")

top10 = corr_with_target.head(10).index.tolist()
fig, axes = plt.subplots(2, 5, figsize=(18, 7))
axes = axes.flatten()

for i, col in enumerate(top10):
    for j, outcome in enumerate([0, 1, 2]):
        subset = df[df["result"] == outcome][col].dropna()
        axes[i].hist(subset, bins=40, alpha=0.5, color=class_colors[j],
                     label=labels_map[outcome], density=True)
    axes[i].set_title(col.replace("_", " ").title()[:25], fontsize=9)
    axes[i].tick_params(labelsize=7)
    if i == 0:
        axes[i].legend(fontsize=7)

plt.suptitle("Feature Distributions by Outcome (Top 10 Features)", fontsize=14, y=1.02)
plt.tight_layout()
fig.savefig(FIG_DIR / "07_feature_distributions.png")
plt.close()
print("  Saved: 07_feature_distributions.png")

# ── 8. Missing value heatmap ──────────────────────────────────────────
print("[8/9] Missing value heatmap...")

missing = df[feature_cols].isnull().mean() * 100
missing = missing[missing > 0].sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(range(len(missing)), missing.values, color="#e76f51", edgecolor="white")
ax.set_yticks(range(len(missing)))
ax.set_yticklabels(missing.index)
ax.set_xlabel("Missing Values (%)")
ax.set_title("Missing Value Percentage by Feature")
for bar, val in zip(bars, missing.values):
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
            f"{val:.1f}%", va="center", fontsize=9)
ax.invert_yaxis()
plt.tight_layout()
fig.savefig(FIG_DIR / "08_missing_values.png")
plt.close()
print("  Saved: 08_missing_values.png")

# ── 9. Class distribution in train/val/test splits ────────────────────
print("[9/9] Split class distributions...")

splits_dir = DATA / "splits"
y_train = pd.read_csv(splits_dir / "y_train.csv")["result"]
y_val = pd.read_csv(splits_dir / "y_val.csv")["result"]
y_test = pd.read_csv(splits_dir / "y_test.csv")["result"]
y_smote = pd.read_csv(splits_dir / "y_train_smote.csv")["result"]

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
split_data = [("Train\n(2004–2019)", y_train), ("Validation\n(2020–Nov 2022)", y_val),
              ("Test\n(Nov 2022+)", y_test), ("Train SMOTE\n(balanced)", y_smote)]

for ax, (name, y) in zip(axes, split_data):
    counts = y.value_counts().sort_index()
    pcts = counts / counts.sum() * 100
    bars = ax.bar([labels_map[i] for i in counts.index], pcts.values, color=class_colors, edgecolor="white")
    for bar, pct, cnt in zip(bars, pcts.values, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{pct:.1f}%\n({cnt})", ha="center", fontsize=8)
    ax.set_title(name, fontsize=10)
    ax.set_ylabel("Percentage (%)" if ax == axes[0] else "")
    ax.set_ylim(0, 55)

plt.suptitle("Class Distribution Across Splits", fontsize=13, y=1.02)
plt.tight_layout()
fig.savefig(FIG_DIR / "09_split_class_distributions.png")
plt.close()
print("  Saved: 09_split_class_distributions.png")

# ── Summary stats ──────────────────────────────────────────────────────
print("\n" + "=" * 50)
print("EDA COMPLETE — 9 figures saved to outputs/figures/")
print("=" * 50)
print(f"Total matches: {len(df)}")
print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
print(f"Features: {len(feature_cols)}")
print(f"Home win rate: {(df['result']==0).mean()*100:.1f}%")
print(f"Draw rate: {(df['result']==1).mean()*100:.1f}%")
print(f"Away win rate: {(df['result']==2).mean()*100:.1f}%")
print(f"\nTop 5 features correlated with target:")
for feat, val in corr_with_target.head(5).items():
    print(f"  {feat}: |r| = {val:.4f}")
