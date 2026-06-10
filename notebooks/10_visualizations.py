#!/usr/bin/env python3
"""
10_visualizations.py — Simulation result visualizations.

1. Win probability bar chart (Plotly)
2. Group advancement heatmap
3. Group standings stacked bars
4. Top contenders Sankey diagram
5. 2022 WC predictions vs actual (side-by-side)
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
import plotly.graph_objects as go
import plotly.express as px

SIM_DIR = BASE / "outputs" / "simulation_results"
FIG_DIR = BASE / "outputs" / "figures"
WC_DIR = BASE / "data" / "raw" / "world_cup_2026"

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": 11, "figure.facecolor": "white",
})

groups_df = pd.read_csv(WC_DIR / "groups.csv")
adv_df = pd.read_csv(SIM_DIR / "advancement_probabilities.csv")
gs_df = pd.read_csv(SIM_DIR / "group_standings_distribution.csv")
win_df = pd.read_csv(SIM_DIR / "win_probabilities.csv")

CONFED_COLORS = {
    "UEFA": "#003f88", "CONMEBOL": "#2a9d8f", "CONCACAF": "#e63946",
    "CAF": "#e9c46a", "AFC": "#f4a261", "OFC": "#8ecae6",
}

# Add confederation info
team_confed = dict(zip(groups_df["team"], groups_df["confederation"]))

# ══════════════════════════════════════════════════════════════════════
# 1. Win probability bar chart (Plotly)
# ══════════════════════════════════════════════════════════════════════
print("[1/5] Win probability bar chart...")

win_df_sorted = win_df.sort_values("win_probability", ascending=True)
win_df_sorted["confederation"] = win_df_sorted["team"].map(team_confed)
colors = win_df_sorted["confederation"].map(CONFED_COLORS).fillna("#adb5bd")

fig = go.Figure()
fig.add_trace(go.Bar(
    y=win_df_sorted["team"],
    x=win_df_sorted["win_probability"],
    orientation="h",
    marker_color=colors.tolist(),
    text=[f"{v:.1f}%" for v in win_df_sorted["win_probability"]],
    textposition="outside",
    textfont_size=9,
))

# Add confederation legend
for confed, color in CONFED_COLORS.items():
    fig.add_trace(go.Bar(
        y=[None], x=[None], orientation="h",
        marker_color=color, name=confed, showlegend=True
    ))

fig.update_layout(
    title="2026 FIFA World Cup — Tournament Win Probability",
    xaxis_title="Win Probability (%)",
    height=1200, width=900,
    showlegend=True, legend=dict(x=0.7, y=0.05),
    margin=dict(l=150),
)
fig.write_html(FIG_DIR / "21_win_probabilities.html")
fig.write_image(FIG_DIR / "21_win_probabilities.png", scale=2)
print("  Saved: 21_win_probabilities.html + .png")

# ══════════════════════════════════════════════════════════════════════
# 2. Group advancement heatmap (12x4)
# ══════════════════════════════════════════════════════════════════════
print("[2/5] Group advancement heatmap...")

# Build a matrix: for each team, their probability of advancing (top 2 + 3rd place)
adv_data = []
for group in sorted(gs_df["group"].unique()):
    grp = gs_df[gs_df["group"] == group].sort_values("pos_1", ascending=False)
    for _, row in grp.iterrows():
        # Advancing = pos_1 + pos_2 + some fraction of pos_3 (best 3rd place)
        team_adv = adv_df[adv_df["team"] == row["team"]]
        if len(team_adv) > 0:
            group_pct = team_adv.iloc[0]["Group"]
        else:
            group_pct = row["pos_1"] + row["pos_2"]
        adv_data.append({
            "group": f"Group {group}",
            "team": row["team"],
            "advance_pct": group_pct
        })

heatmap_df = pd.DataFrame(adv_data)

# Pivot for heatmap
all_groups = sorted(heatmap_df["group"].unique())
fig, ax = plt.subplots(figsize=(14, 10))

# Build matrix manually for proper ordering
group_teams = {}
for group in all_groups:
    teams = heatmap_df[heatmap_df["group"] == group].sort_values("advance_pct", ascending=False)
    group_teams[group] = teams["team"].tolist()

matrix = []
labels = []
for group in all_groups:
    for team in group_teams[group]:
        pct = heatmap_df[(heatmap_df["group"] == group) & (heatmap_df["team"] == team)]["advance_pct"].values[0]
        matrix.append(pct)
        labels.append(f"{team} ({pct:.0f}%)")

# Reshape to 12 groups x 4 teams
matrix = np.array(matrix).reshape(12, 4)
label_matrix = np.array(labels).reshape(12, 4)

sns.heatmap(matrix, annot=label_matrix, fmt="", cmap="RdYlGn",
            xticklabels=["1st seed", "2nd seed", "3rd seed", "4th seed"],
            yticklabels=all_groups, ax=ax,
            cbar_kws={"label": "Knockout Advancement %"},
            linewidths=1, linecolor="white")
ax.set_title("Group Stage Advancement Probability", fontsize=14)
plt.tight_layout()
fig.savefig(FIG_DIR / "22_group_advancement_heatmap.png")
plt.close()
print("  Saved: 22_group_advancement_heatmap.png")

# ══════════════════════════════════════════════════════════════════════
# 3. Group standings stacked bars
# ══════════════════════════════════════════════════════════════════════
print("[3/5] Group standings stacked bars...")

fig, axes = plt.subplots(3, 4, figsize=(20, 12))
pos_colors = ["#2a9d8f", "#264653", "#e9c46a", "#e63946"]

for idx, group in enumerate(sorted(gs_df["group"].unique())):
    ax = axes[idx // 4][idx % 4]
    grp = gs_df[gs_df["group"] == group].sort_values("pos_1", ascending=False)

    teams = grp["team"].tolist()
    bottom = np.zeros(len(teams))
    for pos in [1, 2, 3, 4]:
        vals = grp[f"pos_{pos}"].values
        ax.barh(teams, vals, left=bottom, color=pos_colors[pos-1],
                label=f"{pos}{'st' if pos==1 else 'nd' if pos==2 else 'rd' if pos==3 else 'th'}" if idx == 0 else "")
        bottom += vals

    ax.set_xlim(0, 100)
    ax.set_title(f"Group {group}", fontsize=11)
    ax.invert_yaxis()

axes[0][0].legend(bbox_to_anchor=(0.5, -0.15), ncol=4, fontsize=8)
plt.suptitle("Group Stage Finishing Position Probabilities", fontsize=14, y=1.01)
plt.tight_layout()
fig.savefig(FIG_DIR / "23_group_standings_stacked.png")
plt.close()
print("  Saved: 23_group_standings_stacked.png")

# ══════════════════════════════════════════════════════════════════════
# 4. Top contenders Sankey diagram
# ══════════════════════════════════════════════════════════════════════
print("[4/5] Sankey diagram (top 8 contenders)...")

top8 = adv_df.head(8)
rounds = ["Group", "R32", "R16", "QF", "SF", "Final", "Winner"]

labels = []
sources = []
targets = []
values = []
node_colors = []

# Create nodes: team_name@round
for team in top8["team"]:
    for rnd in rounds:
        labels.append(f"{team}\n{rnd}")
        confed = team_confed.get(team, "Unknown")
        node_colors.append(CONFED_COLORS.get(confed, "#adb5bd"))

n_teams = len(top8)
for i, team in enumerate(top8["team"]):
    row = top8[top8["team"] == team].iloc[0]
    for j in range(len(rounds) - 1):
        src_idx = i * len(rounds) + j
        tgt_idx = i * len(rounds) + j + 1
        val = row[rounds[j + 1]]
        if val > 0:
            sources.append(src_idx)
            targets.append(tgt_idx)
            values.append(val)

fig = go.Figure(go.Sankey(
    node=dict(
        pad=15, thickness=20,
        label=labels,
        color=node_colors,
    ),
    link=dict(
        source=sources,
        target=targets,
        value=values,
        color=[f"rgba(100,100,100,0.2)"] * len(sources)
    )
))

fig.update_layout(
    title="Top 8 Contenders — Tournament Path Probabilities",
    height=700, width=1200,
    font_size=10,
)
fig.write_html(FIG_DIR / "24_sankey_top_contenders.html")
fig.write_image(FIG_DIR / "24_sankey_top_contenders.png", scale=2)
print("  Saved: 24_sankey_top_contenders.html + .png")

# ══════════════════════════════════════════════════════════════════════
# 5. Round-by-round advancement (grouped bar chart)
# ══════════════════════════════════════════════════════════════════════
print("[5/5] Round-by-round advancement chart...")

top12 = adv_df.head(12)

fig, ax = plt.subplots(figsize=(14, 6))
rounds_plot = ["Group", "R32", "R16", "QF", "SF", "Final", "Winner"]
x = np.arange(len(top12))
width = 0.12

colors_rounds = ["#264653", "#2a9d8f", "#e9c46a", "#f4a261", "#e76f51", "#e63946", "#9b2226"]

for i, rnd in enumerate(rounds_plot):
    vals = top12[rnd].values
    offset = (i - len(rounds_plot)/2 + 0.5) * width
    bars = ax.bar(x + offset, vals, width, label=rnd, color=colors_rounds[i])

ax.set_xticks(x)
ax.set_xticklabels(top12["team"], rotation=45, ha="right")
ax.set_ylabel("Probability (%)")
ax.set_title("Round-by-Round Advancement Probability (Top 12 Teams)")
ax.legend(ncol=7, fontsize=8)
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
fig.savefig(FIG_DIR / "25_round_advancement.png")
plt.close()
print("  Saved: 25_round_advancement.png")

print("\n" + "=" * 60)
print("VISUALIZATIONS COMPLETE")
print("=" * 60)
