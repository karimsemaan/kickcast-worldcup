#!/usr/bin/env python3
"""
04_world_cup_simulation.py — 2026 FIFA World Cup Monte Carlo Simulation.

Builds proper feature vectors for 2026 matches by reusing the same
data lookups as 02_build_features.py (Elo, FIFA rank, squad values,
form, H2H, WC history), then runs 10,000 Monte Carlo iterations.
"""

import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

import numpy as np
import pandas as pd
import joblib
from collections import defaultdict, deque
import bisect

from src.simulation import (simulate_group_stage, get_advancing_teams,
                             slot_third_place_teams, simulate_knockout_match,
                             historical_name, display_name, WC_TO_HISTORICAL)

# ── Paths ──────────────────────────────────────────────────────────────
WC_DIR = BASE / "data" / "raw" / "world_cup_2026"
RAW = BASE / "data" / "raw"
PROCESSED = BASE / "data" / "processed"
ARTIFACTS = BASE / "outputs" / "model_artifacts"
SIM_DIR = BASE / "outputs" / "simulation_results"
SIM_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("2026 FIFA World Cup Monte Carlo Simulation")
print("=" * 60)

# ── Load tournament structure ──────────────────────────────────────────
fixtures = pd.read_csv(WC_DIR / "fixtures.csv")
groups = pd.read_csv(WC_DIR / "groups.csv")
bracket = pd.read_csv(WC_DIR / "knockout_bracket.csv")
all_teams = groups["team"].tolist()

print(f"Groups: {len(groups)} teams, {len(fixtures)} group matches, {len(bracket)} knockout matches")

# ── Load model ─────────────────────────────────────────────────────────
model = joblib.load(ARTIFACTS / "KickCastNet_v3.joblib")
print(f"Model loaded: KickCastNet v3 (dual-path + attention + thresholds)")

# ══════════════════════════════════════════════════════════════════════
# LOAD RAW DATA & BUILD TEAM STATE (same approach as 02_build_features.py)
# ══════════════════════════════════════════════════════════════════════
print("\nLoading raw data sources...")

# Reference date: start of World Cup group stage
ref_date = pd.Timestamp("2026-06-10")

# Import name mappings from feature builder
sys.path.insert(0, str(BASE / "data" / "scripts"))
from importlib import import_module
fb_spec = import_module("02_build_features")

# ── 1. Match results + Elo ─────────────────────────────────────────────
results = fb_spec.load_results()
print(f"  Match results: {len(results)} matches")

print("  Computing Elo ratings from scratch...")
elo_hist = fb_spec.compute_elo_ratings(results)

# Current Elo for each team (last recorded value)
# Use historical_name() to look up teams whose names changed (Turkiye→Turkey, etc.)
current_elo = {}
for team in all_teams:
    hist_name = historical_name(team)
    if hist_name in elo_hist and len(elo_hist[hist_name]) > 0:
        current_elo[team] = elo_hist[hist_name][-1][1]
    elif team in elo_hist and len(elo_hist[team]) > 0:
        current_elo[team] = elo_hist[team][-1][1]
    else:
        current_elo[team] = 1500.0  # default
        print(f"    WARNING: no Elo history for {team} (tried {hist_name}), using 1500")

# Print Elo rankings for 48 teams
elo_ranked = sorted(current_elo.items(), key=lambda x: -x[1])
print("\n  Current Elo ratings (48 teams):")
for i, (team, elo) in enumerate(elo_ranked):
    print(f"    {i+1:2d}. {team:<28s} {elo:.0f}")

# Elo momentum (last 5 matches)
current_momentum = {}
for team in all_teams:
    hist_name = historical_name(team)
    mom = fb_spec.get_elo_momentum(hist_name, ref_date, elo_hist, n=5)
    if mom is None:
        mom = fb_spec.get_elo_momentum(team, ref_date, elo_hist, n=5)
    current_momentum[team] = mom if mom is not None else 0.0

# ── 2. FIFA Rankings ───────────────────────────────────────────────────
print("\n  Loading FIFA rankings...")
fifa_dates, fifa_by_date = fb_spec.load_fifa_rankings()

current_rank = {}
current_points = {}
for team in all_teams:
    hist_name = historical_name(team)
    r, p, _ = fb_spec.get_fifa(hist_name, ref_date, fifa_dates, fifa_by_date)
    if r is None:
        r, p, _ = fb_spec.get_fifa(team, ref_date, fifa_dates, fifa_by_date)
    current_rank[team] = r if r is not None else 100
    current_points[team] = p if p is not None else 1000.0

# ── 3. Squad values ───────────────────────────────────────────────────
print("  Loading Transfermarkt squad values...")
players, vals = fb_spec.load_players_and_valuations()
squad_agg, player_snap, top5_snap, quarters = fb_spec.precompute_squad_snapshots(
    players, vals, pd.Timestamp("2004-01-01")
)

current_squad = {}
for team in all_teams:
    hist_name = historical_name(team)
    sq = fb_spec.get_squad(hist_name, ref_date, squad_agg, quarters)
    if sq is None:
        sq = fb_spec.get_squad(team, ref_date, squad_agg, quarters)
    current_squad[team] = sq  # can be None
    if sq is None:
        print(f"    WARNING: no squad values for {team}")

# ── 4. Form (rolling 10 matches) ──────────────────────────────────────
print("  Computing form from match history...")
IMPORTANCE = fb_spec.IMPORTANCE
form_tracker = defaultdict(lambda: deque(maxlen=10))
last_match_date = {}

for _, row in results.iterrows():
    home, away = row["home_team"], row["away_team"]
    hs, aws = int(row["home_score"]), int(row["away_score"])
    date = row["date"]
    imp = IMPORTANCE.get(row["tournament"], 0.3)

    h_result = "win" if hs > aws else ("draw" if hs == aws else "loss")
    a_result = "win" if aws > hs else ("draw" if hs == aws else "loss")

    form_tracker[home].append({"r": h_result, "gf": hs, "ga": aws, "imp": imp})
    form_tracker[away].append({"r": a_result, "gf": aws, "ga": hs, "imp": imp})
    last_match_date[home] = date
    last_match_date[away] = date

def compute_form_stats(team):
    """Compute win rate, weighted form, and goal diff from rolling form."""
    # Form tracker is keyed by historical names (from results.csv)
    hist_name = historical_name(team)
    team_form = form_tracker.get(hist_name) or form_tracker.get(team)
    if not team_form or len(team_form) == 0:
        return 0.5, 0.25, 0.0, 30.0  # defaults

    wins = sum(1 for m in team_form if m["r"] == "win")
    wr = wins / len(team_form)

    wts = []
    for i, m in enumerate(reversed(list(team_form))):
        decay = 0.9 ** i
        score = 1.0 if m["r"] == "win" else (0.5 if m["r"] == "draw" else 0.0)
        wts.append(decay * m["imp"] * score)
    wf = np.mean(wts) if wts else 0.25

    gds = [m["gf"] - m["ga"] for m in team_form]
    gd = np.mean(gds)

    lm = last_match_date.get(hist_name) or last_match_date.get(team)
    days_rest = (ref_date - lm).days if lm else 30

    return wr, wf, gd, days_rest

# ── 5. H2H ─────────────────────────────────────────────────────────────
print("  Building H2H history...")
h2h_db = defaultdict(list)
for _, row in results.iterrows():
    home, away = row["home_team"], row["away_team"]
    hs, aws = int(row["home_score"]), int(row["away_score"])
    pair = tuple(sorted([home, away]))
    winner = home if hs > aws else (away if aws > hs else None)
    h2h_db[pair].append({"date": row["date"], "winner": winner, "home": home})

# ── 6. World Cup history ──────────────────────────────────────────────
print("  Loading World Cup history...")
wc_data = fb_spec.load_wc_history()

# ── 7. Confederation lookup ───────────────────────────────────────────
team_conf = {}
for dd in fifa_by_date.values():
    for t, d in dd.items():
        if d.get("conf"):
            team_conf[t] = d["conf"]

# Also map WC display names to confederations from groups.csv
for _, row in groups.iterrows():
    team_conf[row["team"]] = row["confederation"]


# ══════════════════════════════════════════════════════════════════════
# BUILD FEATURE VECTORS FOR 2026 MATCHES
# ══════════════════════════════════════════════════════════════════════

def log_transform(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    return np.sign(x) * np.log1p(abs(x))


def build_match_features(home, away):
    """Build a proper feature vector for a match, same logic as 02_build_features.py."""
    f = {}

    # CONTEXT
    f["match_importance"] = 1.0  # World Cup
    f["is_neutral"] = 1          # All WC matches are neutral venue

    h_conf = team_conf.get(home, "")
    a_conf = team_conf.get(away, "")
    f["same_confederation"] = int(h_conf != "" and h_conf == a_conf)
    f["home_continent_advantage"] = 0  # neutral venue

    # ELO
    h_elo = current_elo.get(home, 1500)
    a_elo = current_elo.get(away, 1500)
    f["elo_diff"] = h_elo - a_elo

    h_mom = current_momentum.get(home, 0)
    a_mom = current_momentum.get(away, 0)
    f["elo_momentum_diff"] = h_mom - a_mom

    # FIFA RANKINGS
    h_rank = current_rank.get(home, 100)
    a_rank = current_rank.get(away, 100)
    f["rank_diff"] = h_rank - a_rank

    h_pts = current_points.get(home, 1000)
    a_pts = current_points.get(away, 1000)
    f["points_diff"] = h_pts - a_pts

    # SQUAD VALUES
    h_sq = current_squad.get(home)
    a_sq = current_squad.get(away)
    if h_sq and a_sq:
        for key in ["total", "top11", "attack", "mid", "def"]:
            f[f"squad_value_{key}_delta"] = log_transform(h_sq[key] - a_sq[key])
        f["star_player_value_delta"] = log_transform(h_sq["star"] - a_sq["star"])
        f["squad_depth_delta"] = log_transform(h_sq["depth"] - a_sq["depth"])
    else:
        for col in ["squad_value_total_delta", "squad_value_top11_delta",
                     "squad_value_attack_delta", "squad_value_mid_delta",
                     "squad_value_def_delta", "star_player_value_delta",
                     "squad_depth_delta"]:
            f[col] = np.nan

    # FORM
    h_wr, h_wf, h_gd, h_rest = compute_form_stats(home)
    a_wr, a_wf, a_gd, a_rest = compute_form_stats(away)
    f["home_days_rest"] = h_rest
    f["away_days_rest"] = a_rest
    f["form_win_rate_diff"] = h_wr - a_wr
    f["form_weighted_diff"] = h_wf - a_wf
    f["goal_diff_delta"] = h_gd - a_gd

    # H2H (h2h_db is keyed by historical names from results.csv)
    h_hist = historical_name(home)
    a_hist = historical_name(away)
    pair = tuple(sorted([h_hist, a_hist]))
    hist = h2h_db.get(pair, [])
    if not hist:
        pair = tuple(sorted([home, away]))
        hist = h2h_db.get(pair, [])
    if hist:
        h_wins = sum(1 for h in hist if h["winner"] == h_hist or h["winner"] == home)
        draws = sum(1 for h in hist if h["winner"] is None)
        f["h2h_home_win_rate"] = h_wins / len(hist)
        f["h2h_draw_rate"] = draws / len(hist)
        f["h2h_matches_played"] = len(hist)
    else:
        f["h2h_home_win_rate"] = np.nan
        f["h2h_draw_rate"] = np.nan
        f["h2h_matches_played"] = 0

    # TOURNAMENT WIN RATE (use WC-specific)
    f["tournament_wr_delta"] = 0.0  # no prior WC 2026 matches to base this on

    # WC HISTORY (keyed by canonical names — try both WC and historical)
    h_wc = wc_data.get(home, wc_data.get(historical_name(home), {}))
    a_wc = wc_data.get(away, wc_data.get(historical_name(away), {}))
    for key, default in [("appearances", 0), ("knockout_rate", 0),
                          ("best_finish", 1), ("goals_per_game", 0)]:
        f[f"wc_{key}_diff"] = h_wc.get(key, default) - a_wc.get(key, default)

    # INJURIES (set to 0 — no live scrape yet)
    f["injury_count_delta"] = 0
    f["injury_burden_delta"] = 0.0
    f["star_injury_flag"] = 0

    return f


# ── Get the expected column order from the training data ───────────────
feature_cols = pd.read_csv(PROCESSED / "splits" / "X_train.csv", nrows=0).columns.tolist()

print(f"\nBuilding feature vectors for {len(fixtures)} group stage matches...")

match_probas = {}
match_details = []

for _, match in fixtures.iterrows():
    mn = match["match_number"]
    home = match["home_team"]
    away = match["away_team"]

    f = build_match_features(home, away)
    X = pd.DataFrame([f]).reindex(columns=feature_cols)
    proba = model.predict_proba(X)[0]

    match_probas[mn] = (proba[0], proba[1], proba[2])
    match_details.append({
        "match_number": mn, "home_team": home, "away_team": away,
        "group": match["group"],
        "p_home_win": proba[0], "p_draw": proba[1], "p_away_win": proba[2],
        "predicted": ["Home Win", "Draw", "Away Win"][np.argmax(proba)]
    })

pred_df = pd.DataFrame(match_details)
pred_df.to_csv(SIM_DIR / "group_match_predictions.csv", index=False)

print("\nGroup stage predictions:")
for _, row in pred_df.iterrows():
    print(f"  M{int(row['match_number']):02d} {row['group']} {row['home_team']:<28s} vs {row['away_team']:<28s} "
          f"[HW:{row['p_home_win']:.2f} D:{row['p_draw']:.2f} AW:{row['p_away_win']:.2f}] -> {row['predicted']}")


# ── Feature builder for knockout matches ───────────────────────────────
def build_knockout_features(team_a, team_b):
    """Build feature vector for a knockout match."""
    f = build_match_features(team_a, team_b)
    return pd.DataFrame([f]).reindex(columns=feature_cols)


# ══════════════════════════════════════════════════════════════════════
# MONTE CARLO SIMULATION
# ══════════════════════════════════════════════════════════════════════
N_ITERATIONS = 10000
print(f"\nRunning {N_ITERATIONS} Monte Carlo iterations...")

rng = np.random.default_rng(42)

tournament_wins = defaultdict(int)
round_advancement = defaultdict(lambda: defaultdict(int))
group_standings_dist = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
final_matchups = defaultdict(int)

for team in all_teams:
    for rnd in ["Group", "R32", "R16", "QF", "SF", "Final", "Winner"]:
        round_advancement[team][rnd] = 0

for iteration in range(N_ITERATIONS):
    if iteration % 2000 == 0:
        print(f"  Iteration {iteration}/{N_ITERATIONS}...")

    # 1. Group stage
    standings = simulate_group_stage(fixtures, match_probas, groups, rng)

    for group, teams in standings.items():
        for pos, t in enumerate(teams):
            group_standings_dist[group][t["team"]][pos + 1] += 1

    # 2. Advancing teams
    advancing, third_place_groups = get_advancing_teams(standings)
    for code, team in advancing.items():
        round_advancement[team]["Group"] += 1

    # 3. Slot 3rd place
    slot_assignments = slot_third_place_teams(advancing, third_place_groups, bracket)

    # 4. Play knockout rounds
    match_winners = {}

    for rnd in ["R32", "R16", "QF", "SF", "F"]:
        rnd_label = "Final" if rnd == "F" else rnd
        rnd_matches = bracket[bracket["round"] == rnd]

        for _, match in rnd_matches.iterrows():
            mn = match["match_number"]
            home_ref = str(match["home_source"])
            away_ref = str(match["away_source"])

            if home_ref.startswith("W") or home_ref.startswith("L"):
                home_team = match_winners.get(home_ref)
            else:
                home_team = advancing.get(home_ref)

            if "/" in away_ref:
                code = slot_assignments.get(mn)
                away_team = advancing.get(code) if code else None
            elif away_ref.startswith("W") or away_ref.startswith("L"):
                away_team = match_winners.get(away_ref)
            else:
                away_team = advancing.get(away_ref)

            if home_team and away_team:
                winner = simulate_knockout_match(
                    home_team, away_team, model, build_knockout_features, rng
                )
                match_winners[f"W{mn}"] = winner
                match_winners[f"L{mn}"] = away_team if winner == home_team else home_team
                round_advancement[winner][rnd_label] += 1

                if rnd == "F":
                    tournament_wins[winner] += 1
                    round_advancement[winner]["Winner"] += 1
                    matchup = tuple(sorted([home_team, away_team]))
                    final_matchups[matchup] += 1


# ══════════════════════════════════════════════════════════════════════
# RESULTS
# ══════════════════════════════════════════════════════════════════════
print("\nProcessing results...")

win_probs = pd.DataFrame([
    {"team": team, "win_probability": count / N_ITERATIONS * 100}
    for team, count in sorted(tournament_wins.items(), key=lambda x: -x[1])
])
# Add teams with 0 wins
for team in all_teams:
    if team not in win_probs["team"].values:
        win_probs = pd.concat([win_probs, pd.DataFrame([{"team": team, "win_probability": 0.0}])], ignore_index=True)

print("\nTournament Win Probabilities (top 20):")
for _, row in win_probs.head(20).iterrows():
    bar = "#" * int(row["win_probability"] * 2)
    print(f"  {row['team']:<28s} {row['win_probability']:5.1f}% {bar}")

adv_data = []
for team in all_teams:
    row = {"team": team}
    for rnd in ["Group", "R32", "R16", "QF", "SF", "Final", "Winner"]:
        row[rnd] = round_advancement[team].get(rnd, 0) / N_ITERATIONS * 100
    adv_data.append(row)

adv_df = pd.DataFrame(adv_data).sort_values("Winner", ascending=False)
adv_df.to_csv(SIM_DIR / "advancement_probabilities.csv", index=False)

print("\nAdvancement Probabilities (top 15):")
print(adv_df.head(15)[["team", "Group", "R32", "R16", "QF", "SF", "Final", "Winner"]].to_string(
    index=False, float_format="{:.1f}".format))

gs_data = []
for group in sorted(group_standings_dist.keys()):
    for team, positions in group_standings_dist[group].items():
        row = {"group": group, "team": team}
        for pos in [1, 2, 3, 4]:
            row[f"pos_{pos}"] = positions.get(pos, 0) / N_ITERATIONS * 100
        gs_data.append(row)

gs_df = pd.DataFrame(gs_data).sort_values(["group", "pos_1"], ascending=[True, False])
gs_df.to_csv(SIM_DIR / "group_standings_distribution.csv", index=False)

print("\nGroup Standings (% chance of each position):")
for group in sorted(gs_df["group"].unique()):
    print(f"\n  Group {group}:")
    grp = gs_df[gs_df["group"] == group].sort_values("pos_1", ascending=False)
    for _, row in grp.iterrows():
        print(f"    {row['team']:<28s} 1st:{row['pos_1']:5.1f}%  2nd:{row['pos_2']:5.1f}%  "
              f"3rd:{row['pos_3']:5.1f}%  4th:{row['pos_4']:5.1f}%")

print("\nMost Likely Finals:")
for matchup, count in sorted(final_matchups.items(), key=lambda x: -x[1])[:10]:
    pct = count / N_ITERATIONS * 100
    print(f"  {matchup[0]} vs {matchup[1]}: {pct:.2f}%")

win_probs.to_csv(SIM_DIR / "win_probabilities.csv", index=False)

print("\n" + "=" * 60)
print("SIMULATION COMPLETE")
print(f"All results saved to {SIM_DIR}")
print("=" * 60)
