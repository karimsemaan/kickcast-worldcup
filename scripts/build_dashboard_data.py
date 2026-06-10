#!/usr/bin/env python3
"""Pre-compute all 1128 team matchups + tournament data for the static Vercel dashboard."""
import sys, json
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "data" / "scripts"))

import numpy as np
import pandas as pd
import joblib
from collections import defaultdict, deque
from importlib import import_module
from src.simulation import historical_name

fb = import_module("02_build_features")

# Load model + data
model = joblib.load(BASE / "outputs/model_artifacts/CatBoost_tuned_balanced.joblib")
groups = pd.read_csv(BASE / "data/raw/world_cup_2026/groups.csv")
feature_cols = pd.read_csv(BASE / "data/processed/splits/X_train.csv", nrows=0).columns.tolist()
win_probs = pd.read_csv(BASE / "outputs/simulation_results/win_probabilities.csv")
adv_probs = pd.read_csv(BASE / "outputs/simulation_results/advancement_probabilities.csv")
match_preds = pd.read_csv(BASE / "outputs/simulation_results/group_match_predictions.csv")
gs_dist = pd.read_csv(BASE / "outputs/simulation_results/group_standings_distribution.csv")

all_teams = sorted(groups["team"].tolist())
team_confed = dict(zip(groups["team"], groups["confederation"]))
team_group = dict(zip(groups["team"], groups["group"]))

print("Loading raw data...")
results = fb.load_results()
elo_hist = fb.compute_elo_ratings(results)
fifa_dates, fifa_by_date = fb.load_fifa_rankings()
players, vals = fb.load_players_and_valuations()
squad_agg, _, _, quarters = fb.precompute_squad_snapshots(players, vals, pd.Timestamp("2004-01-01"))
wc_data = fb.load_wc_history()
IMPORTANCE = fb.IMPORTANCE
ref_date = pd.Timestamp("2026-06-10")

# Build form + H2H
form_tracker = defaultdict(lambda: deque(maxlen=10))
last_match_date = {}
h2h_db = defaultdict(list)
for _, row in results.iterrows():
    home, away = row["home_team"], row["away_team"]
    hs, aws = int(row["home_score"]), int(row["away_score"])
    date = row["date"]
    imp = IMPORTANCE.get(row["tournament"], 0.3)
    h_r = "win" if hs > aws else ("draw" if hs == aws else "loss")
    a_r = "win" if aws > hs else ("draw" if hs == aws else "loss")
    form_tracker[home].append({"r": h_r, "gf": hs, "ga": aws, "imp": imp})
    form_tracker[away].append({"r": a_r, "gf": aws, "ga": hs, "imp": imp})
    last_match_date[home] = date
    last_match_date[away] = date
    pair = tuple(sorted([home, away]))
    winner = home if hs > aws else (away if aws > hs else None)
    h2h_db[pair].append({"winner": winner})

team_conf = {}
for dd in fifa_by_date.values():
    for t, d in dd.items():
        if d.get("conf"):
            team_conf[t] = d["conf"]
for _, row in groups.iterrows():
    team_conf[row["team"]] = row["confederation"]

log_t = lambda x: float(np.sign(x) * np.log1p(abs(x))) if x is not None and not (isinstance(x, float) and np.isnan(x)) else 0.0

# Team profiles
print("Building team profiles...")
profiles = {}
for team in all_teams:
    hn = historical_name(team)
    ed = elo_hist.get(hn) or elo_hist.get(team)
    elo = ed[-1][1] if ed else 1500
    mom = fb.get_elo_momentum(hn, ref_date, elo_hist, n=5) or 0
    r, p, _ = fb.get_fifa(hn, ref_date, fifa_dates, fifa_by_date)
    if r is None:
        r, p, _ = fb.get_fifa(team, ref_date, fifa_dates, fifa_by_date)
    sq = fb.get_squad(hn, ref_date, squad_agg, quarters) or fb.get_squad(team, ref_date, squad_agg, quarters)
    wc = wc_data.get(team, wc_data.get(hn, {}))
    tf = form_tracker.get(hn) or form_tracker.get(team)
    wr, wf, gd = 0.5, 0.25, 0.0
    if tf and len(tf) > 0:
        wins = sum(1 for m in tf if m["r"] == "win")
        wr = wins / len(tf)
        wts = [0.9**i * m["imp"] * (1.0 if m["r"]=="win" else (0.5 if m["r"]=="draw" else 0.0))
               for i, m in enumerate(reversed(list(tf)))]
        wf = float(np.mean(wts))
        gd = float(np.mean([m["gf"]-m["ga"] for m in tf]))
    lm = last_match_date.get(hn) or last_match_date.get(team)
    rest = (ref_date - lm).days if lm else 30
    profiles[team] = {
        "elo": round(elo), "rank": r or 100, "points": round(p or 1000, 1),
        "squad": round(sq["total"]/1e6, 1) if sq else 0,
        "form_wr": round(wr, 3), "form_gd": round(gd, 2), "rest": rest,
        "wc_apps": wc.get("appearances", 0), "wc_best": wc.get("best_finish", 1),
        "conf": team_confed.get(team, ""), "group": team_group.get(team, ""),
    }

# All matchups
print("Computing 1128 matchups...")
matchups = {}
h2h_records = {}

for i, ta in enumerate(all_teams):
    for tb in all_teams[i+1:]:
        hn_a, hn_b = historical_name(ta), historical_name(tb)
        pa, pb = profiles[ta], profiles[tb]

        f = {"match_importance": 1.0, "is_neutral": 1, "home_continent_advantage": 0}
        f["same_confederation"] = int(pa["conf"] != "" and pa["conf"] == pb["conf"])

        ea = elo_hist.get(hn_a, elo_hist.get(ta, []))
        eb = elo_hist.get(hn_b, elo_hist.get(tb, []))
        h_elo = ea[-1][1] if ea else 1500
        a_elo = eb[-1][1] if eb else 1500
        f["elo_diff"] = h_elo - a_elo
        f["elo_momentum_diff"] = (fb.get_elo_momentum(hn_a, ref_date, elo_hist, n=5) or 0) - (fb.get_elo_momentum(hn_b, ref_date, elo_hist, n=5) or 0)
        f["rank_diff"] = pa["rank"] - pb["rank"]
        f["points_diff"] = pa["points"] - pb["points"]

        h_sq = fb.get_squad(hn_a, ref_date, squad_agg, quarters) or fb.get_squad(ta, ref_date, squad_agg, quarters)
        a_sq = fb.get_squad(hn_b, ref_date, squad_agg, quarters) or fb.get_squad(tb, ref_date, squad_agg, quarters)
        if h_sq and a_sq:
            for k in ["total","top11","attack","mid","def"]:
                f[f"squad_value_{k}_delta"] = log_t(h_sq[k] - a_sq[k])
            f["star_player_value_delta"] = log_t(h_sq["star"] - a_sq["star"])
            f["squad_depth_delta"] = log_t(h_sq["depth"] - a_sq["depth"])
        else:
            for c in ["squad_value_total_delta","squad_value_top11_delta","squad_value_attack_delta",
                       "squad_value_mid_delta","squad_value_def_delta","star_player_value_delta","squad_depth_delta"]:
                f[c] = 0.0

        f["home_days_rest"] = pa["rest"]
        f["away_days_rest"] = pb["rest"]
        f["form_win_rate_diff"] = pa["form_wr"] - pb["form_wr"]
        f["form_weighted_diff"] = 0.0
        f["goal_diff_delta"] = pa["form_gd"] - pb["form_gd"]

        pair = tuple(sorted([hn_a, hn_b]))
        hist = h2h_db.get(pair, h2h_db.get(tuple(sorted([ta, tb])), []))
        if hist:
            a_w = sum(1 for h in hist if h["winner"]==hn_a or h["winner"]==ta)
            b_w = sum(1 for h in hist if h["winner"]==hn_b or h["winner"]==tb)
            d = len(hist) - a_w - b_w
            f["h2h_home_win_rate"] = a_w / len(hist)
            f["h2h_draw_rate"] = d / len(hist)
            f["h2h_matches_played"] = len(hist)
            h2h_records[f"{ta}|{tb}"] = [a_w, d, b_w, len(hist)]
        else:
            f["h2h_home_win_rate"] = 0.0
            f["h2h_draw_rate"] = 0.0
            f["h2h_matches_played"] = 0

        f["tournament_wr_delta"] = 0.0
        h_wc = wc_data.get(ta, wc_data.get(hn_a, {}))
        a_wc = wc_data.get(tb, wc_data.get(hn_b, {}))
        for k, dv in [("appearances",0),("knockout_rate",0),("best_finish",1),("goals_per_game",0)]:
            f[f"wc_{k}_diff"] = h_wc.get(k, dv) - a_wc.get(k, dv)
        f["injury_count_delta"] = 0
        f["injury_burden_delta"] = 0.0
        f["star_injury_flag"] = 0

        # Symmetric prediction
        X_ab = pd.DataFrame([f]).reindex(columns=feature_cols).fillna(0)
        f_rev = dict(f)
        for c in feature_cols:
            if "diff" in c or "delta" in c or c == "star_injury_flag":
                f_rev[c] = -f.get(c, 0)
            if c == "h2h_home_win_rate":
                f_rev[c] = -f.get(c, 0)
        f_rev["home_days_rest"], f_rev["away_days_rest"] = f["away_days_rest"], f["home_days_rest"]
        X_ba = pd.DataFrame([f_rev]).reindex(columns=feature_cols).fillna(0)

        p_ab = model.predict_proba(X_ab)[0]
        p_ba = model.predict_proba(X_ba)[0]
        p_a = (p_ab[0] + p_ba[2]) / 2
        p_d = (p_ab[1] + p_ba[1]) / 2
        p_b = (p_ab[2] + p_ba[0]) / 2
        tot = p_a + p_d + p_b

        matchups[f"{ta}|{tb}"] = [round(p_a/tot*100, 1), round(p_d/tot*100, 1), round(p_b/tot*100, 1)]

# Output
out = {
    "teams": all_teams,
    "profiles": profiles,
    "matchups": matchups,
    "h2h": h2h_records,
    "winProbs": dict(zip(win_probs["team"], [round(x, 2) for x in win_probs["win_probability"]])),
    "advancement": {row["team"]: {c: round(row[c], 1) for c in ["Group","R32","R16","QF","SF","Final","Winner"]} for _, row in adv_probs.iterrows()},
    "groupMatches": [{k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()} for _, row in match_preds.iterrows()],
    "standings": [{k: (round(v, 1) if isinstance(v, float) else v) for k, v in row.items()} for _, row in gs_dist.iterrows()],
}

outpath = BASE / "public" / "data.json"
with open(outpath, "w") as fp:
    json.dump(out, fp, separators=(",", ":"))

print(f"Saved {outpath} ({outpath.stat().st_size // 1024} KB)")
print(f"{len(matchups)} matchups, {len(all_teams)} teams")
