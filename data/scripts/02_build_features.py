#!/usr/bin/env python3
"""
02_build_features.py — Build feature matrix from raw data sources.

Reads from data/raw/ (7 sources), computes ~35 features per match,
outputs data/processed/feature_matrix.csv.

Training window: 2004-01-01 to present. All team-specific features
are deltas (home minus away). No data leakage: every feature uses
only pre-match information.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict, deque
import bisect
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent.parent
RAW = BASE / "data" / "raw"
OUT = BASE / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

START_DATE = pd.Timestamp("2004-01-01")

# ── Tournament importance (CLAUDE.md spec) ───────────────────────────
IMPORTANCE = {
    "Friendly": 0.2,
    "FIFA World Cup qualification": 0.6,
    "UEFA Euro qualification": 0.6,
    "African Cup of Nations qualification": 0.6,
    "AFC Asian Cup qualification": 0.6,
    "UEFA Nations League": 0.6,
    "CONCACAF Nations League": 0.6,
    "CONMEBOL–UEFA Cup of Champions": 0.6,
    "FIFA World Cup": 1.0,
    "Confederations Cup": 0.8,
    "UEFA Euro": 0.8,
    "Copa América": 0.8,
    "African Cup of Nations": 0.8,
    "AFC Asian Cup": 0.8,
    "Gold Cup": 0.8,
    "CONCACAF Nations League Finals": 0.8,
}

# Elo K factors by tournament type
ELO_K = {
    "FIFA World Cup": 60,
    "UEFA Euro": 50,
    "Copa América": 50,
    "African Cup of Nations": 50,
    "AFC Asian Cup": 50,
    "Gold Cup": 50,
    "Confederations Cup": 50,
    "FIFA World Cup qualification": 40,
    "UEFA Euro qualification": 40,
    "African Cup of Nations qualification": 40,
    "AFC Asian Cup qualification": 40,
    "UEFA Nations League": 30,
    "CONCACAF Nations League": 30,
    "Friendly": 20,
}

# ── Name mappings (results.csv canonical → other sources) ────────────
FIFA_TO_CANONICAL = {
    "Korea Republic": "South Korea",
    "Korea DPR": "North Korea",
    "USA": "United States",
    "Turkey": "Turkiye",
    "IR Iran": "Iran",
    "Cape Verde Islands": "Cape Verde",
    "Congo DR": "DR Congo",
    "Côte d'Ivoire": "Ivory Coast",
    "Cote d'Ivoire": "Ivory Coast",
    "China PR": "China PR",
    "eSwatini": "Eswatini",
    "Brunei Darussalam": "Brunei",
    "Kyrgyz Republic": "Kyrgyzstan",
}

TM_TO_CANONICAL = {
    "Korea, South": "South Korea",
    "Korea, North": "North Korea",
    "Czech Republic": "Czechia",
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "Turkey": "Turkiye",
    "Cote d'Ivoire": "Ivory Coast",
    "Côte d'Ivoire": "Ivory Coast",
    "Congo DR": "DR Congo",
    "Congo, DR": "DR Congo",
    "Cabo Verde": "Cape Verde",
    "Curacao": "Curacao",
    "Curaçao": "Curacao",
}

WC_TO_CANONICAL = {
    "West Germany": "Germany",
    "Soviet Union": "Russia",
    "Yugoslavia": "Serbia",
    "Serbia and Montenegro": "Serbia",
    "Zaire": "DR Congo",
    "Dutch East Indies": "Indonesia",
    "China": "China PR",
    "Czechoslovakia": "Czechia",
}

CONFEDERATION_MAP = {
    "UEFA": "Europe",
    "CONMEBOL": "South America",
    "CAF": "Africa",
    "AFC": "Asia",
    "CONCACAF": "North America",
    "OFC": "Oceania",
}

# Country → continent (for home_continent_advantage)
COUNTRY_CONTINENT = {}
_europe = [
    "England", "France", "Germany", "Spain", "Italy", "Portugal",
    "Netherlands", "Belgium", "Croatia", "Switzerland", "Austria",
    "Scotland", "Sweden", "Norway", "Denmark", "Poland", "Czechia",
    "Turkiye", "Romania", "Greece", "Hungary", "Serbia", "Ukraine",
    "Russia", "Finland", "Iceland", "Wales", "Ireland", "Northern Ireland",
    "Bosnia and Herzegovina", "Albania", "North Macedonia", "Montenegro",
    "Slovenia", "Slovakia", "Bulgaria", "Lithuania", "Latvia", "Estonia",
    "Luxembourg", "Georgia", "Armenia", "Azerbaijan", "Cyprus", "Malta",
    "Kosovo", "Andorra", "Liechtenstein", "San Marino", "Gibraltar",
    "Faroe Islands", "Moldova", "Belarus",
]
_south_america = [
    "Brazil", "Argentina", "Uruguay", "Colombia", "Chile", "Ecuador",
    "Paraguay", "Peru", "Venezuela", "Bolivia", "Guyana", "Suriname",
]
_north_america = [
    "United States", "Mexico", "Canada", "Costa Rica", "Jamaica",
    "Honduras", "Panama", "El Salvador", "Haiti", "Trinidad and Tobago",
    "Guatemala", "Curacao", "Cuba", "Bermuda",
]
_africa = [
    "Morocco", "Egypt", "Senegal", "Nigeria", "Algeria", "Tunisia",
    "Cameroon", "Ghana", "Ivory Coast", "South Africa", "Mali",
    "DR Congo", "Cape Verde", "Burkina Faso", "Guinea",
]
_asia = [
    "Japan", "South Korea", "Iran", "Australia", "Saudi Arabia", "Qatar",
    "Iraq", "Jordan", "Uzbekistan", "China PR", "India", "Vietnam",
    "Thailand", "Indonesia", "United Arab Emirates", "Oman", "Bahrain",
    "Syria", "Palestine", "Lebanon", "Kuwait", "Yemen",
]
_oceania = ["New Zealand", "Fiji", "Papua New Guinea", "Tahiti"]

for _t in _europe:
    COUNTRY_CONTINENT[_t] = "Europe"
for _t in _south_america:
    COUNTRY_CONTINENT[_t] = "South America"
for _t in _north_america:
    COUNTRY_CONTINENT[_t] = "North America"
for _t in _africa:
    COUNTRY_CONTINENT[_t] = "Africa"
for _t in _asia:
    COUNTRY_CONTINENT[_t] = "Asia"
for _t in _oceania:
    COUNTRY_CONTINENT[_t] = "Oceania"

# Match host country to continent
HOST_CONTINENT = {
    "England": "Europe", "France": "Europe", "Germany": "Europe",
    "Spain": "Europe", "Italy": "Europe", "Portugal": "Europe",
    "Netherlands": "Europe", "Belgium": "Europe", "Switzerland": "Europe",
    "Austria": "Europe", "Sweden": "Europe", "Norway": "Europe",
    "Denmark": "Europe", "Poland": "Europe", "Scotland": "Europe",
    "Wales": "Europe", "Ireland": "Europe", "Turkey": "Europe",
    "Russia": "Europe", "Romania": "Europe", "Hungary": "Europe",
    "Greece": "Europe", "Czech Republic": "Europe", "Czechia": "Europe",
    "Croatia": "Europe", "Serbia": "Europe", "Bulgaria": "Europe",
    "Ukraine": "Europe", "Finland": "Europe", "Iceland": "Europe",
    "Cyprus": "Europe", "Georgia": "Europe",
    "Brazil": "South America", "Argentina": "South America",
    "Colombia": "South America", "Chile": "South America",
    "Peru": "South America", "Ecuador": "South America",
    "Uruguay": "South America", "Paraguay": "South America",
    "Venezuela": "South America", "Bolivia": "South America",
    "United States": "North America", "USA": "North America",
    "Mexico": "North America", "Canada": "North America",
    "Costa Rica": "North America", "Jamaica": "North America",
    "Honduras": "North America", "Panama": "North America",
    "Trinidad and Tobago": "North America",
    "Morocco": "Africa", "Egypt": "Africa", "South Africa": "Africa",
    "Nigeria": "Africa", "Cameroon": "Africa", "Ghana": "Africa",
    "Tunisia": "Africa", "Algeria": "Africa", "Senegal": "Africa",
    "Ivory Coast": "Africa", "Kenya": "Africa", "Ethiopia": "Africa",
    "Japan": "Asia", "South Korea": "Asia", "China PR": "Asia",
    "China": "Asia", "India": "Asia", "Qatar": "Asia",
    "United Arab Emirates": "Asia", "Saudi Arabia": "Asia",
    "Iran": "Asia", "Iraq": "Asia", "Thailand": "Asia",
    "Australia": "Oceania", "New Zealand": "Oceania",
}


# ══════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════


def load_results():
    """Load international match results, sorted by date. Drop unplayed matches."""
    pre = OUT / "sim_inputs" / "results.parquet"
    if pre.exists():
        return pd.read_parquet(pre)
    df = pd.read_csv(RAW / "international_results" / "results.csv")
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["home_score", "away_score"])
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    return df.sort_values("date").reset_index(drop=True)


def load_fifa_rankings():
    """Load FIFA rankings → (sorted_dates, {date: {team: {rank, points, conf}}})."""
    pre = OUT / "sim_inputs" / "fifa_rankings.parquet"
    if pre.exists():
        df = pd.read_parquet(pre)
        dates = sorted(df["rank_date"].unique())
        by_date = {}
        for d, grp in df.groupby("rank_date"):
            by_date[d] = {
                row["team"]: {
                    "rank": int(row["rank"]),
                    "points": float(row["points"]),
                    "conf": row["conf"] or "",
                }
                for _, row in grp.iterrows()
            }
        return dates, by_date

    df = pd.read_csv(RAW / "fifa_rankings" / "fifa_ranking.csv")
    df["rank_date"] = pd.to_datetime(df["rank_date"])
    df = df.dropna(subset=["rank", "total_points"])
    df["team"] = df["country_full"].map(lambda x: FIFA_TO_CANONICAL.get(x, x))

    dates = sorted(df["rank_date"].unique())
    by_date = {}
    for d, grp in df.groupby("rank_date"):
        by_date[d] = {
            row["team"]: {
                "rank": int(row["rank"]),
                "points": float(row["total_points"]),
                "conf": row.get("confederation", ""),
            }
            for _, row in grp.iterrows()
        }
    return dates, by_date


def load_players_and_valuations():
    """Load Transfermarkt players + valuations for squad/injury features."""
    pre_p = OUT / "sim_inputs" / "players.parquet"
    pre_v = OUT / "sim_inputs" / "valuations.parquet"
    if pre_p.exists() and pre_v.exists():
        return pd.read_parquet(pre_p), pd.read_parquet(pre_v)

    players = pd.read_csv(
        RAW / "transfermarkt" / "players.csv",
        usecols=[
            "player_id", "country_of_citizenship", "position",
            "sub_position", "date_of_birth",
        ],
    )
    players["team"] = (
        players["country_of_citizenship"]
        .map(lambda x: TM_TO_CANONICAL.get(x, x) if pd.notna(x) else None)
    )

    pos_map = {
        "Centre-Forward": "attack", "Left Winger": "attack",
        "Right Winger": "attack", "Second Striker": "attack",
        "Central Midfield": "mid", "Defensive Midfield": "mid",
        "Attacking Midfield": "mid", "Right Midfield": "mid",
        "Left Midfield": "mid",
        "Centre-Back": "def", "Right-Back": "def", "Left-Back": "def",
        "Goalkeeper": "def",
    }
    players["pos_cat"] = players["sub_position"].map(pos_map).fillna("mid")

    vals = pd.read_csv(
        RAW / "transfermarkt" / "player_valuations.csv",
        usecols=["player_id", "date", "market_value_in_eur"],
    )
    vals["date"] = pd.to_datetime(vals["date"])
    vals = vals.dropna(subset=["market_value_in_eur"])

    return players, vals


def precompute_squad_snapshots(players, vals, start):
    """
    Pre-compute quarterly squad value snapshots per national team.
    Returns:
        squad_agg: {(team, quarter): {total, top11, attack, mid, def, star, depth}}
        player_snap: {(team, quarter): {player_id: value}}
        top5_snap:   {(team, quarter): set(top5_player_ids)}
        quarters:    sorted list of quarter timestamps
    """
    print("  Joining players with valuations...")
    info = players[["player_id", "team", "pos_cat"]].dropna(subset=["team"])
    merged = vals.merge(info, on="player_id", how="inner").sort_values("date")

    quarters = list(
        pd.date_range(start=start - pd.DateOffset(months=6),
                       end=pd.Timestamp.now(), freq="QS")
    )

    squad_agg = {}
    player_snap = {}
    top5_snap = {}

    for q in tqdm(quarters, desc="  Quarterly squad values"):
        cutoff = q - pd.DateOffset(years=2)
        recent = merged[(merged["date"] <= q) & (merged["date"] >= cutoff)]
        latest = recent.sort_values("date").groupby("player_id").last().reset_index()

        for team, grp in latest.groupby("team"):
            v = grp["market_value_in_eur"].values
            p = grp["pos_cat"].values

            total = float(v.sum())
            top11 = float(np.sort(v)[-11:].sum()) if len(v) >= 11 else float(v.sum())
            star = float(v.max()) if len(v) > 0 else 0.0
            depth = len(v)

            squad_agg[(team, q)] = {
                "total": total,
                "top11": top11,
                "attack": float(v[p == "attack"].sum()),
                "mid": float(v[p == "mid"].sum()),
                "def": float(v[p == "def"].sum()),
                "star": star,
                "depth": depth,
            }

            # Per-player values (for injury lookups)
            pvals = dict(zip(grp["player_id"].values, grp["market_value_in_eur"].values))
            player_snap[(team, q)] = pvals

            # Top 5 most valuable players
            top_ids = grp.nlargest(5, "market_value_in_eur")["player_id"].values
            top5_snap[(team, q)] = set(top_ids)

    return squad_agg, player_snap, top5_snap, quarters


def load_injuries(players):
    """Load injury data, mapped to national teams via players.csv."""
    inj = pd.read_csv(RAW / "injuries" / "all_injuries.csv")
    inj["from_date"] = pd.to_datetime(inj["from_date"], errors="coerce")
    inj["end_date"] = pd.to_datetime(inj["end_date"], errors="coerce")
    inj = inj.dropna(subset=["from_date", "end_date"])

    team_map = players.set_index("player_id")["team"].to_dict()
    inj["team"] = inj["player_id"].map(team_map)
    inj = inj.dropna(subset=["team"])

    # Index by team for fast lookup
    team_injuries = defaultdict(list)
    for _, row in inj.iterrows():
        team_injuries[row["team"]].append(
            (row["from_date"], row["end_date"], row["player_id"])
        )

    # Sort each team's injuries by from_date
    for team in team_injuries:
        team_injuries[team].sort(key=lambda x: x[0])

    return dict(team_injuries)


def load_wc_history():
    """Build World Cup track record per team."""
    pre = OUT / "sim_inputs" / "wc_history.json"
    if pre.exists():
        with open(pre, encoding="utf-8") as f:
            return json.load(f)

    matches = pd.read_csv(RAW / "world_cup" / "matches.csv")
    standings = pd.read_csv(RAW / "world_cup" / "group_standings.csv")

    wc = {}
    all_teams = set(matches["home_team_name"]) | set(matches["away_team_name"])

    for team in all_teams:
        tm = matches[
            (matches["home_team_name"] == team) | (matches["away_team_name"] == team)
        ]
        tournaments = tm["tournament_id"].nunique()
        n_games = len(tm)

        # Goals
        hg = tm.loc[tm["home_team_name"] == team, "home_team_score"].sum()
        ag = tm.loc[tm["away_team_name"] == team, "away_team_score"].sum()
        gpg = (hg + ag) / n_games if n_games > 0 else 0

        # Knockout rate
        ts = standings[standings["team_name"] == team]
        adv = int(ts["advanced"].sum()) if len(ts) > 0 else 0
        ko_rate = adv / tournaments if tournaments > 0 else 0

        # Best finish
        best = 2 if tournaments > 0 else 1  # 2=group stage, 1=never qualified
        ko = tm[tm["knockout_stage"] == 1]
        stage_map = {
            "final": 7, "third": 5, "semi": 5, "quarter": 4,
            "round of 16": 3, "second round": 3,
        }
        for stage_key, score in stage_map.items():
            if any(ko["stage_name"].str.contains(stage_key, case=False, na=False)):
                best = max(best, score)

        # Check if won the final
        finals = ko[ko["stage_name"].str.lower() == "final"]
        for _, f in finals.iterrows():
            if (f["home_team_name"] == team and f["home_team_win"] == 1) or \
               (f["away_team_name"] == team and f["away_team_win"] == 1):
                best = 7  # winner
            else:
                best = max(best, 6)  # runner-up

        wc[team] = {
            "appearances": tournaments,
            "knockout_rate": ko_rate,
            "best_finish": best,
            "goals_per_game": gpg,
        }

    # Map historical names to canonical
    mapped = {}
    for team, data in wc.items():
        canonical = WC_TO_CANONICAL.get(team, team)
        if canonical in mapped:
            # Merge (e.g., West Germany + Germany)
            existing = mapped[canonical]
            existing["appearances"] += data["appearances"]
            total_games = existing.get("_games", 0) + (
                data["goals_per_game"] * data["appearances"]
            )
            existing["goals_per_game"] = (
                total_games / existing["appearances"]
                if existing["appearances"] > 0
                else 0
            )
            existing["knockout_rate"] = max(
                existing["knockout_rate"], data["knockout_rate"]
            )
            existing["best_finish"] = max(
                existing["best_finish"], data["best_finish"]
            )
        else:
            mapped[canonical] = data.copy()

    return mapped


# ══════════════════════════════════════════════════════════════════════
# ELO COMPUTATION (from scratch for full coverage)
# ══════════════════════════════════════════════════════════════════════


def compute_elo_ratings(results):
    """
    Compute Elo ratings from scratch for all teams.
    Returns elo_history: {team: [(date, elo_before_match), ...]} sorted by date.
    """
    elo = defaultdict(lambda: 1500.0)
    history = defaultdict(list)

    for _, row in tqdm(results.iterrows(), total=len(results),
                       desc="  Computing Elo ratings"):
        home, away = row["home_team"], row["away_team"]
        date = row["date"]
        hs, aws = row["home_score"], row["away_score"]

        # Record pre-match rating
        history[home].append((date, elo[home]))
        history[away].append((date, elo[away]))

        # Expected score (home gets +100 advantage)
        dr = (elo[home] + 100) - elo[away]
        we_home = 1.0 / (10 ** (-dr / 400) + 1)

        # Actual score
        if hs > aws:
            w_home, w_away = 1.0, 0.0
        elif hs < aws:
            w_home, w_away = 0.0, 1.0
        else:
            w_home, w_away = 0.5, 0.5

        # K factor × goal-difference multiplier
        tourn = row.get("tournament", "Friendly")
        K = ELO_K.get(tourn, 20)
        gd = abs(hs - aws)
        if gd <= 1:
            gd_mult = 1.0
        elif gd == 2:
            gd_mult = 1.5
        elif gd == 3:
            gd_mult = 1.75
        else:
            gd_mult = 1.75 + (gd - 3) / 8.0
        K *= gd_mult

        elo[home] += K * (w_home - we_home)
        elo[away] += K * (w_away - (1 - we_home))

    return dict(history)


# ══════════════════════════════════════════════════════════════════════
# FEATURE LOOKUPS
# ══════════════════════════════════════════════════════════════════════


def _bisect_lookup(sorted_pairs, date):
    """Find most recent value in [(date, val), ...] before `date`."""
    idx = bisect.bisect_right([d for d, _ in sorted_pairs], date) - 1
    return sorted_pairs[idx][1] if idx >= 0 else None


def get_elo(team, date, elo_hist):
    if team not in elo_hist:
        return None
    return _bisect_lookup(elo_hist[team], date)


def get_elo_momentum(team, date, elo_hist, n=5):
    if team not in elo_hist or len(elo_hist[team]) == 0:
        return None
    pairs = elo_hist[team]
    idx = bisect.bisect_right([d for d, _ in pairs], date) - 1
    if idx < 0:
        return None
    current = pairs[idx][1]
    past_idx = max(0, idx - n)
    return current - pairs[past_idx][1]


def get_fifa(team, date, fifa_dates, fifa_by_date):
    idx = bisect.bisect_right(fifa_dates, date) - 1
    if idx < 0:
        return None, None, None
    data = fifa_by_date[fifa_dates[idx]]
    if team in data:
        d = data[team]
        return d["rank"], d["points"], d["conf"]
    return None, None, None


def get_squad(team, date, squad_agg, quarters):
    idx = bisect.bisect_right(quarters, date) - 1
    if idx < 0:
        return None
    return squad_agg.get((team, quarters[idx]))


def get_nearest_quarter(date, quarters):
    idx = bisect.bisect_right(quarters, date) - 1
    return quarters[max(0, idx)] if quarters else None


def get_injuries_at_date(team, date, team_injuries, player_snap, top5_snap,
                         squad_agg, quarters):
    """
    Compute injury features for a team at a given date.
    Returns: (count, burden, star_flag)
    """
    records = team_injuries.get(team, [])
    if not records:
        return 0, 0.0, 0

    # Find active injuries: from_date <= date <= end_date
    active_pids = []
    for from_d, end_d, pid in records:
        if from_d > date:
            break  # sorted by from_date, no more can match
        if end_d >= date:
            active_pids.append(pid)

    if not active_pids:
        return 0, 0.0, 0

    q = get_nearest_quarter(date, quarters)
    key = (team, q)
    pv = player_snap.get(key, {})
    sa = squad_agg.get(key, {})
    t5 = top5_snap.get(key, set())

    total_squad = sa.get("total", 1)  # avoid div by zero
    injured_value = sum(pv.get(pid, 0) for pid in active_pids)
    burden = injured_value / total_squad if total_squad > 0 else 0
    star_flag = 1 if any(pid in t5 for pid in active_pids) else 0

    return len(active_pids), burden, star_flag


def log_transform(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    return np.sign(x) * np.log1p(abs(x))


# ══════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════


def main():
    print("=" * 60)
    print("  FEATURE ENGINEERING PIPELINE")
    print("=" * 60)

    # ── Load data ────────────────────────────────────────────────
    print("\n[1/7] Loading match results...")
    results = load_results()
    print(f"  {len(results)} total matches")
    print(f"  {len(results[results['date'] >= START_DATE])} from 2004")

    print("\n[2/7] Computing Elo ratings (all teams, from scratch)...")
    elo_hist = compute_elo_ratings(results)
    print(f"  {len(elo_hist)} teams tracked")

    print("\n[3/7] Loading FIFA rankings...")
    fifa_dates, fifa_by_date = load_fifa_rankings()
    print(f"  {len(fifa_dates)} ranking snapshots")

    print("\n[4/7] Loading Transfermarkt players + valuations...")
    players, vals = load_players_and_valuations()
    print(f"  {len(players)} players, {len(vals)} valuations")

    print("  Pre-computing quarterly squad snapshots...")
    squad_agg, player_snap, top5_snap, quarters = precompute_squad_snapshots(
        players, vals, START_DATE
    )
    print(f"  {len(quarters)} quarters, {len(squad_agg)} team-quarter combos")

    print("\n[5/7] Loading injury data...")
    team_injuries = load_injuries(players)
    inj_count = sum(len(v) for v in team_injuries.values())
    print(f"  {inj_count} injury records across {len(team_injuries)} teams")

    print("\n[6/7] Loading World Cup history...")
    wc_data = load_wc_history()
    print(f"  {len(wc_data)} teams with WC history")

    # ── Build confederations lookup from FIFA data ───────────────
    team_conf = {}
    for dd in fifa_by_date.values():
        for t, d in dd.items():
            if d.get("conf"):
                team_conf[t] = d["conf"]

    # ── Process matches ──────────────────────────────────────────
    print("\n[7/7] Computing features for each match...")

    # Rolling trackers (updated for ALL matches, features saved only 2004+)
    form = defaultdict(lambda: deque(maxlen=10))
    last_match = {}
    h2h = defaultdict(list)
    tournament_record = defaultdict(lambda: defaultdict(lambda: [0, 0]))  # team -> tourn -> [wins, total]

    features = []

    for _, row in tqdm(results.iterrows(), total=len(results),
                       desc="  Processing"):
        date = row["date"]
        home = row["home_team"]
        away = row["away_team"]
        hs = int(row["home_score"])
        aws = int(row["away_score"])
        tourn = row["tournament"]
        neutral = row.get("neutral", False)
        country = row.get("country", "")
        imp = IMPORTANCE.get(tourn, 0.3)

        # Determine result
        if hs > aws:
            h_result, a_result = "win", "loss"
        elif hs < aws:
            h_result, a_result = "loss", "win"
        else:
            h_result, a_result = "draw", "draw"

        # ── Only save features for 2004+ matches ────────────────
        if date >= START_DATE:
            f = {
                "date": date,
                "home_team": home,
                "away_team": away,
                "home_score": hs,
                "away_score": aws,
                "tournament": tourn,
            }

            # TARGET
            f["result"] = 0 if hs > aws else (1 if hs == aws else 2)

            # CONTEXT
            f["match_importance"] = imp
            f["is_neutral"] = int(bool(neutral))

            h_conf = team_conf.get(home, "")
            a_conf = team_conf.get(away, "")
            f["same_confederation"] = int(h_conf != "" and h_conf == a_conf)

            # Home continent advantage
            host_cont = HOST_CONTINENT.get(country, "")
            home_cont = CONFEDERATION_MAP.get(h_conf, "")
            f["home_continent_advantage"] = int(
                host_cont != "" and host_cont == home_cont and not neutral
            )

            # ELO
            h_elo = get_elo(home, date, elo_hist)
            a_elo = get_elo(away, date, elo_hist)
            f["elo_diff"] = (h_elo - a_elo) if (
                h_elo is not None and a_elo is not None
            ) else np.nan

            h_mom = get_elo_momentum(home, date, elo_hist)
            a_mom = get_elo_momentum(away, date, elo_hist)
            f["elo_momentum_diff"] = (h_mom - a_mom) if (
                h_mom is not None and a_mom is not None
            ) else np.nan

            # FIFA RANKINGS
            h_rank, h_pts, _ = get_fifa(home, date, fifa_dates, fifa_by_date)
            a_rank, a_pts, _ = get_fifa(away, date, fifa_dates, fifa_by_date)
            f["rank_diff"] = (
                (h_rank - a_rank) if (h_rank and a_rank) else np.nan
            )
            f["points_diff"] = (
                (h_pts - a_pts) if (h_pts and a_pts) else np.nan
            )

            # SQUAD VALUES
            h_sq = get_squad(home, date, squad_agg, quarters)
            a_sq = get_squad(away, date, squad_agg, quarters)
            if h_sq and a_sq:
                for key in ["total", "top11", "attack", "mid", "def"]:
                    f[f"squad_value_{key}_delta"] = log_transform(
                        h_sq[key] - a_sq[key]
                    )
                f["star_player_value_delta"] = log_transform(
                    h_sq["star"] - a_sq["star"]
                )
                f["squad_depth_delta"] = log_transform(
                    h_sq["depth"] - a_sq["depth"]
                )
            else:
                for col in [
                    "squad_value_total_delta", "squad_value_top11_delta",
                    "squad_value_attack_delta", "squad_value_mid_delta",
                    "squad_value_def_delta", "star_player_value_delta",
                    "squad_depth_delta",
                ]:
                    f[col] = np.nan

            # FORM (rolling 10 games)
            for prefix, team in [("home", home), ("away", away)]:
                team_form = form[team]
                if len(team_form) > 0:
                    wins = sum(1 for m in team_form if m["r"] == "win")
                    f[f"__{prefix}_wr"] = wins / len(team_form)

                    # Weighted form: exponential decay × importance
                    wts = []
                    for i, m in enumerate(reversed(list(team_form))):
                        decay = 0.9 ** i
                        score = 1.0 if m["r"] == "win" else (
                            0.5 if m["r"] == "draw" else 0.0
                        )
                        wts.append(decay * m["imp"] * score)
                    f[f"__{prefix}_wf"] = np.mean(wts) if wts else np.nan

                    gds = [m["gf"] - m["ga"] for m in team_form]
                    f[f"__{prefix}_gd"] = np.mean(gds)
                else:
                    f[f"__{prefix}_wr"] = np.nan
                    f[f"__{prefix}_wf"] = np.nan
                    f[f"__{prefix}_gd"] = np.nan

                # Days rest
                if team in last_match:
                    f[f"{prefix}_days_rest"] = (date - last_match[team]).days
                else:
                    f[f"{prefix}_days_rest"] = np.nan

            # Convert form to deltas
            f["form_win_rate_diff"] = (
                f.pop("__home_wr", np.nan) - f.pop("__away_wr", np.nan)
            )
            f["form_weighted_diff"] = (
                f.pop("__home_wf", np.nan) - f.pop("__away_wf", np.nan)
            )
            f["goal_diff_delta"] = (
                f.pop("__home_gd", np.nan) - f.pop("__away_gd", np.nan)
            )

            # H2H
            pair = tuple(sorted([home, away]))
            hist = [h for h in h2h[pair] if h["date"] < date]
            if hist:
                h_wins = sum(1 for h in hist if h["winner"] == home)
                draws = sum(1 for h in hist if h["winner"] is None)
                f["h2h_home_win_rate"] = h_wins / len(hist)
                f["h2h_draw_rate"] = draws / len(hist)
                f["h2h_matches_played"] = len(hist)
            else:
                f["h2h_home_win_rate"] = np.nan
                f["h2h_draw_rate"] = np.nan
                f["h2h_matches_played"] = 0

            # TOURNAMENT WIN RATE
            h_trec = tournament_record[home][tourn]
            a_trec = tournament_record[away][tourn]
            h_twr = h_trec[0] / h_trec[1] if h_trec[1] > 0 else np.nan
            a_twr = a_trec[0] / a_trec[1] if a_trec[1] > 0 else np.nan
            if not np.isnan(h_twr) and not np.isnan(a_twr):
                f["tournament_wr_delta"] = h_twr - a_twr
            else:
                f["tournament_wr_delta"] = np.nan

            # WC HISTORY
            h_wc = wc_data.get(home, {})
            a_wc = wc_data.get(away, {})
            for key, default in [
                ("appearances", 0), ("knockout_rate", 0),
                ("best_finish", 1), ("goals_per_game", 0),
            ]:
                f[f"wc_{key}_diff"] = (
                    h_wc.get(key, default) - a_wc.get(key, default)
                )

            # INJURIES
            h_cnt, h_burden, h_star = get_injuries_at_date(
                home, date, team_injuries, player_snap, top5_snap,
                squad_agg, quarters,
            )
            a_cnt, a_burden, a_star = get_injuries_at_date(
                away, date, team_injuries, player_snap, top5_snap,
                squad_agg, quarters,
            )
            f["injury_count_delta"] = h_cnt - a_cnt
            f["injury_burden_delta"] = h_burden - a_burden
            f["star_injury_flag"] = h_star - a_star

            features.append(f)

        # ── Update rolling trackers (ALL matches, for full history) ──
        form[home].append({"r": h_result, "gf": hs, "ga": aws, "imp": imp})
        form[away].append({"r": a_result, "gf": aws, "ga": hs, "imp": imp})
        last_match[home] = date
        last_match[away] = date

        pair = tuple(sorted([home, away]))
        winner = home if hs > aws else (away if aws > hs else None)
        h2h[pair].append({"date": date, "winner": winner})

        tournament_record[home][tourn][1] += 1
        tournament_record[away][tourn][1] += 1
        if h_result == "win":
            tournament_record[home][tourn][0] += 1
        if a_result == "win":
            tournament_record[away][tourn][0] += 1

    # ── Build DataFrame ──────────────────────────────────────────
    df = pd.DataFrame(features)

    # ── Coverage report ──────────────────────────────────────────
    meta_cols = {"date", "home_team", "away_team", "tournament",
                 "home_score", "away_score"}
    feature_cols = sorted(c for c in df.columns if c not in meta_cols)

    print("\n" + "=" * 60)
    print("  FEATURE COVERAGE (% non-null)")
    print("=" * 60)
    for col in feature_cols:
        pct = df[col].notna().mean() * 100
        bar = "█" * int(pct // 5) + "░" * (20 - int(pct // 5))
        print(f"  {col:35s} {bar} {pct:5.1f}%")

    # ── Class distribution ───────────────────────────────────────
    print("\n  Target distribution:")
    for label, name in [(0, "Home Win"), (1, "Draw"), (2, "Away Win")]:
        count = (df["result"] == label).sum()
        pct = count / len(df) * 100
        print(f"    {name:10s}: {count:6d} ({pct:.1f}%)")

    # ── Date range ───────────────────────────────────────────────
    print(f"\n  Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"  Total rows: {len(df)}")

    # ── Save ─────────────────────────────────────────────────────
    out_path = OUT / "feature_matrix.csv"
    df.to_csv(out_path, index=False)
    print(f"\n  Saved to {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
