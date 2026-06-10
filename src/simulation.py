"""
simulation.py — 2026 FIFA World Cup Monte Carlo Simulation.

Core functions for:
- Building prediction feature vectors for 2026 matches
- Running group stage simulations
- Knockout bracket resolution (with penalty shootout model)
- 3rd place advancement logic
- Full tournament Monte Carlo
"""

import numpy as np
import pandas as pd
from collections import defaultdict


# ── Team Name Mapping ─────────────────────────────────────────────────
# WC 2026 official names → names used in historical results.csv / Elo data.
# These three teams changed names or use diacriticals not in the match database.
WC_TO_HISTORICAL = {
    "Turkiye": "Turkey",
    "Czechia": "Czech Republic",
    "Curacao": "Curaçao",
}

# Reverse mapping: historical name → WC display name
HISTORICAL_TO_WC = {v: k for k, v in WC_TO_HISTORICAL.items()}


def historical_name(team):
    """Convert WC display name to the name used in historical data."""
    return WC_TO_HISTORICAL.get(team, team)


def display_name(team):
    """Convert historical data name to WC display name."""
    return HISTORICAL_TO_WC.get(team, team)


# ── 3rd Place Advancement Mapping ──────────────────────────────────────
# FIFA's predetermined mapping of which 3rd-place teams go into which R32 slots,
# depending on which groups they come from. There are many valid combinations.
# We simplify: rank all 12 third-place teams, take top 8, slot them by group letter.

THIRD_PLACE_SLOTS = {
    # R32 match -> list of possible source group letters for 3rd place
    79: ["A", "C", "E", "F", "H", "I"],   # 1A vs 3rd
    75: ["A", "B", "C", "D", "F"],         # 1E vs 3rd
    78: ["C", "D", "F", "G", "H"],         # 1I vs 3rd
    80: ["E", "H", "I", "J", "K"],         # 1L vs 3rd
    81: ["A", "E", "H", "I", "J"],         # 1G vs 3rd
    82: ["B", "E", "F", "I", "J"],         # 1D vs 3rd
    85: ["E", "F", "G", "I", "J"],         # 1B vs 3rd
    88: ["D", "E", "I", "J", "L"],         # 1K vs 3rd
}


def simulate_match_goals(p_home, p_draw, p_away, rng):
    """Sample a match outcome and approximate goals using Poisson.

    Returns (home_goals, away_goals).
    """
    r = rng.random()
    if r < p_home:
        # Home win — sample goals where home > away
        home_g = rng.poisson(1.8)
        away_g = rng.poisson(0.8)
        if home_g <= away_g:
            home_g = away_g + 1
        return home_g, away_g
    elif r < p_home + p_draw:
        # Draw
        goals = rng.poisson(1.0)
        return goals, goals
    else:
        # Away win
        home_g = rng.poisson(0.8)
        away_g = rng.poisson(1.8)
        if away_g <= home_g:
            away_g = home_g + 1
        return home_g, away_g


def simulate_group_stage(fixtures_df, match_probas, groups_df, rng):
    """Simulate all 72 group stage matches.

    Args:
        fixtures_df: DataFrame with match_number, group, home_team, away_team
        match_probas: dict of match_number -> (p_home, p_draw, p_away)
        groups_df: DataFrame with group, team, confederation columns
        rng: numpy random generator

    Returns:
        standings: dict of group -> list of (team, points, gd, gf, team_rank_for_tiebreak)
    """
    # Initialize team stats
    team_stats = {}
    for _, row in groups_df.iterrows():
        team_stats[row["team"]] = {
            "group": row["group"], "points": 0, "gf": 0, "ga": 0,
            "gd": 0, "fifa_rank": row.get("pot", 4)  # approximate
        }

    # Play each match
    for _, match in fixtures_df.iterrows():
        mn = match["match_number"]
        home = match["home_team"]
        away = match["away_team"]
        ph, pd_, pa = match_probas[mn]

        hg, ag = simulate_match_goals(ph, pd_, pa, rng)

        team_stats[home]["gf"] += hg
        team_stats[home]["ga"] += ag
        team_stats[home]["gd"] += (hg - ag)
        team_stats[away]["gf"] += ag
        team_stats[away]["ga"] += hg
        team_stats[away]["gd"] += (ag - hg)

        if hg > ag:
            team_stats[home]["points"] += 3
        elif hg == ag:
            team_stats[home]["points"] += 1
            team_stats[away]["points"] += 1
        else:
            team_stats[away]["points"] += 3

    # Build standings per group
    standings = defaultdict(list)
    for team, stats in team_stats.items():
        standings[stats["group"]].append({
            "team": team, "points": stats["points"],
            "gd": stats["gd"], "gf": stats["gf"],
            "fifa_rank": stats["fifa_rank"]
        })

    # Sort each group: points > GD > GF > FIFA ranking (lower is better)
    for group in standings:
        standings[group].sort(
            key=lambda x: (-x["points"], -x["gd"], -x["gf"], x["fifa_rank"])
        )

    return standings


def get_advancing_teams(standings):
    """Determine the 32 teams advancing from group stage.

    Top 2 per group (24 teams) + 8 best 3rd-place teams.

    Returns:
        advancing: dict mapping position codes (e.g. '1A', '2A', '3A') to team names
        third_place_groups: list of group letters from which 3rd-place teams advance
    """
    advancing = {}

    # Top 2 per group
    for group, teams in standings.items():
        advancing[f"1{group}"] = teams[0]["team"]
        advancing[f"2{group}"] = teams[1]["team"]

    # Collect all 3rd-place teams
    third_place = []
    for group, teams in standings.items():
        t = teams[2]
        t["group"] = group
        third_place.append(t)

    # Rank 3rd-place teams
    third_place.sort(key=lambda x: (-x["points"], -x["gd"], -x["gf"], x["fifa_rank"]))

    # Top 8 advance
    advancing_3rd = third_place[:8]
    third_place_groups = sorted([t["group"] for t in advancing_3rd])

    for t in advancing_3rd:
        advancing[f"3{t['group']}"] = t["team"]

    return advancing, third_place_groups


def slot_third_place_teams(advancing, third_place_groups, bracket_df):
    """Assign 3rd-place teams to R32 slots based on which groups they came from.

    Uses a simplified assignment: for each R32 match that has a 3rd-place slot,
    find the first available 3rd-place group from that match's possible sources.
    """
    available_3rd = set(third_place_groups)
    r32_matches = bracket_df[bracket_df["round"] == "R32"].copy()

    slot_assignments = {}
    for _, match in r32_matches.iterrows():
        mn = match["match_number"]
        away_src = match["away_source"]

        if "/" in str(away_src):
            # This is a 3rd-place slot
            possible_groups = [g for g in away_src.replace("3", "").split("/") if g]
            # Pick the first available group
            assigned = None
            for g in possible_groups:
                if g in available_3rd:
                    assigned = g
                    break
            if assigned:
                slot_assignments[mn] = f"3{assigned}"
                available_3rd.discard(assigned)
            else:
                # Fallback: pick any remaining
                if available_3rd:
                    assigned = available_3rd.pop()
                    slot_assignments[mn] = f"3{assigned}"

    return slot_assignments


def simulate_knockout_match(team_a, team_b, model, feature_builder, rng,
                            penalty_home_win_rate=0.5, symmetric=True):
    """Simulate a single knockout match. If draw, resolve via penalties.

    Args:
        symmetric: If True, average predictions from both orientations (A-as-home
                   and B-as-home) to remove positional bias on neutral venues.

    Returns winner team name.
    """
    features_ab = feature_builder(team_a, team_b)
    if features_ab is not None and symmetric:
        features_ba = feature_builder(team_b, team_a)
        if features_ba is not None:
            proba_ab = model.predict_proba(features_ab)[0]
            proba_ba = model.predict_proba(features_ba)[0]
            # Average: AB's "home win" = A wins; BA's "away win" = A wins
            p_a_win = (proba_ab[0] + proba_ba[2]) / 2
            p_draw = (proba_ab[1] + proba_ba[1]) / 2
            p_b_win = (proba_ab[2] + proba_ba[0]) / 2
            # Renormalize
            total = p_a_win + p_draw + p_b_win
            ph, pd_, pa = p_a_win / total, p_draw / total, p_b_win / total
        else:
            proba = model.predict_proba(features_ab)[0]
            ph, pd_, pa = proba[0], proba[1], proba[2]
    elif features_ab is not None:
        proba = model.predict_proba(features_ab)[0]
        ph, pd_, pa = proba[0], proba[1], proba[2]
    else:
        ph, pd_, pa = 0.4, 0.3, 0.3

    r = rng.random()
    if r < ph:
        return team_a
    elif r < ph + pd_:
        # Penalty shootout — 50/50 on neutral venue
        if rng.random() < penalty_home_win_rate:
            return team_a
        else:
            return team_b
    else:
        return team_b


def run_full_simulation(fixtures_df, groups_df, bracket_df, match_probas,
                        model, feature_builder, n_iterations=10000, seed=42):
    """Run the full Monte Carlo simulation.

    Returns:
        results: dict with win counts, advancement counts, group standings distributions
    """
    rng = np.random.default_rng(seed)

    # Track results
    tournament_wins = defaultdict(int)
    round_advancement = defaultdict(lambda: defaultdict(int))
    group_standings_dist = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    all_teams = groups_df["team"].tolist()
    for team in all_teams:
        for rnd in ["Group", "R32", "R16", "QF", "SF", "Final", "Winner"]:
            round_advancement[team][rnd] = 0

    for iteration in range(n_iterations):
        if iteration % 1000 == 0:
            print(f"  Iteration {iteration}/{n_iterations}...")

        # 1. Group stage
        standings = simulate_group_stage(fixtures_df, match_probas, groups_df, rng)

        # Track group position distributions
        for group, teams in standings.items():
            for pos, t in enumerate(teams):
                group_standings_dist[group][t["team"]][pos + 1] += 1

        # 2. Determine advancing teams
        advancing, third_place_groups = get_advancing_teams(standings)

        for code, team in advancing.items():
            round_advancement[team]["Group"] += 1

        # 3. Slot 3rd-place teams
        slot_assignments = slot_third_place_teams(advancing, third_place_groups, bracket_df)

        # 4. R32
        r32_matches = bracket_df[bracket_df["round"] == "R32"].copy()
        r32_winners = {}

        for _, match in r32_matches.iterrows():
            mn = match["match_number"]
            home_src = match["home_source"]
            away_src = match["away_source"]

            # Resolve home team
            home_team = advancing.get(home_src)

            # Resolve away team (may be direct or 3rd-place slot)
            if "/" in str(away_src):
                code = slot_assignments.get(mn)
                away_team = advancing.get(code) if code else None
            else:
                away_team = advancing.get(away_src)

            if home_team and away_team:
                winner = simulate_knockout_match(
                    home_team, away_team, model, feature_builder, rng
                )
                r32_winners[f"W{mn}"] = winner
                round_advancement[winner]["R32"] += 1
            elif home_team:
                r32_winners[f"W{mn}"] = home_team
                round_advancement[home_team]["R32"] += 1
            elif away_team:
                r32_winners[f"W{mn}"] = away_team
                round_advancement[away_team]["R32"] += 1

        # 5. R16, QF, SF, Final
        # Single dict tracks all knockout winners across all rounds
        match_winners = dict(r32_winners)

        for rnd, rnd_label in [("R16", "R16"), ("QF", "QF"), ("SF", "SF"), ("F", "Final")]:
            rnd_matches = bracket_df[bracket_df["round"] == rnd]

            for _, match in rnd_matches.iterrows():
                mn = match["match_number"]
                home_ref = str(match["home_source"])
                away_ref = str(match["away_source"])

                home_team = match_winners.get(home_ref)
                away_team = match_winners.get(away_ref)

                if home_team and away_team:
                    winner = simulate_knockout_match(
                        home_team, away_team, model, feature_builder, rng
                    )
                    match_winners[f"W{mn}"] = winner
                    round_advancement[winner][rnd_label] += 1

                    if rnd == "F":
                        tournament_wins[winner] += 1
                        round_advancement[winner]["Winner"] += 1

    return {
        "tournament_wins": dict(tournament_wins),
        "round_advancement": {t: dict(v) for t, v in round_advancement.items()},
        "group_standings_dist": {
            g: {t: dict(p) for t, p in teams.items()}
            for g, teams in group_standings_dist.items()
        },
        "n_iterations": n_iterations
    }
