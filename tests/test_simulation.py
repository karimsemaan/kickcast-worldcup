"""
test_simulation.py — Tests for the World Cup simulation engine.

Covers:
- Team name mapping (WC ↔ historical)
- Group stage simulation logic
- Knockout match resolution (including symmetric prediction)
- 3rd-place advancement
- Probability sanity checks
- Feature vector column alignment
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from src.simulation import (
    WC_TO_HISTORICAL,
    HISTORICAL_TO_WC,
    historical_name,
    display_name,
    simulate_match_goals,
    simulate_group_stage,
    get_advancing_teams,
    slot_third_place_teams,
    simulate_knockout_match,
)


# ══════════════════════════════════════════════════════════════════════
# 1. Team Name Mapping
# ══════════════════════════════════════════════════════════════════════

class TestNameMapping:
    def test_wc_to_historical_mapping_exists(self):
        assert "Turkiye" in WC_TO_HISTORICAL
        assert "Czechia" in WC_TO_HISTORICAL
        assert "Curacao" in WC_TO_HISTORICAL

    def test_historical_name_converts(self):
        assert historical_name("Turkiye") == "Turkey"
        assert historical_name("Czechia") == "Czech Republic"
        assert historical_name("Curacao") == "Curaçao"

    def test_historical_name_passthrough(self):
        assert historical_name("France") == "France"
        assert historical_name("Brazil") == "Brazil"
        assert historical_name("United States") == "United States"

    def test_display_name_converts(self):
        assert display_name("Turkey") == "Turkiye"
        assert display_name("Czech Republic") == "Czechia"
        assert display_name("Curaçao") == "Curacao"

    def test_display_name_passthrough(self):
        assert display_name("France") == "France"
        assert display_name("Argentina") == "Argentina"

    def test_roundtrip(self):
        for wc_name, hist_name in WC_TO_HISTORICAL.items():
            assert display_name(historical_name(wc_name)) == wc_name
            assert historical_name(display_name(hist_name)) == hist_name

    def test_all_48_teams_resolvable(self):
        """Every team in groups.csv must map to a name that exists in results.csv."""
        groups = pd.read_csv(BASE / "data" / "raw" / "world_cup_2026" / "groups.csv")
        results = pd.read_csv(BASE / "data" / "raw" / "international_results" / "results.csv")
        all_result_teams = set(results["home_team"]) | set(results["away_team"])

        missing = []
        for team in groups["team"]:
            hist = historical_name(team)
            if hist not in all_result_teams and team not in all_result_teams:
                missing.append(team)
        assert missing == [], f"Teams not found in results.csv: {missing}"


# ══════════════════════════════════════════════════════════════════════
# 2. Match Goal Simulation
# ══════════════════════════════════════════════════════════════════════

class TestMatchGoals:
    def test_home_win_goals(self):
        rng = np.random.default_rng(42)
        for _ in range(100):
            hg, ag = simulate_match_goals(1.0, 0.0, 0.0, rng)
            assert hg > ag, "Home win must have home goals > away goals"

    def test_away_win_goals(self):
        rng = np.random.default_rng(42)
        for _ in range(100):
            hg, ag = simulate_match_goals(0.0, 0.0, 1.0, rng)
            assert ag > hg, "Away win must have away goals > home goals"

    def test_draw_goals(self):
        rng = np.random.default_rng(42)
        for _ in range(100):
            hg, ag = simulate_match_goals(0.0, 1.0, 0.0, rng)
            assert hg == ag, "Draw must have equal goals"

    def test_goals_non_negative(self):
        rng = np.random.default_rng(42)
        for _ in range(1000):
            hg, ag = simulate_match_goals(0.4, 0.3, 0.3, rng)
            assert hg >= 0 and ag >= 0

    def test_probability_distribution(self):
        """Over many samples, outcome frequencies should match input probabilities."""
        rng = np.random.default_rng(42)
        n = 50000
        hw, d, aw = 0, 0, 0
        for _ in range(n):
            hg, ag = simulate_match_goals(0.5, 0.25, 0.25, rng)
            if hg > ag:
                hw += 1
            elif hg == ag:
                d += 1
            else:
                aw += 1
        # Tolerance: 2% absolute
        assert abs(hw / n - 0.5) < 0.02
        assert abs(d / n - 0.25) < 0.02
        assert abs(aw / n - 0.25) < 0.02


# ══════════════════════════════════════════════════════════════════════
# 3. Group Stage
# ══════════════════════════════════════════════════════════════════════

class TestGroupStage:
    @pytest.fixture
    def mini_group(self):
        """4 teams, 6 matches (a full round-robin group)."""
        groups_df = pd.DataFrame({
            "team": ["Alpha", "Beta", "Gamma", "Delta"],
            "group": ["X", "X", "X", "X"],
            "confederation": ["A", "A", "A", "A"],
            "pot": [1, 2, 3, 4],
        })
        fixtures_df = pd.DataFrame([
            {"match_number": 1, "group": "X", "home_team": "Alpha", "away_team": "Beta"},
            {"match_number": 2, "group": "X", "home_team": "Gamma", "away_team": "Delta"},
            {"match_number": 3, "group": "X", "home_team": "Alpha", "away_team": "Gamma"},
            {"match_number": 4, "group": "X", "home_team": "Beta", "away_team": "Delta"},
            {"match_number": 5, "group": "X", "home_team": "Alpha", "away_team": "Delta"},
            {"match_number": 6, "group": "X", "home_team": "Beta", "away_team": "Gamma"},
        ])
        return groups_df, fixtures_df

    def test_group_stage_all_teams_present(self, mini_group):
        groups_df, fixtures_df = mini_group
        rng = np.random.default_rng(42)
        # All home wins
        probas = {i: (0.99, 0.005, 0.005) for i in range(1, 7)}
        standings = simulate_group_stage(fixtures_df, probas, groups_df, rng)
        assert "X" in standings
        teams = [t["team"] for t in standings["X"]]
        assert set(teams) == {"Alpha", "Beta", "Gamma", "Delta"}

    def test_group_stage_points_add_up(self, mini_group):
        groups_df, fixtures_df = mini_group
        rng = np.random.default_rng(42)
        probas = {i: (0.4, 0.3, 0.3) for i in range(1, 7)}
        standings = simulate_group_stage(fixtures_df, probas, groups_df, rng)
        total_points = sum(t["points"] for t in standings["X"])
        # 6 matches × 3 points each = 18 max, or fewer with draws
        assert 12 <= total_points <= 18  # minimum: all draws = 12 points

    def test_group_stage_sorted_by_points(self, mini_group):
        groups_df, fixtures_df = mini_group
        rng = np.random.default_rng(42)
        probas = {i: (0.4, 0.3, 0.3) for i in range(1, 7)}
        standings = simulate_group_stage(fixtures_df, probas, groups_df, rng)
        points = [t["points"] for t in standings["X"]]
        assert points == sorted(points, reverse=True)


# ══════════════════════════════════════════════════════════════════════
# 4. Advancing Teams
# ══════════════════════════════════════════════════════════════════════

class TestAdvancement:
    def test_24_plus_8_equals_32(self):
        """Top 2 per group (24) + 8 best 3rd-place = 32 advancing teams."""
        standings = {}
        for g in "ABCDEFGHIJKL":
            standings[g] = [
                {"team": f"{g}1", "points": 9, "gd": 5, "gf": 8, "fifa_rank": 1, "group": g},
                {"team": f"{g}2", "points": 6, "gd": 2, "gf": 5, "fifa_rank": 2, "group": g},
                {"team": f"{g}3", "points": 3, "gd": 0, "gf": 3, "fifa_rank": 3, "group": g},
                {"team": f"{g}4", "points": 0, "gd": -7, "gf": 1, "fifa_rank": 4, "group": g},
            ]
        advancing, third_place_groups = get_advancing_teams(standings)
        assert len(advancing) == 32  # 24 top-2 + 8 best third-place
        assert len(third_place_groups) == 8

    def test_top2_always_advance(self):
        standings = {}
        for g in "ABCDEFGHIJKL":
            standings[g] = [
                {"team": f"{g}_first", "points": 9, "gd": 5, "gf": 8, "fifa_rank": 1, "group": g},
                {"team": f"{g}_second", "points": 6, "gd": 2, "gf": 5, "fifa_rank": 2, "group": g},
                {"team": f"{g}_third", "points": 3, "gd": 0, "gf": 3, "fifa_rank": 3, "group": g},
                {"team": f"{g}_fourth", "points": 0, "gd": -7, "gf": 1, "fifa_rank": 4, "group": g},
            ]
        advancing, _ = get_advancing_teams(standings)
        for g in "ABCDEFGHIJKL":
            assert advancing[f"1{g}"] == f"{g}_first"
            assert advancing[f"2{g}"] == f"{g}_second"


# ══════════════════════════════════════════════════════════════════════
# 5. Knockout Match (Symmetric Prediction)
# ══════════════════════════════════════════════════════════════════════

class TestKnockoutMatch:
    def _make_model(self, p_home=0.5, p_draw=0.2, p_away=0.3):
        """Fake model that returns fixed probabilities."""
        class FakeModel:
            def predict_proba(self, X):
                return np.array([[p_home, p_draw, p_away]])
        return FakeModel()

    def _identity_builder(self, team_a, team_b):
        """Feature builder that returns a dummy DataFrame."""
        return pd.DataFrame([{"x": 1.0}])

    def test_returns_one_of_two_teams(self):
        model = self._make_model()
        rng = np.random.default_rng(42)
        for _ in range(100):
            winner = simulate_knockout_match(
                "A", "B", model, self._identity_builder, rng, symmetric=False
            )
            assert winner in ("A", "B")

    def test_strong_favorite_wins_mostly(self):
        model = self._make_model(p_home=0.8, p_draw=0.1, p_away=0.1)
        rng = np.random.default_rng(42)
        a_wins = sum(
            1 for _ in range(1000)
            if simulate_knockout_match(
                "A", "B", model, self._identity_builder, rng, symmetric=False
            ) == "A"
        )
        # 80% win + 50% of 10% draw = 85% expected
        assert a_wins > 700, f"A should win most: got {a_wins}/1000"

    def test_symmetric_mode_averages_orientations(self):
        """With symmetric=True and an asymmetric model, A-as-home and B-as-home
        should partially cancel, making results more balanced than asymmetric mode."""
        # Model gives 70% to whoever is "home"
        model = self._make_model(p_home=0.7, p_draw=0.1, p_away=0.2)
        rng = np.random.default_rng(42)

        a_wins_asym = sum(
            1 for _ in range(2000)
            if simulate_knockout_match(
                "A", "B", model, self._identity_builder, rng, symmetric=False
            ) == "A"
        )
        rng = np.random.default_rng(42)
        a_wins_sym = sum(
            1 for _ in range(2000)
            if simulate_knockout_match(
                "A", "B", model, self._identity_builder, rng, symmetric=True
            ) == "A"
        )
        # Symmetric should be closer to 50% than asymmetric
        assert abs(a_wins_sym / 2000 - 0.5) < abs(a_wins_asym / 2000 - 0.5)

    def test_fallback_when_features_none(self):
        """When feature builder returns None, use fallback probabilities."""
        model = self._make_model()
        rng = np.random.default_rng(42)

        def none_builder(a, b):
            return None

        winners = set()
        for _ in range(100):
            winners.add(simulate_knockout_match("A", "B", model, none_builder, rng))
        assert winners == {"A", "B"}  # Both should appear


# ══════════════════════════════════════════════════════════════════════
# 6. Data Integrity
# ══════════════════════════════════════════════════════════════════════

class TestDataIntegrity:
    def test_groups_have_48_teams(self):
        groups = pd.read_csv(BASE / "data" / "raw" / "world_cup_2026" / "groups.csv")
        assert len(groups) == 48
        assert groups["group"].nunique() == 12
        assert all(groups.groupby("group").size() == 4)

    def test_fixtures_have_72_matches(self):
        fixtures = pd.read_csv(BASE / "data" / "raw" / "world_cup_2026" / "fixtures.csv")
        assert len(fixtures) == 72
        assert fixtures["match_number"].is_unique

    def test_bracket_has_32_matches(self):
        bracket = pd.read_csv(BASE / "data" / "raw" / "world_cup_2026" / "knockout_bracket.csv")
        assert len(bracket) == 32
        rounds = bracket["round"].value_counts()
        assert rounds["R32"] == 16
        assert rounds["R16"] == 8
        assert rounds["QF"] == 4
        assert rounds["SF"] == 2
        assert rounds["F"] == 1
        assert rounds["3P"] == 1

    def test_fixture_teams_in_groups(self):
        groups = pd.read_csv(BASE / "data" / "raw" / "world_cup_2026" / "groups.csv")
        fixtures = pd.read_csv(BASE / "data" / "raw" / "world_cup_2026" / "fixtures.csv")
        group_teams = set(groups["team"])
        fixture_teams = set(fixtures["home_team"]) | set(fixtures["away_team"])
        assert fixture_teams == group_teams, f"Mismatch: {fixture_teams - group_teams}"

    def test_feature_matrix_columns(self):
        """Feature matrix must have all expected columns."""
        fm = pd.read_csv(BASE / "data" / "processed" / "feature_matrix.csv", nrows=5)
        expected_features = [
            "elo_diff", "elo_momentum_diff", "rank_diff", "points_diff",
            "squad_value_total_delta", "squad_value_top11_delta",
            "form_win_rate_diff", "form_weighted_diff", "goal_diff_delta",
            "h2h_home_win_rate", "h2h_draw_rate", "h2h_matches_played",
            "match_importance", "is_neutral", "result",
        ]
        for col in expected_features:
            assert col in fm.columns, f"Missing column: {col}"

    def test_splits_consistent_columns(self):
        """X_train, X_val, X_test must have identical columns."""
        splits = BASE / "data" / "processed" / "splits"
        X_train = pd.read_csv(splits / "X_train.csv", nrows=0)
        X_val = pd.read_csv(splits / "X_val.csv", nrows=0)
        X_test = pd.read_csv(splits / "X_test.csv", nrows=0)
        assert list(X_train.columns) == list(X_val.columns)
        assert list(X_train.columns) == list(X_test.columns)

    def test_splits_row_counts(self):
        """Train + val + test must equal feature matrix total."""
        fm = pd.read_csv(BASE / "data" / "processed" / "feature_matrix.csv")
        splits = BASE / "data" / "processed" / "splits"
        n_train = len(pd.read_csv(splits / "X_train.csv"))
        n_val = len(pd.read_csv(splits / "X_val.csv"))
        n_test = len(pd.read_csv(splits / "X_test.csv"))
        assert n_train + n_val + n_test == len(fm)

    def test_target_encoding(self):
        """Target must be 0, 1, or 2."""
        splits = BASE / "data" / "processed" / "splits"
        for f in ["y_train.csv", "y_val.csv", "y_test.csv"]:
            y = pd.read_csv(splits / f)["result"]
            assert set(y.unique()) == {0, 1, 2}


# ══════════════════════════════════════════════════════════════════════
# 7. Simulation Results Sanity
# ══════════════════════════════════════════════════════════════════════

class TestSimulationResults:
    @pytest.fixture
    def win_probs(self):
        return pd.read_csv(BASE / "outputs" / "simulation_results" / "win_probabilities.csv")

    @pytest.fixture
    def adv_probs(self):
        return pd.read_csv(BASE / "outputs" / "simulation_results" / "advancement_probabilities.csv")

    def test_win_probabilities_sum_to_100(self, win_probs):
        total = win_probs["win_probability"].sum()
        assert abs(total - 100.0) < 1.0, f"Win probabilities sum to {total}, expected ~100"

    def test_all_48_teams_present(self, win_probs):
        groups = pd.read_csv(BASE / "data" / "raw" / "world_cup_2026" / "groups.csv")
        expected_teams = set(groups["team"])
        actual_teams = set(win_probs["team"])
        assert expected_teams == actual_teams

    def test_no_negative_probabilities(self, win_probs):
        assert (win_probs["win_probability"] >= 0).all()

    def test_advancement_monotonically_decreasing(self, adv_probs):
        """Each team's advancement probability must decrease through rounds."""
        rounds = ["Group", "R32", "R16", "QF", "SF", "Final", "Winner"]
        for _, row in adv_probs.iterrows():
            for i in range(len(rounds) - 1):
                assert row[rounds[i]] >= row[rounds[i + 1]], (
                    f"{row['team']}: {rounds[i]}={row[rounds[i]]} < "
                    f"{rounds[i+1]}={row[rounds[i+1]]}"
                )

    def test_top_favorites_are_plausible(self, win_probs):
        """Top 5 should include known football powerhouses."""
        top5 = set(win_probs.nlargest(5, "win_probability")["team"])
        powerhouses = {"France", "Spain", "Argentina", "England", "Brazil",
                       "Germany", "Portugal", "Netherlands"}
        overlap = top5 & powerhouses
        assert len(overlap) >= 4, f"Top 5 {top5} doesn't include enough favorites"

    def test_renamed_teams_not_at_default_elo(self, adv_probs):
        """Turkiye, Czechia, Curacao must not be stuck at near-zero advancement
        (which would indicate the name mapping bug is still present)."""
        turkiye = adv_probs[adv_probs["team"] == "Turkiye"]["Group"].values[0]
        czechia = adv_probs[adv_probs["team"] == "Czechia"]["Group"].values[0]
        # Turkiye in group D with USA/Australia/Paraguay should advance > 30%
        assert turkiye > 30, f"Turkiye group advancement {turkiye}% is too low — name mapping bug?"
        # Czechia in group A with Mexico/South Korea/South Africa should advance > 20%
        assert czechia > 20, f"Czechia group advancement {czechia}% is too low — name mapping bug?"
