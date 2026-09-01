"""
test_group_stage_scoring.py — Locks the out-of-sample 2026 group-stage score.

Recomputes, from the committed forecast (group_match_predictions.csv) joined to
the real results (group_stage_actual_results.csv), the headline numbers the
README and the portfolio publish, so they can never silently drift:

- 47 / 72 correct (top-1), 65.3%
- always-home baseline 34 / 72
- log-loss 0.850, beating the uniform ln 3 baseline

Also checks data integrity: every predicted match has a result and vice versa,
each group is a valid round-robin (four teams, every team plays three), and each
declared outcome matches its recorded goals (load_actuals enforces the last one).
"""

import sys
from collections import Counter
from pathlib import Path

import pytest

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "scripts"))

from score_group_stage import (  # noqa: E402
    load_actuals,
    load_predictions,
    log_loss,
    top_pick,
)


@pytest.fixture(scope="module")
def joined():
    preds, actuals = load_predictions(), load_actuals()
    assert set(preds) == set(actuals), "prediction/result match-number mismatch"
    assert len(preds) == 72
    return preds, actuals


def test_reproduces_published_headline(joined):
    preds, actuals = joined
    n = correct = home_wins = 0
    ll = 0.0
    for number, pred in preds.items():
        actual = actuals[number]["outcome"]
        n += 1
        correct += int(top_pick(pred["p"]) == actual)
        home_wins += int(actual == "home")
        ll += log_loss(pred["p"], actual)

    assert n == 72
    assert correct == 47  # 65.3% top-1
    assert correct / n == pytest.approx(0.6528, abs=1e-3)
    assert home_wins == 34  # always-home 47.2%
    assert ll / n == pytest.approx(0.850, abs=1e-3)
    # Beats the uniform (1/3,1/3,1/3) baseline whose per-match log-loss is ln 3.
    import math

    assert ll / n < math.log(3)


def test_every_group_is_a_valid_round_robin(joined):
    preds, _ = joined
    by_group = {}
    for pred in preds.values():
        plays = by_group.setdefault(pred["group"], Counter())
        plays[pred["home_team"]] += 1
        plays[pred["away_team"]] += 1
    assert len(by_group) == 12
    for group, counts in by_group.items():
        assert len(counts) == 4, f"group {group} does not have four teams"
        assert all(v == 3 for v in counts.values()), f"group {group} not a round-robin"
