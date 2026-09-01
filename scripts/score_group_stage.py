"""Score the committed group-stage forecast against the real 2026 World Cup.

Joins the pre-committed per-match probabilities
(``outputs/simulation_results/group_match_predictions.csv`` — the tuned
class-balanced XGBoost course-submission run, Spain 16.51% favourite) to the
final full-time results of all 72 group-stage matches
(``group_stage_actual_results.csv``) and scores them with proper scoring rules:
top-1 accuracy against an always-home baseline, plus log-loss and multiclass
Brier against a uniform (1/3, 1/3, 1/3) baseline whose log-loss is ln 3.

The forecast was locked on 2026-04-20, weeks before the 2026-06-11 kickoff, so
this is a genuine out-of-sample test. Actual results were compiled from public
match reporting (ESPN, FIFA, FOX, Yahoo, Sky) and cross-checked as a consistent
round-robin per group; the outcome column is derived from the recorded goals.

Pure stdlib (csv + math), no dependencies. Writes:
  - outputs/simulation_results/group_stage_scored.csv        (per-match)
  - outputs/simulation_results/group_stage_score_summary.json (aggregate)

Run: python scripts/score_group_stage.py
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

SIM_DIR = Path(__file__).resolve().parent.parent / "outputs" / "simulation_results"
PRED_CSV = SIM_DIR / "group_match_predictions.csv"
ACTUAL_CSV = SIM_DIR / "group_stage_actual_results.csv"
SCORED_CSV = SIM_DIR / "group_stage_scored.csv"
SUMMARY_JSON = SIM_DIR / "group_stage_score_summary.json"

EPS = 1e-15
UNIFORM = {"home": 1 / 3, "draw": 1 / 3, "away": 1 / 3}
# The predictions file spells outcomes as "Home Win" / "Draw" / "Away Win".
LABEL_TO_KEY = {"home win": "home", "draw": "draw", "away win": "away"}


def outcome_from_goals(home_goals: int, away_goals: int) -> str:
    if home_goals > away_goals:
        return "home"
    if home_goals < away_goals:
        return "away"
    return "draw"


def top_pick(p: dict[str, float]) -> str:
    """Argmax outcome; home wins ties, then draw (matches the site's scorer)."""
    if p["home"] >= p["draw"] and p["home"] >= p["away"]:
        return "home"
    return "draw" if p["draw"] >= p["away"] else "away"


def log_loss(p: dict[str, float], actual: str) -> float:
    return -math.log(min(1.0, max(EPS, p[actual])))


def brier(p: dict[str, float], actual: str) -> float:
    return sum((p[o] - (1.0 if o == actual else 0.0)) ** 2 for o in ("home", "draw", "away"))


def load_predictions() -> dict[int, dict]:
    preds: dict[int, dict] = {}
    with PRED_CSV.open(newline="") as f:
        for row in csv.DictReader(f):
            n = int(row["match_number"])
            preds[n] = {
                "home_team": row["home_team"],
                "away_team": row["away_team"],
                "group": row["group"],
                "p": {
                    "home": float(row["p_home_win"]),
                    "draw": float(row["p_draw"]),
                    "away": float(row["p_away_win"]),
                },
                "predicted": LABEL_TO_KEY[row["predicted"].strip().lower()],
            }
    return preds


def load_actuals() -> dict[int, dict]:
    actuals: dict[int, dict] = {}
    with ACTUAL_CSV.open(newline="") as f:
        for row in csv.DictReader(f):
            n = int(row["match_number"])
            hg, ag = int(row["home_goals"]), int(row["away_goals"])
            declared = LABEL_TO_KEY[row["actual"].strip().lower()]
            derived = outcome_from_goals(hg, ag)
            if declared != derived:
                raise ValueError(
                    f"match {n}: actual '{row['actual']}' disagrees with goals {hg}-{ag}"
                )
            actuals[n] = {"home_goals": hg, "away_goals": ag, "outcome": derived}
    return actuals


def main() -> None:
    preds, actuals = load_predictions(), load_actuals()
    missing = sorted(set(preds) ^ set(actuals))
    if missing:
        raise SystemExit(f"prediction/result mismatch on match numbers: {missing}")

    n = correct = home_wins = 0
    ll = ull = br = 0.0
    scored_rows = []
    for match_number in sorted(preds):
        pred, act = preds[match_number], actuals[match_number]
        p, actual = pred["p"], act["outcome"]
        pick = top_pick(p)
        is_correct = pick == actual
        n += 1
        correct += int(is_correct)
        home_wins += int(actual == "home")
        ll += log_loss(p, actual)
        ull += log_loss(UNIFORM, actual)
        br += brier(p, actual)
        scored_rows.append(
            {
                "match_number": match_number,
                "group": pred["group"],
                "home_team": pred["home_team"],
                "away_team": pred["away_team"],
                "score": f"{act['home_goals']}-{act['away_goals']}",
                "predicted": pick,
                "actual": actual,
                "correct": is_correct,
                "log_loss": round(log_loss(p, actual), 6),
                "brier": round(brier(p, actual), 6),
            }
        )

    summary = {
        "n": n,
        "correct": correct,
        "accuracy": correct / n,
        "always_home_accuracy": home_wins / n,
        "log_loss": ll / n,
        "uniform_log_loss": ull / n,
        "beats_uniform": ll / n < ull / n,
        "brier": br / n,
        "forecast": "committed course-submission run (tuned class-balanced XGBoost, Spain 16.51% favourite), locked 2026-04-20 before the 2026-06-11 kickoff",
        "scope": "group stage only (72 matches); the committed per-match forecast does not cover knockout ties",
        "actual_results_provenance": "compiled from public match reporting (ESPN, FIFA, FOX, Yahoo, Sky) and cross-checked as a consistent round-robin per group; outcome derived from recorded goals",
    }

    with SCORED_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(scored_rows[0].keys()))
        writer.writeheader()
        writer.writerows(scored_rows)
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2) + "\n")

    print(f"Scored {n}/72 committed group-stage forecasts against real results:")
    print(f"  Correct (top-1):  {correct}/{n} = {100 * correct / n:.1f}%")
    print(f"  Always-home base: {home_wins}/{n} = {100 * home_wins / n:.1f}%")
    print(f"  Log-loss:         {ll / n:.4f}  (uniform {ull / n:.4f} = ln 3)")
    print(f"  Brier:            {br / n:.4f}")
    print(f"  Wrote {SCORED_CSV.name} and {SUMMARY_JSON.name}")


if __name__ == "__main__":
    main()
