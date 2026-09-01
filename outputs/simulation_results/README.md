# Simulation results — which run is which

Two different model runs produce tournament odds in this project, and their
headline numbers differ. This note exists so nobody has to guess why.

| Artifact | Model | Spain win % (headline) |
| --- | --- | --- |
| `win_probabilities.csv` / `advancement_probabilities.csv` (this folder) | Tuned class-balanced **XGBoost** (raw probabilities) — the committed course-submission run | **16.51%** |
| Live dashboard (kickcast-dashboard.vercel.app) | **Calibrated ensemble** — a later run combining the model zoo with post-hoc calibration, baked into the dashboard at build time | **20.5%** |

The calibrated-ensemble run's raw output files were consumed directly by the
dashboard build and were not exported to this repository — an honest gap, noted
here rather than papered over. The calibration methodology behind that run is
published in [`../../calibration_study/`](../../calibration_study/) (study
script, bootstrap CIs, results JSON), and the committed XGBoost run remains
fully reproducible from this repo's pipeline.

If you are diffing the dashboard against this folder: you are looking at two
different runs, not a discrepancy within one.

## Scored against the real tournament (group stage)

Now that the 2026 World Cup has been played, the committed per-match forecast in
`group_match_predictions.csv` is scored against the real results:

- `group_stage_actual_results.csv` — final full-time score and outcome of all 72
  group-stage matches, compiled from public match reporting (ESPN, FIFA, FOX,
  Yahoo, Sky) and cross-checked as a consistent round-robin per group. The
  outcome column is derived from the recorded goals.
- `group_stage_scored.csv` — per-match join of prediction vs. actual, with a
  `correct` flag, log-loss and Brier per match.
- `group_stage_score_summary.json` — the aggregate: 47/72 = 65.3% top-1 (vs.
  47.2% always-home), log-loss 0.850 beating the 1.099 uniform (ln 3), Brier
  0.507.

Regenerate the two scored files with `python ../../scripts/score_group_stage.py`.
Group stage only: the committed per-match forecast does not cover the knockout
bracket, so this scores no knockout tie or the tournament-winner odds.
