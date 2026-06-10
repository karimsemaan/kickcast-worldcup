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
