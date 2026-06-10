# Published 2026-06-10 from the portfolio calibration-study run (paths
# generalized to this repo; methodology byte-identical to the original).
"""Bootstrap confidence intervals for the KickCast calibration study.

Extends study.py (same data, same model artifact, same isotonic recipe) with
the study's own stated next step: percentile bootstrap CIs on the n=64 holdout
metrics and their before/after deltas.

Method:
  - Refit nothing: the calibrator is the SAME per-class isotonic regression
    fit once on the 2,324-match validation split (exactly as study.py does).
  - Resample the 64 holdout matches with replacement B=10,000 times (seeded);
    for each resample compute log-loss, pooled 10-bin OvR ECE, and multiclass
    Brier for the raw and recalibrated probabilities, plus the deltas.
  - Report percentile 95% CIs.

FIDELITY GATE: before computing anything, the script must reproduce the
published point estimates exactly (log-loss 1.347 -> 1.093, ECE 0.157 -> 0.120,
Brier 0.739 -> 0.646 at 3 decimals). If it cannot, it aborts rather than
publish numbers from a different setup.

Output: bootstrap_ci.json next to this script.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss

BASE = Path(__file__).resolve().parent.parent  # repo root (kickcast-worldcup)
SPLITS = BASE / "data" / "processed" / "splits"
ARTIFACTS = BASE / "outputs" / "model_artifacts"
OUT = Path(__file__).resolve().parent

N_BINS = 10
B = 10_000
SEED = 5644  # course number; fixed for reproducibility

# ---------------------------------------------------------------- data
# joblib.load is safe here: first-party artifact trained and saved by Karim's
# own pipeline in this local repo, not an untrusted source (same note as
# study.py, which loads the identical file).
model = joblib.load(ARTIFACTS / "XGBoost_tuned_balanced.joblib")

fm = pd.read_csv(BASE / "data" / "processed" / "feature_matrix.csv", parse_dates=["date"])
wc22 = fm[
    (fm["tournament"].str.contains("FIFA World Cup", na=False))
    & (fm["date"] >= "2022-11-20")
    & (fm["date"] <= "2022-12-18")
].copy()
assert len(wc22) == 64, f"expected 64 WC matches, got {len(wc22)}"

DROP_COLS = ["date", "home_team", "away_team", "home_score", "away_score", "tournament", "result"]
X_wc = wc22.drop(columns=DROP_COLS, errors="ignore")
y_wc = wc22["result"].values.astype(int)

X_val = pd.read_csv(SPLITS / "X_val.csv")
y_val = pd.read_csv(SPLITS / "y_val.csv")["result"].values.astype(int)

p_wc = model.predict_proba(X_wc)
p_val = model.predict_proba(X_val)

# ------------------------------------------------- isotonic (as study.py)
calibrators = []
for k in range(3):
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(p_val[:, k], (y_val == k).astype(float))
    calibrators.append(iso)


def recalibrate(p):
    q = np.column_stack([calibrators[k].predict(p[:, k]) for k in range(3)])
    q = np.clip(q, 1e-9, None)
    return q / q.sum(axis=1, keepdims=True)


q_wc = recalibrate(p_wc)


# ------------------------------------------------------------- metrics
def brier_multiclass(y, p):
    onehot = np.eye(3)[y]
    return float(np.mean(np.sum((p - onehot) ** 2, axis=1)))


def ece_pooled(y, p, n_bins=N_BINS):
    conf = p.reshape(-1)
    hit = np.eye(3)[y].reshape(-1)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, n_bins - 1)
    n = len(conf)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        ece += (m.sum() / n) * abs(hit[m].mean() - conf[m].mean())
    return float(ece)


def metrics(y, p):
    return {
        "log_loss": float(log_loss(y, p, labels=[0, 1, 2])),
        "ece": ece_pooled(y, p),
        "brier": brier_multiclass(y, p),
    }


# ------------------------------------------------------- fidelity gate
before = metrics(y_wc, p_wc)
after = metrics(y_wc, q_wc)
published = {
    "before": {"log_loss": 1.347, "ece": 0.157, "brier": 0.739},
    "after": {"log_loss": 1.093, "ece": 0.120, "brier": 0.646},
}
for phase, got in (("before", before), ("after", after)):
    for k, want in published[phase].items():
        if abs(round(got[k], 3) - want) > 0.0005:
            raise SystemExit(
                f"FIDELITY GATE FAILED: {phase}.{k} = {got[k]:.4f}, published {want}"
            )
print("fidelity gate PASSED: point estimates reproduce the published numbers")

# ------------------------------------------------------------ bootstrap
rng = np.random.default_rng(SEED)
n = len(y_wc)
rows = {key: [] for key in ("ll_b", "ll_a", "ll_d", "ece_b", "ece_a", "ece_d", "br_b", "br_a", "br_d")}
for _ in range(B):
    idx = rng.integers(0, n, n)
    yb = y_wc[idx]
    if len(np.unique(yb)) < 2:
        continue  # degenerate resample; log_loss undefined for single class
    mb = metrics(yb, p_wc[idx])
    ma = metrics(yb, q_wc[idx])
    rows["ll_b"].append(mb["log_loss"])
    rows["ll_a"].append(ma["log_loss"])
    rows["ll_d"].append(ma["log_loss"] - mb["log_loss"])
    rows["ece_b"].append(mb["ece"])
    rows["ece_a"].append(ma["ece"])
    rows["ece_d"].append(ma["ece"] - mb["ece"])
    rows["br_b"].append(mb["brier"])
    rows["br_a"].append(ma["brier"])
    rows["br_d"].append(ma["brier"] - mb["brier"])


def ci(v):
    lo, hi = np.percentile(v, [2.5, 97.5])
    return {"lo": round(float(lo), 3), "hi": round(float(hi), 3), "mean": round(float(np.mean(v)), 3)}


result = {
    "_provenance": (
        "Percentile bootstrap (B=10,000, seed 5644) over the 64-match 2022 WC holdout; "
        "calibrator fit once on the 2,324-match validation split and held fixed across "
        "resamples (no refitting). Point estimates verified to reproduce the published "
        "study numbers before resampling. Resamples with a single outcome class are "
        "skipped (log-loss undefined)."
    ),
    "n_holdout": n,
    "n_resamples_used": len(rows["ll_d"]),
    "point_estimates": {"before": before, "after": after},
    "ci95": {
        "log_loss_before": ci(rows["ll_b"]),
        "log_loss_after": ci(rows["ll_a"]),
        "log_loss_delta": ci(rows["ll_d"]),
        "ece_before": ci(rows["ece_b"]),
        "ece_after": ci(rows["ece_a"]),
        "ece_delta": ci(rows["ece_d"]),
        "brier_before": ci(rows["br_b"]),
        "brier_after": ci(rows["br_a"]),
        "brier_delta": ci(rows["br_d"]),
    },
}

out_path = OUT / "bootstrap_ci.json"
out_path.write_text(json.dumps(result, indent=2))
print(json.dumps(result["ci95"], indent=2))
print(f"\nwritten: {out_path}")
