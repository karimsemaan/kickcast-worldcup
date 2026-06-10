# Published 2026-06-10 from the portfolio calibration-study run (paths
# generalized to this repo's layout: outputs/model_artifacts + data/processed/
# splits; methodology and every computation byte-identical to the original).
"""KickCast calibration study (real data, real model).

Pipeline:
  1. Load Karim's own saved XGBoost_tuned_balanced.joblib (first-party artifact in
     this local repo; joblib.load is safe here, not an untrusted source).
     Fidelity verified: it reproduces test_results.csv to 6 decimals.
  2. Holdout = the 64 matches of the 2022 World Cup (2022-11-20 .. 2022-12-18),
     inside the chronological test split. Never used for training or tuning.
  3. BEFORE: raw model probabilities on the 64 matches -> reliability (10 bins,
     pooled one-vs-rest, 64x3 = 192 (match, class) pairs), ECE, log-loss,
     multiclass Brier.
  4. Recalibration: per-class isotonic regression fit on the VALIDATION split
     (2,324 matches, 2020-01-01 .. 2022-11-19, chronologically earlier), then
     row-renormalized. The WC-64 holdout is never touched during fitting.
  5. AFTER: same metrics + diagram on the recalibrated probabilities.
  6. Robustness: same before/after metrics on the full test split (n=3,552).

Outputs: reliability-before.png, reliability-after.png, cover.png (2560x1600 —
12.8x8.0in at dpi=200, i.e. the original 1280x800 layout rendered at 2x for
retina; dark #0b0d12, accents #bff73a / #7da7ff) + results.json with every number.
"""
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score, log_loss

BASE = Path(__file__).resolve().parent.parent  # repo root (kickcast-worldcup)
SPLITS = BASE / "data" / "processed" / "splits"
ARTIFACTS = BASE / "outputs" / "model_artifacts"
OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(parents=True, exist_ok=True)
SCRATCH = OUT / "scratch"

N_BINS = 10
RNG_LABELS = ["Home Win", "Draw", "Away Win"]

# ---------------------------------------------------------------- data
model = joblib.load(ARTIFACTS / "XGBoost_tuned_balanced.joblib")

fm = pd.read_csv(BASE / "data" / "processed" / "feature_matrix.csv", parse_dates=["date"])
wc22 = fm[(fm["tournament"].str.contains("FIFA World Cup", na=False)) &
          (fm["date"] >= "2022-11-20") & (fm["date"] <= "2022-12-18")].copy()
assert len(wc22) == 64, f"expected 64 WC matches, got {len(wc22)}"

DROP_COLS = ["date", "home_team", "away_team", "home_score", "away_score", "tournament", "result"]
X_wc = wc22.drop(columns=DROP_COLS, errors="ignore")
y_wc = wc22["result"].values.astype(int)

X_val = pd.read_csv(SPLITS / "X_val.csv")
y_val = pd.read_csv(SPLITS / "y_val.csv")["result"].values.astype(int)
X_test = pd.read_csv(SPLITS / "X_test.csv")
y_test = pd.read_csv(SPLITS / "y_test.csv")["result"].values.astype(int)

p_wc = model.predict_proba(X_wc)
p_val = model.predict_proba(X_val)
p_test = model.predict_proba(X_test)

# ------------------------------------------------------------- metrics
def brier_multiclass(y, p):
    onehot = np.eye(3)[y]
    return float(np.mean(np.sum((p - onehot) ** 2, axis=1)))

def pooled_ovr(y, p):
    """Flatten to (n*3,) one-vs-rest pairs."""
    conf = p.reshape(-1)
    hit = np.eye(3)[y].reshape(-1)
    return conf, hit

def ece_pooled(y, p, n_bins=N_BINS):
    conf, hit = pooled_ovr(y, p)
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

def ece_toplabel(y, p, n_bins=N_BINS):
    conf = p.max(axis=1)
    hit = (np.argmax(p, axis=1) == y).astype(float)
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

def bin_stats(y, p, n_bins=N_BINS):
    conf, hit = pooled_ovr(y, p)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, n_bins - 1)
    centers, accs, confs, counts = [], [], [], []
    for b in range(n_bins):
        m = idx == b
        centers.append((edges[b] + edges[b + 1]) / 2)
        counts.append(int(m.sum()))
        if m.sum() > 0:
            accs.append(float(hit[m].mean()))
            confs.append(float(conf[m].mean()))
        else:
            accs.append(np.nan)
            confs.append(np.nan)
    return np.array(centers), np.array(accs), np.array(confs), np.array(counts)

def metric_block(y, p):
    return {
        "log_loss": float(log_loss(y, p, labels=[0, 1, 2])),
        "brier_multiclass": brier_multiclass(y, p),
        "ece_pooled_ovr_10bin": ece_pooled(y, p),
        "ece_toplabel_10bin": ece_toplabel(y, p),
        "accuracy": float(accuracy_score(y, np.argmax(p, axis=1))),
    }

# -------------------------------------------------- isotonic (fit on val)
def fit_isotonic(p_fit, y_fit):
    isos = []
    for k in range(3):
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(p_fit[:, k], (y_fit == k).astype(float))
        isos.append(iso)
    return isos

def apply_isotonic(isos, p):
    q = np.column_stack([isos[k].predict(p[:, k]) for k in range(3)])
    q = np.clip(q, 1e-6, 1.0)
    return q / q.sum(axis=1, keepdims=True)

isos = fit_isotonic(p_val, y_val)
p_wc_cal = apply_isotonic(isos, p_wc)
p_test_cal = apply_isotonic(isos, p_test)

results = {
    "model": "XGBoost_tuned_balanced (Karim's saved artifact, repro-verified vs test_results.csv)",
    "calibrator": "per-class isotonic regression fit on the validation split (n=2324, 2020-01-01..2022-11-19), row-renormalized; WC-64 holdout untouched during fitting",
    "wc2022_holdout_n": int(len(y_wc)),
    "wc2022_class_counts_HW_D_AW": np.bincount(y_wc, minlength=3).tolist(),
    "wc2022_before": metric_block(y_wc, p_wc),
    "wc2022_after": metric_block(y_wc, p_wc_cal),
    "full_test_n": int(len(y_test)),
    "full_test_before": metric_block(y_test, p_test),
    "full_test_after": metric_block(y_test, p_test_cal),
}
print(json.dumps(results, indent=2))
(SCRATCH / "results.json").write_text(json.dumps(results, indent=2))

# ---------------------------------------------------------------- plots
BG = "#0b0d12"
PANEL = "#11141c"
LIME = "#bff73a"
BLUE = "#7da7ff"
TEXT = "#e7ebf3"
MUTED = "#8b93a7"
GRIDC = "#2a2f3d"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": PANEL,
    "axes.edgecolor": GRIDC, "axes.labelcolor": TEXT,
    "text.color": TEXT, "xtick.color": MUTED, "ytick.color": MUTED,
    "font.family": "DejaVu Sans", "axes.grid": True,
    "grid.color": GRIDC, "grid.linewidth": 0.6, "grid.alpha": 0.6,
    "savefig.facecolor": BG,
})

def reliability_figure(path, p, y, accent, title, subtitle, mb):
    centers, accs, confs, counts = bin_stats(y, p)
    # dpi=200 with the same 12.8x8.0in figsize -> 2560x1600 PNGs (2x for retina;
    # layout/typography identical to the original 1280x800 render).
    fig = plt.figure(figsize=(12.8, 8.0), dpi=200)
    gs = fig.add_gridspec(2, 1, height_ratios=[3.2, 1.0], hspace=0.28,
                          left=0.09, right=0.96, top=0.83, bottom=0.10)
    ax = fig.add_subplot(gs[0])
    axb = fig.add_subplot(gs[1])

    ax.plot([0, 1], [0, 1], "--", color=MUTED, linewidth=1.2, label="Perfect calibration")
    m = ~np.isnan(accs)
    ax.plot(confs[m], accs[m], "o-", color=accent, linewidth=2.4, markersize=8,
            markeredgecolor=BG, markeredgewidth=1.2, label="Model")
    ax.fill_between(confs[m], confs[m], accs[m], color=accent, alpha=0.12)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted probability (per bin)")
    ax.set_ylabel("Observed frequency")
    ax.legend(loc="upper left", frameon=False, fontsize=11)

    fig.text(0.09, 0.955, title, fontsize=21, fontweight="bold", color=TEXT, ha="left")
    fig.text(0.09, 0.935, subtitle, fontsize=11.5, color=MUTED, ha="left", va="top")
    mtxt = (f"log-loss {mb['log_loss']:.3f}    Brier {mb['brier_multiclass']:.3f}    "
            f"ECE (10-bin, pooled OvR) {mb['ece_pooled_ovr_10bin']:.3f}")
    fig.text(0.09, 0.865, mtxt, fontsize=12.5, color=accent, ha="left")

    axb.bar(centers, counts, width=0.085, color=accent, alpha=0.75, edgecolor=BG)
    for c, n in zip(centers, counts):
        if n > 0:
            axb.text(c, n + max(counts) * 0.04, str(n), ha="center", va="bottom",
                     fontsize=8.5, color=MUTED)
    axb.set_xlim(0, 1)
    axb.set_ylim(0, max(counts) * 1.25)
    axb.set_xlabel("Predicted probability bin")
    axb.set_ylabel("Pairs per bin")

    fig.savefig(path)
    plt.close(fig)
    print("saved", path)

sub_before = ("2022 World Cup holdout, 64 matches x 3 outcomes = 192 (match, class) pairs, pooled one-vs-rest,\n"
              "10 uniform bins. Raw XGBoost (tuned, class-balanced) probabilities.")
sub_after = ("Same 64-match holdout after per-class isotonic recalibration. Calibrator fit only on the 2020 to\n"
             "Nov 2022 validation split (n=2,324), then row-renormalized. Holdout never seen during fitting.")

reliability_figure(OUT / "reliability-before.png", p_wc, y_wc, BLUE,
                   "KickCast reliability: before recalibration",
                   sub_before, results["wc2022_before"])
reliability_figure(OUT / "reliability-after.png", p_wc_cal, y_wc, LIME,
                   "KickCast reliability: after isotonic recalibration",
                   sub_after, results["wc2022_after"])

# ------------------------------------------------------------- cover
cb, ab_, fb, nb = bin_stats(y_wc, p_wc)
ca, aa, fa, na = bin_stats(y_wc, p_wc_cal)
b, a = results["wc2022_before"], results["wc2022_after"]

fig = plt.figure(figsize=(12.8, 8.0), dpi=200)
gs = fig.add_gridspec(2, 2, width_ratios=[1.45, 1.0], height_ratios=[3.0, 1.0],
                      hspace=0.30, wspace=0.22, left=0.08, right=0.95, top=0.82, bottom=0.10)
ax = fig.add_subplot(gs[:, 0])
axm = fig.add_subplot(gs[0, 1])
axb2 = fig.add_subplot(gs[1, 1])

ax.plot([0, 1], [0, 1], "--", color=MUTED, linewidth=1.2, label="Perfect calibration")
mB = ~np.isnan(ab_)
mA = ~np.isnan(aa)
ax.plot(fb[mB], ab_[mB], "o-", color=BLUE, linewidth=2.4, markersize=8,
        markeredgecolor=BG, markeredgewidth=1.2, label="Before (raw XGBoost)")
ax.plot(fa[mA], aa[mA], "o-", color=LIME, linewidth=2.4, markersize=8,
        markeredgecolor=BG, markeredgewidth=1.2, label="After (isotonic)")
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.set_xlabel("Mean predicted probability")
ax.set_ylabel("Observed frequency")
ax.legend(loc="upper left", frameon=False, fontsize=11)
ax.set_title("Reliability, 2022 World Cup holdout (n=64)", fontsize=13, color=TEXT, pad=10)

# metric deltas panel
axm.axis("off")
rows = [
    ("log-loss", b["log_loss"], a["log_loss"]),
    ("Brier (multiclass)", b["brier_multiclass"], a["brier_multiclass"]),
    ("ECE (10-bin OvR)", b["ece_pooled_ovr_10bin"], a["ece_pooled_ovr_10bin"]),
]
axm.text(0.0, 0.95, "Holdout metrics", fontsize=14, fontweight="bold", color=TEXT, va="top")
yy = 0.78
for name, vb, va_ in rows:
    axm.text(0.0, yy, name, fontsize=11.5, color=MUTED, va="center")
    axm.text(0.52, yy, f"{vb:.3f}", fontsize=12.5, color=BLUE, va="center", fontweight="bold")
    axm.text(0.72, yy, "->", fontsize=11.5, color=MUTED, va="center")
    axm.text(0.82, yy, f"{va_:.3f}", fontsize=12.5, color=LIME, va="center", fontweight="bold")
    yy -= 0.16
axm.text(0.0, yy - 0.02,
         "Isotonic fit on the 2020 to Nov 2022\nvalidation split (n=2,324) only.\nSmall sample: n=64 matches.",
         fontsize=10, color=MUTED, va="top")

axb2.bar(cb - 0.022, nb, width=0.042, color=BLUE, alpha=0.8, edgecolor=BG, label="Before")
axb2.bar(ca + 0.022, na, width=0.042, color=LIME, alpha=0.8, edgecolor=BG, label="After")
axb2.set_xlim(0, 1)
axb2.set_xlabel("Probability bin")
axb2.set_ylabel("Pairs")
axb2.legend(frameon=False, fontsize=9, loc="upper right")

fig.text(0.08, 0.945, "KickCast calibration study", fontsize=22, fontweight="bold", color=TEXT)
fig.text(0.08, 0.90, "Real model, real holdout: XGBoost (tuned, class-balanced) probabilities on the 64 matches "
                     "of the 2022 World Cup,\nbefore and after per-class isotonic recalibration.",
         fontsize=11.5, color=MUTED, va="top")

fig.savefig(OUT / "cover.png")
plt.close(fig)
print("saved", OUT / "cover.png")
