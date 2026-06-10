# Lessons Learned — 2026 FIFA World Cup Prediction Project

## What Worked

### Elo Ratings Are King
- Elo difference is the single strongest predictor (|r| = 0.504 with target)
- The Elo system's built-in K-factor weighting (World Cup=60, Friendly=20) naturally captures match importance
- Live data from eloratings.net is more accurate than static Kaggle snapshots

### Transfermarkt Market Values > EA FC Ratings
- Market values from Transfermarkt are updated weekly and reflect real-world player assessment
- Log-transforming squad values (np.sign(x) * np.log1p(np.abs(x))) handled the heavy-tailed distribution well
- Positional splits (attack/mid/def) add signal beyond raw squad totals

### Class Balancing Is Essential
- Without balancing, models predict almost zero draws (1-3% recall) and inflate accuracy by defaulting to Home Win
- Balanced sample weights (sklearn's n_samples / (n_classes * n_class_samples)) gave the best F1 trade-off
- SMOTE performed worse than simple sample weighting — synthetic draws don't capture the real decision boundary

### Chronological Splits Prevent Leakage
- Random splits on time-series sports data would leak future match patterns into training
- Our expanding window CV (train to 2010 -> test 2014, etc.) showed consistent improvement over time

## What Didn't Work

### Draw Prediction Remains Hard (~26% Recall at Best)
- Draws don't have a distinctive feature signature — they look like "close Home Wins" or "close Away Wins"
- The two-stage approach (Win/Not-Win then Draw/Loss) didn't beat the direct 3-class model
- This is consistent with the academic literature — draws are the fundamental challenge in football prediction

### Some Features Added Noise
- SHAP ablation showed removing 9 of 31 features (injury metrics, match_importance, some WC history features) slightly *improved* performance
- Injury data is sparse and noisy in the historical dataset — the salimt dataset is static and doesn't capture partial fitness
- star_injury_flag rarely triggers in the historical data

### High Accuracy ≠ Good Model
- Stacking Ensemble had the highest accuracy (61.2%) but worst macro F1 (0.465) because it ignored draws entirely
- Log loss is a better metric than accuracy for this problem — it penalizes confident wrong predictions

### 2022 World Cup Upsets Are Unpredictable
- Model gave Argentina vs Saudi Arabia 97% home win probability — the biggest upset still happened
- Tournament football has structural randomness (one bad day = elimination) that no model captures
- 45% match accuracy on the 2022 WC is consistent with the state of the art

## Technical Notes for Reproducing

### Environment
- Python 3.11+ with venv
- `pip install -r requirements.txt` then `brew install libomp` (macOS, for XGBoost)
- All scripts run from project root: `python data/scripts/01_download_data.py`

### Data Pipeline Order (Must Be Sequential)
1. `data/scripts/01_download_data.py` — downloads all raw data
2. `data/scripts/02_build_features.py` — builds feature_matrix.csv
3. `data/scripts/03_create_splits.py` — creates train/val/test splits
4. `notebooks/02_model_training.py` — trains all models (~5 min total)
5. Everything else (evaluation, SHAP, simulation) can run in any order after Step 4

### Key Files
- `data/processed/feature_matrix.csv` — the master feature matrix (21,371 rows x 38 cols)
- `data/processed/splits/` — ready-to-use train/val/test splits (raw + median-imputed + SMOTE)
- `outputs/model_artifacts/results_summary.csv` — all model metrics in one table
- `outputs/model_artifacts/optuna_results.joblib` — best hyperparameters from tuning
- `outputs/simulation_results/` — Monte Carlo simulation outputs

### What's NOT in Git (Re-generate Yourself)
- `data/raw/` (except world_cup_2026/) — run `01_download_data.py` to re-download
- `outputs/model_artifacts/StackingEnsemble.joblib` (307MB) — retrain via `02_model_training.py`
- `outputs/model_artifacts/RandomForest.joblib` (300MB) — retrain via `02_model_training.py`

### sklearn 1.8 Note
- `LogisticRegression(multi_class="multinomial")` is removed in sklearn 1.8 — multinomial is now the default, just drop the parameter
