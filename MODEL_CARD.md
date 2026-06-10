# Model Card: KickCast match-outcome models

## Model details

- **Developers:** Karim Semaan & Ramzi Zeineddine (EECE 5644 final project, Northeastern University, Spring 2026).
- **Task:** 3-class classification of international football matches (Home win / Draw / Away win), feeding a Monte-Carlo tournament simulator.
- **Architectures:**
  - Classical zoo: Logistic Regression, KNN (k = 5/10/20/50), Random Forest, XGBoost, HistGradientBoosting, SVM-RBF, stacking ensemble, LightGBM, CatBoost. Implemented in scikit-learn / XGBoost / LightGBM / CatBoost; hyperparameters tuned with Optuna (best configs in `outputs/model_artifacts/*optuna_results.joblib`).
  - KickCastNet (PyTorch): per-feature tokenizer, multi-head self-attention over feature tokens, residual MLP head; later variants add adaptive focal loss and mixup (`src/kickcast_net*.py`).
- **Version/date:** Spring 2026 course submission.
- **License:** MIT (code). Trained weights are not distributed.

## Intended use

- **Primary:** educational demonstration of a leakage-safe sports-prediction pipeline and probability-driven tournament simulation; the public dashboard shows per-team 2026 World Cup advancement odds.
- **Out of scope:** betting or any financial decision-making. Probabilities are course-project grade, calibrated on historical internationals, and carry irreducible tournament randomness. Do not use them to wager money.

## Training data

- 21,371 international matches (2004 onward for training windows) with 38 features: Elo ratings and deltas, FIFA rankings, Transfermarkt squad market values (log-transformed, positional splits), recent form, head-to-head history, manager experience, injury aggregates.
- Splits are strictly chronological: train 2004-2019, validation 2020 to pre-2022-WC, test = 2022 World Cup onward. No random shuffling, no future leakage.
- Class balance: ~45% home win, ~25% draw, ~30% away win. Balanced sample weights used in the "balanced" model variants; SMOTE was evaluated and rejected (worse macro-F1 than simple weighting).
- Raw sources and their licenses are documented in `data/README.md`; raw data is re-downloaded by `data/scripts/01_download_data.py`, not redistributed.

## Evaluation and metrics

- **Primary metric: log-loss** (a proper scoring rule), with calibration plots (`outputs/figures/12_calibration_plots.png`). Macro-F1 and per-class recall are reported because accuracy alone is misleading on this task.
- **Headline holdout:** 2022 World Cup, 64 matches: best top-1 accuracy 29/64 = 45.3%, at the always-home baseline (~44%) and below a naive Elo-favorite rule (~52%), precisely because the model spends probability mass on draws that favorite-picking baselines never predict.
- Validation log-loss bottoms out around 0.86-0.92; expanding-window CV (train to 2010 → train to 2022) shows monotonic improvement (log-loss 1.172 → 1.000).
- Known pathology, kept on purpose as a cautionary result: the stacking ensemble maximizes accuracy (61.2%) with near-zero draw recall (3%), the worst macro-F1 (0.465) of the zoo.

## Explainability

SHAP (beeswarm, bar, dependence, per-match waterfalls in `outputs/figures/`). Elo difference is the single strongest predictor (|r| = 0.504). SHAP ablation removed 9 noisy features (mostly injury and match-importance flags) and slightly improved performance.

## Simulation layer

Match-level class probabilities drive 10,000 Monte-Carlo iterations of the full 2026 tournament (48 teams, 104 matches): Poisson goal sampling conditioned on the predicted outcome, full group tiebreakers, third-place advancement, knockout bracket with a penalty-shootout model (`src/simulation.py`, unit-tested in `tests/test_simulation.py`).

## Limitations and ethical considerations

- Draw prediction is the structural weakness of the field; best draw recall here is roughly 26-37% depending on model and split.
- Single-elimination football has irreducible variance: a 97%-favorite lost in the 2022 data (Argentina vs Saudi Arabia). Point estimates of "who wins" are the wrong product; distributions are the product.
- Historical injury data is sparse and static; those features were mostly noise.
- Training data reflects historical conditions (squad values, Elo dynamics); performance degrades for teams with little recent international history.
- Outputs are probabilities about sporting events and involve no personal data. The repository ships no scraped raw data and no trained weights; everything is rebuilt from documented public sources.
