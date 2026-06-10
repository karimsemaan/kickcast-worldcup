"""
models.py — Model definitions and training utilities.

7 models for 3-class match outcome prediction (Home Win / Draw / Away Win):
1. Logistic Regression (multinomial) — baseline
2. KNN (k=5,10,20,50)
3. Random Forest (500 trees)
4. XGBoost (multi:softprob)
5. HistGradientBoosting
6. SVM (RBF kernel, StandardScaler)
7. Stacking Ensemble (top base models + LR meta-learner)
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import (RandomForestClassifier, HistGradientBoostingClassifier,
                               StackingClassifier)
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, f1_score, log_loss,
                              roc_auc_score, classification_report)
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier


def get_base_models():
    """Return dict of model name -> (model, needs_imputation) tuples."""
    return {
        "LogisticRegression": (
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(
                    max_iter=1000, random_state=42))
            ]),
            False  # pipeline handles imputation internally
        ),
        "KNN_k5": (
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("clf", KNeighborsClassifier(n_neighbors=5))
            ]),
            False
        ),
        "KNN_k10": (
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("clf", KNeighborsClassifier(n_neighbors=10))
            ]),
            False
        ),
        "KNN_k20": (
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("clf", KNeighborsClassifier(n_neighbors=20))
            ]),
            False
        ),
        "KNN_k50": (
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("clf", KNeighborsClassifier(n_neighbors=50))
            ]),
            False
        ),
        "RandomForest": (
            RandomForestClassifier(
                n_estimators=500, random_state=42, n_jobs=-1),
            False  # handles NaN natively in sklearn 1.8+? No — RF needs imputation or we use HistGBM
            # Actually sklearn RF doesn't support NaN. We'll handle in training.
        ),
        "XGBoost": (
            XGBClassifier(
                objective="multi:softprob", num_class=3,
                eval_metric="mlogloss", random_state=42,
                n_jobs=-1, tree_method="hist"),
            False  # XGBoost handles NaN natively
        ),
        "HistGBM": (
            HistGradientBoostingClassifier(
                random_state=42, max_iter=500),
            False  # handles NaN natively
        ),
        "SVM_RBF": (
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("clf", SVC(kernel="rbf", probability=True, random_state=42))
            ]),
            False
        ),
    }


def get_xgb_tuned(best_params):
    """Return XGBoost model with tuned hyperparameters."""
    return XGBClassifier(
        objective="multi:softprob", num_class=3,
        eval_metric="mlogloss", random_state=42,
        n_jobs=-1, tree_method="hist",
        **best_params
    )


def get_histgbm_tuned(best_params):
    """Return HistGBM model with tuned hyperparameters."""
    return HistGradientBoostingClassifier(
        random_state=42,
        **best_params
    )


def build_stacking(base_models_dict):
    """Build stacking ensemble from top base models.

    Uses LR as meta-learner. Base models should be pre-fitted estimator names
    from the training results.
    """
    estimators = [(name, model) for name, model in base_models_dict.items()]
    return StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(
            max_iter=1000, random_state=42),
        cv=5,
        stack_method="predict_proba",
        n_jobs=-1,
        passthrough=False
    )


class CalibratedEnsemble:
    """Weighted ensemble of calibrated models with optimized weights."""

    def __init__(self, models, weights):
        self.models = models
        self.weights = weights
        self.classes_ = np.array([0, 1, 2])

    def predict_proba(self, X):
        proba = np.zeros((len(X), 3))
        for (name, model), w in zip(self.models.items(), self.weights):
            proba += w * model.predict_proba(X)
        proba = np.clip(proba, 1e-15, 1.0)
        proba = proba / proba.sum(axis=1, keepdims=True)
        return proba

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


def evaluate_model(model, X, y, model_name=""):
    """Compute all metrics for a fitted model."""
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)

    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y, y_pred),
        "macro_f1": f1_score(y, y_pred, average="macro"),
        "log_loss": log_loss(y, y_proba),
    }

    # Per-class accuracy
    for cls in [0, 1, 2]:
        mask = y == cls
        if mask.sum() > 0:
            metrics[f"acc_class_{cls}"] = accuracy_score(y[mask], y_pred[mask])

    # AUC-ROC (one-vs-rest)
    try:
        metrics["auc_roc_ovr"] = roc_auc_score(y, y_proba, multi_class="ovr", average="macro")
    except ValueError:
        metrics["auc_roc_ovr"] = np.nan

    return metrics
