#!/usr/bin/env python3
"""
03_create_splits.py — Create chronological train/val/test splits.

Splits from CLAUDE.md:
  Train:      2004-01-01 to 2019-12-31
  Validation: 2020-01-01 to 2022-11-19
  Test:       2022-11-20 onward (2022 World Cup holdout)

Outputs to data/processed/splits/:
  X_train.csv, y_train.csv
  X_val.csv,   y_val.csv
  X_test.csv,  y_test.csv
  X_train_smote.csv, y_train_smote.csv   (SMOTE-oversampled training set)
  X_train_median.csv, X_val_median.csv, X_test_median.csv  (median-imputed for non-tree models)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from imblearn.over_sampling import SMOTE

# ── Paths ──────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent.parent
FEATURE_MATRIX = BASE / "data" / "processed" / "feature_matrix.csv"
SPLITS_DIR = BASE / "data" / "processed" / "splits"
SPLITS_DIR.mkdir(parents=True, exist_ok=True)

# ── Split boundaries (CLAUDE.md spec) ─────────────────────────────────
TRAIN_END = "2019-12-31"
VAL_START = "2020-01-01"
VAL_END = "2022-11-19"
TEST_START = "2022-11-20"

# Columns to drop from features (non-feature metadata + target)
DROP_COLS = ["date", "home_team", "away_team", "home_score", "away_score",
             "tournament", "result"]


def load_and_split(df):
    """Split dataframe chronologically into train/val/test."""
    df["date"] = pd.to_datetime(df["date"])

    train = df[df["date"] <= TRAIN_END].copy()
    val = df[(df["date"] >= VAL_START) & (df["date"] <= VAL_END)].copy()
    test = df[df["date"] >= TEST_START].copy()

    return train, val, test


def separate_xy(split_df):
    """Separate features (X) and target (y), dropping non-feature columns."""
    y = split_df["result"].values
    X = split_df.drop(columns=DROP_COLS, errors="ignore")
    return X, y


def print_class_dist(name, y):
    """Print class distribution for a split."""
    unique, counts = np.unique(y, return_counts=True)
    total = len(y)
    labels = {0: "Home Win", 1: "Draw", 2: "Away Win"}
    print(f"\n  {name}: {total} samples")
    for cls, cnt in zip(unique, counts):
        pct = 100.0 * cnt / total
        print(f"    Class {cls} ({labels.get(cls, '?')}): {cnt:>6} ({pct:5.1f}%)")


def apply_smote(X_train, y_train):
    """Apply SMOTE to balance the training set."""
    # SMOTE needs no NaN — fill with median for SMOTE fitting,
    # but we store the SMOTE-resampled data with the synthetic values.
    X_filled = X_train.copy()
    for col in X_filled.columns:
        if X_filled[col].isnull().any():
            X_filled[col] = X_filled[col].fillna(X_filled[col].median())

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_filled, y_train)

    X_resampled = pd.DataFrame(X_resampled, columns=X_train.columns)
    return X_resampled, y_resampled


def median_impute(X_train, X_val, X_test):
    """Impute missing values with training-set medians (for non-tree models).

    Medians are computed from training data only to prevent leakage.
    """
    medians = X_train.median()
    X_train_imp = X_train.fillna(medians)
    X_val_imp = X_val.fillna(medians)
    X_test_imp = X_test.fillna(medians)
    return X_train_imp, X_val_imp, X_test_imp


def main():
    print("=" * 60)
    print("03_create_splits.py — Chronological train/val/test splits")
    print("=" * 60)

    # Load feature matrix
    df = pd.read_csv(FEATURE_MATRIX)
    print(f"\nLoaded feature matrix: {df.shape[0]} rows, {df.shape[1]} columns")

    # Split chronologically
    train_df, val_df, test_df = load_and_split(df)
    print(f"\nChronological splits:")
    print(f"  Train: {train_df['date'].min().date()} to {train_df['date'].max().date()} ({len(train_df)} matches)")
    print(f"  Val:   {val_df['date'].min().date()} to {val_df['date'].max().date()} ({len(val_df)} matches)")
    print(f"  Test:  {test_df['date'].min().date()} to {test_df['date'].max().date()} ({len(test_df)} matches)")

    # Separate X and y
    X_train, y_train = separate_xy(train_df)
    X_val, y_val = separate_xy(val_df)
    X_test, y_test = separate_xy(test_df)

    print(f"\nFeature columns ({len(X_train.columns)}):")
    for col in X_train.columns:
        null_pct = 100.0 * X_train[col].isnull().sum() / len(X_train)
        print(f"  {col}: {null_pct:.1f}% missing in train")

    # Class distributions
    print("\n" + "-" * 40)
    print("Class distributions:")
    print_class_dist("Train", y_train)
    print_class_dist("Validation", y_val)
    print_class_dist("Test", y_test)

    # Save raw splits (NaN preserved — tree models handle these natively)
    print("\n" + "-" * 40)
    print("Saving raw splits (NaN preserved for tree models)...")
    X_train.to_csv(SPLITS_DIR / "X_train.csv", index=False)
    pd.Series(y_train, name="result").to_csv(SPLITS_DIR / "y_train.csv", index=False)
    X_val.to_csv(SPLITS_DIR / "X_val.csv", index=False)
    pd.Series(y_val, name="result").to_csv(SPLITS_DIR / "y_val.csv", index=False)
    X_test.to_csv(SPLITS_DIR / "X_test.csv", index=False)
    pd.Series(y_test, name="result").to_csv(SPLITS_DIR / "y_test.csv", index=False)
    print("  Saved: X_train.csv, y_train.csv, X_val.csv, y_val.csv, X_test.csv, y_test.csv")

    # Median-imputed splits (for non-tree models: LR, KNN, SVM)
    print("\nCreating median-imputed splits (for non-tree models)...")
    X_train_imp, X_val_imp, X_test_imp = median_impute(X_train, X_val, X_test)
    X_train_imp.to_csv(SPLITS_DIR / "X_train_median.csv", index=False)
    X_val_imp.to_csv(SPLITS_DIR / "X_val_median.csv", index=False)
    X_test_imp.to_csv(SPLITS_DIR / "X_test_median.csv", index=False)
    remaining_nulls = X_train_imp.isnull().sum().sum() + X_val_imp.isnull().sum().sum() + X_test_imp.isnull().sum().sum()
    print(f"  Medians computed from training set only (no leakage)")
    print(f"  Remaining NaNs after imputation: {remaining_nulls}")
    print("  Saved: X_train_median.csv, X_val_median.csv, X_test_median.csv")

    # SMOTE-oversampled training set
    print("\nApplying SMOTE to training set...")
    X_train_smote, y_train_smote = apply_smote(X_train, y_train)
    X_train_smote.to_csv(SPLITS_DIR / "X_train_smote.csv", index=False)
    pd.Series(y_train_smote, name="result").to_csv(SPLITS_DIR / "y_train_smote.csv", index=False)
    print_class_dist("Train (SMOTE)", y_train_smote)
    print("  Saved: X_train_smote.csv, y_train_smote.csv")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Features:       {len(X_train.columns)}")
    print(f"  Train:          {X_train.shape}")
    print(f"  Validation:     {X_val.shape}")
    print(f"  Test:           {X_test.shape}")
    print(f"  Train (SMOTE):  {X_train_smote.shape}")
    print(f"  Train (median): {X_train_imp.shape}")
    print(f"\n  All files saved to: {SPLITS_DIR}")


if __name__ == "__main__":
    main()
