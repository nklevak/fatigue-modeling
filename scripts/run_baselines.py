"""
Run the baseline models (Ridge + Gradient Boosting) and print MAE and R².
These are first run without history features, then run again with history (these history features
are previous epoch's performance, rt, and overall accuracy per task)

Usage:
  uv run python scripts/run_baselines.py
  uv run python scripts/run_baselines.py --split subject
  uv run python scripts/run_baselines.py --split subject --n_folds 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.preprocess import get_data, get_feature_columns
from src.split import split_by_dataset, kfold_by_subject


def main() -> None:
    def format_metric(values: list[float]) -> str:
        a = np.array(values)
        if len(a) == 1:
            return f"{a[0]:.4f}"
        return f"{a.mean():.4f} ± {a.std():.4f}"

    parser = argparse.ArgumentParser(description="Run baseline fatigue prediction models")
    parser.add_argument(
        "--split",
        choices=["dataset", "subject"],
        default="dataset",
        help="By dataset (train=original, val=replication) or by subject (k-fold)",
    )
    parser.add_argument("--n_folds", type=int, default=10, help="Folds when --split subject")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # One preprocessing pipeline
    print("Loading and building features...")
    df = get_data()
    baseline_cols = get_feature_columns(baseline_only=True)
    ext_cols = get_feature_columns(baseline_only=False)
    # Use complete cases only (blockwise can have NA avg_rt when no valid RTs; we do not impute)
    df_baseline = df.dropna(subset=baseline_cols)
    df_extended = df.dropna(subset=ext_cols)

    # Splits (by subject: no mixing epochs across subjects)
    if args.split == "dataset":
        folds_baseline = [split_by_dataset(df_baseline, "original", "replication")]
        folds_extended = [split_by_dataset(df_extended, "original", "replication")]
        n_folds_str = "1 (train=original, val=replication)"
    else:
        folds_baseline = list(kfold_by_subject(df_baseline, n_splits=args.n_folds, random_state=args.seed))
        folds_extended = list(kfold_by_subject(df_extended, n_splits=args.n_folds, random_state=args.seed))
        n_folds_str = str(args.n_folds)

    print(f"Split: {args.split} ({n_folds_str} fold(s))\n")

    # Baseline features: Ridge and GBM
    mae_ridge, r2_ridge = [], []
    mae_gbm, r2_gbm = [], []
    for train, val in folds_baseline:
        X_tr = train[baseline_cols].to_numpy()
        y_tr = train["rest_length"].to_numpy()
        X_va = val[baseline_cols].to_numpy()
        y_va = val["rest_length"].to_numpy()

        ridge = Ridge(alpha=1.0, random_state=args.seed).fit(X_tr, y_tr)
        mae_ridge.append(mean_absolute_error(y_va, ridge.predict(X_va)))
        r2_ridge.append(r2_score(y_va, ridge.predict(X_va)))

        gbm = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=args.seed).fit(X_tr, y_tr)
        mae_gbm.append(mean_absolute_error(y_va, gbm.predict(X_va)))
        r2_gbm.append(r2_score(y_va, gbm.predict(X_va)))

    print("--- Ridge (baseline features) ---")
    print(f"MAE: {format_metric(mae_ridge)}\nR²:  {format_metric(r2_ridge)}")
    print("\n--- Gradient Boosting (baseline features) ---")
    print(f"MAE: {format_metric(mae_gbm)}\nR²:  {format_metric(r2_gbm)}")

    # Extended features: Ridge and GBM
    mae_re, r2_re = [], []
    mae_ge, r2_ge = [], []
    for train, val in folds_extended:
        X_tr = train[ext_cols].to_numpy().astype(float)
        y_tr = train["rest_length"].to_numpy()
        X_va = val[ext_cols].to_numpy().astype(float)
        y_va = val["rest_length"].to_numpy()

        ridge = Ridge(alpha=1.0, random_state=args.seed).fit(X_tr, y_tr)
        mae_re.append(mean_absolute_error(y_va, ridge.predict(X_va)))
        r2_re.append(r2_score(y_va, ridge.predict(X_va)))

        gbm = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=args.seed).fit(X_tr, y_tr)
        mae_ge.append(mean_absolute_error(y_va, gbm.predict(X_va)))
        r2_ge.append(r2_score(y_va, gbm.predict(X_va)))

    print("\n--- Extended features (with history) ---")
    print("Ridge + history:")
    print(f"  MAE: {format_metric(mae_re)}\n  R²:  {format_metric(r2_re)}")
    print("GBM + history:")
    print(f"  MAE: {format_metric(mae_ge)}\n  R²:  {format_metric(r2_ge)}")
    print("\nDone.")


if __name__ == "__main__":
    main()
