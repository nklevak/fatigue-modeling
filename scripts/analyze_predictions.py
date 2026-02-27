"""
Analyze per-subject prediction distributions: compare predicted vs actual rest_length
for each test subject across epochs. Requires models saved via run_baselines.py --save-models.

Run baselines first to save models and splits:
  uv run python scripts/run_baselines.py --save-models models --save-split splits/compare
 
Then run this script (use same load-split for pooled so test set matches):
  uv run python scripts/analyze_predictions.py --models models --load-split splits/compare_pooled --out analysis

Produces: analysis_dataset_mean.png, analysis_dataset_std.png, analysis_dataset_examples.png
         analysis_pooled_mean.png, analysis_pooled_std.png, analysis_pooled_examples.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.preprocess import get_data, get_feature_columns
from src.split import split_by_dataset, train_test_split_pooled, load_split_ids, split_df_by_subject_ids


def _test_fair_df(val_df: pd.DataFrame, train_df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Replace subject-level features that would leak at test time (e.g. mean_rest_length_subj)."""
    out = val_df.copy()
    if "mean_rest_length_subj" in cols:
        out["mean_rest_length_subj"] = train_df["mean_rest_length_subj"].mean()
    return out


def load_models_and_predict(
    models_prefix: str,
    split: str,
    train_b: pd.DataFrame,
    test_b: pd.DataFrame,
    train_e: pd.DataFrame,
    test_e: pd.DataFrame,
    baseline_cols: list[str],
    ext_cols: list[str],
) -> dict[str, pd.DataFrame]:
    """
    Load saved models and predict on test. Returns dict mapping model_name -> test_df with pred columns.
    Baseline models use test_b; extended use test_e (different rows due to history NA).
    """
    # Match run_baselines save format: {prefix}_{split}_*.pkl
    p = Path(models_prefix)
    base = p.parent / f"{p.name}_{split}"
    test_b_fair = _test_fair_df(test_b, train_b, baseline_cols)
    test_e_fair = _test_fair_df(test_e, train_e, ext_cols)

    out = {}

    # Ridge baseline
    ridge_b, scaler_b = joblib.load(f"{base}_ridge_baseline.pkl")
    X_b = scaler_b.transform(test_b_fair[baseline_cols].to_numpy().astype(float))
    out["ridge_baseline"] = test_b.assign(pred=ridge_b.predict(X_b))

    # GBM baseline (raw features)
    gbm_b = joblib.load(f"{base}_gbm_baseline.pkl")
    X_b_raw = test_b_fair[baseline_cols].to_numpy().astype(float)
    out["gbm_baseline"] = test_b.assign(pred=gbm_b.predict(X_b_raw))

    # Ridge extended
    ridge_e, scaler_e = joblib.load(f"{base}_ridge_extended.pkl")
    X_e = scaler_e.transform(test_e_fair[ext_cols].to_numpy().astype(float))
    out["ridge_extended"] = test_e.assign(pred=ridge_e.predict(X_e))

    # GBM extended
    gbm_e = joblib.load(f"{base}_gbm_extended.pkl")
    X_e_raw = test_e_fair[ext_cols].to_numpy().astype(float)
    out["gbm_extended"] = test_e.assign(pred=gbm_e.predict(X_e_raw))

    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze per-subject predicted vs actual rest distributions"
    )
    parser.add_argument(
        "--models",
        type=str,
        default="models",
        metavar="PREFIX",
        help="Model path prefix (e.g. models → models_dataset_ridge_baseline.pkl)",
    )
    parser.add_argument(
        "--load-split",
        type=str,
        default=None,
        help="Load pooled split from this path (e.g. splits/compare_pooled.json). Required for pooled.",
    )
    parser.add_argument(
        "--test-frac",
        type=float,
        default=0.1,
        help="Test fraction for pooled when not using --load-split",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--out",
        type=str,
        default="analysis",
        help="Output prefix for plots (e.g. analysis → analysis_dataset_*.png)",
    )
    args = parser.parse_args()

    print("Loading data...")
    df = get_data()
    df_baseline = df.dropna(subset=get_feature_columns(baseline_only=True))
    df_extended = df.dropna(subset=get_feature_columns(baseline_only=False))

    out_prefix = Path(args.out)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    for split in ["dataset", "pooled"]:
        include_dataset = split == "pooled"
        baseline_cols = get_feature_columns(baseline_only=True, include_dataset=include_dataset)
        ext_cols = get_feature_columns(baseline_only=False, include_dataset=include_dataset)

        if split == "dataset":
            train_b, test_b = split_by_dataset(df_baseline, "original", "replication")
            train_e, test_e = split_by_dataset(df_extended, "original", "replication")
        else:
            if args.load_split:
                train_ids, test_ids = load_split_ids(args.load_split)
                train_b, test_b = split_df_by_subject_ids(df_baseline, train_ids, test_ids)
                train_e, test_e = split_df_by_subject_ids(df_extended, train_ids, test_ids)
            else:
                train_b, test_b = train_test_split_pooled(
                    df_baseline, test_frac=args.test_frac, random_state=args.seed
                )
                train_e, test_e = train_test_split_pooled(
                    df_extended, test_frac=args.test_frac, random_state=args.seed
                )

        print(f"\n--- Split: {split} ---")
        preds = load_models_and_predict(
            args.models, split, train_b, test_b, train_e, test_e,
            baseline_cols, ext_cols,
        )

        # Per-subject stats: mean and std of rest_length (actual) vs pred
        # Use test_e for subject list (extended has slightly fewer rows; both have same subjects in test)
        subjects = np.union1d(test_b["subject_id"].unique(), test_e["subject_id"].unique())
        model_names = ["ridge_baseline", "gbm_baseline", "ridge_extended", "gbm_extended"]

        # 1. Mean predicted vs mean actual per subject (2x2 grid)
        fig1, axes = plt.subplots(2, 2, figsize=(10, 10))
        axes = axes.flatten()
        for ax, name in zip(axes, model_names):
            p = preds[name]
            agg = p.groupby("subject_id").agg(
                mean_actual=("rest_length", "mean"),
                mean_pred=("pred", "mean"),
            )
            ax.scatter(agg["mean_actual"], agg["mean_pred"], alpha=0.7, s=40)
            mx = max(agg["mean_actual"].max(), agg["mean_pred"].max())
            mn = min(agg["mean_actual"].min(), agg["mean_pred"].min())
            ax.plot([mn, mx], [mn, mx], "k--", alpha=0.5, label="Perfect")
            ax.set_xlabel("Mean actual rest per subject")
            ax.set_ylabel("Mean predicted rest per subject")
            ax.set_title(name.replace("_", " ").title())
            ax.set_aspect("equal")
            ax.legend()
        fig1.suptitle(f"Mean rest per subject: predicted vs actual ({split})", fontsize=12)
        fig1.tight_layout()
        path1 = out_prefix.parent / f"{out_prefix.stem}_{split}_mean.png"
        fig1.savefig(path1, dpi=150)
        print(f"Saved {path1}")
        plt.close(fig1)

        # 2. Std predicted vs std actual per subject
        fig2, axes = plt.subplots(2, 2, figsize=(10, 10))
        axes = axes.flatten()
        for ax, name in zip(axes, model_names):
            p = preds[name]
            agg = p.groupby("subject_id").agg(
                std_actual=("rest_length", "std"),
                std_pred=("pred", "std"),
            )
            agg = agg.dropna()  # some subjects may have std=NaN (single epoch)
            if len(agg) == 0:
                ax.text(0.5, 0.5, "No data", ha="center", va="center")
                ax.set_title(name.replace("_", " ").title())
                continue
            ax.scatter(agg["std_actual"], agg["std_pred"], alpha=0.7, s=40)
            mx = max(agg["std_actual"].max(), agg["std_pred"].max())
            mn = min(agg["std_actual"].min(), agg["std_pred"].min())
            ax.plot([mn, mx], [mn, mx], "k--", alpha=0.5, label="Perfect")
            ax.set_xlabel("Std actual rest per subject")
            ax.set_ylabel("Std predicted rest per subject")
            ax.set_title(name.replace("_", " ").title())
            ax.set_aspect("equal")
            ax.legend()
        fig2.suptitle(f"Std rest per subject: predicted vs actual ({split})", fontsize=12)
        fig2.tight_layout()
        path2 = out_prefix.parent / f"{out_prefix.stem}_{split}_std.png"
        fig2.savefig(path2, dpi=150)
        print(f"Saved {path2}")
        plt.close(fig2)

        # 3. Example subjects: epoch-level predicted vs actual (pick first 3 from extended test)
        test_subjects = test_e["subject_id"].unique()
        n_example = min(3, len(test_subjects))
        example_subjects = test_subjects[:n_example]
        fig3, axes = plt.subplots(n_example, 1, figsize=(10, 3 * n_example))
        if n_example == 1:
            axes = [axes]
        for ax, sid in zip(axes, example_subjects):
            p = preds["ridge_extended"]
            sub = p[p["subject_id"] == sid].sort_values("epoch_num")
            ax.plot(sub["epoch_num"], sub["rest_length"], "o-", label="Actual", color="C0")
            ax.plot(sub["epoch_num"], sub["pred"], "s--", label="Predicted", color="C1")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Rest length")
            ax.set_title(f"Subject {sid} (Ridge extended)")
            ax.legend()
            ax.set_xlim(sub["epoch_num"].min() - 0.5, sub["epoch_num"].max() + 0.5)
        fig3.suptitle(f"Example subjects: predicted vs actual over epochs ({split})", fontsize=12)
        fig3.tight_layout()
        path3 = out_prefix.parent / f"{out_prefix.stem}_{split}_examples.png"
        fig3.savefig(path3, dpi=150)
        print(f"Saved {path3}")
        plt.close(fig3)

    print("\nDone.")


if __name__ == "__main__":
    main()
