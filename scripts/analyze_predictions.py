"""
Analyze per-subject prediction distributions: compare predicted vs actual rest_length
for each test subject across epochs. Requires models saved via run_baselines or run_all_models.

With run_all_models output (baselines + LSTM in models/):
  uv run python scripts/analyze_predictions.py --models-dir models --load-split splits/compare_pooled --split pooled

Legacy (baselines only):
  uv run python scripts/analyze_predictions.py --models models --load-split splits/compare_pooled --out analysis

Produces: mean/scatter plots, std plots, example trajectories, and trajectory plots per subject.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.preprocess import get_data, get_feature_columns
from src.split import split_by_dataset, train_test_split_pooled, load_split_ids, split_df_by_subject_ids
from src.lstm_model import RestLSTM, load_state_dict
from scripts.train_lstm import build_sequences


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


def load_lstm_and_predict(
    lstm_path: str | Path,
    train_e: pd.DataFrame,
    test_e: pd.DataFrame,
    ext_cols: list[str],
    hidden: int = 16,
    num_layers: int = 2,
) -> pd.DataFrame:
    """
    Load saved LSTM (.npz) and predict on test. Returns test_e with pred column.
    """
    path = Path(lstm_path)
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=True)
    features_path = path.parent / (path.stem + "_features.txt")
    if features_path.exists():
        feature_cols = [c.strip() for c in features_path.read_text().strip().split("\n")]
    else:
        feature_cols = ext_cols

    X_test, y_test, lengths = build_sequences(test_e, feature_cols, train_df=train_e)

    if "scaler_mean" in data:
        scaler_mean = data["scaler_mean"]
        scaler_scale = data["scaler_scale"]
        X_flat = X_test.reshape(-1, X_test.shape[-1])
        X_flat = (X_flat - scaler_mean) / scaler_scale
        X_test = X_flat.reshape(X_test.shape).astype(np.float32)

    n_features = len(feature_cols)
    model = RestLSTM(n_features=n_features, hidden_size=hidden, num_layers=num_layers)
    state = {"layers": [], "fc_W": data["fc_W"], "fc_b": data["fc_b"]}
    i = 0
    while f"layer{i}_W" in data:
        state["layers"].append({"W": data[f"layer{i}_W"], "b": data[f"layer{i}_b"]})
        i += 1
    load_state_dict(model, state)

    # Subject order matches build_sequences (df["subject_id"].unique())
    test_e_fair = _test_fair_df(test_e, train_e, feature_cols)
    subjects_ordered = test_e_fair["subject_id"].unique()

    pred_by_subj_epoch = {}
    for j in range(len(X_test)):
        mask = np.zeros((1, X_test.shape[1]), dtype=bool)
        mask[0, : lengths[j]] = True
        out = model.predict(X_test[j : j + 1], mask=mask).squeeze(-1)[0]
        preds = out[: lengths[j]].tolist()
        sid = subjects_ordered[j]
        sub = test_e[test_e["subject_id"] == sid].sort_values("epoch_num")
        for k, (_, row) in enumerate(sub.iterrows()):
            ep = int(row["epoch_num"])
            pred_by_subj_epoch[(sid, ep)] = preds[k] if k < len(preds) else np.nan

    pred_col = test_e.apply(
        lambda r: pred_by_subj_epoch.get((r["subject_id"], r["epoch_num"]), np.nan), axis=1
    )
    return test_e.assign(pred=pred_col.values)


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
        default=None,
        help="Output directory/prefix for plots. Default: --models-dir if set, else 'analysis'",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default=None,
        help="Models directory from run_all_models (models/baselines_*, models/lstm_*.npz). Sets --models and --out.",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["dataset", "pooled", "both"],
        default="both",
        help="Which split(s) to analyze. Default: both",
    )
    parser.add_argument(
        "--lstm-hidden",
        type=int,
        default=16,
        help="LSTM hidden size (must match saved model)",
    )
    parser.add_argument(
        "--lstm-layers",
        type=int,
        default=2,
        help="LSTM num layers (must match saved model)",
    )
    parser.add_argument(
        "--n-trajectory-examples",
        type=int,
        default=6,
        help="Number of example subjects for trajectory plots",
    )
    args = parser.parse_args()

    if args.models_dir:
        models_dir = Path(args.models_dir)
        args.models = str(models_dir / "baselines")
        args.out = args.out or str(models_dir)
    args.out = args.out or "analysis"

    print("Loading data...")
    df = get_data()
    df_baseline = df.dropna(subset=get_feature_columns(baseline_only=True))
    df_extended = df.dropna(subset=get_feature_columns(baseline_only=False))

    out_prefix = Path(args.out)
    out_prefix.mkdir(parents=True, exist_ok=True)

    splits_to_run = ["dataset", "pooled"] if args.split == "both" else [args.split]

    for split in splits_to_run:
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

        # Load LSTM if available (run_all_models saves models/lstm_{split}.npz)
        lstm_path = Path(args.out) / f"lstm_{split}.npz"
        if not lstm_path.exists():
            lstm_path = Path(args.models).parent / f"lstm_{split}.npz"
        lstm_df = load_lstm_and_predict(
            lstm_path, train_e, test_e, ext_cols,
            hidden=args.lstm_hidden, num_layers=args.lstm_layers,
        )
        if lstm_df is not None:
            preds["lstm"] = lstm_df
            print(f"  Loaded LSTM from {lstm_path}")

        subjects = np.union1d(test_b["subject_id"].unique(), test_e["subject_id"].unique())
        model_names = [n for n in ["ridge_baseline", "gbm_baseline", "ridge_extended", "gbm_extended", "lstm"] if n in preds]

        # 1. Mean predicted vs mean actual per subject
        n_models = len(model_names)
        ncols = 2 if n_models <= 4 else 3
        nrows = max(1, (n_models + ncols - 1) // ncols)
        fig1, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows), squeeze=False)
        axes = axes.flatten()[:n_models]
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
        path1 = Path(args.out) / f"analysis_{split}_mean.png"
        fig1.savefig(path1, dpi=150)
        print(f"Saved {path1}")
        plt.close(fig1)

        # 2. Std predicted vs std actual per subject
        fig2, axes2 = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows), squeeze=False)
        axes2 = axes2.flatten()[:n_models]
        for ax, name in zip(axes2, model_names):
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
        path2 = Path(args.out) / f"analysis_{split}_std.png"
        fig2.savefig(path2, dpi=150)
        print(f"Saved {path2}")
        plt.close(fig2)

        # 3. Example subjects: one model per subplot (ridge_extended or first available)
        ref_model = "ridge_extended" if "ridge_extended" in preds else model_names[0]
        test_subjects = test_e["subject_id"].unique()
        n_example = min(3, len(test_subjects))
        example_subjects = test_subjects[:n_example]
        fig3, axes3 = plt.subplots(n_example, 1, figsize=(10, 3 * n_example))
        if n_example == 1:
            axes3 = [axes3]
        for ax, sid in zip(axes3, example_subjects):
            p = preds[ref_model]
            sub = p[p["subject_id"] == sid].sort_values("epoch_num")
            ax.plot(sub["epoch_num"], sub["rest_length"], "o-", label="Actual", color="C0")
            ax.plot(sub["epoch_num"], sub["pred"], "s--", label="Predicted", color="C1")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Rest length")
            ax.set_title(f"Subject {sid} ({ref_model})")
            ax.legend()
            ax.set_xlim(sub["epoch_num"].min() - 0.5, sub["epoch_num"].max() + 0.5)
        fig3.suptitle(f"Example subjects: predicted vs actual over epochs ({split})", fontsize=12)
        fig3.tight_layout()
        path3 = Path(args.out) / f"analysis_{split}_examples.png"
        path3.parent.mkdir(parents=True, exist_ok=True)
        fig3.savefig(path3, dpi=150)
        print(f"Saved {path3}")
        plt.close(fig3)

        # 4. Trajectory plots: predicted vs actual for each example subject, all models
        n_traj = min(args.n_trajectory_examples, len(test_subjects))
        traj_subjects = test_subjects[:n_traj]
        colors = plt.cm.tab10(np.linspace(0, 1, len(model_names) + 1))
        for ti, sid in enumerate(traj_subjects):
            fig4, ax = plt.subplots(figsize=(10, 5))
            sub_actual = test_e[test_e["subject_id"] == sid].sort_values("epoch_num")
            epochs = sub_actual["epoch_num"].values
            actuals = sub_actual["rest_length"].values
            ax.plot(epochs, actuals, "o-", label="Actual", color="black", linewidth=2, markersize=8)
            for mi, name in enumerate(model_names):
                p = preds[name]
                sub = p[p["subject_id"] == sid].sort_values("epoch_num")
                if len(sub) > 0:
                    ax.plot(sub["epoch_num"], sub["pred"], "--", label=name.replace("_", " "), color=colors[mi], alpha=0.8)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Rest length")
            ax.set_title(f"Subject {sid}: predicted vs actual trajectory ({split})")
            ax.legend(loc="best", fontsize=8)
            ax.set_xlim(epochs.min() - 0.5, epochs.max() + 0.5)
            fig4.tight_layout()
            safe_sid = str(sid).replace("/", "_").replace(" ", "_")
            path4 = Path(args.out) / f"trajectory_{split}_{safe_sid}.png"
            path4.parent.mkdir(parents=True, exist_ok=True)
            fig4.savefig(path4, dpi=150)
            plt.close(fig4)
        print(f"Saved {n_traj} trajectory plots to {Path(args.out)}")

    print("\nDone.")


if __name__ == "__main__":
    main()
