"""
Run baselines (Ridge + GBM with tuning) and LSTM on both dataset and pooled splits.
Saves all models, a description document, MAE/R² comparison plot, and results for analysis.

Usage:
  uv run python scripts/run_all_models.py
  uv run python scripts/run_all_models.py --models-dir models/run1 --load-split splits/compare_pooled
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.preprocess import get_feature_columns

from scripts.run_baselines import run_one_config  # noqa: E402
from scripts.train_lstm import run_lstm  # noqa: E402


def plot_comparison_all(
    baseline_results: list[dict],
    lstm_results: list[dict],
    out_path: Path,
) -> None:
    """Plot MAE and R² comparison including baselines + LSTM."""
    model_labels = [
        "Ridge\n(baseline)",
        "GBM\n(baseline)",
        "Ridge\n(extended)",
        "GBM\n(extended)",
        "LSTM",
    ]
    keys_mae = [
        "ridge_baseline_mae",
        "gbm_baseline_mae",
        "ridge_extended_mae",
        "gbm_extended_mae",
    ]
    keys_r2 = [
        "ridge_baseline_r2",
        "gbm_baseline_r2",
        "ridge_extended_r2",
        "gbm_extended_r2",
    ]
    x = np.arange(len(model_labels))
    width = 0.35

    fig1, ax1 = plt.subplots(figsize=(10, 5))
    for i, res in enumerate(baseline_results):
        mae_vals = [res[k] for k in keys_mae]
        lstm_res = next((r for r in lstm_results if r["split"] == res["split"]), None)
        mae_vals = mae_vals + ([lstm_res["lstm_mae"]] if lstm_res else [np.nan])
        offset = (i - 0.5) * width
        label = "train=original, test=replication" if res["split"] == "dataset" else "pooled (90% train, 10% test)"
        ax1.bar(x + offset, mae_vals, width, label=label)
    ax1.set_ylabel("Test MAE")
    ax1.set_xlabel("Model")
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_labels)
    ax1.set_title("Test MAE by split method and model")
    ax1.legend()
    fig1.tight_layout()
    fig1.savefig(out_path / "comparison_mae.png", dpi=150)
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    for i, res in enumerate(baseline_results):
        r2_vals = [res[k] for k in keys_r2]
        lstm_res = next((r for r in lstm_results if r["split"] == res["split"]), None)
        r2_vals = r2_vals + ([lstm_res["lstm_r2"]] if lstm_res else [np.nan])
        offset = (i - 0.5) * width
        label = "train=original, test=replication" if res["split"] == "dataset" else "pooled (90% train, 10% test)"
        ax2.bar(x + offset, r2_vals, width, label=label)
    ax2.set_ylabel("Test R²")
    ax2.set_xlabel("Model")
    ax2.set_xticks(x)
    ax2.set_xticklabels(model_labels)
    ax2.set_title("Test R² by split method and model")
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(out_path / "comparison_r2.png", dpi=150)
    plt.close(fig2)


def write_description(
    models_dir: Path,
    baseline_results: list[dict],
    lstm_results: list[dict],
    baseline_args: dict,
    lstm_params: dict,
) -> None:
    """Write MODELS_README.md with features and params for all models."""
    lines = [
        "# Model Configuration Summary",
        "",
        "Features and hyperparameters used for baselines and LSTM.",
        "",
        "## Splits",
        "- **dataset**: train=original, test=replication",
        "- **pooled**: 90% train / 10% test (using pre-determined split when --load-split provided)",
        "",
        "## Baseline Features (baseline_only=True)",
        "",
    ]
    baseline_cols = baseline_results[0].get("baseline_cols", get_feature_columns(baseline_only=True))
    lines.extend([f"- {c}" for c in baseline_cols])
    lines.append("")
    lines.append("## Extended Features (baseline_only=False, for Ridge extended, GBM extended, LSTM)")
    lines.append("")
    ext_cols = baseline_results[0].get("ext_cols", get_feature_columns(baseline_only=False))
    lines.extend([f"- {c}" for c in ext_cols])
    lines.append("")
    lines.append("## Baseline Tuning & Params")
    lines.append("")
    lines.append(f"- Ridge alpha sweep: {baseline_args.get('alphas', '0.01,0.05,0.1,0.5,1,10,20,100')}")
    lines.append(f"- GBM n_estimators: {baseline_args.get('gbm_n_estimators', '100,200,300,400,500')}")
    lines.append(f"- GBM max_depth: {baseline_args.get('gbm_max_depth', '1,2,3,4')}")
    lines.append(f"- GBM learning_rate: {baseline_args.get('gbm_learning_rate', 0.05)}")
    lines.append(f"- n_folds: {baseline_args.get('n_folds', 15)}")
    lines.append("")
    for res in baseline_results:
        s = res["split"]
        lines.append(f"### {s} split (tuned values)")
        lines.append(f"- Ridge alpha (baseline): {res.get('ridge_alpha_baseline', 'N/A')}")
        lines.append(f"- Ridge alpha (extended): {res.get('ridge_alpha_extended', 'N/A')}")
        lines.append(f"- GBM n_estimators (baseline): {res.get('gbm_n_baseline', 'N/A')}")
        lines.append(f"- GBM max_depth (baseline): {res.get('gbm_d_baseline', 'N/A')}")
        lines.append(f"- GBM n_estimators (extended): {res.get('gbm_n_extended', 'N/A')}")
        lines.append(f"- GBM max_depth (extended): {res.get('gbm_d_extended', 'N/A')}")
        lines.append("")
    lines.append("## LSTM Params")
    lines.append("")
    for k, v in lstm_params.items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    (models_dir / "MODELS_README.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run baselines + LSTM on dataset and pooled splits, save models and results"
    )
    parser.add_argument("--models-dir", type=str, default="models", help="Output directory for models and results")
    parser.add_argument("--load-split", type=str, default="splits/compare_pooled", help="Load pooled split from this path")
    parser.add_argument("--n-folds", type=int, default=15)
    parser.add_argument("--test-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lstm-epochs", type=int, default=60)
    parser.add_argument("--lstm-hidden", type=int, default=16)
    parser.add_argument("--lstm-layers", type=int, default=2)
    parser.add_argument("--lstm-dropout", type=float, default=0.5)
    parser.add_argument("--lstm-lr", type=float, default=2e-3)
    parser.add_argument("--lstm-weight-decay", type=float, default=1e-3)
    args = parser.parse_args()

    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("RUNNING BASELINES (Ridge + GBM with tuning)")
    print("=" * 60)

    from src.preprocess import get_data  # noqa: E402

    df = get_data()
    df_baseline = df.dropna(subset=get_feature_columns(baseline_only=True))
    df_extended = df.dropna(subset=get_feature_columns(baseline_only=False))

    baseline_results = []
    for split in ["dataset", "pooled"]:
        print(f"\n--- Split: {split} ---")
        include_dataset = split == "pooled"
        baseline_cols = get_feature_columns(baseline_only=True, include_dataset=include_dataset)
        ext_cols = get_feature_columns(baseline_only=False, include_dataset=include_dataset)
        load_split_path = args.load_split if split == "pooled" else None
        save_prefix = str(models_dir / "baselines")
        res = run_one_config(
            df_baseline,
            df_extended,
            baseline_cols,
            ext_cols,
            split=split,
            n_folds=args.n_folds,
            test_frac=args.test_frac,
            seed=args.seed,
            tune_alpha=True,
            tune_gbm=True,
            alphas_str="0.01,0.05,0.1,0.5,1,10,20,100",
            gbm_n_estimators_str="100,200,300,400,500",
            gbm_max_depth_str="1,2,3,4",
            gbm_learning_rate=0.05,
            gbm_early_stopping=True,
            gbm_final_squeeze=False,
            no_scale=False,
            load_split=load_split_path,
            save_split=None,
            save_models=save_prefix,
            verbose=True,
        )
        baseline_results.append(res)
        print(f"  Test MAE  Ridge base: {res['ridge_baseline_mae']:.4f}  GBM base: {res['gbm_baseline_mae']:.4f}  Ridge ext: {res['ridge_extended_mae']:.4f}  GBM ext: {res['gbm_extended_mae']:.4f}")
        print(f"  Test R²   Ridge base: {res['ridge_baseline_r2']:.4f}  GBM base: {res['gbm_baseline_r2']:.4f}  Ridge ext: {res['ridge_extended_r2']:.4f}  GBM ext: {res['gbm_extended_r2']:.4f}")

    print("\n" + "=" * 60)
    print("RUNNING LSTM")
    print("=" * 60)

    lstm_results = []
    for split in ["dataset", "pooled"]:
        print(f"\n--- LSTM Split: {split} ---")
        load_split_path = args.load_split if split == "pooled" else None
        save_path = str(models_dir / f"lstm_{split}.npz")
        res = run_lstm(
            split=split,
            load_split=load_split_path,
            test_frac=args.test_frac,
            seed=args.seed,
            epochs=args.lstm_epochs,
            hidden=args.lstm_hidden,
            layers=args.lstm_layers,
            dropout=args.lstm_dropout,
            lr=args.lstm_lr,
            weight_decay=args.lstm_weight_decay,
            batch_size=8,
            save=save_path,
            no_scale=False,
            verbose=True,
        )
        lstm_results.append(res)
        print(f"  Final Test MAE: {res['lstm_mae']:.4f}, R²: {res['lstm_r2']:.4f}")

    # Build combined results for JSON (serializable)
    results_for_json = []
    for br, lr in zip(baseline_results, lstm_results):
        results_for_json.append({
            "split": br["split"],
            "ridge_baseline_mae": br["ridge_baseline_mae"],
            "ridge_baseline_r2": br["ridge_baseline_r2"],
            "gbm_baseline_mae": br["gbm_baseline_mae"],
            "gbm_baseline_r2": br["gbm_baseline_r2"],
            "ridge_extended_mae": br["ridge_extended_mae"],
            "ridge_extended_r2": br["ridge_extended_r2"],
            "gbm_extended_mae": br["gbm_extended_mae"],
            "gbm_extended_r2": br["gbm_extended_r2"],
            "lstm_mae": lr["lstm_mae"],
            "lstm_r2": lr["lstm_r2"],
        })
    with open(models_dir / "results.json", "w") as f:
        json.dump(results_for_json, f, indent=2)
    print(f"\nSaved results to {models_dir / 'results.json'}")

    # Comparison plot
    plot_comparison_all(baseline_results, lstm_results, models_dir)
    print(f"Saved comparison_mae.png and comparison_r2.png to {models_dir}")

    # Description document
    write_description(
        models_dir,
        baseline_results,
        lstm_results,
        baseline_args={
            "n_folds": args.n_folds,
            "alphas": "0.01,0.05,0.1,0.5,1,10,20,100",
            "gbm_n_estimators": "100,200,300,400,500",
            "gbm_max_depth": "1,2,3,4",
            "gbm_learning_rate": 0.05,
        },
        lstm_params={
            "epochs": args.lstm_epochs,
            "hidden": args.lstm_hidden,
            "layers": args.lstm_layers,
            "dropout": args.lstm_dropout,
            "lr": args.lstm_lr,
            "weight_decay": args.lstm_weight_decay,
            "batch_size": 8,
        },
    )
    print(f"Saved MODELS_README.md to {models_dir}")

    # Run trajectory analysis for pooled: predict + plot predicted vs actual
    print("\n" + "=" * 60)
    print("RUNNING TRAJECTORY ANALYSIS (pooled)")
    print("=" * 60)
    try:
        from scripts import analyze_predictions
        old_argv = sys.argv
        sys.argv = [
            "analyze_predictions",
            "--models-dir", str(models_dir),
            "--load-split", args.load_split,
            "--split", "pooled",
        ]
        analyze_predictions.main()
        sys.argv = old_argv
    except Exception as e:
        print(f"Skipping trajectory analysis: {e}")
        print("Run manually: uv run python scripts/analyze_predictions.py --models-dir models --load-split splits/compare_pooled --split pooled")

    print("\nDone.")


if __name__ == "__main__":
    main()
