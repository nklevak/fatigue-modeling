"""
Run the baseline models (Ridge + Gradient Boosting) and print MAE and R².
Compares both split methods: dataset (train=original, test=replication) and pooled (90/10).
Tunes Ridge alpha and GBM params via k-fold CV, then fits final models and evaluates on held-out test.

Usage:
  uv run python scripts/run_baselines.py
  uv run python scripts/run_baselines.py --compare-out comparison --gbm-final-squeeze
  uv run python scripts/run_baselines.py --compare-out comparison --save-models models --save-split splits/compare
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.preprocess import get_data, get_feature_columns
from src.split import (
    split_by_dataset,
    kfold_by_subject,
    train_test_split_pooled,
    save_split_ids,
    load_split_ids,
    split_df_by_subject_ids,
)


def sweep_ridge_alpha(
    folds: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    alphas: List[float],
    seed: int = 42,
    scale: bool = True,
) -> List[dict]:
    """
    For each alpha, fit Ridge on each fold and collect validation MAE and R².
    Used internally for Ridge alpha tuning (best R² on CV).
    """
    results = []
    for alpha in alphas:
        mae_list, r2_list = [], []
        for X_tr, y_tr, X_va, y_va in folds:
            if scale:
                scaler = StandardScaler().fit(X_tr)
                X_tr = scaler.transform(X_tr)
                X_va = scaler.transform(X_va)
            model = Ridge(alpha=alpha, random_state=seed).fit(X_tr, y_tr)
            pred = model.predict(X_va)
            mae_list.append(mean_absolute_error(y_va, pred))
            r2_list.append(r2_score(y_va, pred))
        results.append({
            "alpha": alpha,
            "mae_mean": float(np.mean(mae_list)),
            "mae_std": float(np.std(mae_list)),
            "r2_mean": float(np.mean(r2_list)),
            "r2_std": float(np.std(r2_list)),
        })
    return results


def _gbm_kwargs(n_estimators: int, max_depth: int, learning_rate: float, seed: int, early_stopping: bool = True):
    """Common GBM args: small learning_rate and early stopping to reduce overfitting on small data."""
    kwargs = dict(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=seed,
    )
    if early_stopping:
        kwargs["validation_fraction"] = 0.1
        kwargs["n_iter_no_change"] = 10
        kwargs["tol"] = 1e-4
    return kwargs


def tune_gbm_params(
    folds: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    n_estimators_list: List[int],
    max_depth_list: List[int],
    seed: int = 42,
    learning_rate: float = 0.1,
    early_stopping: bool = True,
) -> Tuple[int, int]:
    """
    Grid search over n_estimators and max_depth using mean validation R².
    Uses learning_rate=0.1 by default ('start high') and early stopping to find optimal trees.
    n_estimators is the max; early stopping may stop before that. Returns (best_n_estimators, best_max_depth).
    """
    best_r2 = -np.inf
    best_n, best_d = n_estimators_list[0], max_depth_list[0]
    for n_est in n_estimators_list:
        for d in max_depth_list:
            r2_list = []
            for X_tr, y_tr, X_va, y_va in folds:
                model = GradientBoostingRegressor(
                    **_gbm_kwargs(n_est, d, learning_rate, seed, early_stopping)
                ).fit(X_tr, y_tr)
                r2_list.append(r2_score(y_va, model.predict(X_va)))
            mean_r2 = float(np.mean(r2_list))
            if mean_r2 > best_r2:
                best_r2 = mean_r2
                best_n, best_d = n_est, d
    return best_n, best_d


def _to_arrays_fair(train_dfs, val_dfs, cols):
    out = []
    for tr, va in zip(train_dfs, val_dfs):
        va_fair = va.copy()
        if "mean_rest_length_subj" in cols:
            va_fair["mean_rest_length_subj"] = tr["mean_rest_length_subj"].mean()
        out.append((
            tr[cols].to_numpy().astype(float), tr["rest_length"].to_numpy(),
            va_fair[cols].to_numpy().astype(float), va["rest_length"].to_numpy(),
        ))
    return out


def _test_fair_df(val_df, train_df, cols):
    out = val_df.copy()
    if "mean_rest_length_subj" in cols:
        out["mean_rest_length_subj"] = train_df["mean_rest_length_subj"].mean()
    return out


def run_one_config(
    df_baseline,
    df_extended,
    baseline_cols: List[str],
    ext_cols: List[str],
    split: str,
    n_folds: int,
    test_frac: float,
    seed: int,
    tune_alpha: bool,
    tune_gbm: bool,
    alphas_str: str,
    gbm_n_estimators_str: str,
    gbm_max_depth_str: str,
    gbm_learning_rate: float = 0.1,
    gbm_early_stopping: bool = True,
    gbm_final_squeeze: bool = False,
    no_scale: bool = False,
    load_split: str | None = None,
    save_split: str | None = None,
    save_models: str | None = None,
    verbose: bool = True,
) -> dict:
    """
    Run one train/test config (dataset or pooled) with optional tuning.
    Returns dict with keys: split, ridge_baseline_mae, ridge_baseline_r2, gbm_baseline_mae, gbm_baseline_r2,
    ridge_extended_mae, ridge_extended_r2, gbm_extended_mae, gbm_extended_r2.
    """
    if split == "dataset":
        train_baseline, test_baseline = split_by_dataset(df_baseline, "original", "replication")
        train_extended, test_extended = split_by_dataset(df_extended, "original", "replication")
        if save_split:
            save_split_ids(
                train_baseline["subject_id"].unique().tolist(),
                test_baseline["subject_id"].unique().tolist(),
                save_split,
            )
    else:
        if load_split:
            train_ids, test_ids = load_split_ids(load_split)
            train_baseline, test_baseline = split_df_by_subject_ids(df_baseline, train_ids, test_ids)
            train_extended, test_extended = split_df_by_subject_ids(df_extended, train_ids, test_ids)
        else:
            train_baseline, test_baseline = train_test_split_pooled(
                df_baseline, test_frac=test_frac, random_state=seed
            )
            train_extended, test_extended = train_test_split_pooled(
                df_extended, test_frac=test_frac, random_state=seed
            )
        if save_split:
            save_split_ids(
                train_baseline["subject_id"].unique().tolist(),
                test_baseline["subject_id"].unique().tolist(),
                save_split,
            )

    folds_baseline = list(kfold_by_subject(train_baseline, n_splits=n_folds, random_state=seed))
    folds_extended = list(kfold_by_subject(train_extended, n_splits=n_folds, random_state=seed))
    folds_b_arrays = _to_arrays_fair([f[0] for f in folds_baseline], [f[1] for f in folds_baseline], baseline_cols)
    folds_e_arrays = _to_arrays_fair([f[0] for f in folds_extended], [f[1] for f in folds_extended], ext_cols)

    ridge_alpha_baseline = 1.0
    ridge_alpha_extended = 1.0
    if tune_alpha:
        alphas = [float(x.strip()) for x in alphas_str.split(",")]
        use_scale = not no_scale
        rb = sweep_ridge_alpha(folds_b_arrays, alphas, seed=seed, scale=use_scale)
        re = sweep_ridge_alpha(folds_e_arrays, alphas, seed=seed, scale=use_scale)
        ridge_alpha_baseline = max(rb, key=lambda r: r["r2_mean"])["alpha"]
        ridge_alpha_extended = max(re, key=lambda r: r["r2_mean"])["alpha"]
        if verbose:
            print(f"  Tuned Ridge alpha: baseline={ridge_alpha_baseline}, extended={ridge_alpha_extended}")

    gbm_n_baseline, gbm_d_baseline = 100, 3
    gbm_n_extended, gbm_d_extended = 100, 3
    if tune_gbm:
        n_est_list = [int(x.strip()) for x in gbm_n_estimators_str.split(",")]
        max_d_list = [int(x.strip()) for x in gbm_max_depth_str.split(",")]
        gbm_n_baseline, gbm_d_baseline = tune_gbm_params(
            folds_b_arrays, n_est_list, max_d_list, seed=seed,
            learning_rate=gbm_learning_rate, early_stopping=gbm_early_stopping,
        )
        gbm_n_extended, gbm_d_extended = tune_gbm_params(
            folds_e_arrays, n_est_list, max_d_list, seed=seed,
            learning_rate=gbm_learning_rate, early_stopping=gbm_early_stopping,
        )
        if verbose:
            print(f"  Tuned GBM: baseline n_est={gbm_n_baseline} max_d={gbm_d_baseline}, extended n_est={gbm_n_extended} max_d={gbm_d_extended}")
    if gbm_final_squeeze and verbose:
        print(f"  GBM final squeeze: LR=0.01, n_est=2×tuned (min 1000) for final model")

    # Final models on full train, evaluate on test
    test_b_fair = _test_fair_df(test_baseline, train_baseline, baseline_cols)
    test_e_fair = _test_fair_df(test_extended, train_extended, ext_cols)
    X_train_b = train_baseline[baseline_cols].to_numpy().astype(float)
    y_train_b = train_baseline["rest_length"].to_numpy()
    X_test_b = test_b_fair[baseline_cols].to_numpy().astype(float)
    y_test_b = test_baseline["rest_length"].to_numpy()
    X_train_e = train_extended[ext_cols].to_numpy().astype(float)
    y_train_e = train_extended["rest_length"].to_numpy()
    X_test_e = test_e_fair[ext_cols].to_numpy().astype(float)
    y_test_e = test_extended["rest_length"].to_numpy()
    X_test_b_raw = test_b_fair[baseline_cols].to_numpy().astype(float)
    X_test_e_raw = test_e_fair[ext_cols].to_numpy().astype(float)

    scaler_b = scaler_e = None
    if not no_scale:
        scaler_b = StandardScaler().fit(X_train_b)
        X_train_b = scaler_b.transform(X_train_b)
        X_test_b = scaler_b.transform(X_test_b)
        scaler_e = StandardScaler().fit(X_train_e)
        X_train_e = scaler_e.transform(X_train_e)
        X_test_e = scaler_e.transform(X_test_e)

    ridge_b = Ridge(alpha=ridge_alpha_baseline, random_state=seed).fit(X_train_b, y_train_b)
    lr_b, n_b = gbm_learning_rate, gbm_n_baseline
    if gbm_final_squeeze:
        lr_b, n_b = 0.01, max(gbm_n_baseline * 2, 1000)
    gbm_b = GradientBoostingRegressor(
        **_gbm_kwargs(n_b, gbm_d_baseline, lr_b, seed, gbm_early_stopping)
    ).fit(train_baseline[baseline_cols].to_numpy().astype(float), y_train_b)
    ridge_e = Ridge(alpha=ridge_alpha_extended, random_state=seed).fit(X_train_e, y_train_e)
    lr_e, n_e = gbm_learning_rate, gbm_n_extended
    if gbm_final_squeeze:
        lr_e, n_e = 0.01, max(gbm_n_extended * 2, 1000)
    gbm_e = GradientBoostingRegressor(
        **_gbm_kwargs(n_e, gbm_d_extended, lr_e, seed, gbm_early_stopping)
    ).fit(train_extended[ext_cols].to_numpy().astype(float), y_train_e)

    if save_models:
        prefix = Path(save_models)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        base = str(prefix)
        joblib.dump((ridge_b, scaler_b), f"{base}_{split}_ridge_baseline.pkl")
        joblib.dump(gbm_b, f"{base}_{split}_gbm_baseline.pkl")
        joblib.dump((ridge_e, scaler_e), f"{base}_{split}_ridge_extended.pkl")
        joblib.dump(gbm_e, f"{base}_{split}_gbm_extended.pkl")
        if verbose:
            print(f"  Saved models to {base}_{split}_*.pkl")

    return {
        "split": split,
        "ridge_baseline_mae": mean_absolute_error(y_test_b, ridge_b.predict(X_test_b)),
        "ridge_baseline_r2": r2_score(y_test_b, ridge_b.predict(X_test_b)),
        "gbm_baseline_mae": mean_absolute_error(y_test_b, gbm_b.predict(X_test_b_raw)),
        "gbm_baseline_r2": r2_score(y_test_b, gbm_b.predict(X_test_b_raw)),
        "ridge_extended_mae": mean_absolute_error(y_test_e, ridge_e.predict(X_test_e)),
        "ridge_extended_r2": r2_score(y_test_e, ridge_e.predict(X_test_e)),
        "gbm_extended_mae": mean_absolute_error(y_test_e, gbm_e.predict(X_test_e_raw)),
        "gbm_extended_r2": r2_score(y_test_e, gbm_e.predict(X_test_e_raw)),
    }


def plot_comparison(results: List[dict], out_prefix: str | Path | None = None) -> None:
    """Plot test MAE and test R² comparison across splits (dataset vs pooled). results = list of run_one_config outputs."""
    model_labels = ["Ridge\n(baseline)", "GBM\n(baseline)", "Ridge\n(extended)", "GBM\n(extended)"]
    keys_mae = ["ridge_baseline_mae", "gbm_baseline_mae", "ridge_extended_mae", "gbm_extended_mae"]
    keys_r2 = ["ridge_baseline_r2", "gbm_baseline_r2", "ridge_extended_r2", "gbm_extended_r2"]
    x = np.arange(len(model_labels))
    width = 0.35

    fig1, ax1 = plt.subplots(figsize=(8, 5))
    for i, res in enumerate(results):
        mae_vals = [res[k] for k in keys_mae]
        offset = (i - 0.5) * width if len(results) == 2 else (i - len(results) / 2 + 0.5) * width
        label = "train=original, test=replication" if res["split"] == "dataset" else "pooled (90% train, 10% test)"
        ax1.bar(x + offset, mae_vals, width, label=label)
    ax1.set_ylabel("Test MAE")
    ax1.set_xlabel("Model")
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_labels)
    ax1.set_title("Test MAE by split method and model")
    ax1.legend()
    fig1.tight_layout()
    if out_prefix:
        p = Path(out_prefix)
        path_mae = (p.parent / (p.stem + "_mae.png")) if p.suffix else Path(str(out_prefix) + "_mae.png")
        path_mae.parent.mkdir(parents=True, exist_ok=True)
        fig1.savefig(path_mae, dpi=150)
        print(f"Saved {path_mae}")
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    for i, res in enumerate(results):
        r2_vals = [res[k] for k in keys_r2]
        offset = (i - 0.5) * width if len(results) == 2 else (i - len(results) / 2 + 0.5) * width
        label = "train=original, test=replication" if res["split"] == "dataset" else "pooled (90% train, 10% test)"
        ax2.bar(x + offset, r2_vals, width, label=label)
    ax2.set_ylabel("Test R²")
    ax2.set_xlabel("Model")
    ax2.set_xticks(x)
    ax2.set_xticklabels(model_labels)
    ax2.set_title("Test R² by split method and model")
    ax2.legend()
    fig2.tight_layout()
    if out_prefix:
        p = Path(out_prefix)
        path_r2 = (p.parent / (p.stem + "_r2.png")) if p.suffix else Path(str(out_prefix) + "_r2.png")
        path_r2.parent.mkdir(parents=True, exist_ok=True)
        fig2.savefig(path_r2, dpi=150)
        print(f"Saved {path_r2}")
    plt.close(fig2)


def main() -> None:
    def format_metric(values: list[float]) -> str:
        a = np.array(values)
        if len(a) == 1:
            return f"{a[0]:.4f}"
        return f"{a.mean():.4f} ± {a.std():.4f}"

    parser = argparse.ArgumentParser(description="Run baseline fatigue prediction models (dataset + pooled splits)")
    parser.add_argument("--n_folds", type=int, default=15, help="K-fold CV on training set (default 25)")
    parser.add_argument("--test_frac", type=float, default=0.1, help="Test fraction for pooled split (default 0.1 → 90/10)")
    parser.add_argument("--alphas", type=str, default="0.01,0.05,0.1,0.5,1,10,20,100", help="Comma-separated alphas for Ridge tuning")
    parser.add_argument("--gbm-n-estimators", type=str, default="100,200,300,400,500", help="Comma-separated n_estimators for GBM tune")
    parser.add_argument("--gbm-max-depth", type=str, default="1,2,3,4", help="Comma-separated max_depth for GBM tune")
    parser.add_argument("--gbm-learning-rate", type=float, default=0.05, help="GBM learning rate for tuning")
    parser.add_argument("--gbm-final-squeeze", action="store_true", help="Final model: LR=0.01 and 2x trees for 'final squeeze'")
    parser.add_argument("--gbm-no-early-stopping", action="store_true", help="Disable GBM early stopping")
    parser.add_argument("--no-scale", action="store_true", help="Don't standardize features for Ridge (not recommended)")
    parser.add_argument("--save-split", type=str, default=None, metavar="PATH", help="Save train/test IDs to PATH_dataset.json and PATH_pooled.json")
    parser.add_argument("--load-split", type=str, default=None, metavar="PATH", help="Load pooled split from PATH.json (only affects pooled run)")
    parser.add_argument("--save-models", type=str, default=None, metavar="PREFIX", help="Save fitted models to PREFIX_dataset_*.pkl and PREFIX_pooled_*.pkl")
    parser.add_argument("--compare-out", type=str, default="comparison", metavar="PREFIX", help="Output prefix for comparison plots (→ PREFIX_mae.png, PREFIX_r2.png)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # One preprocessing pipeline
    print("Loading and building features...")
    df = get_data()
    # Complete cases only for extended (history has NA at first epoch); baseline has no NA after imputation
    df_baseline = df.dropna(subset=get_feature_columns(baseline_only=True))
    df_extended = df.dropna(subset=get_feature_columns(baseline_only=False))

    n_folds = args.n_folds
    test_frac = args.test_frac
    print(f"Running comparison: {n_folds}-fold CV, tune alpha, tune GBM, both splits (dataset + pooled {int((1-test_frac)*100)}/{int(test_frac*100)})\n")
    results = []
    for split in ["dataset", "pooled"]:
        print(f"--- Split: {split} ---")
        include_dataset = split == "pooled"
        baseline_cols = get_feature_columns(baseline_only=True, include_dataset=include_dataset)
        ext_cols = get_feature_columns(baseline_only=False, include_dataset=include_dataset)
        save_split_path = None
        if args.save_split:
            p = Path(args.save_split)
            save_split_path = str(p.parent / (p.stem + f"_{split}" + p.suffix)) if p.suffix else f"{args.save_split}_{split}"
        load_split_path = args.load_split if split == "pooled" else None
        res = run_one_config(
            df_baseline, df_extended, baseline_cols, ext_cols,
            split=split, n_folds=n_folds, test_frac=test_frac, seed=args.seed,
            tune_alpha=True, tune_gbm=True,
            alphas_str=args.alphas, gbm_n_estimators_str=args.gbm_n_estimators, gbm_max_depth_str=args.gbm_max_depth,
            gbm_learning_rate=args.gbm_learning_rate, gbm_early_stopping=not args.gbm_no_early_stopping,
            gbm_final_squeeze=args.gbm_final_squeeze,
            no_scale=args.no_scale, load_split=load_split_path, save_split=save_split_path,
            save_models=args.save_models, verbose=True,
        )
        results.append(res)
        print(f"  Test MAE  Ridge base: {res['ridge_baseline_mae']:.4f}  GBM base: {res['gbm_baseline_mae']:.4f}  Ridge ext: {res['ridge_extended_mae']:.4f}  GBM ext: {res['gbm_extended_mae']:.4f}")
        print(f"  Test R²   Ridge base: {res['ridge_baseline_r2']:.4f}  GBM base: {res['gbm_baseline_r2']:.4f}  Ridge ext: {res['ridge_extended_r2']:.4f}  GBM ext: {res['gbm_extended_r2']:.4f}\n")
    plot_comparison(results, out_prefix=args.compare_out)
    print("Done.")


if __name__ == "__main__":
    main()
