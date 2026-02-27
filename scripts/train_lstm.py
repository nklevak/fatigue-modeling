"""
Train NumPy-based LSTM to predict rest_length at each epoch from full history (many-to-many).
Uses same splits as baselines: dataset (train=original, test=replication) or pooled (90/10).
No PyTorch - pure NumPy only. Run with: uv run python scripts/train_lstm.py

Usage:
  uv run python scripts/train_lstm.py
  uv run python scripts/train_lstm.py --split pooled --load-split splits/compare_pooled
  uv run python scripts/train_lstm.py --split dataset --epochs 100 --save model.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.preprocess import get_data, get_feature_columns
from src.split import split_by_dataset, train_test_split_pooled, load_split_ids, split_df_by_subject_ids
from src.lstm_model import RestLSTM, NumPyAdam, get_state_dict, load_state_dict


def _test_fair_df(val_df: pd.DataFrame, train_df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Replace subject-level features that leak at test time."""
    out = val_df.copy()
    if "mean_rest_length_subj" in cols:
        out["mean_rest_length_subj"] = train_df["mean_rest_length_subj"].mean()
    return out


def build_sequences(
    df: pd.DataFrame,
    feature_cols: list[str],
    train_df: pd.DataFrame | None = None,
    max_len: int = 30,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build padded sequences (X, y, lengths) per subject.
    X: (n_subjects, max_len, n_features), y: (n_subjects, max_len), lengths: (n_subjects,)
    """
    if train_df is not None:
        df = _test_fair_df(df, train_df, feature_cols)

    subjects = df["subject_id"].unique()
    X_list, y_list, lengths = [], [], []

    for sid in subjects:
        sub = df[df["subject_id"] == sid].sort_values("epoch_num")
        x = sub[feature_cols].to_numpy().astype(np.float32)
        y = sub["rest_length"].to_numpy().astype(np.float32)
        L = len(x)
        if L == 0:
            continue
        x_pad = np.zeros((max_len, x.shape[1]), dtype=np.float32)
        y_pad = np.full(max_len, np.nan, dtype=np.float32)
        x_pad[:L] = x
        y_pad[:L] = y
        X_list.append(x_pad)
        y_list.append(y_pad)
        lengths.append(L)

    X = np.stack(X_list)
    y = np.stack(y_list)
    lengths = np.array(lengths)
    return X, y, lengths


def train_epoch(
    model: RestLSTM,
    X: np.ndarray,
    y: np.ndarray,
    lengths: np.ndarray,
    optimizer: NumPyAdam,
    batch_size: int,
    seed_offset: int,
) -> float:
    n = len(X)
    indices = np.arange(n)
    np.random.seed(seed_offset)
    np.random.shuffle(indices)
    total_loss = 0.0
    n_valid = 0

    for start in range(0, n, batch_size):
        batch_idx = indices[start : start + batch_size]
        B = len(batch_idx)
        mask = np.zeros((B, X.shape[1]), dtype=bool)
        for i, bi in enumerate(batch_idx):
            mask[i, : lengths[bi]] = True

        x_batch = X[batch_idx]
        y_batch = y[batch_idx].copy()
        y_batch[~mask] = 0.0  # masked positions not used in loss

        out, caches = model.forward(x_batch, mask=mask, training=True, return_caches=True)
        pred = out.squeeze(-1)
        diff = pred - y_batch
        diff[~mask] = 0
        loss = np.mean(diff[mask] ** 2)
        n_valid += mask.sum()
        total_loss += loss * mask.sum()

        scale = 2.0 / mask.sum()
        d_out = np.where(mask[:, :, np.newaxis], scale * (pred - y_batch)[:, :, np.newaxis], 0.0).astype(np.float32)

        grads = model.backward(x_batch, mask, d_out, caches)
        optimizer.step(model, grads)

    return float(total_loss / n_valid) if n_valid > 0 else 0.0


def evaluate(
    model: RestLSTM,
    X: np.ndarray,
    y: np.ndarray,
    lengths: np.ndarray,
    batch_size: int = 32,
) -> tuple[float, float, float]:
    preds, actuals = [], []
    for start in range(0, len(X), batch_size):
        batch_idx = np.arange(start, min(start + batch_size, len(X)))
        mask = np.zeros((len(batch_idx), X.shape[1]), dtype=bool)
        for i, bi in enumerate(batch_idx):
            mask[i, : lengths[bi]] = True
        x_batch = X[batch_idx]
        out = model.predict(x_batch, mask=mask).squeeze(-1)
        for i, bi in enumerate(batch_idx):
            L = lengths[bi]
            preds.extend(out[i, :L].tolist())
            actuals.extend(y[bi, :L].tolist())
    preds = np.array(preds)
    actuals = np.array(actuals)
    mae = mean_absolute_error(actuals, preds)
    r2 = r2_score(actuals, preds)
    mse = float(np.mean((actuals - preds) ** 2))
    return mae, r2, mse


def run_lstm(
    split: str = "dataset",
    load_split: str | None = None,
    test_frac: float = 0.1,
    seed: int = 42,
    epochs: int = 60,
    hidden: int = 16,
    layers: int = 2,
    dropout: float = 0.5,
    lr: float = 2e-3,
    weight_decay: float = 1e-3,
    batch_size: int = 8,
    save: str | None = None,
    no_scale: bool = False,
    verbose: bool = True,
) -> dict:
    """
    Train LSTM and return results. For programmatic use (e.g. run_all_models.py).
    Returns dict with: split, lstm_mae, lstm_r2, feature_cols, params.
    """
    np.random.seed(seed)
    df = get_data()
    include_dataset = split == "pooled"
    feature_cols = get_feature_columns(baseline_only=False, include_dataset=include_dataset)
    df_ext = df.dropna(subset=feature_cols + ["rest_length"])

    if split == "dataset":
        train_df, test_df = split_by_dataset(df_ext, "original", "replication")
    else:
        if load_split:
            train_ids, test_ids = load_split_ids(load_split)
            train_df, test_df = split_df_by_subject_ids(df_ext, train_ids, test_ids)
        else:
            train_df, test_df = train_test_split_pooled(
                df_ext, test_frac=test_frac, random_state=seed
            )

    X_train, y_train, len_train = build_sequences(train_df, feature_cols, train_df=None)
    X_test, y_test, len_test = build_sequences(test_df, feature_cols, train_df=train_df)

    scaler = None
    if not no_scale:
        scaler = StandardScaler()
        n_train, seq_len, n_feat = X_train.shape
        X_train_flat = X_train.reshape(-1, n_feat)
        scaler.fit(X_train_flat)
        X_train = scaler.transform(X_train_flat).reshape(n_train, seq_len, n_feat).astype(np.float32)
        X_test = scaler.transform(X_test.reshape(-1, n_feat)).reshape(len(X_test), seq_len, n_feat).astype(np.float32)

    n_features = len(feature_cols)
    model = RestLSTM(n_features=n_features, hidden_size=hidden, num_layers=layers, dropout=dropout)
    optimizer = NumPyAdam(model, lr=lr, weight_decay=weight_decay)

    best_mae = float("inf")
    best_state = None
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(
            model, X_train, y_train, len_train, optimizer, batch_size, seed + epoch
        )
        mae, r2, _ = evaluate(model, X_test, y_test, len_test)
        if mae < best_mae:
            best_mae = mae
            best_state = get_state_dict(model)
        if verbose and (epoch % 10 == 0 or epoch == 1):
            print(f"  Epoch {epoch}: train_loss={train_loss:.4f} test_MAE={mae:.4f} test_R²={r2:.4f}")

    if save and best_state is not None:
        load_state_dict(model, best_state)
        save_obj: dict = {
            "fc_W": best_state["fc_W"], "fc_b": best_state["fc_b"],
        }
        for i, layer in enumerate(best_state["layers"]):
            save_obj[f"layer{i}_W"] = layer["W"]
            save_obj[f"layer{i}_b"] = layer["b"]
        if scaler is not None:
            save_obj["scaler_mean"] = scaler.mean_
            save_obj["scaler_scale"] = scaler.scale_
        np.savez(save, **save_obj)
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        with open(str(save).replace(".npz", "_features.txt"), "w") as f:
            f.write("\n".join(feature_cols))

    if best_state is not None:
        load_state_dict(model, best_state)
    mae, r2, _ = evaluate(model, X_test, y_test, len_test)

    return {
        "split": split,
        "lstm_mae": float(mae),
        "lstm_r2": float(r2),
        "feature_cols": feature_cols,
        "params": {
            "epochs": epochs, "hidden": hidden, "layers": layers, "dropout": dropout,
            "lr": lr, "weight_decay": weight_decay, "batch_size": batch_size,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train NumPy LSTM for rest-length prediction")
    parser.add_argument("--split", choices=["dataset", "pooled"], default="dataset")
    parser.add_argument("--load-split", type=str, default=None, help="Load pooled split (e.g. splits/compare_pooled.json)")
    parser.add_argument("--test-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3, help="L2 regularization (Ridge-style)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--save", type=str, default=None, help="Path to save best model (.npz)")
    parser.add_argument("--no-scale", action="store_true", help="Don't standardize features")
    args = parser.parse_args()

    print("Using NumPy-based LSTM (no PyTorch)")
    res = run_lstm(
        split=args.split,
        load_split=args.load_split,
        test_frac=args.test_frac,
        seed=args.seed,
        epochs=args.epochs,
        hidden=args.hidden,
        layers=args.layers,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        save=args.save,
        no_scale=args.no_scale,
        verbose=True,
    )
    print(f"\nFinal Test MAE: {res['lstm_mae']:.4f}, R²: {res['lstm_r2']:.4f}")


if __name__ == "__main__":
    main()
