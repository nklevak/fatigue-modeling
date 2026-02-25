"""
Train/validation splits for fatigue prediction.

We always split by subject: a subject's entire rest pattern (all 30 epochs) is
either fully in train or fully in test. We never mix epochs from the same
subject across train and test.

Strategies:
  1) split_by_dataset: train on one dataset, test on the other.
  2) split_by_subject: fraction of (all) subjects as test.
  3) train_test_split_pooled: stratified 20% of each dataset as test, 80% train (for k-fold on train only).
  4) kfold_by_subject: k-fold over a single pool (e.g. train only).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, List, Set

import numpy as np
import pandas as pd


def split_by_dataset(
    df: pd.DataFrame,
    train_dataset: str = "original",
    val_dataset: str = "replication",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Option 1: Train on one dataset, test on the other.
    All epochs from all subjects in train_dataset go to train; all epochs from
    all subjects in val_dataset go to test. Each subject's full rest pattern
    stays in one set.
    """
    train = df[df["dataset"] == train_dataset].copy()
    val = df[df["dataset"] == val_dataset].copy()
    if train.empty or val.empty:
        raise ValueError(
            f"split_by_dataset: train_dataset={train_dataset} or val_dataset={val_dataset} not found in df['dataset']"
        )
    return train, val


def split_by_subject(
    df: pd.DataFrame,
    val_frac: float = 0.2,
    random_state: int = 42,
    subject_col: str = "subject_id",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Option 2: Train on a fraction of subjects (from both datasets), test on the rest.
    No subject appears in both sets; we predict the entire rest pattern for
    held-out subjects. val_frac is the fraction of subjects (not epochs) in test.
    Uses subject_col (default subject_id: dataset + subj_id) so original and replication
    never share identity.
    """
    subjects = df[subject_col].unique()
    rng = np.random.default_rng(random_state)
    n_val = max(1, int(len(subjects) * val_frac))
    val_subjects = rng.choice(subjects, size=n_val, replace=False)
    train_mask = ~df[subject_col].isin(val_subjects)
    val_mask = df[subject_col].isin(val_subjects)
    return df[train_mask].copy(), df[val_mask].copy()


def train_test_split_pooled(
    df: pd.DataFrame,
    test_frac: float = 0.2,
    random_state: int = 42,
    subject_col: str = "subject_id",
    dataset_col: str = "dataset",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Stratified train/test by subject: take test_frac of subjects from each dataset
    (original, replication) as test; the rest as train. Use for "pooled" evaluation
    where you want 20% of each study held out, then k-fold CV on train only.
    """
    rng = np.random.default_rng(random_state)
    train_parts, test_parts = [], []
    for dataset_name in df[dataset_col].unique():
        sub = df[df[dataset_col] == dataset_name]
        subjects = np.array(sub[subject_col].unique())
        n_test = max(1, int(len(subjects) * test_frac))
        rng.shuffle(subjects)
        test_subjects = set(subjects[:n_test])
        train_subjects = set(subjects[n_test:])
        train_parts.append(sub[~sub[subject_col].isin(test_subjects)])
        test_parts.append(sub[sub[subject_col].isin(test_subjects)])
    train = pd.concat(train_parts, ignore_index=True)
    test = pd.concat(test_parts, ignore_index=True)
    return train, test


def save_split_ids(
    train_ids: List[str],
    test_ids: List[str],
    path: str | Path,
) -> None:
    """Save train and test subject IDs to a JSON file for reproducible splits."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix != ".json":
        path = path.with_suffix(".json")
    with open(path, "w") as f:
        json.dump({"train": sorted(train_ids), "test": sorted(test_ids)}, f, indent=2)
    print(f"Saved split IDs to {path}")


def load_split_ids(path: str | Path) -> tuple[Set[str], Set[str]]:
    """Load train and test subject IDs from a JSON file. Returns (train_ids, test_ids) as sets."""
    path = Path(path)
    if path.suffix != ".json":
        path = path.with_suffix(".json")
    with open(path) as f:
        data = json.load(f)
    return set(data["train"]), set(data["test"])


def split_df_by_subject_ids(
    df: pd.DataFrame,
    train_ids: Set[str],
    test_ids: Set[str],
    subject_col: str = "subject_id",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split dataframe into train and test by subject ID membership."""
    train = df[df[subject_col].isin(train_ids)].copy()
    test = df[df[subject_col].isin(test_ids)].copy()
    return train, test


def kfold_by_subject(
    df: pd.DataFrame,
    n_splits: int = 5,
    random_state: int = 42,
    subject_col: str = "subject_id",
) -> Iterator[tuple[pd.DataFrame, pd.DataFrame]]:
    """
    K-fold cross-validation by subject. Each subject's full rest pattern is
    either entirely in train or entirely in val for a given fold. Yields
    (train_df, val_df) for each of n_splits folds.
    Uses subject_col (default subject_id: dataset + subj_id) so original and
    replication subjects with the same numeric id are never treated as the same.
    """
    subjects = np.array(df[subject_col].unique())
    n = len(subjects)
    if n_splits > n:
        raise ValueError(f"n_splits ({n_splits}) cannot be greater than number of subjects ({n})")
    rng = np.random.default_rng(random_state)
    rng.shuffle(subjects)
    fold_sizes = np.full(n_splits, n // n_splits, dtype=int)
    fold_sizes[: n % n_splits] += 1
    start = 0
    for fold_size in fold_sizes:
        val_subjects = set(subjects[start : start + fold_size])
        start += fold_size
        train_mask = ~df[subject_col].isin(val_subjects)
        val_mask = df[subject_col].isin(val_subjects)
        yield df[train_mask].copy(), df[val_mask].copy()
