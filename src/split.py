"""
Train/validation splits for fatigue prediction.

We always split by subject: a subject's entire rest pattern (all 30 epochs) is
either fully in train or fully in test. We never mix epochs from the same
subject across train and test.

Two strategies:
  1) Train on original, test on replication (split_by_dataset).
  2) Train on a fraction of subjects, test on the rest (split_by_subject), or use
     k-fold over subjects (kfold_by_subject) for a more stable estimate.
"""

from __future__ import annotations

from typing import Iterator

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
