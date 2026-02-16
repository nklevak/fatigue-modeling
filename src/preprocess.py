"""
Load blockwise epoch-level data and build features for fatigue prediction.

Data source: original_blockwise_cleaned.csv and replication_blockwise_cleaned.csv
(cleaned blockwise data derived from original/replication main trials).
Subject IDs match the trial-level data: original uses the same subj_id as
original_main_trials; replication uses the same subj_id as replication_main_trials.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd
import numpy as np

_DATA_DIR = Path(__file__).resolve().parents[1] / "cleaned_exp_data"


def load_original() -> pd.DataFrame:
    """Load original experiment blockwise (epoch-level) data."""
    path = _DATA_DIR / "original_blockwise_cleaned.csv"
    df = pd.read_csv(path)
    df["dataset"] = "original"
    return df


def load_replication() -> pd.DataFrame:
    """Load replication experiment blockwise (epoch-level) data."""
    path = _DATA_DIR / "replication_blockwise_cleaned.csv"
    df = pd.read_csv(path)
    df["dataset"] = "replication"
    return df


def get_epoch_table() -> pd.DataFrame:
    """
    Build a single epoch-level table from both blockwise files.
    Includes rest_length (target) = num_rest_in_chunk, and all blockwise variables.
    """
    orig = load_original()
    rep = load_replication()
    df = pd.concat([orig, rep], ignore_index=True)
    df["rest_length"] = df["num_rest_in_chunk"]
    # Composite subject ID so original and replication never share identity (e.g. subj_id 14 in both are different people)
    df["subject_id"] = df["dataset"].astype(str) + "_" + df["subj_id"].astype(str)
    df = df.sort_values(["dataset", "subj_id", "epoch_num"]).reset_index(drop=True)
    return df


def add_baseline_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add baseline model features: epoch/block, accuracy, rt, game_type (one-hot), cue (switch vs stay)."""
    out = df.copy()
    # Cue: switch vs stay from cue_type
    out["cue_switch"] = (out["cue_type"].astype(str).str.strip().str.lower() == "switch").astype(int)
    # Game type one-hot (digit_span, spatial_recall)
    gt = out["game_type"].astype(str).str.strip().str.lower()
    out["game_type_digit_span"] = (gt == "digit_span").astype(int)
    out["game_type_spatial_recall"] = (gt == "spatial_recall").astype(int)
    # avg_rt can be NA in blockwise when there are no valid RTs (e.g. 0% accuracy epoch);
    # we do not impute here—use dropna(subset=baseline_cols) or equivalent downstream if needed.
    return out


def add_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add previous-epoch features within each subject: rest_length_prev, accuracy_prev, rt_prev."""
    out = df.copy()
    g = out.groupby("subject_id")
    out["rest_length_prev"] = g["rest_length"].shift(1)
    out["accuracy_prev"] = g["avg_epoch_accuracy"].shift(1)
    out["rt_prev"] = g["avg_rt"].shift(1)
    return out


def get_feature_columns(baseline_only: bool = False) -> List[str]:
    """Column names to use as model features. baseline_only=False adds history columns."""
    baseline = [
        "epoch_num",
        "block_num",
        "avg_epoch_accuracy",
        "avg_rt",
        "game_type_digit_span",
        "game_type_spatial_recall",
        "cue_switch",
    ]
    if baseline_only:
        return baseline
    return baseline + ["rest_length_prev", "accuracy_prev", "rt_prev"]


def get_data() -> pd.DataFrame:
    """Epoch table + baseline features + history features (for run_baselines and downstream)."""
    df = get_epoch_table()
    df = add_baseline_features(df)
    df = add_history_features(df)
    return df
