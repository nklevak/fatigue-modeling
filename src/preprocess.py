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

# Trial types that are main task responses (exclude rt_main_trials for trial/epoch counts).
MAIN_RESPONSE_TRIAL_TYPES = ("ds_main_response", "sr_main_response")

# -----------------------------------------------------------------------------
# Trial-level data (for LSTM / sequence models later)
# -----------------------------------------------------------------------------
# this is from the 2024 FYP experiment (spr_main)
def load_original_trials() -> pd.DataFrame:
    """Load original experiment trial-level data. Adds dataset and subject_id."""
    path = _DATA_DIR / "original_main_trials_cleaned.csv"
    df = pd.read_csv(path)
    df["dataset"] = "original"
    df["subject_id"] = "original_" + df["subj_id"].astype(str)
    return df

# this is from the 2025 replication experiment (cf_ts_rep)
def load_replication_trials() -> pd.DataFrame:
    """Load replication experiment trial-level data. Adds dataset and subject_id."""
    path = _DATA_DIR / "replication_main_trials_cleaned.csv"
    df = pd.read_csv(path)
    df["dataset"] = "replication"
    df["subject_id"] = "replication_" + df["subj_id"].astype(str)
    return df

def _assign_epoch_num_original(trials: pd.DataFrame) -> pd.DataFrame:
    """
    Assign epoch_num (1..30) to original trials from the experiment structure:
    each epoch is one contiguous block of all digit-span OR all spatial-recall trials,
    followed by one or more rt_main_trials; the next epoch starts when we go from rt
    back to main (rt→main boundary). So we detect block start = main and previous row
    was not main (or start of subject); number blocks 1..30; rt trials get the epoch
    of the block they follow (ffill).
    """
    out = trials.copy().sort_values(["subject_id", "trial_index"]).reset_index(drop=True)
    is_main = out["trial_type"].isin(MAIN_RESPONSE_TRIAL_TYPES).values
    subject = out["subject_id"].values
    prev_is_main = np.zeros(len(out), dtype=bool)
    for sid in np.unique(subject):
        mask = subject == sid
        arr = is_main[mask]
        prev_is_main[mask] = np.concatenate([[False], arr[:-1]])
    block_start = is_main & ~prev_is_main
    block_id = np.full(len(out), np.nan, dtype=float)
    for sid in np.unique(subject):
        mask = subject == sid
        starts = block_start[mask].astype(float)
        ids = np.cumsum(starts)
        ids[ids == 0] = np.nan
        block_id[mask] = pd.Series(ids).ffill().values
    out["epoch_num"] = np.nanmax(np.c_[block_id, np.ones(len(out))], axis=1).astype(int)
    return out


def get_trials(include_epoch_cues: bool = True) -> pd.DataFrame:
    """
    Combined trial-level table from both experiments (for LSTM / sequence models).
    Columns differ slightly between original and replication; both have subject_id, dataset,
    subj_id, rt, game_type, is_correct_numeric, block_num, num_rest_in_chunk, etc.
    If include_epoch_cues=True (default), merges cue_transition_type (3-level: stay_within_block,
    stay_between_block, switch_between_block), cue_type, and rest_type from the blockwise
    (epoch-level) data onto each trial via (subject_id, epoch_num).
    Original trials: epoch_num from structure (all-DS or all-SR block then rt): boundaries = task change among main-response trials; rt trials get the epoch of the block they follow.
    Replication trials: epoch_num is native (NaN filled within subject).
    """
    orig = load_original_trials()
    rep = load_replication_trials()
    # Original: epoch = every 10 main-response trials within subject (so 10 per epoch)
    orig = _assign_epoch_num_original(orig)
    if "epoch_num" in rep.columns:
        rep = rep.copy()
        rep["epoch_num"] = (
            rep.groupby("subject_id")["epoch_num"].ffill().bfill().fillna(1).astype(int)
        )
    if not include_epoch_cues:
        return pd.concat([orig, rep], ignore_index=True)
    # Merge blockwise epoch-level cues onto trials
    epoch_table = get_epoch_table()
    cue_cols = ["subject_id", "epoch_num", "cue_transition_type", "cue_type", "rest_type"]
    epoch_cues = epoch_table[cue_cols].drop_duplicates()
    orig = orig.merge(epoch_cues, on=["subject_id", "epoch_num"], how="left")
    rep = rep.merge(epoch_cues, on=["subject_id", "epoch_num"], how="left")
    return pd.concat([orig, rep], ignore_index=True)


# -----------------------------------------------------------------------------
# Blockwise (epoch-level) data – used by baselines
# -----------------------------------------------------------------------------
# this is from the 2024 FYP experiment (spr_main)
def load_original() -> pd.DataFrame:
    """Load original experiment blockwise (epoch-level) data."""
    path = _DATA_DIR / "original_blockwise_cleaned.csv"
    df = pd.read_csv(path)
    df["dataset"] = "original"
    return df

# this is from the 2025 replication experiment (cf_ts_rep)
def load_replication() -> pd.DataFrame:
    """Load replication experiment blockwise (epoch-level) data."""
    path = _DATA_DIR / "replication_blockwise_cleaned.csv"
    df = pd.read_csv(path)
    df["dataset"] = "replication"
    return df

def _get_num_timeouts_per_epoch() -> pd.DataFrame:
    """Number of timed-out trials per (subject_id, epoch_num), main-response trials only."""
    orig = load_original_trials()
    rep = load_replication_trials()
    orig = _assign_epoch_num_original(orig)
    if "epoch_num" in rep.columns:
        rep = rep.copy()
        rep["epoch_num"] = rep.groupby("subject_id")["epoch_num"].ffill().bfill().fillna(1).astype(int)
    both = pd.concat([orig, rep], ignore_index=True)
    main = both[both["trial_type"].isin(MAIN_RESPONSE_TRIAL_TYPES)].copy()
    if "timed_out" not in main.columns:
        out = main.groupby(["subject_id", "epoch_num"], as_index=False).size()
        out["num_timeouts"] = 0
        return out[["subject_id", "epoch_num", "num_timeouts"]]
    timed_out = main["timed_out"].astype(str).str.strip().str.lower().isin(("true", "1", "1.0"))
    main["_timed_out"] = timed_out.astype(int)
    return main.groupby(["subject_id", "epoch_num"], as_index=False)["_timed_out"].sum().rename(columns={"_timed_out": "num_timeouts"})

def get_epoch_table() -> pd.DataFrame:
    """
    Build a single epoch-level table from both blockwise files.
    Includes rest_length (target) = num_rest_in_chunk, all blockwise variables,
    and num_timeouts (from trial-level, main-response only).
    """
    orig = load_original()
    rep = load_replication()
    df = pd.concat([orig, rep], ignore_index=True)
    df["rest_length"] = df["num_rest_in_chunk"]
    # Composite subject ID so original and replication never share identity (e.g. subj_id 14 in both are different people)
    df["subject_id"] = df["dataset"].astype(str) + "_" + df["subj_id"].astype(str)
    df = df.sort_values(["dataset", "subj_id", "epoch_num"]).reset_index(drop=True)
    timeouts = _get_num_timeouts_per_epoch()
    df = df.merge(timeouts, on=["subject_id", "epoch_num"], how="left")
    df["num_timeouts"] = df["num_timeouts"].fillna(0).astype(int)
    return df

def add_baseline_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add baseline model features: epoch/block, avg accuracy, accuracy sd, avg rt, rt variability,
    game_type (one-hot), cue_transition_type (3-level dummies), block_num."""
    out = df.copy()
    # Cue transition type: 3-level (stay_within_block, stay_between_block, switch_between_block); reference = stay_within_block
    ctt = out["cue_transition_type"].astype(str).str.strip().str.lower()
    out["cue_stay_between_block"] = (ctt == "stay_between_block").astype(int)
    out["cue_switch_between_block"] = (ctt == "switch_between_block").astype(int)
    # Game type one-hot (digit_span, spatial_recall)
    gt = out["game_type"].astype(str).str.strip().str.lower()
    out["game_type_digit_span"] = (gt == "digit_span").astype(int)
    out["game_type_spatial_recall"] = (gt == "spatial_recall").astype(int)
    # Blockwise columns: accuracy_sd and rt_sd (per-epoch variability); block_num
    if "accuracy_sd" in out.columns:
        out["accuracy_sd"] = out["accuracy_sd"].fillna(0)
    if "rt_sd" in out.columns:
        out["rt_sd"] = out["rt_sd"].fillna(0)
    if "block_num" in out.columns:
        out["block_num"] = out["block_num"].fillna(1).astype(int)
    # num_timeouts is already in out (from get_epoch_table); ensure int
    if "num_timeouts" in out.columns:
        out["num_timeouts"] = out["num_timeouts"].fillna(0).astype(int)
    # Impute avg_rt with max when NA (e.g. all timeouts in epoch)
    if out["avg_rt"].isna().any():
        out["avg_rt"] = out["avg_rt"].fillna(out["avg_rt"].max())
    # Cumulative rest trials prior to this epoch (within subject)
    out["rests_taken_so_far"] = (
        out.groupby("subject_id")["rest_length"]
        .transform(lambda x: x.shift(1).cumsum().fillna(0))
        .astype(float)
    )
    # Leave-one-out mean rest length per subject (avoids leakage: don't use current epoch's target)
    g = out.groupby("subject_id")["rest_length"]
    total = g.transform("sum") - out["rest_length"]
    n = g.transform("count") - 1
    out["mean_rest_length_subj"] = total / n.replace(0, np.nan)
    # Dataset dummy (used only for pooled split: 1=replication, 0=original)
    out["dataset_replication"] = (out["dataset"] == "replication").astype(int)
    return out


def add_cumulative_accuracy_by_game(df: pd.DataFrame) -> pd.DataFrame:
    """Add avg_accuracy_until_now_digit_span and avg_accuracy_until_now_spatial_recall:
    mean accuracy in all previous epochs (within subject) for that game type. Excludes current epoch.
    When no previous epochs of that game exist, fill with 0.5 (chance)."""
    out = df.copy()
    gt = out["game_type"].astype(str).str.strip().str.lower()
    ds_vals = []
    sr_vals = []
    for sid in out["subject_id"].unique():
        mask = out["subject_id"] == sid
        grp = out.loc[mask].sort_values("epoch_num")
        for i in range(len(grp)):
            prev = grp.iloc[:i]
            ds_prev = prev[gt.loc[prev.index] == "digit_span"]
            sr_prev = prev[gt.loc[prev.index] == "spatial_recall"]
            mean_ds = float(ds_prev["avg_epoch_accuracy"].mean()) if len(ds_prev) > 0 else 0.5
            mean_sr = float(sr_prev["avg_epoch_accuracy"].mean()) if len(sr_prev) > 0 else 0.5
            ds_vals.append(mean_ds)
            sr_vals.append(mean_sr)
    # Reassign in original order (grp iteration was per-subject, need to map back)
    out = out.sort_values(["subject_id", "epoch_num"]).reset_index(drop=True)
    out["avg_accuracy_until_now_digit_span"] = ds_vals
    out["avg_accuracy_until_now_spatial_recall"] = sr_vals
    return out


def add_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add previous-epoch features within each subject: rest_length_prev, accuracy_prev, rt_prev, game_type_prev (one-hot)."""
    out = df.copy()
    g = out.groupby("subject_id")
    out["rest_length_prev"] = g["rest_length"].shift(1)
    out["accuracy_prev"] = g["avg_epoch_accuracy"].shift(1)
    out["rt_prev"] = g["avg_rt"].shift(1)
    if "game_type_digit_span" in out.columns and "game_type_spatial_recall" in out.columns:
        out["game_type_digit_span_prev"] = g["game_type_digit_span"].shift(1)
        out["game_type_spatial_recall_prev"] = g["game_type_spatial_recall"].shift(1)
        out["previous_cue_stay_between_block"] = g["cue_stay_between_block"].shift(1)
        out["previous_cue_switch_between_block"] = g["cue_switch_between_block"].shift(1)
        out["previous_num_timeouts"] = g["num_timeouts"].shift(1)
        out["previous_avg_rt"] = g["avg_rt"].shift(1)
    return out


def get_feature_columns(baseline_only: bool = False, include_dataset: bool = False) -> List[str]:
    """Column names to use as model features. baseline_only=False adds history columns.
    include_dataset=True adds dataset_replication (use only for pooled split)."""
    # cue_transition_type 3-level dummies (stay_within_block = reference)
    baseline = [
        "epoch_num",
        "block_num",
        "avg_epoch_accuracy",
        "accuracy_sd",
        "avg_rt",
        "rt_sd",
        "num_timeouts",
        "game_type_digit_span",
        "game_type_spatial_recall",
        "cue_stay_between_block",
        "cue_switch_between_block",
        "rests_taken_so_far",
        "mean_rest_length_subj",
        "avg_accuracy_until_now_digit_span",
        "avg_accuracy_until_now_spatial_recall",
    ]
    if include_dataset:
        baseline = baseline + ["dataset_replication"]
    if baseline_only:
        return baseline
    return baseline + [
        "rest_length_prev",
        "accuracy_prev",
        "rt_prev",
        "game_type_digit_span_prev",
        "game_type_spatial_recall_prev",
        "previous_cue_stay_between_block",
        "previous_cue_switch_between_block",
    ]


def get_data() -> pd.DataFrame:
    """Epoch table + baseline features + cumulative accuracy by game + history features (for run_baselines and downstream)."""
    df = get_epoch_table()
    df = add_baseline_features(df)
    df = add_cumulative_accuracy_by_game(df)
    df = add_history_features(df)
    return df
