# Project plan: Fatigue modeling

## Data and target

- **Data:** Blockwise epoch-level CSVs: `cleaned_exp_data/original_blockwise_cleaned.csv` (84 subjects × 30 epochs), `cleaned_exp_data/replication_blockwise_cleaned.csv` (103 subjects × 30 epochs). Subject identity is `subject_id` = dataset + subj_id (e.g. `original_4`, `replication_4`) so the same numeric id in different datasets are different people.
- **Epoch structure (verified):**  
  - **Epochs per subject:** **30** in both original and replication.  
  - **Main-response trials per (subject, epoch):** **10** in both (only ds_main_response and sr_main_response; rt_main_trials excluded).  
  - **Per epoch, all 10 trials are the same task:** either all **digit-span** or all **spatial-recall** (unique game_type = 1 per epoch). So each epoch is 10 digit-span trials or 10 spatial-recall trials, never mixed.  
  - **Per subject:** **15 digit-span epochs** and **15 spatial-recall epochs** (30 total) in both datasets. Original: 84 subjects → 2520 epoch-blocks (1260 DS, 1260 SR total). Replication: 103 subjects → 3090 epoch-blocks (1545 DS, 1545 SR total).  
  - Original epoch boundary: **rt→main** (next epoch when we go from rt_main_trials back to main; rt trials are assigned the epoch of the block they follow). Replication uses native epoch_num.
- **Target:** Rest length (1–20 trials) chosen after each epoch (`num_rest_in_chunk`), as a proxy for cognitive fatigue.

## Feature lists

- **Baseline:** `epoch_num`, `block_num`, `avg_epoch_accuracy`, `avg_rt`, `game_type_digit_span`, `game_type_spatial_recall`, `cue_stay_between_block`, `cue_switch_between_block` (3-level cue_transition_type with reference = stay_within_block).
- **Extended (with history):** Baseline + `rest_length_prev`, `accuracy_prev`, `rt_prev` (previous epoch within subject). Rows with any missing feature are dropped (complete-case analysis).

**Missing features (complete-case):** Out of 5,610 rows, 1,376 have at least one missing extended feature and are dropped; 4,234 rows are used for extended models. Reasons: (1) **First epoch per subject** (187 rows)—no previous epoch, so `rest_length_prev`, `accuracy_prev`, and `rt_prev` are NA. (2) **Missing `avg_rt`** (1,141 rows)—blockwise has no RT when there are no valid trials (e.g. 0% accuracy). (3) **Missing `rt_prev`** (1,292 total)—first epoch (187) plus epochs whose previous epoch had missing `avg_rt` (shift carries NA forward).

## Trial-level data (for LSTM / sequence models)

- **Source:** `cleaned_exp_data/original_main_trials_cleaned.csv`, `cleaned_exp_data/replication_main_trials_cleaned.csv` (one row per trial). For trial/epoch counts or task-only analyses use only main task responses: `trial_type` in `ds_main_response`, `sr_main_response` (constant `MAIN_RESPONSE_TRIAL_TYPES` in `preprocess.py`); exclude `rt_main_trials`.
- **Loaders in `src/preprocess.py`:** `load_original_trials()`, `load_replication_trials()`, `get_trials(include_epoch_cues=True)`. Each adds `dataset` and `subject_id`. With `include_epoch_cues=True` (default), each trial gets **cue_transition_type** (3-level), **cue_type**, and **rest_type** from the blockwise data via merge on (subject_id, epoch_num). Original: epoch_num from main-response order (10 per epoch); replication: native epoch_num (NaN filled where needed).

## Methods

- **Ridge** and **Gradient Boosting** (GBM) – implemented in `scripts/run_baselines.py` for baseline and extended features.
- **LSTM** and **k-means** – planned; use trial-level data via `get_trials()` for LSTM.

## Evaluation

- **Metrics:** MAE and R² (reported as single value for one split, or mean ± std over k-fold).
- **Splits:** (1) Train on original, validate on replication (`--split dataset`). (2) K-fold by subject so no subject appears in both train and val (`--split subject --n_folds N`).
- **Distributions / Wasserstein:** Optional follow-up for comparing prediction vs actual rest-length distributions.

## File structure

- `src/preprocess.py` – load blockwise CSVs → epoch table, baseline/history features (`get_data()`, `get_feature_columns()`); load trial-level CSVs for LSTM (`get_trials()`, `load_original_trials()`, `load_replication_trials()`).
- `src/split.py` – `split_by_dataset`, `split_by_subject`, `kfold_by_subject` (by `subject_id`).
- `scripts/run_baselines.py` – run Ridge and GBM, print MAE and R².
