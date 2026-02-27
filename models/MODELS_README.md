# Model Configuration Summary

Features and hyperparameters used for baselines and LSTM.

## Splits
- **dataset**: train=original, test=replication
- **pooled**: 90% train / 10% test (using pre-determined split when --load-split provided)

## Baseline Features (baseline_only=True)

- epoch_num
- block_num
- avg_epoch_accuracy
- accuracy_sd
- avg_rt
- rt_sd
- num_timeouts
- game_type_digit_span
- game_type_spatial_recall
- cue_stay_between_block
- cue_switch_between_block
- rests_taken_so_far
- mean_rest_length_subj
- avg_accuracy_until_now_digit_span
- avg_accuracy_until_now_spatial_recall

## Extended Features (baseline_only=False, for Ridge extended, GBM extended, LSTM)

- epoch_num
- block_num
- avg_epoch_accuracy
- accuracy_sd
- avg_rt
- rt_sd
- num_timeouts
- game_type_digit_span
- game_type_spatial_recall
- cue_stay_between_block
- cue_switch_between_block
- rests_taken_so_far
- mean_rest_length_subj
- avg_accuracy_until_now_digit_span
- avg_accuracy_until_now_spatial_recall
- rest_length_prev
- accuracy_prev
- rt_prev
- game_type_digit_span_prev
- game_type_spatial_recall_prev
- previous_cue_stay_between_block
- previous_cue_switch_between_block

## Baseline Tuning & Params

- Ridge alpha sweep: 0.01,0.05,0.1,0.5,1,10,20,100
- GBM n_estimators: 100,200,300,400,500
- GBM max_depth: 1,2,3,4
- GBM learning_rate: 0.05
- n_folds: 15

### dataset split (tuned values)
- Ridge alpha (baseline): 100.0
- Ridge alpha (extended): 100.0
- GBM n_estimators (baseline): 200
- GBM max_depth (baseline): 4
- GBM n_estimators (extended): 400
- GBM max_depth (extended): 1

### pooled split (tuned values)
- Ridge alpha (baseline): 100.0
- Ridge alpha (extended): 100.0
- GBM n_estimators (baseline): 100
- GBM max_depth (baseline): 4
- GBM n_estimators (extended): 500
- GBM max_depth (extended): 1

## LSTM Params

- epochs: 60
- hidden: 16
- layers: 2
- dropout: 0.5
- lr: 0.002
- weight_decay: 0.001
- batch_size: 8
