# Deep bidirectional LSTM with grouped five-fold CV

Run using the existing CUDA environment:

```powershell
.venv/Scripts/python.exe -u train_bilstm_cv.py
```

The default architecture has **four bidirectional LSTM layers with 256 hidden units**, up from the previous three-layer, 192-unit model. It uses mixed precision, AdamW, masked L1 loss, learning rate 0.001 with cosine decay, 160 maximum epochs, and early stopping with patience 20. All five folds run sequentially on the GPU.

## Features and validation

The script preserves the 67 extended features and adds 13 more:

- Trapezoidal cumulative `u_in` area weighted by actual time differences, plus area divided by RC.
- Within-breath exponential moving averages with spans 3, 5, 10, and 20, plus current input minus each EMA.
- Second-order input differences at offsets 1 and 2.
- A time-adjusted second derivative for irregular time intervals.

Every history operation resets per breath. Features use complete 80-step breaths before filtering the loss/metric to inspiratory rows (`u_out == 0`). `u_in` is a valve command, so these are physics-inspired proxies, not a calibrated pressure equation.

`GroupKFold(n_splits=5, shuffle=True, random_state=42)` keeps whole breaths together. Each fold fits its own scaler on that fold's training breaths only. Categorical lung settings retain the existing one-hot neural encoding. A fold's model seed is 42 plus its zero-based fold index.

The script prints each fold's inspiratory MAE, the fold mean and standard deviation, and **pooled out-of-fold inspiratory MAE**. Every training breath receives exactly one prediction, from the fold that excluded it. It never averages predictions from models that trained on that breath to compute CV MAE.

The **test** predictions are averaged equally across all five best checkpoints. After all folds finish successfully, the script validates sample columns, ID order, and finite predictions, backs up the existing output, and atomically writes `submission.csv`. If another process changes that file during training, it preserves the changed file and keeps a separate candidate. The sample template is protected, and no leaderboard upload is performed.

## Runtime estimate

On this machine's **RTX 4060 Laptop GPU (8 GB)**, the final short CUDA benchmark estimates approximately **5.3 hours**, with a practical **4–7 hour** range, for five folds at the full 160-epoch budget. It measured about **24 seconds per epoch**, including estimated training and validation work. Feature preparation/checkpoint I/O add overhead. Early stopping may reduce the time; throttling, competing GPU work, and machine sleep may increase wall time. The measured settings/results are recorded in `experiments/bilstm_5fold_runtime_estimate.json`.

Re-measure for different architecture or batch settings without starting training:

```powershell
.venv/Scripts/python.exe -u train_bilstm_cv.py --estimate-only
```

Change settings as needed, using a new run name for a different configuration:

```powershell
.venv/Scripts/python.exe -u train_bilstm_cv.py --epochs 120 --run-name bilstm_cv_120
.venv/Scripts/python.exe -u train_bilstm_cv.py --output outputs/five_fold_candidate.csv
```

## Saved files and restart behavior

Artifacts are under `artifacts/experiments/bilstm_5fold/`:

- `config.json`: architecture, training settings, feature schema, and training-data hash.
- `fold_<n>/attempt_<n>/model.pt`: best weights and fold-specific normalization.
- Fold training/validation breath IDs, validation predictions, epoch logs, and fold results.
- `fold_scores.csv`, `oof_predictions.npy`, `oof_counts.npy`, `breath_ids.npy`, and `summary.json`.
- `ensemble.json`: all five checkpoint references and equal weights.
- `submission_candidate.csv` and, if an existing file was replaced, `previous_submission.csv`.

Rerunning the same command reuses completed folds after checking their configuration, IDs, and score. An unfinished fold restarts in a new attempt directory, preserving its partial checkpoint. Once all folds are cached, the same command regenerates metrics and test predictions without retraining. A changed training configuration requires a new `--run-name`.

This CV run is tracked separately from `experiments/best.json`. Its pooled single-model OOF score is **not directly comparable** to the earlier 0.197748 score from three-model averaging on one holdout. No full real-data CV score is claimed until the actual training run finishes.

## Verification

```powershell
.venv/Scripts/python.exe -u train_bilstm_cv.py --smoke-test
```

The smoke test uses ten synthetic breaths, all five folds, four small LSTM layers, and two epochs. It writes only under an artifact directory. Checks verified actual optimizer updates, training-only fold scalers, no overlapping breath groups, exactly one OOF prediction per breath, pooled MAE, the arithmetic five-model test average, and reversed template-ID mapping. Feature checks verified EMA/area/second-difference calculations and isolation from other breaths and target values. The real 0.197748 submission remains unchanged.

The full multi-hour real-data training run has **not** been started.
