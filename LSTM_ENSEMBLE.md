# Three-seed LSTM ensemble

Run from the project directory using the existing CUDA environment:

```powershell
.venv/Scripts/python.exe -u train_lstm_ensemble.py
```

The default model seeds are **42, 43, and 44**. The validation split stays fixed at seed 42 and holds out the same 13,581 complete breaths for all models. Each model uses the extended features, a bidirectional three-layer LSTM with 192 hidden units, 160 maximum epochs, cosine-decayed learning rate starting at 0.001, and patience 20. Scaling is fitted only on training breaths. Features use complete 80-step breaths; loss and MAE use only `u_out == 0` rows.

The script reuses compatible completed checkpoints by default, verifying configuration, validation breath IDs, the saved MAE, and a checkpoint replay batch. For the first run, this reuses the already-trained **raw seed-42 LSTM (MAE 0.216131)** and trains seeds 43 and 44 independently. The older residual correction and pressure rounding are not included in this ensemble: its score measures the **equal-weight mean of the three raw predictions**.

To train all three from scratch in a separate run:

```powershell
.venv/Scripts/python.exe -u train_lstm_ensemble.py --no-reuse-existing --run-name lstm_3seed_fresh
```

Other seeds can be supplied using `--seeds 101 202 303`; use a new run name when changing training settings. Rerunning an interrupted run skips completed members. An unfinished member restarts in a new attempt directory, preserving its earlier partial files. A completed run prints its saved summary without retraining.

Artifacts are saved under `artifacts/experiments/lstm_3seed/`:

- `config.json`: training settings, training-data hash, feature-source hash, and holdout hash.
- `seed_<seed>/attempt_<number>/`: newly trained checkpoints, per-epoch logs, validation predictions, and scores. Reused models remain at their original saved paths.
- `ensemble.json`: equal-weight member references, readable by the existing inference helper.
- `validation_predictions.npy`, `validation_breath_ids.npy`, and error tables: ensemble validation outputs.
- `summary.json`: individual MAEs, ensemble MAE, whether it broke 0.20, and whether the submission was replaced.

The script compares the ensemble with the **best recorded validation score**, initially 0.2140895873. If it improves, it averages those same three checkpoints on test data, validates the sample columns/IDs and finite predictions, backs up the old submission to `previous_submission.csv` in the run directory, and atomically replaces `submission.csv`. It can replace the submission when the score improves even if it remains above 0.20. Otherwise the existing submission and best-model selection are preserved. If another process changes the submission during training, the script saves a candidate but preserves that changed file.

The saved models retain the original training/validation split; there is no unvalidated full-data refit. No leaderboard upload is performed, and `sample_submission.csv` is never changed.

Synthetic integration checks passed for prediction averaging (rather than averaging MAEs), strict improvement/equality handling, reversed template ID order, backups, and changed-file protection.

## Run results

| Model | Best epoch | Validation inspiratory MAE |
|---|---:|---:|
| Seed 42 (verified existing checkpoint) | 157 | 0.216131 |
| Seed 43 | 160 | 0.213762 |
| Seed 44 | 159 | 0.209266 |
| **Equal-weight average of all three** | — | **0.197748** |

The exact ensemble MAE is **0.1977481025**, below 0.20 on the original whole-breath holdout and **7.63% lower** than the prior best of 0.2140895873. This score comes from averaging raw predictions, without residual correction or pressure rounding. It is a validation result, not a new leaderboard score.

The script replaced `submission.csv` with **603,600** averaged test predictions. The previous 0.21409 file is backed up at `artifacts/experiments/lstm_3seed/previous_submission.csv`; `sample_submission.csv` remains the original template. Independent checks confirmed exact sample columns/ID order, finite predictions, and matching hashes between the generated candidate and the final submission. No leaderboard upload was performed.

Reload/replay passed on **7,767 inspiratory validation rows** (mean prediction difference 0.000001, maximum 0.000003, due to floating-point averaging precision). The selected ensemble is registered in `experiments/best.json`, and `models/lstm_3seed_bundle.zip` contains all three checkpoints, their preprocessing, the equal-weight descriptor, source code, and result metadata. Raw CSV datasets and Python packages are not included.

The goal is reached on this holdout. A different whole-breath validation fold remains the appropriate next check before treating the score as an estimate for a future leaderboard result.
