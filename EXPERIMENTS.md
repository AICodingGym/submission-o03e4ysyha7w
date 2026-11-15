# Ventilator pressure experiments

**Latest result:** the three-seed LSTM average reached **0.197748 validation inspiratory MAE**, below 0.20 on the original breath holdout. See [LSTM_ENSEMBLE.md](LSTM_ENSEMBLE.md) for the script, individual scores, saved models, and submission update. `experiments/best.json` now points to that ensemble. Nothing was uploaded to the leaderboard by these scripts.

The sections below preserve the earlier experiment batch, whose best was 0.214090. Its postprocessing comparison and bootstrap tables describe that batch; the latest ensemble is logged in `experiments/results.csv` and its own run summary.

## Recorded history

All validation scores use a 20% whole-breath holdout, `GroupShuffleSplit(random_state=42)`, scored only where `u_out == 0`.

| Version | Validation MAE | Leaderboard MAE (reported by user) | Changes |
|---|---:|---:|---|
| Four-feature baseline | 6.327418 | 6.40828 | Histogram gradient boosting on `u_in`, `u_out`, `R`, `C` |
| Breath-history model | 1.340701 | 1.35652 | Step/time features, input lags/differences and cumulative input, categorical lung settings |
| L1 LightGBM | 0.737145 | 0.73469 | Leads, rolling/breath summaries, physics interactions; inspiratory-only fitting; L1 loss, learning rate 0.05, 4,000 trees |

These are **bundled changes**, not individual-feature ablations. We can attribute improvements to each bundle, but cannot claim how much any individual change contributed. Validation and leaderboard scores track closely. Step-0 validation MAE fell from 10.773621 to 0.695422 to 0.356149.

## Experiment protocol

- Keep exactly the same validation breath IDs for comparable scores. Save their hash and IDs with each experiment.
- Compute input features on complete 80-step breaths before applying inspiratory masks. Future **control inputs** are allowed because complete test breaths are supplied; pressure targets never enter features.
- Trees are fitted on inspiratory rows. Sequence models process complete breaths and mask the loss and validation MAE to inspiratory rows.
- Fit neural preprocessing on training breaths only. Save preprocessing, feature columns, architecture, and weights together.
- Preserve the current `submission.csv`. New candidate files require a separate path. Nothing is submitted by this runner.
- Repeated tuning on this holdout can overfit model selection. A new breath holdout or grouped cross-validation is needed before claiming a durable improvement, especially near 0.20.

## Reproduction

The global Python installation contains CPU-only PyTorch and conflicting OpenMP runtimes. Use the project-local `.venv` rather than bypassing the runtime conflict.

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
.venv/Scripts/python.exe -u experiments.py --name tree_capacity_127 --kind tree --profile base --trees 2000 --leaves 127
.venv/Scripts/python.exe -u experiments.py --name lstm_base --kind lstm --profile base
.venv/Scripts/python.exe -u experiments.py --name lstm_extended --kind lstm --profile extended
.venv/Scripts/python.exe -u experiments.py --name lstm_large_extended --kind lstm --profile extended --hidden 192 --layers 3 --epochs 160 --patience 20
.venv/Scripts/python.exe -u residual_experiments.py --base lstm_large_extended
```

The split seed is fixed at 42; `--seed` controls model initialization. Experiment names must be unique so existing checkpoints cannot be overwritten accidentally.

`experiments/results.csv` contains completed experiment scores. `experiments/configurations.json` stores their full arguments, and `experiments/environment.txt` records installed versions. `experiments/best.json` points to the selected model or ensemble. Model binaries, per-epoch logs, validation predictions, and per-step/per-lung tables are kept in `artifacts/experiments/<name>/`. The saved models are trained on the training partition; their validation scores refer to that exact checkpoint, not to an unvalidated full-data refit. Elapsed times include a long machine pause and overlapping jobs, so they are not compute benchmarks.

Compare, select, and verify saved checkpoints:

```powershell
.venv/Scripts/python.exe -u compare_experiments.py --select-best
.venv/Scripts/python.exe verify_saved_model.py
```

Comparison checks the saved validation breath IDs and recomputes every run's MAE before testing fixed blend weights of 0.1, 0.25, 0.5, 0.75, and 0.9. Pressure-level snapping uses only levels observed in the **training partition**. Bootstrap intervals resample complete breaths, not individual correlated rows; these intervals are conditional on the fitted models and do not account for repeated model selection or training-seed variability.

Optional local candidate generation (never submits):

```powershell
.venv/Scripts/python.exe experiments.py --predict outputs/candidate_submission.csv
```

## New results

All runs below use the same 54,324 training / 13,581 validation breaths. The split hash is `45fc6be0675e269600bc31e117e5848cac60eafcb4c6dfe7318cf3cc55dad91e`.

| Experiment | Validation MAE | Interpretation |
|---|---:|---|
| Previous submitted LightGBM | 0.737145 | Historical reference; 31 leaves, 4,000 trees |
| Tuned L1 tree | 0.657357 | 127 leaves, 2,000 trees, feature fraction 0.9; a parameter bundle, not a single-setting ablation |
| Tuned tree without short leads | 0.668737 | Removing the six lead/lead-difference columns hurts by 0.011379 |
| Tuned tree without two physics interactions | 0.658810 | Removing `area/C` and `u_in*R` changes MAE by only 0.001453 |
| Tuned tree with L2 objective | 0.632856 | Better than L1 by 0.024501 with the other settings fixed |
| Tuned L1 tree trained on all phases | 0.709083 | Worse by 0.051726; validation still scores only inspiration |
| Bidirectional LSTM, existing 35 features | 0.354547 | 128 hidden units, two layers, 60 epochs; best checkpoint at epoch 59 |
| Matched LSTM, extended 67 features | 0.349692 | Same architecture, seed, and epoch budget; best checkpoint at epoch 59 |
| Larger/longer LSTM, extended features | 0.216131 | 192 hidden units, three layers, 160 epochs; best checkpoint at epoch 157 |
| Large LSTM + training-only lung/step offsets | 0.216103 | Essentially no improvement: only 0.000029 |
| Large LSTM + small residual tree | 0.215261 | 15 leaves, 300 trees, learning rate 0.03; training-only residual fitting |
| **Selected pipeline: residual tree + pressure snapping** | **0.214090** | **Lowest measured selection-holdout score; not a new leaderboard score** |

The extended feature bundle adds longer leads/lags, rolling standard deviations, centered rolling means, initial input and peak timing, inspiratory summaries/duration, input slope, trapezoidal area/C, and exponentially decayed input-memory proxies at three RC time scales. These are physically inspired proxies; `u_in` is a valve-control signal, not measured flow.

### What the controlled comparisons show

- **Short leads help**: removing them increases MAE by 0.011379; paired 95% breath-bootstrap interval [0.009719, 0.012860]. Step-0 MAE rises from 0.324877 to 0.385467.
- **Inspiratory-only fitting helps**: including expiration increases MAE by 0.051726; interval [0.049067, 0.054424].
- **L1 is not automatically best**: L2 reduces MAE by 0.024501; interval [-0.028561, -0.020772]. The historical 1.3407-to-0.7371 improvement cannot be credited to L1 alone.
- **The two physics interactions have little demonstrated standalone benefit**: their removal changes MAE by 0.001453; interval [-0.000390, 0.003360], which includes zero.
- **The extended neural features help modestly**: MAE changes by -0.004855; interval [-0.008999, -0.000598]. This is much smaller than the tree-to-sequence-model gain and has only been tested with one initialization.
- **A naive blend can hurt**: a 50/50 blend of the tuned L1 tree and base LSTM scores 0.453769, worse than the LSTM's 0.354547. Training-only pressure snapping improves that LSTM only slightly, to 0.353832.

The large model's increased size and longer training budget are a combined experiment, so that gain cannot be assigned to either factor separately. A blend of the two small neural models also helps (0.333660, or 0.332939 with snapping), but none of the tested blends beats the selected large-model pipeline.

## Previous pipeline and verification (0.21409)

**At the end of that batch, the best validation inspiratory MAE was 0.2140895873.** The <0.20 goal had not yet been reached. This was a 71.0% reduction from the previous model's validation MAE of 0.737145. The later three-seed ensemble above closes the remaining gap.

At that point, `experiments/best.json` pointed to `artifacts/experiments/selected/ensemble.json`. That older descriptor remains available. Despite its generic name, that pipeline is one neural model plus its additive residual tree and pressure-level snapping; it is not an average of multiple neural models. Its saved artifacts:

- `artifacts/experiments/lstm_large_extended/model.pt`: neural weights, training-fitted scaler, feature columns, architecture, and epoch.
- `artifacts/experiments/lstm_large_extended_residual_tree/model.joblib`: residual tree and its neural-checkpoint reference.
- `artifacts/experiments/selected/ensemble.json`: complete pipeline descriptor.
- `artifacts/experiments/selected/pressure_grid.npy`: 950 pressure levels learned only from training targets.

Model binaries are local artifacts, ignored by Git. The scripts, configurations, environment, and result tables are versioned. The saved checkpoint uses only the training partition; there has been no full-data refit falsely assigned this holdout score.

During that batch, checkpoint reload/replay passed on **7,767 inspiratory validation rows**, reproducing the saved predictions exactly. The then-original `submission.csv` was preserved (SHA256 `E0192518690626E23B1297118B264C69DFFF5AFB656DFF9C514C2D36043E36D7`). The separate 0.21409 `outputs/candidate_submission.csv` passed an independent check of all **603,600** rows: exact sample columns and ID order, with finite predictions. Later requests replaced `submission.csv`; its current contents are documented in LSTM_ENSEMBLE.md. No leaderboard upload was performed by these scripts.

`models/best_model_bundle.zip` packages the selected weights, preprocessing/pipeline metadata, source code, environment record, result tables, and validation-replay snapshot. It includes a SHA256 manifest and an inference README; original CSV datasets and Python packages are not included. Other experiments' model binaries remain available locally under `artifacts/experiments/`.

Full results are in `experiments/results.csv`; postprocessing comparisons in `experiments/comparisons.csv`; uncertainty tables in `experiments/ablation_bootstrap.csv` and `experiments/paired_bootstrap.csv`. The selected pipeline's error tables are `artifacts/experiments/selected/mae_by_step.csv` and `mae_by_lung.csv`. Blank step MAEs mean no scored rows exist at that step.

## Recommendations from that earlier batch

1. **Train additional independent neural seeds or grouped folds and ensemble comparable models.** Blending the two small neural models already worked, while mixing a strong neural model with a much weaker tree did not. Two or more models comparable to the large LSTM are the strongest next experiment; getting below 0.20 is plausible, not demonstrated yet.
2. **Target the current hardest lung settings, not just C=10.** The selected pipeline's MAEs are 0.329230 for `(R=50,C=50)` and 0.293580 for `(50,20)`, versus 0.224239 for `(50,10)`. Test shared-model fine-tuning or conditional experts for those two hard groups, retaining the global model elsewhere. Step 0 is now 0.162454, so it is no longer the main problem.
3. **Confirm on another whole-breath fold before any upload.** Architecture, early stopping, blend weights, and postprocessing were selected on the same holdout. Breath-bootstrap intervals do not remove that selection effect. After confirmation, train the chosen fold ensemble or refit at the selected epoch budget, generate a fresh candidate, and only then consider a leaderboard submission.
