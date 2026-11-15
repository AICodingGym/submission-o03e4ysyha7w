"""Test training-only corrections to the best completed neural checkpoint.

The validation targets are used for scoring/early stopping only. Neither offsets
nor residual-tree fitting use validation labels. No submission is written.
"""

import argparse
import time
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import torch
from lightgbm import LGBMRegressor, early_stopping, log_evaluation

from experiments import (ARTIFACTS, CATEGORICAL_FEATURES, ROOT, TRACK, build_network,
                         load_arrays, neural_inputs, record, split_breaths)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base")
    args = parser.parse_args()
    completed = pd.read_csv(TRACK / "results.csv")
    completed = completed.loc[completed["kind"] == "lstm"]
    if args.base:
        completed = completed.loc[completed["name"] == args.base]
    base = completed.sort_values("validation_mae").iloc[0].to_dict()
    print("Correcting checkpoint:", base["name"], flush=True)
    x, y, mask, ids, columns = load_arrays(base["profile"])
    tr, va = split_breaths(ids)
    checkpoint = torch.load(ROOT / base["model"], map_location="cpu", weights_only=False)
    if columns != checkpoint["columns"]:
        raise ValueError("Cached features differ from the checkpoint")
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    model = build_network(checkpoint["input_size"], checkpoint["hidden"], checkpoint["layers"]).cuda()
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    pred = []
    with torch.no_grad():
        for start in range(0, len(x), 256):
            value = neural_inputs(np.asarray(x[start:start + 256]), columns)
            value = np.clip((value - checkpoint["mean"]) / checkpoint["std"], -20, 20)
            with torch.autocast("cuda", dtype=torch.float16):
                p = model(torch.from_numpy(value).cuda())
            pred.append(p.float().cpu().numpy())
    pred = np.concatenate(pred)
    raw_mae = float(np.abs(pred[va] - y[va])[mask[va]].mean())
    if abs(raw_mae - base["validation_mae"]) > 0.001:
        raise AssertionError(f"Checkpoint replay MAE differs: {raw_mae} vs {base['validation_mae']}")
    print(f"Replayed base MAE: {raw_mae:.6f}", flush=True)
    residual = y - pred
    codes = (np.asarray(x[:, :, columns.index("R_C")]) * 80
             + np.asarray(x[:, :, columns.index("step_index")])).astype(int)

    for method in ("offsets", "residual_tree"):
        start = time.monotonic()
        name = base["name"] + "_" + method
        folder = ARTIFACTS / name
        folder.mkdir(exist_ok=False)
        result = dict(base=base, profile=base["profile"], method=method, columns=columns)
        if method == "offsets":
            training_residual = pd.DataFrame(dict(code=codes[tr][mask[tr]],
                                                  residual=residual[tr][mask[tr]]))
            grouped = training_residual.groupby("code")["residual"].median()
            offsets = np.zeros(9 * 80, dtype="float32")
            offsets[grouped.index.to_numpy()] = grouped.to_numpy()
            corrected = pred[va] + offsets[codes[va]]
            result["offsets"] = offsets
            iteration = 0
        else:
            xt = pd.DataFrame(np.asarray(x[tr]).reshape(-1, len(columns))[mask[tr].ravel()], columns=columns)
            xv = pd.DataFrame(np.asarray(x[va]).reshape(-1, len(columns))[mask[va].ravel()], columns=columns)
            tree = LGBMRegressor(objective="regression_l1", metric="l1", num_leaves=15,
                                 n_estimators=300, learning_rate=0.03, min_child_samples=100,
                                 n_jobs=6, random_state=42, verbosity=-1)
            tree.fit(xt, residual[tr][mask[tr]], categorical_feature=CATEGORICAL_FEATURES,
                     eval_set=[(xv, residual[va][mask[va]])], eval_metric="l1",
                     callbacks=[early_stopping(60), log_evaluation(50)])
            corrected = pred[va].copy()
            corrected[mask[va]] += tree.predict(xv)
            result["model"] = tree
            iteration = tree.best_iteration_
        score = float(np.abs(corrected - y[va])[mask[va]].mean())
        joblib.dump(result, folder / "model.joblib")
        np.save(folder / "validation_predictions.npy", corrected)
        provenance = SimpleNamespace(name=name, kind="correction", profile=base["profile"],
                                     seed=42, base=base["name"], method=method)
        record(provenance, score, iteration, "model.joblib", folder, ids, va,
               y, mask, columns, x, time.monotonic() - start)


if __name__ == "__main__":
    main()
