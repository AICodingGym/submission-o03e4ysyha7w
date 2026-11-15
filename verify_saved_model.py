"""Reload the selected checkpoint and replay one saved validation batch."""

import json

import numpy as np
import pandas as pd

from experiments import DATA, RAW_FEATURES, ROOT, TRACK, predict_member


def main():
    best = json.loads((TRACK / "best.json").read_text())
    folder = (ROOT / best["model"]).parent
    ids = np.load(folder / "validation_breath_ids.npy")[:256]
    expected = np.load(folder / "validation_predictions.npy")[:256].ravel()
    data = pd.read_csv(DATA / "train.csv", usecols=RAW_FEATURES + ["breath_id"],
                       dtype=dict.fromkeys(RAW_FEATURES, "float32"))
    data = data.loc[data["breath_id"].isin(ids)]
    np.testing.assert_array_equal(data["breath_id"].to_numpy().reshape(-1, 80)[:, 0], ids)
    replay = predict_member(best, data)
    scored = data["u_out"].to_numpy() == 0
    difference = np.abs(replay - expected)[scored]
    # Mixed-precision cuDNN kernels can differ slightly across inference batches.
    if difference.mean() > 0.005 or difference.max() > 0.15:
        raise AssertionError(f"Checkpoint replay mismatch: mean={difference.mean()}, max={difference.max()}")
    print(f"Checkpoint replay passed on {scored.sum():,} scored rows: "
          f"mean prediction difference={difference.mean():.6f}, max={difference.max():.6f}")


if __name__ == "__main__":
    main()
