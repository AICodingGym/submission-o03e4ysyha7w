"""Train a four-feature baseline and write a Kaggle-format submission.

Install dependencies: python -m pip install -r requirements.txt
Run: python train.py
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupShuffleSplit


FEATURES = ["u_in", "u_out", "R", "C"]
ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path,
                        default=ROOT / "data" / "ventilator-pressure-prediction")
    parser.add_argument("--output", type=Path, default=ROOT / "submission.csv")
    parser.add_argument("--validation-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-iter", type=int, default=100)
    args = parser.parse_args()
    if not 0 < args.validation_size < 1:
        parser.error("--validation-size must be between 0 and 1")
    if args.max_iter < 1:
        parser.error("--max-iter must be positive")

    train = pd.read_csv(
        args.data_dir / "train.csv",
        usecols=FEATURES + ["breath_id", "pressure"],
        dtype={**dict.fromkeys(FEATURES, "float32"), "pressure": "float32"},
    )
    X, y = train[FEATURES], train["pressure"]
    split = GroupShuffleSplit(n_splits=1, test_size=args.validation_size,
                              random_state=args.seed)
    train_idx, valid_idx = next(split.split(X, y, groups=train["breath_id"]))
    train_breaths = set(train.iloc[train_idx]["breath_id"])
    valid_breaths = set(train.iloc[valid_idx]["breath_id"])
    if train_breaths & valid_breaths:
        raise RuntimeError("Training and validation breaths overlap")
    inspiratory = X.iloc[valid_idx]["u_out"].to_numpy() == 0
    if not inspiratory.any():
        raise ValueError("Validation split has no inspiratory rows (u_out == 0)")
    print(f"Training: {len(train_idx):,} rows / {len(train_breaths):,} breaths; "
          f"validation: {len(valid_idx):,} rows / {len(valid_breaths):,} breaths",
          flush=True)

    # Disable the estimator's internal row-level early-stopping split.
    model = HistGradientBoostingRegressor(
        loss="squared_error", learning_rate=0.1, max_iter=args.max_iter,
        max_leaf_nodes=31, early_stopping=False, random_state=args.seed,
    )
    model.fit(X.iloc[train_idx], y.iloc[train_idx])
    valid_predictions = model.predict(X.iloc[valid_idx])
    score = mean_absolute_error(y.iloc[valid_idx].to_numpy()[inspiratory],
                                valid_predictions[inspiratory])
    print(f"Validation inspiratory MAE (u_out == 0): {score:.6f}", flush=True)

    print("Refitting on all training breaths...", flush=True)
    model.fit(X, y)
    test = pd.read_csv(args.data_dir / "test.csv", usecols=["id"] + FEATURES,
                       dtype=dict.fromkeys(FEATURES, "float32"))
    submission = pd.read_csv(args.data_dir / "sample_submission.csv")
    if list(submission.columns) != ["id", "pressure"]:
        raise ValueError("Expected sample submission columns: id,pressure")
    if (not test["id"].is_unique or not submission["id"].is_unique
            or len(test) != len(submission)
            or set(test["id"]) != set(submission["id"])):
        raise ValueError("Test and sample submission IDs must match uniquely")
    predictions = pd.Series(model.predict(test[FEATURES]), index=test["id"])
    # Map by ID to preserve the template's row order, even if it differs from test.
    submission["pressure"] = submission["id"].map(predictions)
    if not np.isfinite(submission["pressure"]).all():
        raise ValueError("Submission contains missing or non-finite predictions")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output, index=False)
    print(f"Saved {len(submission):,} predictions to {args.output}", flush=True)


if __name__ == "__main__":
    main()
