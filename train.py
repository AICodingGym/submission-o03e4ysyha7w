"""Train a breath-history model and submit only if it beats the baseline.

Install dependencies: python -m pip install -r requirements.txt
Run: python train.py
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupShuffleSplit


RAW_FEATURES = ["u_in", "u_out", "R", "C", "time_step"]
CATEGORICAL_FEATURES = ["R", "C", "R_C"]
BASELINE_MAE = 1.340701
ROOT = Path(__file__).resolve().parent


def make_features(data):
    """Build input-only features on complete breaths, including future inputs."""
    features = data[RAW_FEATURES].copy()
    breaths = data.groupby("breath_id", sort=False)
    features["step_index"] = breaths.cumcount()
    features["dt"] = breaths["time_step"].diff().fillna(0)
    if (features["dt"] < 0).any():
        raise ValueError("Rows within each breath must be ordered by time_step")
    features["elapsed_time"] = data["time_step"] - breaths["time_step"].transform("first")
    features["u_in_cumsum"] = breaths["u_in"].cumsum()
    features["u_in_integral"] = (data["u_in"] * features["dt"]).groupby(
        data["breath_id"], sort=False
    ).cumsum()
    for lag in (1, 2, 3, 4):
        features[f"u_in_lag_{lag}"] = breaths["u_in"].shift(lag).fillna(0)
        features[f"u_in_diff_{lag}"] = breaths["u_in"].diff(lag).fillna(0)
    for lead in (1, 2, 3):
        features[f"u_in_lead_{lead}"] = breaths["u_in"].shift(-lead).fillna(0)
        features[f"u_in_lead_diff_{lead}"] = (
            breaths["u_in"].shift(-lead) - data["u_in"]
        ).fillna(0)
    for window in (3, 5, 10):
        rolling = breaths["u_in"].rolling(window, min_periods=1)
        for statistic in ("mean", "max"):
            values = getattr(rolling, statistic)().reset_index(level=0, drop=True)
            features[f"u_in_rolling_{statistic}_{window}"] = values.reindex(data.index)
    features["u_in_breath_max"] = breaths["u_in"].transform("max")
    features["u_in_breath_mean"] = breaths["u_in"].transform("mean")
    # Use physical values before R and C are replaced by category codes below.
    features["area_div_C"] = features["u_in_integral"] / data["C"]
    features["u_in_times_R"] = data["u_in"] * data["R"]

    # Fixed category codes are shared by training and test; no target is used.
    r_code = pd.Categorical(data["R"], categories=[5, 20, 50]).codes
    c_code = pd.Categorical(data["C"], categories=[10, 20, 50]).codes
    if (r_code < 0).any() or (c_code < 0).any():
        raise ValueError("Unexpected R or C lung setting")
    features["R"] = r_code
    features["C"] = c_code
    features["R_C"] = r_code * 3 + c_code
    return features.astype("float32")


def print_validation_error_analysis(train, valid_idx, predictions):
    """Report row-weighted inspiratory errors by breath step and lung setting."""
    validation = train.iloc[valid_idx].copy()
    # Count all steps before filtering out expiratory rows so positions stay intact.
    validation["step_index"] = train.groupby("breath_id", sort=False).cumcount().iloc[valid_idx]
    validation["absolute_error"] = np.abs(
        validation["pressure"].to_numpy() - predictions
    )
    inspiratory = validation.loc[validation["u_out"] == 0]

    by_step = inspiratory.groupby("step_index").agg(
        inspiratory_rows=("absolute_error", "size"),
        inspiratory_mae=("absolute_error", "mean"),
    ).reindex(range(80))
    by_step["inspiratory_rows"] = by_step["inspiratory_rows"].fillna(0).astype(int)
    by_lung = inspiratory.groupby(["R", "C"]).agg(
        inspiratory_rows=("absolute_error", "size"),
        inspiratory_mae=("absolute_error", "mean"),
    ).sort_values("inspiratory_mae", ascending=False)

    print("\nValidation inspiratory MAE by step within breath (0-79):", flush=True)
    print(by_step.to_string(float_format="{:.6f}".format, na_rep="N/A"), flush=True)
    print("N/A means no inspiratory validation rows at that step.", flush=True)
    print("\nValidation inspiratory MAE by (R, C), highest error first:", flush=True)
    print(by_lung.to_string(float_format="{:.6f}".format), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path,
                        default=ROOT / "data" / "ventilator-pressure-prediction")
    parser.add_argument("--output", type=Path, default=ROOT / "submission.csv")
    parser.add_argument("--validation-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-iter", "--n-estimators", type=int, default=4000)
    parser.add_argument("--early-stopping-rounds", type=int, default=100)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--baseline-mae", type=float, default=BASELINE_MAE,
                        help="Submission must beat this MAE on the same split")
    args = parser.parse_args()
    if not 0 < args.validation_size < 1:
        parser.error("--validation-size must be between 0 and 1")
    if args.max_iter < 1:
        parser.error("--max-iter must be positive")
    if args.early_stopping_rounds < 1:
        parser.error("--early-stopping-rounds must be positive")

    train = pd.read_csv(
        args.data_dir / "train.csv",
        usecols=RAW_FEATURES + ["breath_id", "pressure"],
        dtype={**dict.fromkeys(RAW_FEATURES, "float32"), "pressure": "float32"},
    )
    X, y = make_features(train), train["pressure"]
    split = GroupShuffleSplit(n_splits=1, test_size=args.validation_size,
                              random_state=args.seed)
    train_idx, valid_idx = next(split.split(X, y, groups=train["breath_id"]))
    train_breaths = set(train.iloc[train_idx]["breath_id"])
    valid_breaths = set(train.iloc[valid_idx]["breath_id"])
    if train_breaths & valid_breaths:
        raise RuntimeError("Training and validation breaths overlap")
    # Feature engineering above uses all 80 rows; only now select scored rows.
    train_idx = train_idx[train.iloc[train_idx]["u_out"].to_numpy() == 0]
    valid_idx = valid_idx[train.iloc[valid_idx]["u_out"].to_numpy() == 0]
    if not len(train_idx) or not len(valid_idx):
        raise ValueError("Training and validation must both contain inspiratory rows")
    print(f"Training: {len(train_idx):,} rows / {len(train_breaths):,} breaths; "
          f"validation: {len(valid_idx):,} rows / {len(valid_breaths):,} breaths",
          flush=True)

    model = LGBMRegressor(
        objective="regression_l1", learning_rate=0.05, n_estimators=args.max_iter,
        num_leaves=31, random_state=args.seed, n_jobs=args.n_jobs, verbosity=-1,
    )
    model.fit(
        X.iloc[train_idx], y.iloc[train_idx],
        categorical_feature=CATEGORICAL_FEATURES,
        eval_set=[(X.iloc[valid_idx], y.iloc[valid_idx])], eval_metric="l1",
        callbacks=[early_stopping(args.early_stopping_rounds), log_evaluation(100)],
    )
    valid_predictions = model.predict(X.iloc[valid_idx])
    score = mean_absolute_error(y.iloc[valid_idx], valid_predictions)
    print(f"Best iteration: {model.best_iteration_}", flush=True)
    print(f"Validation inspiratory MAE (u_out == 0): {score:.6f}", flush=True)
    print_validation_error_analysis(train, valid_idx, valid_predictions)
    print(f"Baseline MAE: {args.baseline_mae:.6f}; "
          f"improvement: {args.baseline_mae - score:.6f}", flush=True)
    if not np.isfinite(score) or score >= args.baseline_mae:
        print("No improvement over baseline; submission file left unchanged.", flush=True)
        return

    print("Refitting on all training breaths...", flush=True)
    # Use the selected tree count; the full-data refit has no validation holdout.
    model = LGBMRegressor(
        objective="regression_l1", learning_rate=0.05,
        n_estimators=model.best_iteration_, num_leaves=31,
        random_state=args.seed, n_jobs=args.n_jobs, verbosity=-1,
    )
    all_inspiratory = train["u_out"] == 0
    model.fit(X.loc[all_inspiratory], y.loc[all_inspiratory],
              categorical_feature=CATEGORICAL_FEATURES)
    test = pd.read_csv(args.data_dir / "test.csv",
                       usecols=["id", "breath_id"] + RAW_FEATURES,
                       dtype=dict.fromkeys(RAW_FEATURES, "float32"))
    submission = pd.read_csv(args.data_dir / "sample_submission.csv")
    if list(submission.columns) != ["id", "pressure"]:
        raise ValueError("Expected sample submission columns: id,pressure")
    if (not test["id"].is_unique or not submission["id"].is_unique
            or len(test) != len(submission)
            or set(test["id"]) != set(submission["id"])):
        raise ValueError("Test and sample submission IDs must match uniquely")
    predictions = pd.Series(model.predict(make_features(test)), index=test["id"])
    # Map by ID to preserve the template's row order, even if it differs from test.
    submission["pressure"] = submission["id"].map(predictions)
    if not np.isfinite(submission["pressure"]).all():
        raise ValueError("Submission contains missing or non-finite predictions")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output, index=False)
    print(f"Saved {len(submission):,} predictions to {args.output}", flush=True)


if __name__ == "__main__":
    main()
