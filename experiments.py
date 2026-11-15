"""Reproducible whole-breath experiments. Never submits or replaces submission.csv.

Examples (use the isolated environment documented in EXPERIMENTS.md):
  python -u experiments.py --name lstm_base --kind lstm --profile base
  python -u experiments.py --name lstm_extended --kind lstm --profile extended
  python -u experiments.py --name tree_capacity --kind tree --profile base
  python experiments.py --predict outputs/candidate_submission.csv
"""

import argparse
import csv
import hashlib
import inspect
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from train import CATEGORICAL_FEATURES, RAW_FEATURES, ROOT, make_features


TRACK = ROOT / "experiments"
ARTIFACTS = ROOT / "artifacts" / "experiments"
DATA = ROOT / "data" / "ventilator-pressure-prediction"


def features(data, profile):
    x = make_features(data)
    if profile == "base":
        return x
    g = data.groupby("breath_id", sort=False)
    for offset in (5, 8, 12):
        for direction in (-1, 1):
            label = "lead" if direction < 0 else "lag"
            shifted = g["u_in"].shift(direction * offset)
            x[f"u_in_{label}_{offset}"] = shifted.fillna(0)
            x[f"u_in_{label}_diff_{offset}"] = (data["u_in"] - shifted).fillna(0)
    for window in (5, 10, 20):
        x[f"u_in_std_{window}"] = g["u_in"].rolling(
            window, min_periods=1
        ).std(ddof=0).reset_index(level=0, drop=True).reindex(data.index)
    for window in (5, 11):
        x[f"u_in_centered_mean_{window}"] = g["u_in"].rolling(
            window, min_periods=1, center=True
        ).mean().reset_index(level=0, drop=True).reindex(data.index)
    x["u_in_initial"] = g["u_in"].transform("first")
    x["u_in_peak_step"] = g["u_in"].transform(
        "idxmax"
    ).map(x["step_index"])
    inspiration = data["u_out"].eq(0)
    insp_time = data["time_step"].where(inspiration).groupby(data["breath_id"]).transform("max")
    x["inspiratory_duration"] = insp_time
    x["time_to_expiration"] = insp_time - data["time_step"]
    for stat in ("mean", "max", "std"):
        x[f"inspiratory_u_in_{stat}"] = data["u_in"].where(inspiration).groupby(
            data["breath_id"]
        ).transform(stat).fillna(0)
    x["input_slope"] = x["u_in_diff_1"] / x["dt"].clip(lower=1e-3)
    x["RC_seconds"] = data["R"] * data["C"] / 1000
    x["relative_time_RC"] = x["elapsed_time"] / x["RC_seconds"]
    x["cumulative_mean_u_in"] = x["u_in_cumsum"] / (x["step_index"] + 1)
    x["trapezoid_area_div_C"] = (
        ((data["u_in"] + g["u_in"].shift().fillna(data["u_in"])) * 0.5 * x["dt"])
        .groupby(data["breath_id"], sort=False).cumsum() / data["C"]
    )
    # RC memory features: exponential response at three physical time scales.
    n = len(data) // 80
    control = data["u_in"].to_numpy().reshape(n, 80)
    dt = x["dt"].to_numpy().reshape(n, 80)
    rc = x["RC_seconds"].to_numpy().reshape(n, 80)
    for factor in (0.5, 1.0, 2.0):
        state = np.zeros_like(control)
        for step in range(1, 80):
            decay = np.exp(-dt[:, step] / (rc[:, step] * factor))
            state[:, step] = decay * state[:, step - 1] + (1 - decay) * control[:, step]
        x[f"RC_memory_{factor}"] = state.ravel()
    return x.astype("float32")


def load_arrays(profile):
    cache = ARTIFACTS / f"cache_{profile}"
    cache.mkdir(parents=True, exist_ok=True)
    source = DATA / "train.csv"
    # Invalidate cached features when their implementation or the data changes.
    signature = hashlib.sha256(
        inspect.getsource(features).encode() + inspect.getsource(make_features).encode()
        + str((source.stat().st_size, source.stat().st_mtime_ns)).encode()
    ).hexdigest()
    marker = cache / "signature.txt"
    if not marker.exists() or marker.read_text() != signature:
        print(f"Building {profile} features from complete 80-step breaths...", flush=True)
        d = pd.read_csv(source, usecols=RAW_FEATURES + ["breath_id", "pressure"],
                        dtype={**dict.fromkeys(RAW_FEATURES, "float32"), "pressure": "float32"})
        ids = d["breath_id"].to_numpy().reshape(-1, 80)
        if not np.all(ids == ids[:, :1]) or len(np.unique(ids[:, 0])) != len(ids):
            raise ValueError("Expected contiguous, unique, complete 80-step breaths")
        x = features(d, profile)
        if not np.isfinite(x.to_numpy()).all():
            raise ValueError("Non-finite features")
        np.save(cache / "X.npy", x.to_numpy().reshape(-1, 80, len(x.columns)))
        np.save(cache / "y.npy", d["pressure"].to_numpy().reshape(-1, 80))
        np.save(cache / "mask.npy", d["u_out"].to_numpy().reshape(-1, 80) == 0)
        np.save(cache / "breaths.npy", ids[:, 0])
        (cache / "columns.json").write_text(json.dumps(list(x.columns)))
        marker.write_text(signature)
    return (
        np.load(cache / "X.npy", mmap_mode="r"), np.load(cache / "y.npy"),
        np.load(cache / "mask.npy"), np.load(cache / "breaths.npy"),
        json.loads((cache / "columns.json").read_text()),
    )


def split_breaths(ids):
    train, valid = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42).split(
        ids, groups=ids
    ))
    assert not set(ids[train]) & set(ids[valid])
    return train, valid


def neural_inputs(x, columns):
    # One-hot categorical codes for the network; preserve numeric lung values too.
    numeric = [i for i, c in enumerate(columns) if c not in CATEGORICAL_FEATURES]
    pieces = [np.asarray(x[:, :, numeric])]
    for column, count in (("R", 3), ("C", 3), ("R_C", 9)):
        codes = np.asarray(x[:, :, columns.index(column)], dtype=int)
        pieces.append(np.eye(count, dtype="float32")[codes])
    return np.concatenate(pieces, axis=-1)


def build_network(input_size, hidden, layers):
    import torch.nn as nn

    class SequenceModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.input = nn.Sequential(nn.Linear(input_size, hidden), nn.LayerNorm(hidden), nn.SiLU())
            self.lstm = nn.LSTM(hidden, hidden, num_layers=layers, batch_first=True,
                                bidirectional=True, dropout=0.1 if layers > 1 else 0)
            self.head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))

        def forward(self, x):
            x, _ = self.lstm(self.input(x))
            return self.head(x).squeeze(-1) * 20.0

    return SequenceModel()


def train_neural(args, x, y, mask, columns, tr, va, folder):
    import torch

    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("GPU PyTorch required for these sequence experiments")
    torch.backends.cudnn.benchmark = True
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    values = neural_inputs(x, columns)
    # Fit scaling on training breaths only, never on the validation breaths.
    mean = values[tr].mean(axis=(0, 1), dtype=np.float64).astype("float32")
    std = values[tr].std(axis=(0, 1), dtype=np.float64).astype("float32")
    std = np.maximum(std, 1e-5)
    values -= mean
    values /= std
    values = np.clip(values, -20, 20)
    gx = torch.from_numpy(values).cuda()
    gy, gm = torch.from_numpy(y).cuda(), torch.from_numpy(mask).cuda()
    train_ids, valid_ids = torch.tensor(tr, device="cuda"), torch.tensor(va, device="cuda")
    model = build_network(values.shape[-1], args.hidden, args.layers).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=1e-5)
    scaler = torch.amp.GradScaler("cuda")
    best, patience, best_epoch = float("inf"), 0, 0
    history = folder / "epochs.csv"
    for epoch in range(1, args.epochs + 1):
        start = time.monotonic()
        model.train()
        shuffled = train_ids[torch.randperm(len(tr), device="cuda")]
        total_loss, total_count = 0.0, 0
        for batch in shuffled.split(args.batch_size):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                pred = model(gx[batch])
                loss = (pred.float() - gy[batch]).abs()[gm[batch]].mean()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            count = int(gm[batch].sum())
            total_loss += float(loss.detach()) * count
            total_count += count
        model.eval()
        predictions = []
        with torch.no_grad():
            for batch in valid_ids.split(args.batch_size):
                with torch.autocast("cuda", dtype=torch.float16):
                    pred = model(gx[batch])
                predictions.append(pred.float().cpu().numpy())
        predictions = np.concatenate(predictions)
        score = float(np.abs(predictions - y[va])[mask[va]].mean())
        lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        row = dict(epoch=epoch, train_mae=total_loss / total_count, validation_mae=score,
                   learning_rate=lr, seconds=time.monotonic() - start)
        with history.open("a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            if epoch == 1:
                writer.writeheader()
            writer.writerow(row)
        print(f"Epoch {epoch:03}: train {row['train_mae']:.6f}, valid {score:.6f}, "
              f"{row['seconds']:.1f}s", flush=True)
        if score < best:
            best, patience, best_epoch = score, 0, epoch
            torch.save(dict(state_dict=model.state_dict(), mean=mean, std=std,
                            columns=columns, profile=args.profile, hidden=args.hidden,
                            layers=args.layers, input_size=values.shape[-1],
                            validation_mae=best, epoch=epoch, seed=args.seed), folder / "model.pt")
            np.save(folder / "validation_predictions.npy", predictions)
        else:
            patience += 1
        if patience >= args.patience:
            print("Neural early stopping", flush=True)
            break
    return best, best_epoch, "model.pt"


def train_tree(args, x, y, mask, columns, tr, va, folder):
    import joblib
    from lightgbm import LGBMRegressor, early_stopping, log_evaluation

    training_mask = np.ones_like(mask[tr]) if args.include_expiratory else mask[tr]
    xt = pd.DataFrame(np.asarray(x[tr]).reshape(-1, len(columns))[training_mask.ravel()], columns=columns)
    xv = pd.DataFrame(np.asarray(x[va]).reshape(-1, len(columns))[mask[va].ravel()], columns=columns)
    yt, yv = y[tr][training_mask], y[va][mask[va]]
    model = LGBMRegressor(objective=args.objective, metric="l1", learning_rate=0.05,
                          n_estimators=args.trees, num_leaves=args.leaves,
                          min_child_samples=20, colsample_bytree=0.9,
                          n_jobs=6, random_state=args.seed, verbosity=-1)
    model.fit(xt, yt, categorical_feature=CATEGORICAL_FEATURES,
              eval_set=[(xv, yv)], eval_metric="l1",
              callbacks=[early_stopping(150), log_evaluation(100)])
    pred = model.predict(xv)
    score = float(np.abs(pred - yv).mean())
    full_pred = np.zeros_like(y[va])
    full_pred[mask[va]] = pred
    np.save(folder / "validation_predictions.npy", full_pred)
    joblib.dump(dict(model=model, columns=columns, profile=args.profile), folder / "model.joblib")
    return score, model.best_iteration_, "model.joblib"


def record(args, score, iteration, model_file, folder, ids, va, y, mask, columns, x, elapsed,
           promote=True):
    pred = np.load(folder / "validation_predictions.npy")
    error = np.abs(pred - y[va])
    rc_codes = np.asarray(x[va, 0, columns.index("R_C")], dtype=int)
    per_step = pd.DataFrame({"step": np.arange(80), "rows": mask[va].sum(axis=0),
                            "mae": np.divide((error * mask[va]).sum(axis=0),
                                             mask[va].sum(axis=0),
                                             out=np.full(80, np.nan), where=mask[va].sum(axis=0) > 0)})
    per_step.to_csv(folder / "mae_by_step.csv", index=False)
    lung_rows = []
    for code in np.unique(rc_codes):
        chosen = rc_codes == code
        lung_rows.append(dict(R=[5, 20, 50][code // 3], C=[10, 20, 50][code % 3],
                              mae=float(error[chosen][mask[va][chosen]].mean())))
    pd.DataFrame(lung_rows).sort_values("mae", ascending=False).to_csv(folder / "mae_by_lung.csv", index=False)
    split_hash = hashlib.sha256(np.sort(ids[va]).tobytes()).hexdigest()
    np.save(folder / "validation_breath_ids.npy", ids[va])
    result = dict(name=args.name, kind=args.kind, profile=args.profile, seed=args.seed,
                  validation_mae=score, step0_mae=float(error[:, 0].mean()),
                  best_iteration=iteration, elapsed_seconds=elapsed, split_sha256=split_hash,
                  model=str((folder / model_file).relative_to(ROOT)))
    (folder / "result.json").write_text(json.dumps({**result, "args": vars(args)}, indent=2))
    TRACK.mkdir(exist_ok=True)
    with (TRACK / "results.csv").open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result))
        if handle.tell() == 0:
            writer.writeheader()
        writer.writerow(result)
    best_file = TRACK / "best.json"
    if promote and (not best_file.exists() or score < json.loads(best_file.read_text())["validation_mae"]):
        best_file.write_text(json.dumps(result, indent=2))
        print("Saved new best model manifest:", best_file, flush=True)
    print(json.dumps(result, indent=2), flush=True)


def snap_to_grid(predictions, grid):
    index = np.searchsorted(grid, predictions).clip(1, len(grid) - 1)
    left, right = grid[index - 1], grid[index]
    return np.where(np.abs(predictions - left) <= np.abs(predictions - right), left, right)


def predict_member(best, data):
    if best["kind"] == "ensemble":
        descriptor = json.loads((ROOT / best["model"]).read_text())
        predictions = sum(member["weight"] * predict_member(member["run"], data)
                          for member in descriptor["members"])
        if descriptor.get("pressure_grid"):
            predictions = snap_to_grid(predictions, np.load(ROOT / descriptor["pressure_grid"]))
        return predictions
    if best["kind"] == "correction":
        import joblib
        checkpoint = joblib.load(ROOT / best["model"])
        base = predict_member(checkpoint["base"], data)
        x = features(data, checkpoint["profile"])
        if checkpoint["method"] == "offsets":
            codes = (x["R_C"] * 80 + x["step_index"]).astype(int)
            return base + checkpoint["offsets"][codes]
        return base + checkpoint["model"].predict(x[checkpoint["columns"]])
    if best["kind"] == "lstm":
        import torch
        torch.set_num_threads(4)
        torch.backends.cudnn.benchmark = True
        checkpoint = torch.load(ROOT / best["model"], map_location="cpu", weights_only=False)
        x = features(data, checkpoint["profile"])
        if list(x.columns) != checkpoint["columns"]:
            raise ValueError("Checkpoint feature columns differ")
        values = neural_inputs(x.to_numpy().reshape(-1, 80, len(x.columns)), list(x.columns))
        values = np.clip((values - checkpoint["mean"]) / checkpoint["std"], -20, 20)
        model = build_network(checkpoint["input_size"], checkpoint["hidden"], checkpoint["layers"]).cuda()
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        predictions = []
        with torch.no_grad():
            for start in range(0, len(values), 256):
                with torch.autocast("cuda", dtype=torch.float16):
                    p = model(torch.from_numpy(values[start:start + 256]).cuda())
                predictions.append(p.float().cpu().numpy())
        predictions = np.concatenate(predictions).ravel()
    else:
        import joblib
        checkpoint = joblib.load(ROOT / best["model"])
        predictions = checkpoint["model"].predict(features(data, checkpoint["profile"])[checkpoint["columns"]])
    return predictions


def predict(output):
    """Create a review candidate from the saved validation-trained winner."""
    best = json.loads((TRACK / "best.json").read_text())
    data = pd.read_csv(DATA / "test.csv", dtype=dict.fromkeys(RAW_FEATURES, "float32"))
    predictions = predict_member(best, data)
    template = pd.read_csv(DATA / "sample_submission.csv")
    if (len(template) != len(data) or not template["id"].is_unique
            or not data["id"].is_unique or set(template["id"]) != set(data["id"])):
        raise ValueError("Test and sample submission IDs differ")
    template["pressure"] = template["id"].map(pd.Series(predictions, index=data["id"]))
    if not np.isfinite(template["pressure"]).all():
        raise ValueError("Invalid candidate predictions")
    path = Path(output)
    if path.resolve() == (ROOT / "submission.csv").resolve():
        raise ValueError("Use a separate candidate path; submission.csv is preserved")
    path.parent.mkdir(parents=True, exist_ok=True)
    template.to_csv(path, index=False)
    print(f"Wrote {len(template):,} candidate rows to {path}; nothing submitted")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name")
    parser.add_argument("--kind", choices=["lstm", "tree"], default="lstm")
    parser.add_argument("--profile", choices=["base", "extended"], default="base")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trees", type=int, default=2000)
    parser.add_argument("--leaves", type=int, default=127)
    parser.add_argument("--objective", choices=["regression_l1", "regression"], default="regression_l1")
    parser.add_argument("--include-expiratory", action="store_true",
                        help="Tree ablation: fit on all phases; still score inspiration only")
    parser.add_argument("--drop-group", choices=["none", "leads", "physics"], default="none",
                        help="Tree-only feature ablation, keeping all other settings fixed")
    parser.add_argument("--predict", metavar="CSV")
    args = parser.parse_args()
    if args.drop_group != "none" and args.kind != "tree":
        parser.error("--drop-group is supported only for tree experiments")
    if args.include_expiratory and args.kind != "tree":
        parser.error("--include-expiratory is supported only for tree experiments")
    if args.predict:
        predict(args.predict)
        return
    if not args.name or Path(args.name).name != args.name:
        parser.error("--name must be a simple directory name")
    folder = ARTIFACTS / args.name
    folder.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    x, y, mask, ids, columns = load_arrays(args.profile)
    if args.drop_group != "none":
        drop = ({c for c in columns if "lead" in c} if args.drop_group == "leads"
                else {"area_div_C", "u_in_times_R"})
        keep = [i for i, c in enumerate(columns) if c not in drop]
        columns = [columns[i] for i in keep]
        x = x[:, :, keep]
    tr, va = split_breaths(ids)
    print(f"{args.name}: {len(tr):,} training / {len(va):,} validation breaths, "
          f"{len(columns)} features; inspiratory MAE only", flush=True)
    trainer = train_neural if args.kind == "lstm" else train_tree
    score, iteration, model_file = trainer(args, x, y, mask, columns, tr, va, folder)
    record(args, score, iteration, model_file, folder, ids, va, y, mask,
           columns, x, time.monotonic() - start)


if __name__ == "__main__":
    main()
