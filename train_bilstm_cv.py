"""Deep BiLSTM with physics features and leakage-free grouped five-fold CV.

Full run: .venv/Scripts/python.exe -u train_bilstm_cv.py
Timing:   .venv/Scripts/python.exe -u train_bilstm_cv.py --estimate-only
Check:    .venv/Scripts/python.exe -u train_bilstm_cv.py --smoke-test
No leaderboard upload is performed. A successful full run writes submission.csv.
"""

import argparse
import gc
import hashlib
import inspect
import json
import math
import shutil
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold

from experiments import (ARTIFACTS, DATA, RAW_FEATURES, ROOT, build_network,
                         features, load_arrays, neural_inputs, train_neural)
from train_lstm_ensemble import digest, infer, write_json


def physics_features(base, columns):
    """All cumulative/history operations reset at the start of each breath."""
    u = np.asarray(base[:, :, columns.index("u_in")], dtype="float32")
    dt = np.asarray(base[:, :, columns.index("dt")], dtype="float32")
    extra = {}
    previous = np.concatenate([u[:, :1], u[:, :-1]], axis=1)
    area = np.cumsum((u + previous) * 0.5 * dt, axis=1)
    extra["u_in_time_cumulative_trapezoid"] = area
    rc = np.asarray(base[:, :, columns.index("RC_seconds")])
    extra["u_in_time_area_over_RC"] = area / rc
    for span in (3, 5, 10, 20):
        alpha = 2.0 / (span + 1)
        ema = np.empty_like(u)
        ema[:, 0] = u[:, 0]
        for step in range(1, u.shape[1]):
            ema[:, step] = alpha * u[:, step] + (1 - alpha) * ema[:, step - 1]
        extra[f"u_in_ema_{span}"] = ema
        extra[f"u_in_minus_ema_{span}"] = u - ema
    for lag in (1, 2):
        second = np.zeros_like(u)
        second[:, 2 * lag:] = u[:, 2 * lag:] - 2 * u[:, lag:-lag] + u[:, :-2 * lag]
        extra[f"u_in_second_difference_{lag}"] = second
    slope = np.zeros_like(u)
    slope[:, 1:] = (u[:, 1:] - u[:, :-1]) / np.maximum(dt[:, 1:], 1e-3)
    acceleration = np.zeros_like(u)
    acceleration[:, 2:] = (slope[:, 2:] - slope[:, 1:-1]) / np.maximum(
        (dt[:, 2:] + dt[:, 1:-1]) * 0.5, 1e-3
    )
    extra["u_in_time_second_derivative"] = acceleration
    result = np.concatenate([base] + [value[:, :, None] for value in extra.values()], axis=2)
    if not np.isfinite(result).all():
        raise ValueError("Non-finite physics features")
    return result.astype("float32"), columns + list(extra)


def sequence_features(data):
    ids = data["breath_id"].to_numpy().reshape(-1, 80)
    if not np.all(ids == ids[:, :1]) or len(np.unique(ids[:, 0])) != len(ids):
        raise ValueError("Expected unique, complete, contiguous 80-step breaths")
    base = features(data, "extended")
    x, columns = physics_features(base.to_numpy().reshape(-1, 80, len(base.columns)), list(base.columns))
    return x, columns, ids[:, 0]


def training_arrays():
    base, y, mask, ids, columns = load_arrays("extended")
    cache = ARTIFACTS / "cache_cv_physics"
    cache.mkdir(exist_ok=True)
    signature = hashlib.sha256((inspect.getsource(physics_features) + (
        ARTIFACTS / "cache_extended" / "signature.txt").read_text()).encode()).hexdigest()
    marker = cache / "signature.txt"
    if not marker.exists() or marker.read_text() != signature:
        print("Adding cumulative time-area, EMA, and second-order features...", flush=True)
        values, names = physics_features(base, columns)
        np.save(cache / "X.npy", values)
        write_json(cache / "columns.json", names)
        marker.write_text(signature)
    return (np.load(cache / "X.npy", mmap_mode="r"), y, mask, ids,
            json.loads((cache / "columns.json").read_text()), signature)


def make_splits(ids, seed):
    if len(np.unique(ids)) != len(ids):
        raise ValueError("Sequence array must contain one entry per unique breath_id")
    splitter = GroupKFold(n_splits=5, shuffle=True, random_state=seed)
    splits = list(splitter.split(ids, groups=ids))
    seen = np.zeros(len(ids), dtype=int)
    for tr, va in splits:
        if set(ids[tr]) & set(ids[va]):
            raise ValueError("Training/validation breath leakage")
        seen[va] += 1
    if not np.all(seen == 1):
        raise ValueError("Every breath must be held out exactly once")
    return splits


def estimate_runtime(args, input_size, n_breaths):
    """Short real CUDA benchmark, extrapolated to full train/validation epochs."""
    batch = args.batch_size
    model = build_network(input_size, args.hidden, args.layers).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", init_scale=args.amp_initial_scale)
    x = torch.randn(batch * 2, 80, input_size, device="cuda")
    y = torch.randn(batch * 2, 80, device="cuda") * 20
    mask = torch.arange(80, device="cuda")[None, :].expand(batch * 2, -1) < 30
    index = torch.arange(batch, device="cuda")

    def step():
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            prediction = model(x[index])
            loss = (prediction.float() - y[index]).abs()[mask[index]].mean()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        scaler.step(optimizer)
        scaler.update()
        return float(loss.detach()), int(mask[index].sum())

    model.train()
    for _ in range(3):
        step()
    torch.cuda.synchronize()
    start = time.monotonic()
    for _ in range(10):
        step()
    torch.cuda.synchronize()
    train_seconds = (time.monotonic() - start) / 10
    model.eval()
    start = time.monotonic()
    with torch.no_grad():
        for _ in range(10):
            with torch.autocast("cuda", dtype=torch.float16):
                model(x[index]).float().cpu().numpy()
    torch.cuda.synchronize()
    valid_seconds = (time.monotonic() - start) / 10
    per_epoch = (math.ceil(n_breaths * 0.8 / batch) * train_seconds
                 + math.ceil(n_breaths * 0.2 / batch) * valid_seconds)
    hours = 5 * args.epochs * per_epoch / 3600
    result = dict(gpu=torch.cuda.get_device_name(0), breaths=n_breaths, input_size=input_size,
                  hidden=args.hidden, layers=args.layers, batch_size=batch, max_epochs=args.epochs,
                  train_step_seconds=train_seconds, validation_step_seconds=valid_seconds,
                  estimated_epoch_seconds=per_epoch, estimated_five_fold_hours=hours,
                  estimated_range_hours=[0.8 * hours, 1.3 * hours],
                  note="Excludes feature preparation/checkpoint overhead; early stopping can shorten this. "
                       "Thermals, other GPU work, and machine sleep can extend wall time.")
    del model, optimizer, scaler, x, y, mask
    gc.collect()
    torch.cuda.empty_cache()
    return result


def synthetic_data(n_breaths, first_id=0):
    rng = np.random.default_rng(123 + first_id)
    step = np.tile(np.arange(80), n_breaths)
    u = np.maximum(0, 15 + 10 * np.sin(step / 8) + rng.normal(0, 1, len(step)))
    return pd.DataFrame(dict(id=np.arange(first_id * 80 + 1, (first_id + n_breaths) * 80 + 1),
                             breath_id=np.repeat(np.arange(first_id, first_id + n_breaths), 80),
                             R=np.repeat(np.resize([5, 20, 50], n_breaths), 80),
                             C=np.repeat(np.resize([10, 20, 50], n_breaths), 80),
                             time_step=step / 30, u_in=u, u_out=(step >= 30).astype(int),
                             pressure=5 + u * 0.15 + step * 0.01))


def fit_fold(args, fold, tr, va, x, y, mask, ids, columns, run, config_hash):
    root = run / f"fold_{fold}"
    root.mkdir(exist_ok=True)
    complete = root / "result.json"
    if complete.exists():
        result = json.loads(complete.read_text())
        if result["config_hash"] != config_hash:
            raise ValueError("Completed fold configuration changed; choose a new run name")
        folder = run / result["attempt"]
        np.testing.assert_array_equal(np.load(folder / "validation_breath_ids.npy"), ids[va])
        prediction = np.load(folder / "validation_predictions.npy")
        score = float(np.abs(prediction - y[va])[mask[va]].mean())
        if prediction.shape != y[va].shape or abs(score - result["mae"]) > 1e-6:
            raise ValueError("Cached fold predictions or targets changed")
        print(f"Reusing fold {fold}: MAE {score:.6f}", flush=True)
        return result, prediction
    attempt = 1
    while (root / f"attempt_{attempt:02}").exists():
        attempt += 1
    folder = root / f"attempt_{attempt:02}"
    folder.mkdir()
    settings = SimpleNamespace(**vars(args))
    settings.seed = args.seed + fold - 1
    settings.profile = "cv_physics"
    print(f"Fold {fold}/5: {len(tr):,} training / {len(va):,} validation breaths", flush=True)
    started = time.monotonic()
    score, epoch, model_file = train_neural(settings, x, y, mask, columns, tr, va, folder)
    np.save(folder / "validation_breath_ids.npy", ids[va])
    np.save(folder / "training_breath_ids.npy", ids[tr])
    result = dict(fold=fold, seed=settings.seed, mae=score, best_epoch=epoch,
                  config_hash=config_hash, seconds=time.monotonic() - started,
                  model=(folder / model_file).relative_to(run).as_posix(),
                  attempt=folder.relative_to(run).as_posix())
    write_json(complete, result)
    gc.collect()
    torch.cuda.empty_cache()
    return result, np.load(folder / "validation_predictions.npy")


def save_submission(args, results, run, columns, test, template, original_hash):
    x, test_columns, _ = sequence_features(test)
    if test_columns != columns:
        raise ValueError("Train/test feature columns differ")
    encoded = neural_inputs(x, columns)
    predictions = np.zeros(len(test), dtype="float64")
    for result in results:
        checkpoint = torch.load(run / result["model"], map_location="cpu", weights_only=False)
        if checkpoint["columns"] != columns:
            raise ValueError("Checkpoint feature schema mismatch")
        predictions += infer(checkpoint, encoded, args.batch_size).ravel().astype("float64") / 5
        torch.cuda.empty_cache()
    submission = template.copy()
    if (list(submission.columns) != ["id", "pressure"] or len(test) != len(submission)
            or not test["id"].is_unique or not submission["id"].is_unique
            or set(test["id"]) != set(submission["id"])):
        raise ValueError("Test IDs and sample submission must match uniquely")
    submission["pressure"] = submission["id"].map(pd.Series(predictions, index=test["id"]))
    if not np.isfinite(submission["pressure"]).all():
        raise ValueError("Invalid final predictions")
    candidate = run / "submission_candidate.csv"
    submission.to_csv(candidate, index=False)
    if (args.output.exists() and digest(args.output) != original_hash):
        print(f"Output changed during training; keeping it and saving candidate to {candidate}", flush=True)
        return False
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and not (run / "previous_submission.csv").exists():
        shutil.copy2(args.output, run / "previous_submission.csv")
    with tempfile.NamedTemporaryFile(dir=args.output.parent, prefix=".cv_submission_", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        shutil.copyfile(candidate, temporary)
        temporary.replace(args.output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Saved {len(submission):,} five-fold averaged test predictions to {args.output}", flush=True)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="bilstm_5fold")
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--amp-initial-scale", type=float, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=ROOT / "submission.csv")
    parser.add_argument("--estimate-only", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    if (not args.run_name or Path(args.run_name).name != args.run_name
            or min(args.hidden, args.layers, args.epochs, args.batch_size, args.patience) < 1
            or args.lr <= 0 or args.amp_initial_scale <= 0 or not 0 <= args.seed < 2**32 - 5):
        parser.error("Use a simple run name, positive training settings, and a valid seed")
    if args.output.resolve() == (DATA / "sample_submission.csv").resolve():
        parser.error("Cannot overwrite the sample submission template")
    if not torch.cuda.is_available():
        raise RuntimeError("Use the project .venv with CUDA PyTorch")
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    if args.estimate_only:
        base, _, _, ids, names = load_arrays("extended")
        sample, names = physics_features(base[:2], names)
        estimate = estimate_runtime(args, neural_inputs(sample, names).shape[-1], len(ids))
        print(json.dumps(estimate, indent=2), flush=True)
        ARTIFACTS.mkdir(exist_ok=True)
        write_json(ARTIFACTS / f"{args.run_name}_estimate.json", estimate)
        return
    if args.smoke_test:
        args.hidden, args.layers, args.epochs, args.batch_size, args.patience = 8, 4, 2, 4, 2
        args.amp_initial_scale = 32
        data, test = synthetic_data(10), synthetic_data(3, 100)
        x, columns, ids = sequence_features(data)
        y = data["pressure"].to_numpy(dtype="float32").reshape(-1, 80)
        mask = data["u_out"].to_numpy().reshape(-1, 80) == 0
        feature_signature = hashlib.sha256(inspect.getsource(physics_features).encode()).hexdigest()
        train_hash = "synthetic-smoke-test"
        run = ARTIFACTS / (args.run_name + "_smoke")
        args.output = run / "submission.csv"
        template = pd.DataFrame(dict(id=test["id"].to_numpy()[::-1], pressure=0))
    else:
        x, y, mask, ids, columns, feature_signature = training_arrays()
        train_hash = digest(DATA / "train.csv")
        run = ARTIFACTS / args.run_name
        test, template = None, None
    run.mkdir(parents=True, exist_ok=True)
    config = dict(hidden=args.hidden, layers=args.layers, epochs=args.epochs, batch_size=args.batch_size,
                  lr=args.lr, patience=args.patience, amp_initial_scale=args.amp_initial_scale,
                  seed=args.seed, folds=5,
                  train_sha256=train_hash, feature_signature=feature_signature, columns=columns)
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    config_path = run / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Run configuration changed; choose another --run-name")
    write_json(config_path, config)
    original_hash = digest(args.output) if args.output.exists() else None
    oof = np.full(y.shape, np.nan, dtype="float32")
    counts = np.zeros(len(ids), dtype=int)
    results = []
    for fold, (tr, va) in enumerate(make_splits(ids, args.seed), start=1):
        result, prediction = fit_fold(args, fold, tr, va, x, y, mask, ids, columns, run, config_hash)
        oof[va] = prediction
        counts[va] += 1
        results.append(result)
        print(f"Fold {fold} inspiratory MAE: {result['mae']:.6f}", flush=True)
        pd.DataFrame(results).to_csv(run / "fold_scores.csv", index=False)
    if not np.all(counts == 1) or not np.isfinite(oof).all():
        raise ValueError("Missing/duplicate out-of-fold predictions")
    score = float(np.abs(oof - y)[mask].mean(dtype=np.float64))
    fold_scores = np.array([result["mae"] for result in results])
    np.save(run / "oof_predictions.npy", oof)
    np.save(run / "breath_ids.npy", ids)
    np.save(run / "oof_counts.npy", counts)
    summary = dict(pooled_oof_inspiratory_mae=score, mean_fold_mae=float(fold_scores.mean()),
                   std_fold_mae=float(fold_scores.std()), fold_scores=fold_scores.tolist(),
                   folds=5, breaths=len(ids), scored_rows=int(mask.sum()), smoke_test=args.smoke_test,
                   note="OOF predictions are held-out only; only test predictions are averaged across five models.")
    print(f"CV inspiratory MAE (pooled OOF): {score:.6f}", flush=True)
    print(f"Fold mean +/- std: {fold_scores.mean():.6f} +/- {fold_scores.std():.6f}", flush=True)
    write_json(run / "ensemble.json", dict(config=config, members=results, weights=[0.2] * 5))
    if test is None:
        test = pd.read_csv(DATA / "test.csv", dtype=dict.fromkeys(RAW_FEATURES, "float32"))
        template = pd.read_csv(DATA / "sample_submission.csv")
    summary["submission_updated"] = save_submission(args, results, run, columns, test, template, original_hash)
    write_json(run / "summary.json", summary)
    print("Finished; no leaderboard upload performed", flush=True)


if __name__ == "__main__":
    main()
