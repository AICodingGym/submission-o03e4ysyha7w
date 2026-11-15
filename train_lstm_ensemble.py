"""Train three seeded LSTMs, validate their equal-weight mean, conditionally submit CSV.

Run: .venv/Scripts/python.exe -u train_lstm_ensemble.py
Uses the original seed-42, 20% whole-breath split regardless of model seed.
Compatible completed models are reused by default; --no-reuse-existing trains
three fresh models. Completed members are also reused after an interrupted run.
No leaderboard upload is performed.
"""

import argparse
import gc
import hashlib
import inspect
import json
import shutil
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from experiments import (ARTIFACTS, DATA, RAW_FEATURES, ROOT, TRACK, build_network,
                         features, load_arrays, make_features, neural_inputs,
                         record, split_breaths, train_neural)


TRAINING_KEYS = ("profile", "hidden", "layers", "epochs", "batch_size", "lr", "patience")


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def infer(checkpoint, encoded, batch_size):
    model = build_network(checkpoint["input_size"], checkpoint["hidden"], checkpoint["layers"]).cuda()
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(encoded), batch_size):
            value = np.clip((encoded[start:start + batch_size] - checkpoint["mean"])
                            / checkpoint["std"], -20, 20)
            with torch.autocast("cuda", dtype=torch.float16):
                prediction = model(torch.from_numpy(value).cuda())
            predictions.append(prediction.float().cpu().numpy())
    return np.concatenate(predictions)


def validate_member(result, args, seed, ids, va, y, mask, x, columns):
    configuration = result["args"]
    if result["kind"] != "lstm" or int(result["seed"]) != seed:
        raise ValueError("Wrong model kind or seed")
    if any(configuration.get(key) != getattr(args, key) for key in TRAINING_KEYS):
        raise ValueError("Training configuration differs")
    folder = (ROOT / result["model"]).parent
    np.testing.assert_array_equal(np.load(folder / "validation_breath_ids.npy"), ids[va])
    prediction = np.load(folder / "validation_predictions.npy")
    if prediction.shape != y[va].shape or not np.isfinite(prediction).all():
        raise ValueError("Invalid saved validation predictions")
    score = float(np.abs(prediction - y[va])[mask[va]].mean())
    if abs(score - result["validation_mae"]) > 1e-6:
        raise ValueError("Saved validation score does not match current data")
    checkpoint = torch.load(ROOT / result["model"], map_location="cpu", weights_only=False)
    if checkpoint["columns"] != columns or checkpoint["seed"] != seed:
        raise ValueError("Checkpoint columns or seed differ")
    sample = va[:args.batch_size]
    replay = infer(checkpoint, neural_inputs(np.asarray(x[sample]), columns), args.batch_size)
    difference = np.abs(replay - prediction[:len(sample)])[mask[sample]]
    if difference.mean() > 0.005 or difference.max() > 0.15:
        raise ValueError("Checkpoint replay failed")
    return score, prediction


def get_member(args, seed, run, x, y, mask, ids, columns, tr, va):
    candidates = list(run.glob(f"seed_{seed}/attempt_*/result.json"))
    if args.reuse_existing and (TRACK / "results.csv").exists():
        completed = pd.read_csv(TRACK / "results.csv")
        for row in completed.loc[completed["kind"] == "lstm"].to_dict("records"):
            candidates.append((ROOT / row["model"]).parent / "result.json")
    for path in dict.fromkeys(candidates):
        result = json.loads(path.read_text())
        if int(result.get("seed", -1)) != seed:
            continue
        if any(result["args"].get(key) != getattr(args, key) for key in TRAINING_KEYS):
            continue
        try:
            score, prediction = validate_member(result, args, seed, ids, va, y, mask, x, columns)
        except (ValueError, AssertionError, FileNotFoundError) as error:
            print(f"Cannot reuse {path.parent.name}: {error}", flush=True)
            continue
        print(f"Reusing seed {seed}: MAE {score:.6f}, {path.parent}", flush=True)
        return result, prediction

    seed_folder = run / f"seed_{seed}"
    seed_folder.mkdir(exist_ok=True)
    attempt = 1
    while (seed_folder / f"attempt_{attempt:02}").exists():
        attempt += 1
    folder = seed_folder / f"attempt_{attempt:02}"
    folder.mkdir()
    member_args = SimpleNamespace(**{key: str(value) if isinstance(value, Path) else value
                                    for key, value in vars(args).items()})
    member_args.name = f"{args.run_name}_seed_{seed}"
    member_args.seed, member_args.kind = seed, "lstm"
    print(f"Training seed {seed} in {folder}...", flush=True)
    start = time.monotonic()
    score, epoch, filename = train_neural(member_args, x, y, mask, columns, tr, va, folder)
    record(member_args, score, epoch, filename, folder, ids, va, y, mask,
           columns, x, time.monotonic() - start, promote=False)
    result = json.loads((folder / "result.json").read_text())
    _, prediction = validate_member(result, args, seed, ids, va, y, mask, x, columns)
    gc.collect()
    torch.cuda.empty_cache()
    return result, prediction


def write_submission(args, members, run, original_hash):
    data = pd.read_csv(DATA / "test.csv", dtype=dict.fromkeys(RAW_FEATURES, "float32"))
    breath_ids = data["breath_id"].to_numpy().reshape(-1, 80)
    if not np.all(breath_ids == breath_ids[:, :1]) or len(np.unique(breath_ids[:, 0])) != len(breath_ids):
        raise ValueError("Test must contain complete contiguous 80-step breaths")
    x = features(data, args.profile)
    encoded = neural_inputs(x.to_numpy().reshape(-1, 80, len(x.columns)), list(x.columns))
    averaged = np.zeros(len(data), dtype="float64")
    for member in members:
        checkpoint = torch.load(ROOT / member["model"], map_location="cpu", weights_only=False)
        if checkpoint["columns"] != list(x.columns):
            raise ValueError("Test feature columns differ from checkpoint")
        averaged += infer(checkpoint, encoded, args.batch_size).ravel() / len(members)
        torch.cuda.empty_cache()
    submission = pd.read_csv(DATA / "sample_submission.csv")
    if (list(submission.columns) != ["id", "pressure"] or len(submission) != len(data)
            or not data["id"].is_unique or not submission["id"].is_unique
            or set(submission["id"]) != set(data["id"])):
        raise ValueError("Sample submission/test IDs or columns differ")
    submission["pressure"] = submission["id"].map(pd.Series(averaged, index=data["id"]))
    if not np.isfinite(submission["pressure"]).all():
        raise ValueError("Non-finite submission predictions")
    candidate = run / "submission_candidate.csv"
    submission.to_csv(candidate, index=False)
    if args.output.exists() and digest(args.output) != original_hash:
        print(f"{args.output} changed during training; preserving it. Candidate: {candidate}", flush=True)
        return False
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and not (run / "previous_submission.csv").exists():
        shutil.copy2(args.output, run / "previous_submission.csv")
    with tempfile.NamedTemporaryFile(dir=args.output.parent, prefix=".ensemble_", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        shutil.copyfile(candidate, temporary)
        if digest(temporary) != digest(candidate):
            raise ValueError("Submission copy verification failed")
        temporary.replace(args.output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Saved {len(submission):,} predictions to {args.output}; no leaderboard upload", flush=True)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs=3, default=[42, 43, 44])
    parser.add_argument("--run-name", default="lstm_3seed")
    parser.add_argument("--profile", choices=["base", "extended"], default="extended")
    parser.add_argument("--hidden", type=int, default=192)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--reuse-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--baseline-mae", type=float)
    parser.add_argument("--output", type=Path, default=ROOT / "submission.csv")
    args = parser.parse_args()
    if len(set(args.seeds)) != 3 or any(not 0 <= seed < 2**32 for seed in args.seeds):
        parser.error("Provide three distinct seeds between 0 and 2**32-1")
    if (Path(args.run_name).name != args.run_name or not args.run_name
            or min(args.hidden, args.layers, args.epochs, args.batch_size, args.patience) < 1
            or args.lr <= 0):
        parser.error("Use a simple run name and positive training settings")
    if args.output.resolve() == (DATA / "sample_submission.csv").resolve():
        parser.error("The sample submission template cannot be overwritten")
    if not torch.cuda.is_available():
        raise RuntimeError("Use the CUDA PyTorch environment documented in EXPERIMENTS.md")
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    x, y, mask, ids, columns = load_arrays(args.profile)
    tr, va = split_breaths(ids)
    split_hash = hashlib.sha256(np.sort(ids[va]).tobytes()).hexdigest()
    baseline = json.loads((TRACK / "best.json").read_text())
    if args.baseline_mae is None and baseline.get("split_sha256") != split_hash:
        raise ValueError("Best recorded score uses a different split; set --baseline-mae explicitly")
    baseline_score = args.baseline_mae if args.baseline_mae is not None else baseline["validation_mae"]
    if not np.isfinite(baseline_score) or baseline_score <= 0:
        parser.error("Baseline MAE must be positive and finite")
    run = ARTIFACTS / args.run_name
    run.mkdir(parents=True, exist_ok=True)
    config = {key: getattr(args, key) for key in TRAINING_KEYS}
    config.update(seeds=args.seeds, split_sha256=split_hash, train_sha256=digest(DATA / "train.csv"),
                  feature_sha256=hashlib.sha256((inspect.getsource(features)
                                                + inspect.getsource(make_features)).encode()).hexdigest())
    config_path = run / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Run configuration changed; choose a different --run-name")
    write_json(config_path, config)
    if (run / "summary.json").exists():
        print("Run already completed:", (run / "summary.json").read_text(), flush=True)
        return
    original_hash = digest(args.output) if args.output.exists() else None
    print(f"Fixed holdout: {len(tr):,} training / {len(va):,} validation breaths; "
          f"baseline {baseline_score:.9f}; seeds {args.seeds}", flush=True)
    members, predictions = [], []
    for seed in args.seeds:
        result, prediction = get_member(args, seed, run, x, y, mask, ids, columns, tr, va)
        members.append({key: value for key, value in result.items() if key != "args"})
        predictions.append(prediction)
        torch.cuda.empty_cache()
    averaged = np.mean(np.stack(predictions), axis=0, dtype=np.float64)
    score = float(np.abs(averaged - y[va])[mask[va]].mean())
    np.save(run / "validation_predictions.npy", averaged)
    descriptor = dict(members=[dict(weight=1 / len(members), run=member) for member in members])
    write_json(run / "ensemble.json", descriptor)
    provenance = SimpleNamespace(**config, name=args.run_name, kind="ensemble",
                                 seed=",".join(map(str, args.seeds)))
    if not (run / "result.json").exists():
        record(provenance, score, 0, "ensemble.json", run, ids, va, y, mask, columns, x, 0,
               promote=False)
    latest = json.loads((TRACK / "best.json").read_text())
    if latest.get("split_sha256") == split_hash:
        baseline_score = min(baseline_score, latest["validation_mae"])
    improved = score < baseline_score
    print(f"Equal-weight ensemble inspiratory MAE: {score:.9f}; below 0.20: {score < 0.20}; "
          f"beats best {baseline_score:.9f}: {improved}", flush=True)
    updated = False
    if improved:
        updated = write_submission(args, members, run, original_hash)
        write_json(TRACK / "best.json", json.loads((run / "result.json").read_text()))
    else:
        print("No improvement; submission.csv and best model preserved", flush=True)
    write_json(run / "summary.json", dict(
        seeds=args.seeds, individual_mae=[member["validation_mae"] for member in members],
        ensemble_mae=score, baseline_mae=baseline_score, below_020=score < 0.20,
        improved=improved, submission_updated=updated, output=str(args.output),
        model=str(run / "ensemble.json"), split_sha256=split_hash,
    ))


if __name__ == "__main__":
    main()
