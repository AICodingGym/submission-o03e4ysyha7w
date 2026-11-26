"""Run staged, deadline-bounded experiments while keeping a verified incumbent CSV."""
import argparse
import csv
from dataclasses import dataclass, replace
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from dataset import CLASSES, DatasetArchive, decode

OUTPUT = Path("outputs")
ROOT = OUTPUT / "architecture"
LOG = OUTPUT / "experiment_summary.csv"
STATE = ROOT / "timed_best.json"
CANDIDATE = "resnet34_unet"
FIELDS = ("change", "resolution", "epochs", "training_seconds", *CLASSES, "macro_dice",
          "leaderboard_score", "hausdorff", "accepted", "run")


@dataclass(frozen=True)
class Candidate:
    name: str
    resolution: int = 256
    context: int = 3
    members: tuple = (f"{CANDIDATE}:{CANDIDATE}",)
    flip: bool = True
    objective: str = "bce"


def read_metrics(candidate):
    return json.loads((ROOT / f"{candidate.name}_assessment.json").read_text())["tuned_cleanup"]


def write_state(candidate, metrics):
    path = STATE.with_suffix(".tmp")
    path.write_text(json.dumps(dict(candidate=candidate.__dict__, metrics=metrics), indent=2))
    path.replace(STATE)


def audit(path):
    data = DatasetArchive()
    with path.open(newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames != ["id", "class", "predicted"]:
            raise ValueError("CSV header differs from sample_submission.csv")
        rows = list(reader)
    keys = [(row["id"], row["class"]) for row in rows]
    if (any(not key for key, _ in keys) or len(set(keys)) != len(keys) or
            keys != [(row["id"], row["class"]) for row in data.sample]):
        raise ValueError("CSV keys differ from the submission template")
    for row in rows:
        decode(row["predicted"], data.shapes[row["id"]])
    print(f"Verified {len(rows)} rows, SHA256 {hashlib.sha256(path.read_bytes()).hexdigest()}",
          flush=True)


def git(*args):
    subprocess.run(["git", "-c", f"safe.directory={Path.cwd().as_posix()}", *args], check=True)


def promote(candidate, metrics):
    source = ROOT / f"{candidate.name}_submission.csv"
    audit(source)
    canonical = OUTPUT / "submission.csv"
    data = source.read_bytes()
    if canonical.read_bytes() != data:
        temporary = OUTPUT / "submission.tmp"
        temporary.write_bytes(data)
        temporary.replace(canonical)
        git("add", "-f", "--", str(canonical))
        git("commit", "--only", "-m", f"Save validated submission {candidate.name}", "--", str(canonical))
    write_state(candidate, metrics)


def record(change, candidate, metrics=None, trained=False, accepted=False, leaderboard=""):
    epochs = ROOT / f"{candidate.name}_epochs.jsonl"
    rows = [json.loads(x) for x in epochs.read_text().splitlines()] if trained and epochs.exists() else []
    columns = []
    if LOG.exists():
        with LOG.open(newline="") as file:
            reader = csv.DictReader(file)
            columns = list(reader.fieldnames or ())
            previous = list(reader)
    else:
        previous = []
    columns += [field for field in FIELDS if field not in columns]
    row = dict(change=change, experiment=candidate.name, run=candidate.name, resolution=candidate.resolution,
               epochs=len(rows), training_seconds=round(sum(x["train_seconds"] for x in rows), 2),
               accepted=accepted, leaderboard_score=leaderboard)
    if metrics is not None:
        row.update(macro_dice=metrics["macro_dice"], hausdorff=metrics["mean_normalized_hausdorff"])
        row.update({organ: metrics["organs"][organ]["dice"] for organ in CLASSES})
    temporary = LOG.with_suffix(".tmp")
    with temporary.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(previous)
        writer.writerow(row)
    temporary.replace(LOG)


def call(candidate, deadline, script, *args):
    if deadline <= time.time():
        return False
    env = dict(os.environ, GI_SEGMENTATION_SIZE=str(candidate.resolution),
               GI_SEGMENTATION_CONTEXT=str(candidate.context))
    cmd = [sys.executable, "-u", script, *map(str, args)]
    print("Launch", candidate.name, " ".join(cmd[3:]), flush=True)
    try:
        result = subprocess.run(cmd, env=env, timeout=max(1, deadline - time.time()))
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"Stopped {candidate.name} at its deadline", flush=True)
        return False


def run(start):
    stop = start + 75 * 60
    candidate = Candidate(CANDIDATE)
    metrics = read_metrics(candidate)
    if not (ROOT / f"{candidate.name}_submission.csv").exists():
        raise RuntimeError("The pretrained baseline export is missing")
    promote(candidate, metrics)
    record("Reuse completed pretrained 256-pixel U-Net", candidate, metrics, accepted=True)
    while time.time() < start + 15 * 60:
        time.sleep(min(10, start + 15 * 60 - time.time()))
    previous = []
    anchor = str(int(start))
    for step, label in (("resolution", "320-pixel continuation"),
                        ("loss", "Tversky instead of Dice under BCE"),
                        ("context", "Five adjacent slices"),
                        ("no_flip", "Flip TTA ablation"),
                        ("threshold", "Organ-specific threshold sweep"),
                        ("largest", "Largest stomach component"),
                        ("ensemble", "Average compatible checkpoints")):
        remaining = stop - time.time()
        reserve = {"resolution": 14, "loss": 12, "context": 10, "no_flip": 5,
                   "threshold": 8, "largest": 5, "ensemble": 6}[step] * 60
        if remaining < reserve:
            print(f"Skipped {step}: {remaining / 60:.1f} minutes to cutoff", flush=True)
            continue
        name = f"timed_{anchor}_{step}"
        proposal = replace(candidate, name=name)
        if step == "resolution":
            proposal = replace(proposal, resolution=320)
        if step == "loss":
            proposal = replace(proposal, objective="tversky")
        if step == "context":
            proposal = replace(proposal, context=5)
        if step == "no_flip":
            proposal = replace(proposal, flip=False)
        trained = step in ("resolution", "loss", "context")
        if trained:
            checkpoint = candidate.members[0].split(":")[1]
            proposal = replace(proposal, members=(f"{CANDIDATE}:{name}",))
            epochs = 2 if step != "context" else 1
            budget = min(18, max(6, (remaining - 7 * 60) / 60))
            local_deadline = min(stop - 4 * 60, time.time() + budget * 60)
            ok = call(proposal, local_deadline + 15, "architecture_trials.py", "train",
                      "--model", CANDIDATE, "--run", name, "--initialize", checkpoint,
                      "--objective", proposal.objective, "--epochs", epochs, "--lr", 0.00006,
                      "--warmup", 0.12, "--budget", budget, "--deadline", local_deadline,
                      "--save-each-epoch", "--stop-below", metrics["macro_dice"] - 0.022)
            if not (ROOT / f"{name}.pt").exists():
                record(f"{label} (no complete epoch)", proposal, trained=True)
                continue
        if step == "ensemble":
            compatible = [p for p in previous if (p.resolution, p.context) ==
                          (candidate.resolution, candidate.context) and p.members != candidate.members]
            if (candidate.resolution, candidate.context) == (256, 3) and candidate.members != Candidate(CANDIDATE).members:
                compatible.append(Candidate(CANDIDATE))
            if not compatible:
                print("No second compatible checkpoint for ensemble", flush=True)
                continue
            proposal = replace(proposal, members=(*candidate.members, *compatible[-1].members))
        settings = [metrics["organs"][organ] for organ in CLASSES]
        threshold_args = [r["threshold"] for r in settings]
        minimum_args = [round(r["minimum"] * (proposal.resolution / candidate.resolution) ** 2)
                        if r["minimum"] > 0 else r["minimum"] for r in settings]
        options = ["--thresholds", *threshold_args, "--minimum", *minimum_args]
        if step != "threshold":
            options.append("--quick")
        if step == "largest":
            options.append("--largest")
        if not proposal.flip:
            options.append("--no-flip")
        if not call(proposal, stop - 140, "architecture_metrics.py", "evaluate", "--name", name,
                    "--models", *proposal.members, *options):
            record(f"{label} (assessment incomplete)", proposal, trained=trained)
            continue
        measured = read_metrics(proposal)
        improves = measured["macro_dice"] > metrics["macro_dice"]
        print(label, measured["macro_dice"], measured["mean_normalized_hausdorff"],
              "improves" if improves else "rejected", flush=True)
        accepted = improves and call(proposal, stop, "architecture_metrics.py", "export", "--name",
                                     name, "--models", *proposal.members, "--dice-only")
        record(label, proposal, measured, trained, accepted)
        if accepted:
            promote(proposal, measured)
            previous.append(candidate)
            candidate, metrics = proposal, measured
        elif trained:
            previous.append(proposal)
    print("Experiments closed. Best", candidate.name, metrics["macro_dice"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True)
    args = parser.parse_args()
    run(datetime.fromisoformat(args.start).timestamp())


if __name__ == "__main__":
    main()
